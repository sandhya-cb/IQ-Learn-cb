import sys
import os
import torch
import cv2
import numpy as np
import omni.timeline
import omni.kit.app
import carb
import asyncio
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

# --- CONFIGURATION ---
PROJECT_ROOT = "/home/clutterbot/lerobot/IQ-Learn/iq_learn"
MODEL_PATH = "/home/clutterbot/lerobot/IQ-Learn/iq_learn/outputs/policy_final.pth"

# Must match Training exactly!
ACTIVE_JOINT_NAMES = ["Left_arm", "Right_arm", "Left_palm", "Right_palm"]
CAMERA_PRIM_PATH = "/Dusty_beta/base_link/cam0/Camera"

# ---------------------

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import agent.sac
    import importlib
    importlib.reload(agent.sac)
    print("♻️  Reloaded Agent Code")
except: pass

from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.sensor import Camera
from agent.sac import SAC
from pxr import UsdPhysics

DEFAULT_ACTOR_CFG = {'_target_': 'agent.sac_models.DiagGaussianActor', 'hidden_dim': 256, 'hidden_depth': 2, 'log_std_bounds': [-5, 2]}
DEFAULT_CRITIC_CFG = {'_target_': 'agent.sac_models.DoubleQCritic', 'hidden_dim': 256, 'hidden_depth': 2}

class RobotController:
    def __init__(self):
        self.agent = None
        self.robot = None
        self.camera = None
        self.active_indices = []
        self.device = torch.device("cuda")
        self.sub = None 

    async def setup(self):
        print("🚀 Initializing...")
        
        # --- 1. Load Agent ---
        # CRITICAL: This must match your training config EXACTLY.
        # If you trained with state=[4], this must be 4.
        STATE_DIM = 4 
        OBS_DIM = {
            'image': (3, 84, 84),
            'state': (STATE_DIM,) 
        }
        ACTION_DIM = 4
        
        actor_cfg = DEFAULT_ACTOR_CFG.copy()
        actor_cfg['obs_dim'] = OBS_DIM; actor_cfg['action_dim'] = ACTION_DIM
        critic_cfg = DEFAULT_CRITIC_CFG.copy()
        critic_cfg['obs_dim'] = OBS_DIM; critic_cfg['action_dim'] = ACTION_DIM

        self.agent = SAC(
            obs_dim=OBS_DIM, action_dim=ACTION_DIM, action_range=1.0, batch_size=1, device=self.device,
            actor_cfg=actor_cfg, critic_cfg=critic_cfg, actor_lr=0, critic_lr=0, init_temp=0.1, learn_temp=False
        )

        # 2. Load Weights
        if os.path.exists(MODEL_PATH):
            ckpt = torch.load(MODEL_PATH, map_location=self.device)
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            
            # Strict=False helps skip missing keys if slightly mismatched, 
            # but usually you want Strict=True to ensure correctness.
            try:
                self.agent.actor.load_state_dict(state_dict, strict=True)
                self.agent.actor.eval()
                print("✅ Weights loaded successfully.")
            except Exception as e:
                carb.log_error(f"Weights Load Error: {e}")
                return
        else:
            carb.log_error(f"Model not found at {MODEL_PATH}")
            return

        # 3. Setup World
        World.clear_instance()
        self.world = World(stage_units_in_meters=1.0)
        
        # 4. Find Robot
        stage = omni.usd.get_context().get_stage()
        articulations = [p.GetPath().pathString for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if not articulations:
            carb.log_error("No Articulation Root found!")
            return
        
        try:
            # Warmup
            omni.timeline.get_timeline_interface().play()
            for _ in range(5): await omni.kit.app.get_app().next_update_async()

            self.robot = self.world.scene.add(Robot(prim_path=articulations[0], name="dusty"))
            self.camera = Camera(prim_path=CAMERA_PRIM_PATH, resolution=(84, 84))
            self.camera.initialize()
            self.robot.initialize()
            
            # Map Joints
            all_joints = self.robot.dof_names
            print(f"📋 Available joints: {all_joints}")
            self.active_indices = []
            for name in ACTIVE_JOINT_NAMES:
                if name in all_joints:
                    self.active_indices.append(all_joints.index(name))
            print(f"✅ Mapped indices: {self.active_indices}")
            
        except Exception as e:
            carb.log_error(f"Setup Error: {e}")
            return

        # 5. Register Loop
        self.sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self.on_update
        )
        print("✅ Controller Registered.")

    def on_update(self, event):
        if not self.world.is_playing(): return

        try:
            # --- A. Get Image ---
            raw_data = self.camera.get_rgba()
            if raw_data is None or raw_data.size == 0: return

            if raw_data.ndim == 1: raw_data = raw_data.reshape((84, 84, 4))
            rgb = raw_data[:, :, :3]
            if rgb.shape[0] != 84: rgb = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_AREA)
            
            # (H, W, C) -> (C, H, W)
            obs_numpy_img = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)

            # --- B. Get State ---
            # 1. Get full pose (9 joints)
            current_full_pose = self.robot.get_joint_positions()
            
            # 2. Extract ONLY the active joints (4 joints)
            active_state = [current_full_pose[i] for i in self.active_indices]
            obs_numpy_state = np.array(active_state, dtype=np.float32)

            # --- C. Inference (FIXED) ---
            # We convert to Tensors MANUALLY here to avoid the "got dict" error.
            # agent.choose_action() fails because it tries to convert the whole dict at once.
            
            obs_tensor_dict = {}
            
            # Image: (C, H, W) -> Unsqueeze Batch -> (1, C, H, W) -> Float -> GPU
            # Note: We don't divide by 255 here because PixelEncoder handles that internally.
            obs_tensor_dict['image'] = torch.as_tensor(obs_numpy_img, device=self.device).float().unsqueeze(0)
            
            # State: (Dim,) -> Unsqueeze Batch -> (1, Dim) -> Float -> GPU
            obs_tensor_dict['state'] = torch.as_tensor(obs_numpy_state, device=self.device).float().unsqueeze(0)

            with torch.no_grad():
                # Call the ACTOR directly (Bypassing agent.choose_action)
                # The actor returns a Distribution object
                dist = self.agent.actor(obs_tensor_dict)
                
                # We want the deterministic mean for evaluation
                action_tensor = dist.mean
                
                # Convert back to numpy: (1, 4) -> (4,)
                action = action_tensor.cpu().numpy()[0]

            # --- D. Apply Action ---
            target_arm_positions = action 
            
            # Update the full pose vector
            for i, robot_idx in enumerate(self.active_indices):
                current_full_pose[robot_idx] = target_arm_positions[i]
            
            self.robot.set_joint_positions(current_full_pose)
            
        except Exception as e:
            print(f"Update Error: {e}")
            pass
        
    def stop(self):
        if self.sub:
            self.sub = None
            print("🛑 Controller Stopped.")

if 'global_controller' in globals():
    try: globals()['global_controller'].stop()
    except: pass
    del globals()['global_controller'] 

globals()['global_controller'] = RobotController()
asyncio.ensure_future(globals()['global_controller'].setup())