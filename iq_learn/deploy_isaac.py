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
MODEL_PATH = "/home/clutterbot/lerobot/IQ-Learn/iq_learn/outputs/2025-11-27/14-19-00/policy_final.pth"

ACTIVE_JOINT_NAMES = ["Left_arm", "Right_arm", "Left_palm", "Right_palm"]
CAMERA_PRIM_PATH = "/Dusty_beta/base_link/cam0/Camera_01"
# ---------------------

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Force Reload to clear cached errors
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

# ---------------------------------------------------------
# 2. HARDCODED NETWORK CONFIGURATION
# ---------------------------------------------------------
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
        
        # 1. Load Agent
        OBS_DIM = (3, 84, 84)
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
            self.agent.actor.load_state_dict(state_dict)
            self.agent.actor.eval()
            print("✅ Weights loaded.")
        else:
            carb.log_error(f"Model not found at {MODEL_PATH}")
            return

        # 3. Setup World
        World.clear_instance()
        self.world = World(stage_units_in_meters=1.0)
        
        # 4. Find Robot (Auto-detect)
        stage = omni.usd.get_context().get_stage()
        articulations = [p.GetPath().pathString for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        if not articulations:
            carb.log_error("No Articulation Root found!")
            return
        
        try:
            # WARMUP PHYSICS before wrapping
            omni.timeline.get_timeline_interface().play()
            for _ in range(5): await omni.kit.app.get_app().next_update_async()

            self.robot = self.world.scene.add(Robot(prim_path=articulations[0], name="dusty"))
            
            # cam_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
            # if cam_prim:
            #     geom_cam = UsdGeom.Camera(cam_prim)
            #     h_attr = geom_cam.GetHorizontalApertureAttr()
            #     v_attr = geom_cam.GetVerticalApertureAttr()
                
            #     if h_attr.IsValid() and v_attr.IsValid():
            #         width = h_attr.Get()
            #         # Force Height == Width (Square Sensor)
            #         v_attr.Set(width)
            #         print(f"📷 Corrected Camera Sensor to Square: {width}x{width}")
            # # ------------------------------------
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

        # 5. REGISTER UPDATE LOOP
        self.sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self.on_update
        )
        print("✅ Controller Registered. Moving Robot...")

    def on_update(self, event):
        if not self.world.is_playing(): return

        try:
            # A. Get Obs
            print("A")
            raw_data = self.camera.get_rgba()
            if raw_data is None or raw_data.size == 0: return

            if raw_data.ndim == 1:
                raw_data = raw_data.reshape((84, 84, 4))

            rgb = raw_data[:, :, :3]
            if rgb.shape[0] != 84:
                rgb = cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_AREA)
            
            obs_numpy = np.transpose(rgb, (2, 0, 1)).astype(np.uint8)
            if np.random.rand() < 0.05:
                # 1. Reverse the Transpose: (3, 84, 84) -> (84, 84, 3)
                debug_img = np.transpose(obs_numpy, (1, 2, 0))
                
                # 2. Convert RGB to BGR (OpenCV uses BGR)
                debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
                
                # 3. Blow it up so you can see it easily (84x84 is tiny)
                debug_img_large = cv2.resize(debug_img, (256, 256), interpolation=cv2.INTER_NEAREST)
                
                # 4. Save to your home folder
                save_path = os.path.join(os.path.expanduser("~"), "agent_view.png")
                cv2.imwrite(save_path, debug_img_large)
                print(f"📸 Saved agent view to: {save_path}")
            # ---------------------------------------------------------
            # B. Inference
            print("B")
            obs_tensor = torch.from_numpy(obs_numpy).float().to(self.device) / 255.0
            obs_tensor = obs_tensor.unsqueeze(0) 

            with torch.no_grad():
                action = self.agent.choose_action(obs_numpy, sample=False)

            # C. Apply
            # We use this as a base so we don't disturb wheels/plow
            print("C")
            current_full_pose = self.robot.get_joint_positions()
            
            # 2. Identify the Plow Index (Index 3 from your logs)
            # 'Castor_body', 'Left_arm', 'Left_drive', 'Plow_lift'(3)
            # plow_index = 3 


            # 4. Overwrite just the Arm Indices with Agent Data
            # Map Agent Output [-1, 1] -> Radians
            print("Target arm positions unprocessed:", action)
            target_arm_positions = action # * 1.57
            print("Target arm positions processed:", target_arm_positions)
            for i, robot_idx in enumerate(self.active_indices):
                current_full_pose[robot_idx] = target_arm_positions[i]
            print("Current full pose: ", current_full_pose)
            # current_full_pose = np.zeros(9) 

            # Map the 4 Agent outputs to the specific Active Indices
            # target_arm_positions has 4 values
            # self.active_indices has 4 indices (e.g., [1, 4, 6, 8])
 
            # 5. Send the TOTAL command (All 9 joints)
            self.robot.set_joint_positions(current_full_pose)
        except Exception as e:
            print(e)
            pass

    def stop(self):
        if self.sub:
            self.sub = None
            print("🛑 Controller Stopped.")

# Run Once logic
if 'global_controller' in globals():
    try:
        globals()['global_controller'].stop()
    except: pass
    del globals()['global_controller'] # Delete old object

# 2. Create NEW instance with updated code
globals()['global_controller'] = RobotController()

# 3. Schedule Setup
asyncio.ensure_future(globals()['global_controller'].setup())

#['Castor_body', 'Left_arm', 'Left_drive', 'Plow_lift', 'Right_arm', 'Right_drive', 'Left_palm', 'Plow_tilt', 'Right_palm']