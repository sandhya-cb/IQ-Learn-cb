import numpy as np
import cv2
import torch
import gymnasium as gym
from gymnasium import spaces

import omni.isaac.core.utils.extensions as extensions
from omni.isaac.core.utils.stage import add_reference_to_stage

# Force enable the Core Robotics extension
extensions.enable_extension("omni.isaac.core")
# Isaac Sim Imports (Assumes you are running inside the Isaac python environment)
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.robots import Robot

class IsaacDustyEnv(gym.Env):
    def __init__(self, usd_path, camera_path):
        self.world = World(stage_units_in_meters=1.0)
        
        # 1. Load Robot Asset
        # 1. Manually load the USD file onto the stage at the desired path
        add_reference_to_stage(usd_path=usd_path, prim_path="/World/Dusty")

        # 2. Now wrap that existing prim with the Robot class
        # (The Robot class controls the prim, it doesn't load the file)
        self.robot = self.world.scene.add(
            Robot(prim_path="/World/Dusty", name="dusty")
        )
        
        # 2. Setup Camera (MUST MATCH REAL WORLD POSITION EXACTLY)
        self.camera = Camera(prim_path=camera_path, resolution=(640, 480))
        self.camera.initialize()
        
        # 3. Define Spaces (Matching your Training)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)
        
        self.world.reset()

    def get_observation(self):
        """
        1. Get Image from Isaac (RGBA)
        2. Drop Alpha channel (RGB)
        3. Resize to 84x84 (Agent Size)
        4. Transpose to Channel-First (3, 84, 84)
        """
        # Get raw data (H, W, 4)
        rgba = self.camera.get_rgba()[:, :, :3] # Drop Alpha -> (H, W, 3)
        
        # Resize to 84x84
        resized = cv2.resize(rgba, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Transpose (H, W, C) -> (C, H, W)
        obs = np.transpose(resized, (2, 0, 1))
        return obs.astype(np.uint8)

    def step(self, action):
        # 1. Un-scale Action: [-1, 1] -> [Physical Angles]
        # Example: if your robot moves -90 to +90 degrees
        target_angles = action * np.deg2rad(90) 
        
        # 2. Apply to Isaac Robot
        self.robot.set_joint_positions(target_angles)
        
        # 3. Step Physics
        self.world.step(render=True)
        
        # 4. Return Observation
        obs = self.get_observation()
        return obs, 0.0, False, False, {}

    def reset(self, seed=None):
        self.world.reset()
        # Reset robot to home position
        self.robot.set_joint_positions(np.zeros(4))
        self.world.step(render=True)
        return self.get_observation(), {}