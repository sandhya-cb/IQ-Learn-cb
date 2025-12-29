import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DustyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # 1. Action Space
        # Your dataset metadata says shape [4] (arm angles)
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # 2. Observation Space
        # Your dataset has images. After processing, they are (3, 84, 84).
        # We use uint8 [0-255].
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, 
                high=255, 
                shape=(3, 84, 84), 
                dtype=np.uint8
            ),
            "state": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(4,), # Assuming 4 joints
                dtype=np.float32
            )
        })
        
        # (Optional) If you decide to use State vectors instead of Images later:
        # self.observation_space = spaces.Box(low=-inf, high=inf, shape=(4,), dtype=float32)

    def reset(self, seed=None, options=None):
        # Return a dummy zero-array matching the observation shape
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action):
        # Return dummy data. 
        # Since this is Offline training, the agent NEVER calls this.
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        reward = 0.0
        done = False
        truncated = False
        info = {}
        return obs, reward, done, truncated, info