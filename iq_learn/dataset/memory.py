from collections import deque
import numpy as np
import random
import torch

# Try/Except to prevent import errors if running in isolation
try:
    from wrappers.atari_wrapper import LazyFrames
except ImportError:
    class LazyFrames: pass # Dummy class

from dataset.expert_dataset import ExpertDataset


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        # If path has no extension add npy
        if not path.endswith("pkl"):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        # data = np.load(path, allow_pickle=True)
        for i in range(len(data)):
            self.add(data[i])

    def get_samples(self, batch_size, device):
        batch = self.sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

        # --- FIX: Handle Multimodal Dictionary Inputs ---
        # Check if the first element is a Dictionary (Multimodal case)
        if isinstance(batch_state[0], dict):
            # 1. Process Current State
            state_dict = {}
            for key in batch_state[0].keys():
                # Stack the list of arrays for this specific key (e.g., stack all images together)
                # [Batch, C, H, W] for images or [Batch, Dim] for state
                stacked_val = np.stack([s[key] for s in batch_state])
                state_dict[key] = torch.as_tensor(stacked_val, dtype=torch.float, device=device)
            batch_state = state_dict

            # 2. Process Next State
            next_state_dict = {}
            for key in batch_next_state[0].keys():
                stacked_val = np.stack([s[key] for s in batch_next_state])
                next_state_dict[key] = torch.as_tensor(stacked_val, dtype=torch.float, device=device)
            batch_next_state = next_state_dict

        else:
            # --- LEGACY: Standard Vector or Image Array ---
            # Handle Atari LazyFrames scaling
            if isinstance(batch_state[0], LazyFrames):
                batch_state = np.array(batch_state) / 255.0
            if isinstance(batch_next_state[0], LazyFrames):
                batch_next_state = np.array(batch_next_state) / 255.0
            
            # Standard conversion
            batch_state = torch.as_tensor(np.array(batch_state), dtype=torch.float, device=device)
            batch_next_state = torch.as_tensor(np.array(batch_next_state), dtype=torch.float, device=device)
        # ------------------------------------------------

        # Process Actions, Rewards, Dones
        batch_action = torch.as_tensor(np.array(batch_action), dtype=torch.float, device=device)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
            
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward, batch_done