import os
import pickle
import numpy as np
import hydra
import torch
import cv2
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# --- CONFIGURATION ---
TARGET_RESOLUTION = (84, 84) 
# ---------------------

def process_frame(frame_tensor):
    """
    Takes a LeRobot frame (C, H, W) float32 [0,1], 
    resizes it to 84x84, and converts to uint8 [0,255].
    """
    # 1. Convert to Numpy & [0, 255]
    img = (frame_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # 2. Resize
    resized = cv2.resize(img, TARGET_RESOLUTION, interpolation=cv2.INTER_AREA)

    # 3. Transpose back to (C, H, W)
    transposed = np.transpose(resized, (2, 0, 1))
    return transposed

def get_data_stats(d, rewards, lengths):
    print("\n--- Dataset Statistics ---")
    print(f"Total Trajectories: {len(lengths)}")
    print(f"Total Transitions: {sum(lengths)}")
    
    # Check the first trajectory's state structure
    first_state = d["states"][0]
    if isinstance(first_state, dict):
        print("Observation Type: Dict (Image + State)")
        print(f"  Image Shape: {first_state['image'].shape} (Should be T, 3, 84, 84)")
        print(f"  State Shape: {first_state['state'].shape} (Should be T, N_Joints)")
    else:
        print(f"Observation Shape: {first_state.shape}")
        
    print("--------------------------\n")

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    repo_id = cfg.env.demo 
    output_name = cfg.env.name
    
    print(f"⬇️  Loading LeRobot dataset: {repo_id}...")
    ds = LeRobotDataset(repo_id)
    print(f"Dataset Features: {ds.features}")
    
    # --- 1. Detect Keys ---
    # Image Key
    image_keys = [k for k in ds.features if k.startswith("observation.images.camera_raw")]
    if not image_keys:
        raise ValueError("No image features found!")
    obs_key = next((k for k in image_keys if "raw" in k), image_keys[0])

    # State Key (New)
    state_key = "observation.state"
    if state_key not in ds.features:
        print("⚠️ Warning: 'observation.state' not found. Trying to find a fallback...")
        possible_states = [k for k in ds.features if "state" in k and "velocity" not in k]
        if possible_states:
            state_key = possible_states[0]
            print(f"-> Found alternative state key: {state_key}")
        else:
            raise ValueError("Could not find any joint state data in dataset!")

    act_key = "action"
    
    print(f"📸 Image Key: '{obs_key}'")
    print(f"🦾 State Key: '{state_key}'")
    print(f"🤖 Action Key: '{act_key}'")
    
    num_episodes = ds.num_episodes
    print(f"Processing {num_episodes} episodes...")

    expert_trajs = defaultdict(list)
    expert_lengths = []
    expert_rewards = []

    for ep_idx in range(num_episodes):
        if ep_idx % 5 == 0:
            print(f"Processing episode {ep_idx}/{num_episodes}...", end='\r')

        from_idx = ds.episode_data_index["from"][ep_idx].item()
        to_idx = ds.episode_data_index["to"][ep_idx].item()
        
        # Temp lists for this trajectory
        traj_images = []
        traj_joints = []
        traj_actions = []
        
        for i in range(from_idx, to_idx):
            item = ds[i]
            
            # 1. Process Image
            raw_img = item[obs_key] 
            processed_img = process_frame(raw_img)
            traj_images.append(processed_img)

            # 2. Process Joint State (New)
            # LeRobot usually returns this as Float32 Tensor
            raw_state = item[state_key]
            traj_joints.append(raw_state.numpy())

            # 3. Process Action
            traj_actions.append(item[act_key].numpy())

        # Convert lists to numpy arrays
        # Shape: (T, C, H, W)
        images_np = np.array(traj_images)
        # Shape: (T, D_joint)
        joints_np = np.array(traj_joints)
        # Shape: (T, D_action)
        actions_np = np.array(traj_actions)

        path_len = len(images_np)

        # --- Construct Dictionary Observation ---
        # We group image and joints into a single dictionary for this trajectory
        states_dict = {
            "image": images_np,
            "state": joints_np
        }

        # --- Create Next States ---
        # We must shift BOTH image and joints
        next_images = np.zeros_like(images_np)
        next_images[:-1] = images_np[1:]
        next_images[-1] = images_np[-1]

        next_joints = np.zeros_like(joints_np)
        next_joints[:-1] = joints_np[1:]
        next_joints[-1] = joints_np[-1]

        next_states_dict = {
            "image": next_images,
            "state": next_joints
        }

        # Standard RL arrays
        rewards = np.zeros(path_len, dtype=np.float32)
        dones = np.zeros(path_len, dtype=bool)
        dones[-1] = True

        # Append to master dictionary
        # Now 'states' is a list of Dictionaries, not a list of Arrays
        expert_trajs["states"].append(states_dict)
        expert_trajs["next_states"].append(next_states_dict)
        expert_trajs["actions"].append(actions_np)
        expert_trajs["rewards"].append(rewards)
        expert_trajs["dones"].append(dones)
        expert_trajs["lengths"].append(path_len)
        expert_rewards.append(np.sum(rewards))

    print(f"\nProcessing complete!")
    get_data_stats(expert_trajs, np.array(expert_rewards), np.array(expert_lengths))

    save_dir = hydra.utils.to_absolute_path('experts')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{output_name}.pkl')
    
    print(f"Saving to: {save_path}")
    with open(save_path, 'wb') as f:
        pickle.dump(expert_trajs, f)
    print("✅ Done.")

if __name__ == '__main__':
    main()