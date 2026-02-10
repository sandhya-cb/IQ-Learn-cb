import torch
import cv2
import numpy as np
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_encoder(agent, obs, device):
    """
    Uses EigenCAM to visualize the 'attention' of the ResNet Encoder.
    """
    
    # --- FIX START: Handle Multimodal Dictionary ---
    if isinstance(obs, dict):
        # If input is a dictionary, grab only the image tensor.
        # Your MultiModalEncoder fallback logic (state=zeros) will handle 
        # this tensor-only input correctly during the EigenCAM forward pass.
        obs_data = obs['image']
    else:
        obs_data = obs
    # --- FIX END -----------------------------------

    # 1. Prepare Data (Ensure Tensor is [1, C, H, W])
    if isinstance(obs_data, np.ndarray):
        # Normalize if user passes raw 0-255 numpy
        if obs_data.max() > 1.0: 
            obs_data = obs_data / 255.0
        # Numpy is usually (H, W, C), convert to (C, H, W) for Torch
        if obs_data.shape[2] == 3 or obs_data.shape[2] == 9: 
            obs_data = obs_data.transpose(2, 0, 1)
        
        obs_tensor = torch.as_tensor(obs_data, device=device).float()
    else:
        obs_tensor = obs_data

    # Ensure batch dimension exists
    if obs_tensor.dim() == 3:
        obs_tensor = obs_tensor.unsqueeze(0)

    # 2. Target the ResNet's Last Layer
    # We want the output of the last convolutional block in ResNet18
    # agent.critic.encoder is the MultiModalEncoder
    # agent.critic.encoder.resnet is the ResNet18
    if hasattr(agent.critic.encoder, 'resnet'):
        target_layers = [agent.critic.encoder.resnet.layer4[-1]]
    else:
        print("Warning: Could not find ResNet in encoder for visualization.")
        return None

    # 3. Initialize EigenCAM
    cam = EigenCAM(model=agent.critic.encoder, target_layers=target_layers)

    # 4. Generate Map
    # We pass targets=None because EigenCAM is unsupervised (doesn't need class labels)
    grayscale_cam = cam(input_tensor=obs_tensor, targets=None)[0, :]

    # 5. Process Image for Overlay
    # Grab the tensor back to CPU numpy for plotting
    img_viz = obs_tensor.cpu().numpy()[0] 
    img_viz = np.transpose(img_viz, (1, 2, 0)) # Convert back to HWC
    
    # If using Frame Stacking (e.g. 9 channels), take the last 3 (most recent frame)
    if img_viz.shape[2] > 3:
        img_viz = img_viz[:, :, -3:]
        
    # Ensure it's 0-1 for visualization library
    if img_viz.max() > img_viz.min():
        img_viz = (img_viz - img_viz.min()) / (img_viz.max() - img_viz.min())
    
    # Create the heatmap overlay
    visualization = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
    
    return visualization