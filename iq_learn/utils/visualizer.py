import torch
import cv2
import numpy as np
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_encoder(agent, obs, device):
    """
    Uses EigenCAM to visualize the 'attention' of the ResNet Encoder.
    This shows which objects the ResNet has extracted features for.
    
    Args:
        agent: Your SAC agent
        obs: Numpy array (H, W, C) or Tensor (1, C, H, W)
        device: 'cuda' or 'cpu'
    """
    
    # 1. Prepare Data (Ensure Tensor is [1, C, H, W])
    if isinstance(obs, np.ndarray):
        # Normalize if user passes raw 0-255 numpy
        if obs.max() > 1.0: 
            obs = obs / 255.0
        # Numpy is usually (H, W, C), convert to (C, H, W) for Torch
        if obs.shape[2] == 3 or obs.shape[2] == 9: 
            obs = obs.transpose(2, 0, 1)
        
        obs_tensor = torch.as_tensor(obs, device=device).float().unsqueeze(0)
    else:
        obs_tensor = obs.unsqueeze(0) if obs.dim() == 3 else obs

    # 2. Target the ResNet's Last Layer
    # We want the output of the last convolutional block in ResNet18
    target_layers = [agent.critic.encoder.resnet.layer4[-1]]

    # 3. Initialize EigenCAM
    # We pass the encoder directly. No wrapper needed.
    cam = EigenCAM(model=agent.critic.encoder, target_layers=target_layers)

    # 4. Generate Map
    # targets=None is allowed here because EigenCAM doesn't use classes/scores
    grayscale_cam = cam(input_tensor=obs_tensor, targets=None)[0, :]

    # 5. Process Image for Overlay
    # Grab the tensor back to CPU numpy for plotting
    img_viz = obs_tensor.cpu().numpy()[0] 
    img_viz = np.transpose(img_viz, (1, 2, 0)) # Convert back to HWC
    
    # If using Frame Stacking (e.g. 9 channels), take the last 3 (most recent frame)
    if img_viz.shape[2] > 3:
        img_viz = img_viz[:, :, -3:]
        
    # Ensure it's 0-1 for visualization library
    img_viz = (img_viz - img_viz.min()) / (img_viz.max() - img_viz.min() + 1e-8)
    
    # Create the heatmap overlay
    visualization = show_cam_on_image(img_viz, grayscale_cam, use_rgb=True)
    
    return visualization