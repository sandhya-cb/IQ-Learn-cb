import gymnasium as gym
from agent.sac import SAC
from agent.softq import SoftQ
# Ensure omegaconf is imported to handle config updates cleanly if needed
from omegaconf import OmegaConf

def make_agent(env, args):
    # Determine the shape of the observation
    # If it's an image (Rank 3), we keep the tuple (C, H, W)
    # If it's a vector (Rank 1), we usually just want the integer size
    obs_shape = env.observation_space.shape
    
    # Check for Discrete Action Space (SoftQ)
    if isinstance(env.action_space, gym.spaces.Discrete):
        print('--> Using Soft-Q agent')
        action_dim = env.action_space.n
        
        # For SoftQ, we typically assume vector observations
        obs_dim = obs_shape[0] 
        
        args.agent.obs_dim = obs_dim
        args.agent.action_dim = int(action_dim)
        
        agent = SoftQ(obs_dim, action_dim, args.train.batch, args)

    # Check for Continuous Action Space (SAC)
    else:
        print('--> Using SAC agent')
        action_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max())
        ]
        
        # Pass the full shape (e.g., [3, 84, 84]) so the PixelEncoder detects it's an image
        args.agent.obs_dim = obs_shape
        args.agent.action_dim = action_dim
        
        agent = SAC(obs_shape, action_dim, action_range, args.train.batch, args)

    return agent