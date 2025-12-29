# iq_learn/agent/sac.py

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.utils import soft_update

class MockArgs:
    def __init__(self, agent_cfg, gamma, device):
        self.agent = agent_cfg
        self.gamma = gamma
        self.device = device
        # Some legacy code might check args.train.batch
        self.train = DictConfig({'batch': agent_cfg.batch_size})

class SAC(object):
    def __init__(self, obs_dim, action_dim, 
                 action_range=1.0, 
                 batch_size=32, 
                 args=None, 
                 gamma=0.99, 
                 device='cuda',
                 **kwargs):

        # 1. Handle Missing 'args' (Hydra Instantiation Case)
        if args is None:
            # --- FIX STARTS HERE ---
            # We must filter out complex objects (like actor_cfg/critic_cfg) 
            # before converting to DictConfig to avoid "UnsupportedValueType" errors.
            clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['actor_cfg', 'critic_cfg']}
            
            # Create config from the safe primitives
            agent_cfg = DictConfig(clean_kwargs)
            
            # Manually inject the safe defaults we need
            agent_cfg.batch_size = batch_size
            agent_cfg.init_temp = kwargs.get('init_temp', 0.1)
            agent_cfg.learn_temp = kwargs.get('learn_temp', True)
            agent_cfg.actor_lr = kwargs.get('actor_lr', 1e-4)
            agent_cfg.critic_lr = kwargs.get('critic_lr', 1e-4)
            agent_cfg.alpha_lr = kwargs.get('alpha_lr', 1e-4)
            agent_cfg.critic_tau = kwargs.get('critic_tau', 0.005)
            # --- FIX ENDS HERE ---

            args = MockArgs(agent_cfg, gamma, device)
        
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.action_range = action_range
        self.device = torch.device(args.device) if isinstance(args.device, str) else args.device
        self.args = args
        agent_cfg = args.agent

        self.critic_tau = agent_cfg.critic_tau
        self.learn_temp = agent_cfg.learn_temp
        self.actor_update_frequency = kwargs.get('actor_update_frequency', 1)
        self.critic_target_update_frequency = kwargs.get('critic_target_update_frequency', 1)

        # 2. Instantiate Networks
        # We retrieve the configs from kwargs (Hydra passed them in)
        actor_cfg = kwargs.get('actor_cfg')
        critic_cfg = kwargs.get('critic_cfg')

        # Fallback to agent_cfg if not in kwargs (Legacy support)
        if actor_cfg is None: actor_cfg = agent_cfg.get('actor_cfg')
        if critic_cfg is None: critic_cfg = agent_cfg.get('critic_cfg')

        if actor_cfg is None or critic_cfg is None:
            raise ValueError("SAC Config missing 'actor_cfg' or 'critic_cfg'. Check your sac.yaml")

        

        # Now we instantiate. args is passed, but it contains a "clean" agent_cfg
        self.critic = hydra.utils.instantiate(critic_cfg, args=args).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg, args=args).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        # 3. Optimizers
        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim

        self.actor_optimizer = Adam(self.actor.parameters(), lr=agent_cfg.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=agent_cfg.critic_lr)
        self.log_alpha_optimizer = Adam([self.log_alpha], lr=agent_cfg.alpha_lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.critic

    @property
    def critic_target_net(self):
        return self.critic_target

    def choose_action(self, state, sample=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        current_Q = self.critic(obs, action)
        current_V = current_Q - self.alpha.detach() * log_prob
        return current_V

    def get_targetV(self, obs):
        action, log_prob, _ = self.actor.sample(obs)
        target_Q = self.critic_target(obs, action)
        target_V = target_Q - self.alpha.detach() * log_prob
        return target_V

    def update(self, replay_buffer, logger, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device)
        
        losses = self.update_critic(obs, action, reward, next_obs, done, logger, step)

        if step % self.actor_update_frequency == 0:
            actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
            losses.update(actor_alpha_losses)

        if step % self.critic_target_update_frequency == 0:
            soft_update(self.critic, self.critic_target, self.critic_tau)

        return losses

    def update_critic(self, obs, action, reward, next_obs, done, logger, step):
        with torch.no_grad():
            next_action, log_prob, _ = self.actor.sample(next_obs)
            target_Q = self.critic_target(next_obs, next_action)
            target_V = target_Q - self.alpha.detach() * log_prob
            target_Q = reward + (1 - done) * self.gamma * target_V

        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss = F.mse_loss(current_Q1, target_Q)
        q2_loss = F.mse_loss(current_Q2, target_Q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            'critic_loss/critic_1': q1_loss.item(),
            'critic_loss/critic_2': q2_loss.item(),
            'loss/critic': critic_loss.item()}

    def update_actor_and_alpha(self, obs, logger, step):
        action, log_prob, _ = self.actor.sample(obs)
        actor_Q = self.critic(obs, action)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train/actor_loss', actor_loss, step)
        logger.log('train/target_entropy', self.target_entropy, step)
        logger.log('train/actor_entropy', -log_prob.mean(), step)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        losses = {
            'loss/actor': actor_loss.item(),
            'actor_loss/target_entropy': self.target_entropy,
            'actor_loss/entropy': -log_prob.mean().item()}

        if self.learn_temp:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train/alpha_loss', alpha_loss, step)
            logger.log('train/alpha_value', self.alpha, step)

            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            losses.update({
                'alpha_loss/loss': alpha_loss.item(),
                'alpha_loss/value': self.alpha.item(),
            })
        return losses

    def save(self, path, suffix=""):
        actor_path = f"{path}{suffix}_actor"
        critic_path = f"{path}{suffix}_critic"
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, path, suffix=""):
        actor_path = f'{path}/sac{suffix}_actor'
        critic_path = f'{path}/sac{suffix}_critic'

        # Legacy fallback if name exists in config
        if hasattr(self.args.agent, 'name') and self.args.agent.name is not None:
             actor_path = f'{path}/{self.args.agent.name}{suffix}_actor'
             critic_path = f'{path}/{self.args.agent.name}{suffix}_critic'

        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if os.path.isfile(actor_path):
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        if os.path.isfile(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))