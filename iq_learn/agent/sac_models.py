import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions as pyd
from torch.autograd import Variable, grad
from omegaconf import ListConfig, DictConfig
import torchvision.models as models
# from pytorch_grad_cam import EigenCAM # Optional

try:
    import utils.utils as utils
except ImportError:
    pass 

# --- HELPER FUNCTIONS ---
def to_list(x):
    if isinstance(x, (ListConfig, tuple)):
        return list(x)
    return x

def is_image(obs_dim):
    if hasattr(obs_dim, '__len__') and len(obs_dim) == 3:
        return True
    return False

def orthogonal_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

# --- ENCODERS ---
class MultiModalEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=256, pretrained=True):
        super().__init__()
        # print("Multimodal encoder class")
        weights = 'DEFAULT' if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        self.resnet.fc = nn.Identity() 
        self.resnet_out_dim = 512

        # print("State encoder")
        self.state_dim = obs_shape['state'][0]
        self.state_embed_dim = 64
        self.state_mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.state_embed_dim),
            nn.ReLU(),
            nn.Linear(self.state_embed_dim, self.state_embed_dim),
            nn.ReLU()
        )

        fusion_input_dim = self.resnet_out_dim + self.state_embed_dim
        self.fusion_fc = nn.Linear(fusion_input_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)
        self.feature_dim = feature_dim

        if not pretrained:
            self.apply(orthogonal_init_)

    def forward(self, obs):
        if isinstance(obs, dict):
            img = obs['image']
            state = obs['state']
        else:
            img = obs
            state = torch.zeros((img.shape[0], self.state_dim), device=img.device)

        if img.max() > 1.0: img = img / 255.0
        img_embed = self.resnet(img)
        state_embed = self.state_mlp(state)
        fused = torch.cat([img_embed, state_embed], dim=1)
        out = self.fusion_fc(fused)
        out = self.ln(out)
        return torch.tanh(out)

class PixelEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim=50, pretrained=True):
        super().__init__()
        self.feature_dim = feature_dim
        obs_shape = to_list(obs_shape)
        weights = 'DEFAULT' 
        self.resnet = models.resnet18(weights=weights)
        
        input_channels = obs_shape[0]
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)
        
        if not pretrained:
            self.apply(orthogonal_init_)

    def forward(self, obs):
        if obs.max() > 1.0:
            obs = obs / 255.0
        h = self.resnet(obs)
        h = self.ln(h)
        return torch.tanh(h)

# --- CRITIC ---
class DoubleQCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, args):
        super(DoubleQCritic, self).__init__()
        self.args = args

        is_multimodal = isinstance(obs_dim, (dict, DictConfig)) or (hasattr(obs_dim, 'keys'))
        is_img_list = is_image(obs_dim)

        self.encoder = None
        
        if is_multimodal:
            self.encoder = MultiModalEncoder(obs_dim, feature_dim=hidden_dim)
            input_dim = self.encoder.feature_dim + action_dim
        elif is_img_list:
            self.encoder = PixelEncoder(obs_dim, feature_dim=hidden_dim)
            input_dim = self.encoder.feature_dim + action_dim
        else:
            if obs_dim is None:
                raise ValueError("obs_dim is None! Check your Hydra config.")
            dim_val = obs_dim[0] if hasattr(obs_dim, '__len__') else obs_dim
            input_dim = dim_val + action_dim

        self.Q1 = utils.mlp(input_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(input_dim, hidden_dim, 1, hidden_depth)
        self.apply(orthogonal_init_)

    def forward(self, obs, action, both=False):
        if self.encoder is not None:
            obs = self.encoder(obs)
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        if self.args.method.tanh:
            q1 = torch.tanh(q1) * 1/(1-self.args.gamma)
            q2 = torch.tanh(q2) * 1/(1-self.args.gamma)
        if both:
            return q1, q2
        return torch.min(q1, q2)

# --- ACTOR ---
class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, hidden_depth, log_std_bounds):
        super().__init__()
        self.log_std_bounds = log_std_bounds

        # 1. Determine Input Type
        is_multimodal = isinstance(obs_dim, (dict, DictConfig)) or (hasattr(obs_dim, 'keys'))
        is_img_list = is_image(obs_dim) 

        self.encoder = None
        
        if is_multimodal:
            # Case A: Dictionary (Fusion)
            self.encoder = MultiModalEncoder(obs_dim, feature_dim=hidden_dim)
            input_dim = self.encoder.feature_dim
        elif is_img_list:
            # Case B: Standard Image List (Pixel)
            self.encoder = PixelEncoder(obs_dim, feature_dim=hidden_dim)
            input_dim = self.encoder.feature_dim
        else:
            # Case C: Standard Vector
            dim_val = obs_dim[0] if hasattr(obs_dim, '__len__') else obs_dim
            input_dim = dim_val

        self.trunk = utils.mlp(input_dim, hidden_dim, 2 * action_dim, hidden_depth)
        self.outputs = dict()
        self.apply(orthogonal_init_)

    def forward(self, obs):
        if self.encoder is not None:
            obs = self.encoder(obs)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        return dist

    def sample(self, obs):
        dist = self.forward(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob, dist.mean

# --- DISTRIBUTIONS ---
class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu