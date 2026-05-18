import numpy as np
import torch
from torch import nn
from torch import distributions as pyd
import PIL.Image as Image
import gym
from gym.wrappers.time_limit import TimeLimit
from gym.spaces import Box
from VLM import env_clip_prompts
from VLM import phi, clip, deepseekVL
from scipy.special import softmax
import random
from rlkit.envs.wrappers import NormalizedBoxEnv
import os


def obs_to_image(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    obs = np.transpose(obs, (1, 2, 0))
    obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
    obs = obs.astype(np.uint8)
    
    return obs

def obs_to_PIL_image(obs):
    if isinstance(obs, torch.Tensor):
        obs = obs.cpu().numpy()

    obs = np.transpose(obs, (1, 2, 0))
    obs = (obs - obs.min()) / (obs.max() - obs.min()) * 255
    obs = obs.astype(np.uint8)
    return Image.fromarray(obs)
    
class ClipObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, clip_range=(-1e6, 1e6)):
        super().__init__(env)
        self.clip_low, self.clip_high = clip_range
        low = np.clip(env.observation_space.low, self.clip_low, self.clip_high)
        high = np.clip(env.observation_space.high, self.clip_low, self.clip_high)
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        if isinstance(observation, tuple): 
            return observation[0]
        return observation

from softgym.utils.normalized_env import normalize
    
def make_softgym_env(cfg):
    from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
    env_name = cfg.task
    env_kwargs = env_arg_dict[env_name]
    env = normalize(SOFTGYM_ENVS[env_name](**env_kwargs))

    return env

def make_classic_control_env(cfg):
    if "CartPole" in cfg.task:
        from envs.cartpole import CartPoleEnv
        env = CartPoleEnv()
    elif "RingWorld" in cfg.task:
        from envs.ringworld import RingWorldEnv
        env = RingWorldEnv()
    else:
        raise NotImplementedError
    
    return TimeLimit(NormalizedBoxEnv(env), env.horizon)

def make_metaworld_env(cfg):
    import metaworld.envs.mujoco.env_dict as _env_dict

    env_name = cfg.task
    if env_name in _env_dict.ALL_V2_ENVIRONMENTS:
        env_cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = _env_dict.ALL_V1_ENVIRONMENTS[env_name]
    
    env = env_cls(render_mode='rgb_array')
    env.camera_name = env_name
    
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(cfg.seed)

    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)

def make_minedojo_env(task):
    from animal_zoo import HuntCowDenseRewardEnv
    from animal_zoo import MilkCowDenseRewardEnv
    from mob_combat import CombatSpiderDenseRewardEnv
    from animal_zoo import ShearSheepDenseRewardEnv
    from animal_zoo import HarvestWaterDenseRewardEnv

    if task == "combat_spider":
        env = CombatSpiderDenseRewardEnv(step_penalty=0, attack_reward=1, success_reward=10)
    elif task == "milk_cow":
        env = MilkCowDenseRewardEnv(step_penalty=0, nav_reward_scale=0.1, success_reward=10)
    elif task == "hunt_cow":
        env = HuntCowDenseRewardEnv(step_penalty=0, nav_reward_scale=0.1, attack_reward=1, success_reward=10)
    elif task == "shear_sheep":
        env = ShearSheepDenseRewardEnv(step_penalty=0, nav_reward_scale=0.1, success_reward=10)
    elif task == "harvest_water":
        env = HarvestWaterDenseRewardEnv(step_penalty=0, nav_reward_scale=0.1, success_reward=10)
    else:
        raise NotImplementedError
    return env
    
class make_reward_env(gym.Wrapper):
    def __init__(self, env, env_name, task, reward_mode, reward_k:int=16, reward_model=None, pos_reward:float=0.1, neg_reward:float=-0.1, absolute_alpha:float=0.5, confidence_threshold:float=0.52):
        super().__init__(env)
        self.env = env
        self.env_name = env_name
        self._steps = 0

        self._task = task
        self._reward_mode = reward_mode
        self._reward_k = reward_k
        self.reward_model = reward_model
        self._pos_reward = pos_reward
        self._neg_reward = neg_reward
        self._absolute_alpha = absolute_alpha
        self._confidence_threshold = confidence_threshold

        self._prev_image = [None] * self._reward_k
        self._prev_reward = [0] * self._reward_k
        self.vlm_acc = 0
        self.vlm_cnt = 0
        self.episode_success = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self._reward_mode=="phi":
            self.vlm = phi()
        elif self._reward_mode=="deepseekVL":
            self.vlm = deepseekVL()
        elif self._reward_mode=="clip":
            self.vlm = clip()
        elif self._reward_mode=="mineclip":
            from mineclip import MineCLIP
            self.vlm = MineCLIP(
                arch="vit_base_p16_fz.v2.t2",
                resolution=(160, 256),
                pool_type="attn.d2.nh8.glusw",
                image_feature_dim=512,
                mlp_adapter_spec="v0-2.t0",
                hidden_dim=512,
            ).to(self.device)
            self.vlm.load_ckpt("attn.pth")

            # frame stack 16
            from collections import deque
            self._prev_image = deque(maxlen=16)

    def reset(self, **kwargs):
        self._steps = 0
        self._prev_image = [None] * self._reward_k
        if self._reward_mode=="mineclip":
            from collections import deque
            self._prev_image = deque(maxlen=16)

        self._prev_reward = [0] * self._reward_k
        self.episode_success = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # update info
        self.episode_success = max(self.episode_success, bool(reward >= 10))
        info["is_success"] = self.episode_success

        # dense reward: original reward
        if self._reward_mode=="dense":
            pass
        # sparse reward
        elif self._reward_mode=="sparse":
            reward = 0 if reward < 10 else 10
        # vlm reward model
        elif self._reward_mode=="RL-VLM-F" or self._reward_mode=="VLM-AR3L":
            rgb_image = obs['rgb'].transpose(1, 2, 0) # (H, W, C)

            image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0  # (C, H, W)
            image = image.reshape(1, 3, image.shape[1], image.shape[2]) # (1, C, H, W)

            self.reward_model.add_data(obs['rgb'].flatten(), action, reward, done, img=rgb_image)
            self.reward_model.eval()

            if self._reward_mode=="RL-VLM-F":
                reward = self.reward_model.r_hat(image)
            else:
                idx = self._steps % self._reward_k
                
                reward_relative = 0

                if self._steps > 0:
                    prev_image = self._prev_image[idx] if self._prev_image[idx] is not None else self._prev_image[0]
                    logits = self.reward_model.r_hat_pair(prev_image, image)[0] # (2,)
                    logits_inverse = self.reward_model.r_hat_pair(image, prev_image)[0] # (2,)
                    probs  = softmax(logits)
                    probs_inverse = softmax(logits_inverse)
                    if probs[1] > self._confidence_threshold and probs_inverse[0] > self._confidence_threshold:
                        reward_relative = self._pos_reward
                    elif probs[0] > self._confidence_threshold and probs_inverse[1] > self._confidence_threshold:
                        reward_relative = self._neg_reward
                    else:
                        reward_relative = 0

                reward_absolute = self.reward_model.r_hat(image)

                reward = self._absolute_alpha * reward_absolute + (1 - self._absolute_alpha) * reward_relative

                self._prev_image[idx] = image
            self.reward_model.train()          
        # clip similarity score
        elif self._reward_mode=="clip":
            rgb_image = obs['rgb'].transpose(1, 2, 0) # (H, W, C)
            reward = self.vlm.clip_infer_score(rgb_image, env_clip_prompts[self._task]) * 2 - 1 # actually we should scale it [-1, 1] since tanh is used in the reward model
        # mineclip similarity score
        elif self._reward_mode=="mineclip":
            rgb_image = torch.from_numpy(obs['rgb'].copy()).to(dtype=torch.float32, device=self.device) # PyTorch Tensor
            self._prev_image.append(rgb_image)
            if len(self._prev_image) == 16:
                image_tensor = torch.stack(list(self._prev_image))  # (16, C, H, W)
                image_tensor = image_tensor.unsqueeze(0)  # (1, 16, C, H, W)
                image_feats = self.vlm.forward_image_features(image_tensor)
                video_feats = self.vlm.forward_video_features(image_feats)
                reward = self.vlm.forward_reward_head(video_feats, text_tokens=env_clip_prompts[self._task])[0].item()
            else:
                reward = 0
        else:
            raise NotImplementedError
        
        self._steps += 1
        return obs, reward, done, info
    
    def get_attr(self, str):
        if str == 'vlm_acc':
            return self.vlm_acc
        elif str == 'vlm_cnt':
            return self.vlm_cnt
        
def concatenate_images_vertical(images, dist_images):
    # calc max width from imgs
    width = max(img.width for img in images)
    # calc total height of imgs + dist between them
    total_height = sum(img.height for img in images) + dist_images * (len(images) - 1)

    # create new img with calculated dimensions, black bg
    new_img = Image.new('RGB', (width, total_height), (0, 0, 0))

    # init var to track current height pos
    current_height = 0
    for img in images:
        # paste img in new_img at current height
        new_img.paste(img, (0, current_height))
        # update current height for next img
        current_height += img.height + dist_images

    return new_img

def minedojo_transform_action_multi_discrete(action):
    """
    Map agent action to env action.
    """
    # MultiDiscrete([12, 3]) -> multiDiscrete[3, 3, 4, 25, 25, 8, 244, 36]
    # first 12 actions: NO_OP, forward, backward, left, right, jump, sneak, sprint, camera pitch +30, camera pitch -30, camera yaw +30, and camera yaw -30.
    # last 3 actions: NO_OP, use and attack
    action_t = []
    if action[0] == 1: # forward
        action_t.append(1)
    elif action[0] == 2: # backward
        action_t.append(2)
    else:
        action_t.append(0)

    if action[0] == 3: # left
        action_t.append(1)
    elif action[0] == 4: # right
        action_t.append(2)
    else:
        action_t.append(0)

    if action[0] == 5: # jump
        action_t.append(1)
    elif action[0] == 6: # sneak
        action_t.append(2)
    elif action[0] == 7: # sprint
        action_t.append(3)
    else:
        action_t.append(0)

    if action[0] == 8: # camera pitch +30
        action_t.append(11)
    elif action[0] == 9: # camera pitch -30
        action_t.append(13)
    else:
        action_t.append(12)

    if action[0] == 10: # camera yaw +30
        action_t.append(11)
    elif action[0] == 11: # camera yaw -30
        action_t.append(13)
    else:
        action_t.append(12)

    if action[1] == 1: # use
        action_t.append(1)
    elif action[1] == 2: # attack
        action_t.append(3)
    else:
        action_t.append(0)

    action_t.append(0)
    action_t.append(0)
    return action_t

class TorchRunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=(), device=None):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        with torch.no_grad():
            batch_mean = torch.mean(x, axis=0)
            batch_var = torch.var(x, axis=0)
            batch_count = x.shape[0]
            self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    @property
    def std(self):
        return torch.sqrt(self.var)

def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta + batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 hidden_depth,
                 output_mod=None):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
                         output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)

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
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))
    
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
    
def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

def get_info_stats(infos):
    # infos is a list with N_traj x T entries
    N = len(infos)
    T = len(infos[0])

    all_keys = infos[0][0].keys()
    stat_dict = {}
    for key in all_keys:
        stat_dict[key + '_mean'] = []
        stat_dict[key + '_final'] = []
        for traj_idx, ep_info in enumerate(infos):
            for time_idx, info in enumerate(ep_info):
                stat_dict[key + '_mean'].append(info[key])
            stat_dict[key + '_final'].append(info[key])
        stat_dict[key + '_mean'] = np.mean(stat_dict[key + '_mean'])
        stat_dict[key + '_final'] = np.mean(stat_dict[key + '_final'])

    return stat_dict