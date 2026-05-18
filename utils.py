import numpy as np
import torch
import PIL.Image as Image
from gym.wrappers.time_limit import TimeLimit
import gym
from gym.spaces import Box
from VLM import env_clip_prompts
from VLM import phi, clip, deepseekVL
from scipy.special import softmax
import random

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
    def __init__(self, env, env_name, task, reward_mode, reward_steps:int=1, reward_k:int=16, reward_model=None, pos_reward:float=0.1, neg_reward:float=-0.1):
        super().__init__(env)
        self.env = env
        self.env_name = env_name
        self._steps = 0

        self._task = task
        self._reward_mode = reward_mode
        self._reward_steps = reward_steps
        self._reward_k = reward_k
        self.reward_model = reward_model
        self._pos_reward = pos_reward
        self._neg_reward = neg_reward
        print(f"reward_mode: {self._reward_mode}, reward_steps: {self._reward_steps}, reward_k: {self._reward_k}")
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
                    if probs[1] > 0.52 and probs_inverse[0] > 0.52:
                        reward_relative = self._pos_reward
                    elif probs[0] > 0.52 and probs_inverse[1] > 0.52:
                        reward_relative = self._neg_reward
                    else:
                        reward_relative = 0

                reward_absolute = self.reward_model.r_hat(image)

                reward = (reward_relative + reward_absolute) / 2

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

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)