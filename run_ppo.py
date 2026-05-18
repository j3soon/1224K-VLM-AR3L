# minedojo
# gymnasium==0.29.1
# shimmy==1.3.0
# stable_baselines3 2.3.2
# gym==0.21.0

import os
import numpy as np
from stable_baselines3 import PPO, SAC
from PIL import Image
import torch
import gym
from gym.spaces import Box
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import imageio
from mineclip import MineCLIP
import matplotlib.pyplot as plt
import logging
import argparse
import utils
from reward_model import RewardModel
from scipy.special import softmax
import cv2

class MineCLIPFeatureWrapper(gym.ObservationWrapper):
    def __init__(self, env, mineclip_model):
        super(MineCLIPFeatureWrapper, self).__init__(env)
        self.mineclip_model = mineclip_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.observation_space = Box(
            low=-float('inf'), high=float('inf'), shape=(512,), dtype=np.float32
        )
        self.raw_rgb_obs = None

    def observation(self, obs):
        # Get the RGB observation
        self.raw_rgb_obs = obs['rgb']
        # Convert it to a tensor and normalize
        rgb_obs = torch.from_numpy(self.raw_rgb_obs.copy()).to(dtype=torch.float32, device=self.device)
        rgb_obs = rgb_obs.unsqueeze(0) # (1, C, H, W)
        # Encode the observation using MineCLIP image_encoder
        image_feats = self.mineclip_model.forward_image_features(rgb_obs)
        return image_feats.squeeze(0).cpu().detach().numpy()

class CustomActionWrapper(gym.ActionWrapper):
    def __init__(self, env, transform_fn):
        super(CustomActionWrapper, self).__init__(env)
        self.action_space = gym.spaces.MultiDiscrete([12, 3])
        logger.info(f" Custom action space: {self.action_space}")
        self.transform_fn = transform_fn

    def action(self, action):
        # Apply custom transformation to the action
        return self.transform_fn(action)

class EvalCallbackWithGif(EvalCallback):
    def __init__(self, env_name, task, eval_env, n_eval_episodes, best_model_save_path, log_path, eval_freq, gif_path, reward_model, k, pos_reward, neg_reward, **kwargs):
        if reward_model is not None:
            # 創建保存 reward model 的 callback
            save_reward_callback = SaveRewardModelCallback(reward_model, best_model_save_path, env_name)
            super().__init__(eval_env, callback_on_new_best=save_reward_callback, n_eval_episodes=n_eval_episodes,
                             best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq, **kwargs)
        else:
            super().__init__(eval_env, n_eval_episodes=n_eval_episodes, best_model_save_path=best_model_save_path, log_path=log_path, 
                             eval_freq=eval_freq, **kwargs)
        self.gif_path = gif_path
        self.eval_env = eval_env
        self.env_name = env_name
        self.task = task
        self.k = k
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward

        os.makedirs(gif_path, exist_ok=True)

        # reward model
        self.reward_model = reward_model
        # [NOTE] reward update interval
        self.reward_update_interval = 1000
        self.total_feedback = 0
        self.labeled_feedback = 0
    
    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            gif_filename = os.path.join(self.gif_path, f"eval_{self.num_timesteps}.gif") # n_envs = 4
            self._generate_gif(gif_filename)

        # update reward model
        if self.reward_model is not None:
            if self.n_calls % self.reward_update_interval == 0:
                if self.total_feedback < self.reward_model.capacity / 2:
                    train_acc, vlm_label_acc = self.learn_reward()
                    self.model.logger.record("reward_model/train_acc", train_acc)
                    self.model.logger.record("reward_model/vlm_label_acc", vlm_label_acc)
                    self.model.logger.record("reward_model/total_feedback", self.total_feedback)
                    self.model.logger.record("reward_model/labeled_feedback", self.labeled_feedback)
                    self.model.logger.dump(self.num_timesteps)
                    print("Reward model updated.")
                else:
                    print("Get enough feedbacks, stop updating reward model.")
                    
        return result
    
    def _generate_gif(self, filename):
        images = []
        obs = self.eval_env.reset()
        done = False
        while not done:
            # Predict the action for the current observation
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            
            # Capture the raw RGB observation for GIF
            try:
                current_obs = self.eval_env.get_attr('raw_rgb_obs')[0]
            except:
                current_obs = obs  # Adjust this depending on your environment structure
            rgb_image = utils.obs_to_image(current_obs)
            
            image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0  # (C, H, W)
            image = image.reshape(1, 3, image.shape[1], image.shape[2]) # (1, C, H, W)
                
            images.append(image)
        
        # Save the images as a GIF
        imageio.mimsave(filename, images, fps=10, loop=0)
    
    def learn_reward(self, first_flag=0):
        # get feedbacks
        labeled_queries = 0 
        if first_flag == 1:
            # if it is first time to get feedback, need to use random sampling
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            labeled_queries = self.reward_model.uniform_sampling()
            # if self.cfg.feed_type == 0:
            #     labeled_queries = self.reward_model.uniform_sampling()
            # else:
            #     raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        total_acc = 0
        # reward update epoch
        reward_update = 100

        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(reward_update): 
                self.reward_model.train()
                train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break
        
        print("Reward function is updated!! ACC: " + str(total_acc))
        return total_acc, self.reward_model.vlm_label_acc
    
class SaveRewardModelCallback(BaseCallback):
    def __init__(self, reward_model, save_path, env_name, verbose=0):
        super().__init__(verbose)
        self.reward_model = reward_model
        self.save_path = save_path
        self.env_name = env_name

    def _on_step(self) -> bool:
        # save reward when there is a new best model
        steps = self.num_timesteps
        self.reward_model.save(self.save_path, steps)
        if self.verbose >= 1:
            print(f"Saved reward model to {self.save_path} at step {steps}")
        return True
    
def train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_k, seed, pos_reward=0.1, neg_reward=-0.1, absolute_alpha=0.5, confidence_threshold=0.52):
    if reward_mode == "RL-VLM-F":
        name = f"{reward_mode}_{vlm}"
    elif reward_mode == "VLM-AR3L":
        name = f"{reward_mode}_{vlm}_k{reward_k}"
    else:
        name = f"{reward_mode}"
    path = f"{env_name}/{task}/{algo}/{name}/{seed}"
    print(f"[INFO] model path: {path}")

    reward_model = None
    observation_space = 160 * 256 * 3
    action_space = 8
    image_height = 160
    image_width = 256

    if reward_mode == "RL-VLM-F" or reward_mode == "VLM-AR3L":
        reward_model = RewardModel(
            observation_space,
            action_space,
            mb_size = 100, # [NOTE]
            log_dir = f"./model/{path}",
            capacity = 5e5 if "train" in mode else 40000, # [NOTE]
            lr = 3e-4, # [NOTE]
            ### vlm parameters
            vlm=vlm,
            env_name=env_name,
            task=task,
            ### image-based reward model parameters
            image_reward=True,
            image_height=image_height,
            image_width=image_width,
            resize_factor=1,
            resnet=True,
            reward_mode = reward_mode,
        )

    eval_env = utils.make_minedojo_env(task)
    eval_env = utils.make_reward_env(eval_env, env_name, task, "dense")

    if "train" in mode:
        train_env = utils.make_minedojo_env(task)
        train_env = utils.make_reward_env(train_env, env_name, task, reward_mode, reward_k, reward_model, pos_reward, neg_reward, absolute_alpha, confidence_threshold)
    
    # Load the MineCLIP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mineclip_model = MineCLIP(
        arch="vit_base_p16_fz.v2.t2",
        resolution=(160, 256),
        pool_type="attn.d2.nh8.glusw",
        image_feature_dim=512,
        mlp_adapter_spec="v0-2.t0",
        hidden_dim=512,
    ).to(device)
    mineclip_model.load_ckpt("attn.pth")
    logger.info(f" Load MineCLIP model")

    # Define the action transformation function
    transform = utils.minedojo_transform_action_multi_discrete
    eval_env = MineCLIPFeatureWrapper(eval_env, mineclip_model)
    eval_env = CustomActionWrapper(eval_env, transform)
    
    if "random" not in mode:
        eval_env = make_vec_env(lambda: eval_env, n_envs=1)
        eval_env = VecFrameStack(eval_env, n_stack=4)

    if "train" in mode:
        train_env = MineCLIPFeatureWrapper(train_env, mineclip_model)
        train_env = CustomActionWrapper(train_env, transform)
        train_env = make_vec_env(lambda: train_env, n_envs=4)
        train_env = VecFrameStack(train_env, n_stack=4)

    if "train" in mode:
        # tensorboard writer
        tb_log_dir = f"./tensorboard/{path}"

        # Create the evaluation callback
        eval_callback = EvalCallbackWithGif(
            env_name=env_name,
            task=task,
            eval_env=eval_env,
            n_eval_episodes=5,
            best_model_save_path=f'./model/{path}',
            log_path=f'./logs_result/{path}',
            eval_freq=5120,  # n_envs * n_steps = 4 * 5120 = 20480
            gif_path=f'./gifs/{path}',
            deterministic=True,
            render=False,
            reward_model=reward_model,
            k=reward_k,
            pos_reward=pos_reward,
            neg_reward=neg_reward,
        )

    if mode=='noop':
        obs = eval_env.reset()
        done = False
        total_reward = 0
        images = []
        t = 0
        while not done:
            action = eval_env.action_space.no_op()
            if t == 0:
                action[3] = 13
            else:
                action[3] = 12
            t = 1
            print(action)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            rgb_image = utils.obs_to_image(obs)
            images.append(rgb_image)

        print(f"Total reward: {total_reward}")
    elif "random" in mode:
        obs = eval_env.reset()
        done = False
        total_reward = 0
        images = []
        while not done:
            action = eval_env.action_space.sample()
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
            current_obs = eval_env.raw_rgb_obs
            rgb_image = utils.obs_to_image(current_obs)
            images.append(rgb_image)

        print(f"Total reward: {total_reward}")
        imageio.mimsave(f"gifs/{env_name}/{task}/random.gif", images, fps=10, loop=0)
        
    elif "eval" in mode:
        # check if the model exists
        if not os.path.exists(f"model/{path}/best_model.zip"):
            print(f"The model model/{path}/best_model.zip does not exist.")
            return [], []
        
        if "ppo" in algo:
            model = PPO.load(f"model/{path}/best_model", env=eval_env)
        else:
            raise NotImplementedError

        average_success_rate = []
        reward_list = []
        eval_times = 20
        for i in range(eval_times):
            obs = eval_env.reset()
            done = False
            total_reward = []
            images = []
            while not done:
                action, _ = model.predict(obs.copy(), deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward.append(reward)

                try:
                    current_obs = eval_env.get_attr('raw_rgb_obs')[0]
                except:
                    current_obs = obs
                rgb_image = utils.obs_to_image(current_obs)
            
                image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0  # (C, H, W)
                image = image.reshape(1, 3, image.shape[1], image.shape[2]) # (1, C, H, W)

                images.append(image)
        
            # Save the images as a GIF
            imageio.mimsave(f"gifs/{path}/eval-{i}.gif", images, fps=10, loop=0)
            
            print(f"Episode {i} total reward: {sum(total_reward)}")
            average_success_rate.extend(sum(total_reward) >= 10)
            reward_list.append(sum(total_reward))

        eval_env.close()
        return average_success_rate, reward_list
    elif "train" in mode:
        if "ppo" in algo:
            model = PPO("MlpPolicy", train_env, ent_coef=0.01, verbose=1, tensorboard_log=tb_log_dir, seed=seed)
        else:
            raise NotImplementedError

        model.learn(total_timesteps=1000000, callback=eval_callback)
        model.save(f"./model/{path}")

        train_env.close()
        eval_env.close()
        torch.cuda.empty_cache()

def parse_range_or_list(value):
    """
    Parses a string representing a range (e.g., "1-5") or a list (e.g., "1,2,3").
    Returns a list of integers.
    """
    if "-" in value:  # Range format
        start, end = map(int, value.split("-"))
        return list(range(start, end + 1))
    elif "," in value:  # List format
        return list(map(int, value.split(",")))
    else:
        # Single integer
        return [int(value)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='ERROR') # DEBUG, INFO, WARNING, ERROR, CRITICAL
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        help="Mode of operation: e.g., 'train', 'eval', etc."
    )
    parser.add_argument(
        '--task', 
        type=str, 
        required=True, 
        help="task: e.g., 'hunt_cow', 'combat_spider'"
    )
    parser.add_argument(
        '--algo',
        type=str,
        default='ppo',
        help="algo: e.g., 'ppo'"
    )
    parser.add_argument(
        '--reward_mode', 
        type=str, 
        default='dense',
        help="reward_mode: e.g., 'dense', 'sparse', 'phi', 'RL-VLM-F', 'clip', 'mineclip', 'VLM-AR3L'"
    )
    parser.add_argument(
        '--vlm',
        type=str,
        default='',
        help="vlm: e.g., 'phi3.5', 'MiniCPM-o2.6', 'gemini2.0'"
    )
    parser.add_argument(
        '--reward_k', 
        type=int, 
        default=16,
        help="reward_frame_k: e.g., 4, 8, 16 ..."
    )
    parser.add_argument(
        '--seed', 
        type=parse_range_or_list, 
        default=[1, 2, 3],
        help="random seed, e.g., '1,2,3', '1-3'"
    )
    parser.add_argument(
        '--pos_reward',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--neg_reward',
        type=float,
        default=-0.1,
    )
    parser.add_argument(
        '--absolute_alpha',
        type=float,
        default=0.5,
        help="alpha for absolute reward in VLM-AR3L, between 0 and 1"
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        default=0.52,
        help="confidence threshold for reward assignment in VLM-AR3L, between 0 and 1"
    )
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    loglevel = args.log
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)

    mode = args.mode
    env_name = "minedojo"
    task = args.task
    algo = args.algo
    reward_mode = args.reward_mode
    vlm = args.vlm
    reward_k = args.reward_k
    seed_list = args.seed
    absolute_alpha = args.absolute_alpha
    confidence_threshold = args.confidence_threshold

    if mode == 'random':
        train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_k, 1)
    # python run.py --mode train --task shear_sheep --algo ppo --reward_mode VLM-AR3L --seed 1
    elif mode == 'train':
        for seed in seed_list:
            train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_k, seed, args.pos_reward, args.neg_reward, absolute_alpha, confidence_threshold)
    elif mode == 'eval':
        average_success_rate = []
        average_reward = []
        name = f"{reward_mode}_k{reward_k}"
        if reward_mode == "RL-VLM-F":
            name = f"{reward_mode}_{vlm}"
        elif reward_mode == "VLM-AR3L":
            name = f"{reward_mode}_{vlm}_k{reward_k}"
        with open(f"./logs_result/{env_name}/{task}/{algo}/{name}/success_rate.txt", "a") as f:
            for seed in seed_list:
                success_rate, reward = train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_k, seed, args.pos_reward, args.neg_reward, absolute_alpha, confidence_threshold)
                if len(success_rate) == 0:
                    continue
                mean_success_rate = np.mean(success_rate)
                mean_reward = np.mean(reward)
                print(f"seed {seed} success rate: {mean_success_rate}, reward: {mean_reward}")
                f.write(f"seed {seed} success rate: {mean_success_rate}, reward: {mean_reward}\n")
                for i in range(len(reward)):
                    f.write(f"reward {i}: {reward[i]}, success: {success_rate[i]}\n")
                average_success_rate.append(mean_success_rate)
                average_reward.append(mean_reward)
            print(f"success average: {np.mean(average_success_rate) * 100} SE: {np.std(average_success_rate)/ np.sqrt(len(average_success_rate)) * 100}")
            print(f"reward average: {np.mean(average_reward)} SE: {np.std(average_reward)/ np.sqrt(len(average_reward))}")
            f.write(f"success average: {np.mean(average_success_rate) * 100} SE: {np.std(average_success_rate)/ np.sqrt(len(average_success_rate)) * 100}\n")
            f.write(f"reward average: {np.mean(average_reward)} SE: {np.std(average_reward)/ np.sqrt(len(average_reward))}\n")