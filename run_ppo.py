# minedojo
# gymnasium==0.29.1
# shimmy==1.3.0
# stable_baselines3 2.3.2
# gym==0.21.0

import os
import numpy as np
from stable_baselines3 import PPO
import torch
import gym
from gym.spaces import Box
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import imageio
from mineclip import MineCLIP
import logging
import argparse
import utils
from reward_model import RewardModel
import hydra
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
logger = logging.getLogger(__name__)

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
    
def train_and_evaluate(cfg):
    mode = cfg.mode
    env_name = cfg.env
    task = cfg.task
    algo = cfg.algo
    reward_mode = cfg.reward_mode
    vlm = cfg.vlm
    reward_k = cfg.k
    seed = cfg.seed
    pos_reward = cfg.pos_reward
    neg_reward = cfg.neg_reward
    absolute_alpha = cfg.absolute_alpha
    confidence_threshold = cfg.confidence_threshold

    work_dir = HydraConfig.get().runtime.output_dir
    print(f"[INFO] workspace: {work_dir}")

    model_dir = os.path.join(work_dir, "model")
    gif_dir = os.path.join(work_dir, "gifs")
    log_dir = os.path.join(work_dir, "logs_result")
    tb_log_dir = os.path.join(work_dir, "tensorboard")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)
    
    if reward_mode == "RL-VLM-F":
        name = f"{reward_mode}_{vlm}"
    elif reward_mode == "VLM-AR3L":
        name = f"{reward_mode}_{vlm}_k{reward_k}"
    else:
        name = f"{reward_mode}"
    path = f"{env_name}/{task}/{algo}/{name}/{seed}"
    print(f"[INFO] model path: {path}")

    reward_model = None

    if reward_mode == "RL-VLM-F" or reward_mode == "VLM-AR3L":
        reward_model = RewardModel(
            cfg.observation_space,
            cfg.action_space,
            mb_size=cfg.reward_batch,
            log_dir=model_dir,
            capacity=cfg.reward_capacity_train if "train" in mode else cfg.reward_capacity_eval,
            lr=cfg.reward_lr,

            vlm=cfg.vlm,
            env_name=cfg.env,
            task=cfg.task,

            image_reward=True,
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            resize_factor=1,
            resnet=True,
            reward_mode=cfg.reward_mode,
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
    mineclip_ckpt = cfg.mineclip_ckpt

    if not os.path.isabs(mineclip_ckpt):
        mineclip_ckpt = os.path.join(get_original_cwd(), mineclip_ckpt)

    print(f"[INFO] MineCLIP ckpt: {mineclip_ckpt}")
    mineclip_model.load_ckpt(mineclip_ckpt)

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
        eval_callback = EvalCallbackWithGif(
            env_name=cfg.env,
            task=cfg.task,
            eval_env=eval_env,
            n_eval_episodes=cfg.num_eval_episodes,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=cfg.eval_frequency,
            gif_path=gif_dir,
            deterministic=True,
            render=False,
            reward_model=reward_model,
            k=cfg.k,
            pos_reward=cfg.pos_reward,
            neg_reward=cfg.neg_reward,
        )

    if "eval" in mode:
        eval_model_dir = cfg.get("eval_model_dir", model_dir)
        eval_gif_dir = os.path.join(work_dir, "eval_gifs")
        os.makedirs(eval_gif_dir, exist_ok=True)

        best_model_path = os.path.join(eval_model_dir, "best_model.zip")
        final_model_path = os.path.join(eval_model_dir, "final_model.zip")

        if os.path.exists(best_model_path):
            load_path = best_model_path
        elif os.path.exists(final_model_path):
            load_path = final_model_path
        else:
            print(f"[ERROR] No model found in {eval_model_dir}")
            print(f"Checked: {best_model_path}")
            print(f"Checked: {final_model_path}")
            return [], []

        print(f"[INFO] Loading model from: {load_path}")

        if "ppo" in algo:
            model = PPO.load(load_path, env=eval_env)
        else:
            raise NotImplementedError

        average_success_rate = []
        reward_list = []
        eval_times = cfg.get("eval_times", 20)

        for i in range(eval_times):
            obs = eval_env.reset()
            done = False
            total_reward = []
            images = []

            while not done:
                action, _ = model.predict(obs.copy(), deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward.append(float(np.asarray(reward).mean()))

                try:
                    current_obs = eval_env.get_attr("raw_rgb_obs")[0]
                except Exception:
                    current_obs = obs

                rgb_image = utils.obs_to_image(current_obs)
                images.append(rgb_image)

            episode_reward = float(np.sum(total_reward))
            success = episode_reward >= 10

            gif_path = os.path.join(eval_gif_dir, f"eval-{i}.gif")
            imageio.mimsave(gif_path, images, fps=10, loop=0)

            print(f"Episode {i} total reward: {episode_reward}, success: {success}")

            average_success_rate.append(success)
            reward_list.append(episode_reward)

        eval_env.close()

        mean_success_rate = np.mean(average_success_rate) * 100
        se_success_rate = np.std(average_success_rate) / np.sqrt(len(average_success_rate)) * 100
        mean_reward = np.mean(reward_list)
        se_reward = np.std(reward_list) / np.sqrt(len(reward_list))

        result_path = os.path.join(work_dir, "eval_result.txt")
        with open(result_path, "w") as f:
            f.write(f"success average: {mean_success_rate} SE: {se_success_rate}\n")
            f.write(f"reward average: {mean_reward} SE: {se_reward}\n")
            for i, (r, s) in enumerate(zip(reward_list, average_success_rate)):
                f.write(f"episode {i}: reward={r}, success={s}\n")

        print(f"success average: {mean_success_rate} SE: {se_success_rate}")
        print(f"reward average: {mean_reward} SE: {se_reward}")
        print(f"[INFO] Save eval result to {result_path}")

        return average_success_rate, reward_list
    elif "train" in mode:
        if "ppo" in algo:
            model = PPO(
                "MlpPolicy",
                train_env,
                ent_coef=cfg.ent_coef,
                verbose=1,
                tensorboard_log=tb_log_dir,
                seed=cfg.seed,
            )
        else:
            raise NotImplementedError

        model.learn(
            total_timesteps=int(cfg.num_train_steps),
            callback=eval_callback
        )

        model.save(os.path.join(model_dir, "final_model"))

        train_env.close()
        eval_env.close()
        torch.cuda.empty_cache()

@hydra.main(config_path="config", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    numeric_level = getattr(logging, cfg.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {cfg.log_level}")
    logging.basicConfig(level=numeric_level)

    train_and_evaluate(cfg)

if __name__ == "__main__":
    main()    