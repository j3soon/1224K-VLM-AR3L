# gymnasium==0.29.1
# shimmy==1.3.0
# stable_baselines3 2.3.2

# minedojo
# gym==0.21.0

# metaworld
# gym==0.25.2

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
        self.reward_update_interval = 1000  if env_name=='minedojo' else 4000
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
        images_with_reward = []
        rewards = []
        done = False
        while not done:
            # Predict the action for the current observation
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = self.eval_env.step(action)
            
            if "minedojo" in self.env_name:
                # Capture the raw RGB observation for GIF
                try:
                    current_obs = self.eval_env.get_attr('raw_rgb_obs')[0]
                except:
                    current_obs = obs  # Adjust this depending on your environment structure
                rgb_image = utils.obs_to_image(current_obs)
            elif "metaworld" in self.env_name:  
                rgb_image = self.eval_env.render()[::-1, :, :]
            
            image = rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0  # (C, H, W)
            image = image.reshape(1, 3, image.shape[1], image.shape[2]) # (1, C, H, W)

            if self.reward_model is not None:
                if self.reward_model.R3L_or_PBRL == "R3L":
                    if len(images) > 0:
                        if len(images) >= self.k:
                            pre_image = images[-self.k]
                        else:
                            pre_image = images[0]
                        logits = self.reward_model.r_hat_pair(pre_image, image)[0] # (2,)
                        logits_inverse = self.reward_model.r_hat_pair(image, pre_image)[0] # (2,)
                        probs  = softmax(logits)
                        probs_inverse = softmax(logits_inverse)
                        if probs[1] > 0.52 and probs_inverse[0] > 0.52:
                            reward = self.pos_reward
                        elif probs[0] > 0.52 and probs_inverse[1] > 0.52:
                            reward = self.neg_reward
                        else:
                            reward = 0
                    else:
                        reward = 0
                else:
                    reward = self.reward_model.r_hat(image)
                
            # Draw the reward on the image
            rewards.append(reward)
            import matplotlib.pyplot as plt
            if self.reward_model == None or self.reward_model.R3L_or_PBRL == "R3L":
                plt.plot(rewards, color='C4')
            else:
                plt.plot(rewards, color='C0')
            plt.title("VLM rewards")
            plt.ylabel("reward")
            if self.reward_model == None or self.reward_model.R3L_or_PBRL == "R3L":
                plt.ylim(self.neg_reward - 0.1, self.pos_reward + 0.1)
            else:
                plt.ylim(-1, 1.1)
            # plt.xlim(0, 50)
            plt.savefig("vlm_rewards.png")
            plt.close()
            rewards_img = plt.imread("vlm_rewards.png")
            rewards_img = (rewards_img * 255).astype(np.uint8)
            rewards_img = cv2.resize(rewards_img, (rgb_image.shape[1] , int(rewards_img.shape[0] * (rgb_image.shape[1] / rewards_img.shape[1]))))

            image_with_reward = utils.concatenate_images_vertical([Image.fromarray(rgb_image), Image.fromarray(rewards_img)], 10)

            images.append(image)
            images_with_reward.append(image_with_reward)
        
        # Save the images as a GIF
        imageio.mimsave(filename, images_with_reward, fps=10, loop=0)
    
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
            # elif self.cfg.feed_type == 1:
            #     labeled_queries = self.reward_model.disagreement_sampling()
            # elif self.cfg.feed_type == 2:
            #     labeled_queries = self.reward_model.entropy_sampling()
            # elif self.cfg.feed_type == 3:
            #     labeled_queries = self.reward_model.kcenter_sampling()
            # elif self.cfg.feed_type == 4:
            #     labeled_queries = self.reward_model.kcenter_disagree_sampling()
            # elif self.cfg.feed_type == 5:
            #     labeled_queries = self.reward_model.kcenter_entropy_sampling()
            # else:
            #     raise NotImplementedError
        
        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries
        
        train_acc = 0
        total_acc = 0
        # [NOTE] reward update epoch
        if "minedojo" in self.env_name:
            reward_update = 50
        elif "metaworld" in self.env_name:  
            if "soccer" in self.task:
                reward_update = 5
            else:
                reward_update = 10

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
    
def train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_steps, reward_noise, reward_k, seed, pos_reward=0.1, neg_reward=-0.1):
    name = f"{reward_mode}_n{reward_steps}_k{reward_k}_noise{reward_noise}"
    if reward_mode == "RL-VLM-F":
        name = f"{reward_mode}_{vlm}"
    elif reward_mode == "VLM-R3L":
        name = f"{reward_mode}_{vlm}_k{reward_k}"
        if neg_reward != -0.1 or pos_reward != 0.1:
            name = f"{reward_mode}_{vlm}_k{reward_k}_pos{pos_reward}_neg{abs(neg_reward)}"
    path = f"{env_name}/{task}/{algo}/{name}/{seed}"
    print(f"[INFO] model path: {path}")

    reward_model = None
    if env_name == "minedojo":
        observation_space = 160 * 256 * 3
        action_space = 8
        image_height = 160
        image_width = 256
    elif env_name == "metaworld":
        observation_space = 39
        action_space = 4
        image_height = 300
        image_width = 300

    if reward_mode == "RL-VLM-F" or reward_mode == "VLM-R3L":
        reward_model = RewardModel(
            observation_space,
            action_space,
            mb_size = 100, # [NOTE]
            log_dir = f"./model/{path}",
            capacity= 5e5 if "minedojo" in env_name else 40000, # [NOTE]
            lr = 1e-4 if "soccer" in task else 3e-4, # [NOTE]
            ### vlm parameters
            vlm=vlm,
            env_name=task,
            ### image-based reward model parameters
            image_reward=True,
            image_height=image_height,
            image_width=image_width,
            resize_factor=1,
            resnet=True,
            cached_label_path= f"data/gemini1.0/{task}/seed_{seed}/" if "gemini1.0" == vlm else None,
            R3L_or_PBRL= "PBRL" if reward_mode == "RL-VLM-F" else "R3L",
        )
        ## load the reward model
        # reward_model.load(f"./model/{path}", 24)

    if "minedojo" in env_name:
        eval_env = utils.make_minedojo_env(task)
        eval_env = utils.make_reward_env(eval_env, env_name, task, "dense")
        # eval_env = utils.make_reward_env(eval_env, env_name, task, 'phi')


        if "train" in mode:
            train_env = utils.make_minedojo_env(task)
            train_env = utils.make_reward_env(train_env, env_name, task, reward_mode, reward_steps, reward_noise, reward_k, reward_model, pos_reward, neg_reward)
        
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
        transform = utils.minedojo_transform_action_multi_discrete2
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

    elif "metaworld" in env_name:
        eval_env = utils.make_metaworld_env(task, seed)
        eval_env = utils.make_reward_env(eval_env, env_name, task, "dense")
        if "train" in mode:
            train_env = utils.make_metaworld_env(task, seed)
            train_env = utils.make_reward_env(train_env, env_name, task, reward_mode, reward_steps, reward_noise, reward_k, reward_model, pos_reward, neg_reward)
    
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
            eval_freq=(5120 if env_name=="minedojo" else 20480),  # n_envs * n_steps = 4 * 5120 = 20480
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
            if "minedojo" in env_name:
                current_obs = eval_env.raw_rgb_obs
                rgb_image = utils.obs_to_image(current_obs)
            elif "metaworld" in env_name:
                rgb_image = eval_env.render()[::-1, :, :]
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
        elif "sac" in algo:
            model = SAC.load(f"model/{path}/best_model", env=eval_env)
        average_success_rate = []
        reward_list = []
        eval_times = 20 if "minedojo" in env_name else 1
        for i in range(eval_times):
            obs = eval_env.reset()
            done = False
            total_reward = []
            images = []
            success = 0
            while not done:
                action, _ = model.predict(obs.copy(), deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward.append(reward)
                # use non-stacked observation to save gif
                if "minedojo" in env_name:
                    current_obs = eval_env.get_attr('raw_rgb_obs')[0]
                    rgb_image = utils.obs_to_image(current_obs)
                elif "metaworld" in env_name:
                    rgb_image = eval_env.render()[::-1, :, :]
                    success = max(success, info['success'])
                ## draw & add reward curve
                # plt.plot(total_reward, color='C0' if "RL-VLM-F" in reward_mode else 'C4')
                # plt.title("VLM rewards")
                # plt.ylabel("reward")
                # if "RL-VLM-F" in reward_mode:
                #     plt.ylim(-1.2, 1.2)
                # else:
                #     plt.ylim(-0.1, 0.2)
                # plt.savefig("vlm_rewards.png")
                # plt.close()
                # vlm_rewards_img = plt.imread("vlm_rewards.png")
                # vlm_rewards_img = (vlm_rewards_img * 255).astype(np.uint8)
                # import cv2
                # vlm_rewards_img = cv2.resize(vlm_rewards_img, (rgb_image.shape[1], int(vlm_rewards_img.shape[0] * (rgb_image.shape[1] / vlm_rewards_img.shape[1]))))
                # rgb_image = utils.concatenate_images_vertical([Image.fromarray(rgb_image), Image.fromarray(vlm_rewards_img)], 10)

                images.append(rgb_image)

            total_reward = sum(total_reward)
            print(f"Total reward: {total_reward}")
            imageio.mimsave(f"gifs/{path}/eval-{i}.gif", images, fps=60, loop=0)
            if "minedojo" in env_name:
                average_success_rate.extend(total_reward >= 10)
            elif "metaworld" in env_name:
                average_success_rate.append(success)
            reward_list.append(total_reward)

        eval_env.close()
        return average_success_rate, reward_list
    elif "train" in mode:
        if "ppo" in algo:
            model = PPO("MlpPolicy", train_env, ent_coef=0.01, verbose=1, tensorboard_log=tb_log_dir, seed=seed)
        elif "sac" in algo:
            policy_kwargs = dict(
                net_arch=dict(
                    pi=[256, 256, 256],  # Actor network
                    qf=[256, 256, 256],  # Critic network
                ),
                activation_fn=torch.nn.ReLU,  # Activation function
            )
            model = SAC("MlpPolicy", train_env, verbose=1, tensorboard_log=tb_log_dir, seed=seed, policy_kwargs=policy_kwargs,
                        learning_rate=0.0003, batch_size=512, learning_starts=9000) # NOTE num_unsup_steps = 9000
            print(model.policy)

        model.learn(total_timesteps=1000000, callback=eval_callback)
        model.save(f"./model/{path}")
        if reward_mode == "phi" or reward_mode == "deepseekVL" or reward_mode == "VLM-R3L":
            # save the accuracy
            if not os.path.exists(f"./logs_result/{path}"):
                os.makedirs(f"./logs_result/{path}")
            with open(f"./logs_result/{path}/vlm_accuracy.txt", "w") as f:
                total_acc = 0
                total_cnt = 0
                if 'minedojo' in env_name:
                    for i in range(4):
                        total_acc += train_env.get_attr('vlm_acc')[i]
                        total_cnt += train_env.get_attr('vlm_cnt')[i]
                elif 'metaworld' in env_name:
                    total_acc = train_env.get_attr('vlm_acc')
                    total_cnt = train_env.get_attr('vlm_cnt')
                if total_cnt > 0:
                    f.write(f"vlm_accuracy: {total_acc / total_cnt}")

        train_env.close()
        eval_env.close()
        torch.cuda.empty_cache()

    elif mode=='gen_data':
        if "ppo" in algo:
            model =  PPO.load(f"model/{path}/best_model", env=eval_env)
        elif "sac" in algo:
            model =  SAC.load(f"model/{path}/best_model", env=eval_env)
        obs = eval_env.reset()
        done = False
        total_reward = 0
        images = []
        rewards = []
        step = 0
        n_step = 10
        image_path = f"./VLM/data/{task}/"
        label0_path = image_path + "0/"
        label1_path = image_path + "1/"
        os.makedirs(label0_path, exist_ok=True)
        os.makedirs(label1_path, exist_ok=True)
        label0_count = len(os.listdir(label0_path)) / 2
        label1_count = len(os.listdir(label1_path)) / 2
        print("original label 0 count: ", label0_count)
        print("original label 1 count: ", label1_count)
        # rename the images
        i0 = 0 
        i = 0
        while i0 < label0_count:
            if os.path.exists(label0_path + str(int(i)) + "0.png"):
                os.rename(label0_path + str(int(i)) + "0.png", label0_path + str(int(i0)) + "0.png")
                os.rename(label0_path + str(int(i)) + "1.png", label0_path + str(int(i0)) + "1.png")
                i0 += 1
            i += 1
        i1 = 0
        i = 0
        while i1 < label1_count:
            if os.path.exists(label1_path + str(int(i)) + "0.png"):
                os.rename(label1_path + str(int(i)) + "0.png", label1_path + str(int(i1)) + "0.png")
                os.rename(label1_path + str(int(i)) + "1.png", label1_path + str(int(i1)) + "1.png")
                i1 += 1
            i += 1

        # 21
        while label1_count < 50:
            obs = eval_env.reset()
            done = False
            step = 0
            while not done:
                action, _ = model.predict(obs.copy())
                obs, reward, done, info = eval_env.step(action)
                if "minedojo" in env_name:
                    current_obs = eval_env.get_attr('raw_rgb_obs')[0]
                    rgb_image = utils.obs_to_image(current_obs)
                elif "metaworld" in env_name:
                    rgb_image = eval_env.render()[::-1, :, :]
                images.append(rgb_image)
                rewards.append(reward)
                if step >= n_step and (step % 4 == 0 or reward >=1):
                    if rewards[step] > rewards[step-n_step] and label1_count < 50:
                        # save images[step] and images[step-n_step] to label 1
                        image = Image.fromarray(images[step-n_step])
                        image.save(label1_path + str(int(label1_count)) + "0.png")
                        image = Image.fromarray(images[step])
                        image.save(label1_path + str(int(label1_count)) + "1.png")
                        label1_count += 1

                        # addition save images[step-n_step] and images[step] to label 0
                        if label0_count < 50:
                            image = Image.fromarray(images[step])
                            image.save(label0_path + str(int(label0_count)) + "0.png")
                            image = Image.fromarray(images[step-n_step])
                            image.save(label0_path + str(int(label0_count)) + "1.png")
                            label0_count += 1

                    elif rewards[step] <= rewards[step-n_step] and label0_count < 50:
                        # save images[step] and images[step-n_step] to label 0
                        image = Image.fromarray(images[step-n_step])
                        image.save(label0_path + str(int(label0_count)) + "0.png")
                        image = Image.fromarray(images[step])
                        image.save(label0_path + str(int(label0_count)) + "1.png")
                        label0_count += 1
                step += 1
        

def draw(name, env_name, task, exp, max_timestep=1000000, sigma=1):
    mean_rewards = {}
    se_rewards = {}
    
    for exp_name in exp:
        if "inRLVLMF" in exp_name:
            if "metaworld" in env_name or "softgym" in env_name:
                env_task = f"{env_name}_{task}"
            else:
                env_task = task
            import pandas as pd
            if not os.path.exists(f'../RL-VLM-F/exp/{exp_name.replace("inRLVLMF/", "")}/{env_task}'):
                print(f'[ERROR] The experiment ../RL-VLM-F/exp/{exp_name.replace("inRLVLMF/", "")}/{env_task} does not exist.')
                return
            rewards_timestep = {}
            # find all the eval.csv
            for date_time in os.listdir(f'../RL-VLM-F/exp/{exp_name.replace("inRLVLMF/", "")}/{env_task}'):
                path = f'../RL-VLM-F/exp/{exp_name.replace("inRLVLMF/", "")}/{env_task}/{date_time}'
                while True:
                    if os.path.exists(f'{path}/eval.csv'):
                        csv = pd.read_csv(f'{path}/eval.csv')
                        print(f"[INFO] Load data from {path}/eval.csv")
                        break
                    else:
                        for dir in os.listdir(path):
                            path = f'{path}/{dir}'
                if env_name=="metaworld":
                    reward = csv['success_rate']
                elif env_name=="minedojo":
                    reward = csv['true_episode_reward']
                elif "gym" in env_name or "softgym" in env_name:
                    reward = csv['episode_reward']
                else:
                    reward = csv['success']
                timestep = csv['step']
                # add 0 reward for the first timestep
                # rewards_timestep[0] = [0]

                for i in range(len(reward)):
                    if timestep[i] > max_timestep:
                        break
                    if timestep[i] not in rewards_timestep:
                        rewards_timestep[timestep[i]] = []
                    rewards_timestep[timestep[i]].append(reward[i])      
                
        else:
            if not os.path.exists(f'logs/{env_name}/{task}/{exp_name}'):
                print(f"[ERROR] The experiment {env_name}/{task}/{exp_name} does not exist.")
                return
            rewards_timestep = {}
            for file in os.listdir(f'logs/{env_name}/{task}/{exp_name}'):
                if os.path.isdir(f'logs/{env_name}/{task}/{exp_name}/{file}') != True:
                    continue
                data = np.load(f'logs/{env_name}/{task}/{exp_name}/{file}/evaluations.npz')
                print(f"[INFO] Load data from logs/{env_name}/{task}/{exp_name}/{file}/evaluations.npz")
                if env_name=="metaworld":
                    reward = data['successes'] * 100
                elif env_name=="minedojo":
                    if task == "shear_sheep":
                        reward = data['successes'] * 100
                    else:    
                        reward = data['results'] / 11 * 100
                timestep = data['timesteps']
                for i in range(len(reward)):
                    if timestep[i] > max_timestep:
                        break
                    if timestep[i] not in rewards_timestep:
                        rewards_timestep[timestep[i]] = []
                    rewards_timestep[timestep[i]].append(np.mean(reward[i]))

        mean_rewards[exp_name] = {t: np.mean(rewards_timestep[t]) for t in rewards_timestep}
        se_rewards[exp_name] = {t: np.std(rewards_timestep[t]) / np.sqrt(len(rewards_timestep[t])) for t in rewards_timestep}
    
    if "no_human" in name:
        colors = ['C4', 'C0', 'tab:orange', 'tab:green', 'C1']
    else:
        if env_name == "minedojo":
            colors = ['C4', 'C0', 'tab:orange', 'tab:green', 'C5', 'C3']
            # colors = ['tab:orange', 'tab:green', 'C5', 'C3']
        else:
            colors = ['C4', 'C0', 'tab:orange', 'C5', 'C3']
            # colors = [ 'C0', 'C5', 'C3']
        # colors = ['C0', 'tab:orange']
        # colors = ['tab:orange', 'C4']
        # colors = ['C0', 'C4']
        # colors = ['C4', 'tab:orange', 'tab:green', 'tab:blue', 'C3']
    # colors = ['tab:blue', 'tab:orange', 'tab:green']
    for exp_name in exp:
        label_name = exp_name
        if "_noise0.0" in exp_name:
            label_name = label_name.replace("_noise0.0", "")
        if "sac/" in exp_name:
            label_name =  label_name.replace("sac/", "")
        if "ppo/" in  exp_name:
            label_name =  label_name.replace("ppo/", "")
        if "_n16" in exp_name:
            label_name = label_name.replace("_n16", "")
        if "_n4" in exp_name:
            label_name = label_name.replace("_n4", "")
        # if "_k16" in exp_name:
        #     label_name = label_name.replace("_k16", "")
        # if "_k4" in exp_name:
        #     label_name = label_name.replace("_k4", "")
        if "mineclip" in label_name:
            label_name = "MineCLIP"
        if "clip" in label_name:
            label_name = "CLIP"
        if "dense" in label_name:
            label_name = "GT Dense Reward (Oracle)"
        if "sparse" in label_name:
            label_name = "GT Sparse Reward (Oracle)"
        if "phi_k16" == label_name:
            label_name = "VLM-R3L w/o Siamese"
        if "VLM-R3L_phi3.5_n1" == label_name:
            label_name = "VLM-R3L Siamese w/o Symmetric Comparison"
        if "phi" == label_name:
            label_name = "VLM-R3L (" + label_name + ")"
        if "VLM-R3L (phi)" == label_name:
            label_name = "VLM-R3L (phi3.5)"
        if "deepseekVL" == label_name:
            label_name = "VLM-R3L (deepseekVL)"
        if "RL-VLM-F/phi" in label_name: 
            label_name = "RL-VLM-F (phi)"
        if "RL-VLM-F_phi3.5" == label_name: 
            label_name = "RL-VLM-F (ResNet)"
        if "RL-VLM-F_n1" == label_name: 
            label_name = "RL-VLM-F (CNN)"
        if "oracle_n1_" in exp_name:
            label_name = "VLM-R3L (oracle*)"
        if "VLM-R3L_phi4" in label_name:
            label_name = "VLM-R3L* (phi4)"
        if "inRLVLMF/VLM-R3L_k16_phi3.5_v3.1" == label_name:
            label_name = "VLM-R3L (Phi-3.5)"
        if "VLM-R3L_gemini2.0" == label_name:
            label_name = "VLM-R3L (Ours)"
        if "RL-VLM-F_gemini2.0" == label_name:
            label_name = "RL-VLM-F"
        if "inRLVLMF/gemini1.0" == exp_name:
            label_name = "RL-VLM-F (gemini1.0)"
        if "inRLVLMF/phi" == exp_name:
            label_name = "RL-VLM-F (phi3.5)"
        if "inRLVLMF/VLM-R3L_k16_gemini2.0" == label_name:
            label_name = "VLM-R3L (Gemini-2.0)"
        if "inRLVLMF/RL-VLM-F_MiniCPM-o2.6" == label_name:
            label_name = "RL-VLM-F (MiniCPM-o2.6)"
        if "inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6_v3.1" == label_name:
            label_name = "VLM-R3L (MiniCPM-o2.6)"
        if "phi_n16_k16_noise0.0" == exp_name:
            label_name = "VLM-R3L (phi3.5)"
        # if "VLM-R3L_phi3.5_k16" == label_name:
        #     label_name = "VLM-R3L"
        # if "VLM-R3L_phi3.5_n1_k16" == label_name:
        #     label_name = "VLM-R3L w/o Symmetric Comparison"
        from scipy.ndimage import gaussian_filter1d
        smooth_mean = gaussian_filter1d(list(mean_rewards[exp_name].values()), sigma=sigma)
        smooth_se = gaussian_filter1d(list(se_rewards[exp_name].values()), sigma=sigma)
        if "noise" in label_name or "Dense" in label_name or "Sparse" in label_name or  "gt_task_reward" in label_name :
            plt.plot(list(mean_rewards[exp_name].keys()), smooth_mean, label=label_name, color=colors[exp.index(exp_name)%len(colors)], linestyle="--")
            plt.fill_between(list(mean_rewards[exp_name].keys()), smooth_mean - smooth_se, smooth_mean + smooth_se, color=colors[exp.index(exp_name)%len(colors)], alpha=0.1)
        else:
            plt.plot(list(mean_rewards[exp_name].keys()), smooth_mean, label=label_name, color=colors[exp.index(exp_name)%len(colors)])
            plt.fill_between(list(mean_rewards[exp_name].keys()), smooth_mean - smooth_se, smooth_mean + smooth_se, color=colors[exp.index(exp_name)%len(colors)], alpha=0.2)

    if max_timestep != 1000000:
        import matplotlib.ticker as mticker
        ax = plt.gca()
        # ax.set_xlim(-0.05 * max_timestep, max_timestep * 1.05)
        ax.set_xticks([0, 0.2*max_timestep, 0.4*max_timestep, 0.6*max_timestep, 0.8*max_timestep, max_timestep])
        ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.text(max_timestep, -0.08, format(max_timestep,".0e").replace("e+0", "e"), transform=ax.get_xaxis_transform(),ha='center', va='top', fontsize=10)

    font1 = {'family': 'serif', 'weight': 'normal', 'size': 18}
    font2 = {'family': 'serif', 'weight': 'normal', 'size': 16}
    plt.subplots_adjust(left = 0.15,bottom=0.128)
    plt.xlabel('Timesteps', fontdict=font2)
    if "gym" in env_name or "softgym" in env_name:
        plt.ylabel('Episode Reward', fontdict=font2)
    else:
        plt.ylabel('Success Rate', fontdict=font2)
        if task != "hunt_cow":
            plt.ylim(-5, 105)

    task = task.replace("_", " ")
    print(task)
    if task == "CartPole-v1":
        task = "Cart Pole"
    elif task == "PassWater":
        task = "Pass Water"
    elif task == "RopeFlattenEasy":
        task = "Straighten Rope"
    elif task == "soccer-v2":
        task = "Soccer"
    elif task == "drawer-open-v2":
        task = "Drawer Open"
    elif task == "sweep-into-v2":
        task = "Sweep Into"
    else:
        task = task.title()
    plt.title(task, fontdict=font1)
    # save the figure
    if not os.path.exists(f'plot/{env_name}/{task}'):
        os.makedirs(f'plot/{env_name}/{task}')
    plt.savefig(f"plot/{env_name}/{task}/{name}.png")
    print(f"[INFO] Save the plot to plot/{env_name}/{task}/{name}.png")

    plt.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.15), borderaxespad=0.)
    plt.savefig(f"plot/{env_name}/{task}/{name}_legend.png", bbox_inches='tight')
    print(f"[INFO] Save the plot to plot/{env_name}/{task}/{name}_legend.png")

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
        '--env', 
        type=str, 
        required=True, 
        help="env: e.g., 'minedojo', 'metaworld'"
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
        help="algo: e.g., 'ppo', 'sac'"
    )
    parser.add_argument(
        '--reward_mode', 
        type=str, 
        default='dense',
        help="reward_mode: e.g., 'dense', 'sparse', 'phi', 'RL-VLM-F', 'clip', 'mineclip', 'deepseekVL', 'VLM-R3L'"
    )
    parser.add_argument(
        '--vlm',
        type=str,
        default='',
        help="vlm: e.g., 'phi', 'deepseekVL'"
    )
    parser.add_argument(
        '--reward_steps', 
        type=int, 
        default=1,
        help="reward_steps: e.g., 1, 2, 4 ..."
    )
    parser.add_argument(
        '--reward_noise', 
        type=float, 
        default=0.,
        help="reward_noise: e.g., 0., 0.3, ..."
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
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    loglevel = args.log
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level)

    mode = args.mode
    env_name = args.env
    task = args.task
    algo = args.algo
    reward_mode = args.reward_mode
    vlm = args.vlm
    reward_steps = args.reward_steps
    reward_noise = args.reward_noise
    reward_k = args.reward_k
    seed_list = args.seed
    
    # python ppo_mineclip.py --mode random --env metaworld --task drawer-open-v2
    if mode == 'random':
        train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_steps, reward_noise, reward_k, 1)
    # ln -s ./logs ./logs_result
    # xvfb-run -a python ppo_mineclip.py --mode train --env minedojo --task combat_spider --algo ppo --reward_mode VLM-R3L --vlm phi3.5 --reward_k 16 --seed 1
    # python ppo_mineclip.py --mode train --env metaworld --task drawer-open-v2 --algo sac --reward_mode VLM-R3L --vlm gemini1.0 --seed 1
    # xvfb-run python ppo_mineclip.py --mode train --env minedojo --task combat_spider --algo ppo --reward_mode mineclip --seed 1
    # xvfb-run python ppo_mineclip.py --mode train --env minedojo --task shear_sheep --algo ppo --reward_mode sparse --seed 1-3
    elif mode == 'train':
        for seed in seed_list:
            train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_steps, reward_noise, reward_k, seed, args.pos_reward, args.neg_reward)
    # xvfb-run python ppo_mineclip.py --mode eval --env minedojo --task combat_spider --reward_mode oracle
    # python ppo_mineclip.py --mode eval --env metaworld --task drawer-open-v2 --algo sac --reward_mode phi --reward_steps 16 --reward_k 16 --seed 1
    elif mode == 'eval':
        average_success_rate = []
        average_reward = []
        name = f"{reward_mode}_n{reward_steps}_k{reward_k}_noise{reward_noise}"
        if reward_mode == "RL-VLM-F":
            name = f"{reward_mode}_{vlm}"
        elif reward_mode == "VLM-R3L":
            name = f"{reward_mode}_{vlm}_k{reward_k}"
        with open(f"./logs_result/{env_name}/{task}/{algo}/{name}/success_rate.txt", "a") as f:
            for seed in seed_list:
                success_rate, reward = train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_steps, reward_noise, reward_k, seed)
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
    # python ppo_mineclip.py --mode draw --env minedojo --task milk_cow
    elif mode == 'draw':
        if env_name=="minedojo":
            # python ppo_mineclip.py --mode draw --env minedojo --task shear_sheep
                # draw(f'compare', env_name, task, ['ppo/VLM-R3L_phi3.5_k16', 'ppo/RL-VLM-F_n1_k16_noise0.0', 'ppo/clip_n1_k16_noise0.0', 'ppo/mineclip_n1_k16_noise0.0',
                #                                 'ppo/dense_n1_k16_noise0.0', 'ppo/sparse_n1_k16_noise0.0'], sigma=2)
                # draw(f'compare', env_name, task, ['ppo/VLM-R3L_phi3.5_k16', 'ppo/RL-VLM-F_phi3.5', 'ppo/clip_n1_k16_noise0.0', 'ppo/mineclip_n1_k16_noise0.0',
                #                                 'ppo/dense_n1_k16_noise0.0', 'ppo/sparse_n1_k16_noise0.0'], sigma=2)
                # draw(f'ablation VLM', env_name, task, ['ppo/VLM-R3L_gemini2.0_k16', 'ppo/VLM-R3L_phi3.5_k16', 'ppo/VLM-R3L_MiniCPM-o2.6_k16'], sigma=2)
                draw(f'compare_gemini2.0', env_name, task, ['ppo/VLM-R3L_gemini2.0_k16', 'ppo/RL-VLM-F_gemini2.0', 'ppo/clip_n1_k16_noise0.0', 'ppo/mineclip_n1_k16_noise0.0',
                                                'ppo/dense_n1_k16_noise0.0', 'ppo/sparse_n1_k16_noise0.0'], sigma=2)
                # draw(f'compare_no_human', env_name, task, ['ppo/VLM-R3L_phi3.5_k16', 'ppo/RL-VLM-F_n1_k16_noise0.0', 'ppo/clip_n1_k16_noise0.0', 
                #                                            'ppo/mineclip_n1_k16_noise0.0'], sigma=2)
                # draw(f'compare_no_human deepseek', env_name, task, ['ppo/deepseekVL_n4_k4_noise0.0', 'ppo/RL-VLM-F_n1_k16_noise0.0', 'ppo/clip_n1_k16_noise0.0', 
                #                                            'ppo/mineclip_n1_k16_noise0.0'], sigma=2)
                # draw(f'RL-VLM-F_resnet', env_name, task, ['ppo/RL-VLM-F_n1_k16_noise0.0', 'ppo/RL-VLM-F_phi3.5'], sigma=2)
                # metaworld
                # draw(f'compare', env_name, task, ['sac/dense_n1_k16_noise0.0', 'sac/sparse_n1_k16_noise0.0', 'sac/phi_n16_k16_noise0.0', 
                #                                   'inRLVLMF/phi', 'inRLVLMF/clip'], sigma=3)
                # draw(f'ablation_siamese', env_name, task, ['ppo/VLM-R3L_phi3.5_k16', 'ppo/phi_n16_k16_noise0.0', 'ppo/VLM-R3L_phi3.5_n1_k16'], sigma=2)
                # draw(f'MiniCPM-o2.6', env_name, task, ['ppo/VLM-R3L_phi3.5_k16', 'ppo/VLM-R3L_MiniCPM-o2.6_k16'], sigma=2)
                # draw(f'phi3.5_neg', env_name, task, ['ppo/VLM-R3L_phi3.5_k16', 'ppo/VLM-R3L_phi3.5_k16_neg0.0'], sigma=2)
                # draw(f'RL-VLM-F resnet', env_name, task, ['ppo/RL-VLM-F_n1_k16_noise0.0', 'ppo/RL-VLM-F_phi3.5'], sigma=2)
        # python ppo_mineclip.py --mode draw --env metaworld --task sweep-into-v2
        elif env_name=="metaworld":
            # draw(f'compare', env_name, task, ['inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6_v3.1', 'inRLVLMF/RL-VLM-F_MiniCPM-o2.6_v3', 'inRLVLMF/clip',
            #                                 'inRLVLMF/gt_task_reward', 'inRLVLMF/sparse_task_reward'], sigma=3)
            # if task == "drawer-open-v2":
            #     draw(f'compare_gemini2.0', env_name, task, ['inRLVLMF/VLM-R3L_k16_gemini2.0', 'inRLVLMF/RL-VLM-F_gemini2.0', 'inRLVLMF/clip',
            #                                     'inRLVLMF/gt_task_reward', 'inRLVLMF/sparse_task_reward'], sigma=2, max_timestep=300000)
            # else:
            #     draw(f'compare_gemini2.0', env_name, task, ['inRLVLMF/VLM-R3L_k16_gemini2.0', 'inRLVLMF/RL-VLM-F_gemini2.0', 'inRLVLMF/clip',
            #                                     'inRLVLMF/gt_task_reward', 'inRLVLMF/sparse_task_reward'], sigma=3)
                
            draw(f'ablation VLM', env_name, task, ['inRLVLMF/VLM-R3L_k16_gemini2.0', 'inRLVLMF/VLM-R3L_k16_phi3.5_v3.1', 'inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6_v3.1'], sigma=3)
            # draw(f'compare_no_human', env_name, task, ['inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6_v3.1', 'inRLVLMF/RL-VLM-F_MiniCPM-o2.6_v3', 'inRLVLMF/clip'], sigma=3)
            # draw(f'VLM-R3L_n', env_name, task, ['ppo/phi_n1_k16_noise0.0', 'ppo/phi_n4_k16_noise0.0', 'ppo/phi_n16_k16_noise0.0'], max_timestep=800000, sigma=2)
            # draw(f'VLM-R3L_k', env_name, task, ['ppo/phi_n4_k4_noise0.0', 'ppo/phi_n4_k16_noise0.0'], sigma=2)
            # draw(f'R3L_vs_dense', env_name, task, [ 'ppo/oracle_n1_k16_noise0.0', 'ppo/dense_n1_k16_noise0.0'], sigma=2)
            # draw(f'phi_vs_deepseekVL', env_name, task, [ 'ppo/phi_n4_k4_noise0.0', 'ppo/deepseekVL_n4_k4_noise0.0'], sigma=2)
            # draw(f'VLM-R3L* oracle', env_name, task, [ 'ppo/VLM-R3L_oracle_n1_k16', 'ppo/VLM-R3L_phi4_n1_k16'], sigma=2)
            # draw(f'RL-VLM-F_vs_R3L_phi3.5', env_name, task, ['inRLVLMF/phi', 'sac/phi_n16_k16_noise0.0'], sigma=2)
            # draw(f'RL-VLM-F_vs_R3L_gemini1.0', env_name, task, ['inRLVLMF/gemini1.0', 'inRLVLMF/VLM-R3L_k4_gemini1.0'], sigma=2)
            # draw(f'R3L_gemini1.0_k', env_name, task, ['inRLVLMF/VLM-R3L_gemini1.0', 'inRLVLMF/VLM-R3L_k4_gemini1.0'], sigma=2)
            # draw(f'RL-VLM-F_vs_R3L_MiniCPM-o2.6', env_name, task, ['inRLVLMF/RL-VLM-F_MiniCPM-o2.6', 'inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6'], sigma=2)
            # draw(f'RL-VLM-F_vs_R3L_phi3.5', env_name, task, ['ppo/RL-VLM-F_n1_k16_noise0.0', 'ppo/VLM-R3L_phi3.5_n1_k16'], sigma=2)
            # draw(f'VLM-R3L_MiniCPM-o2.6', env_name, task, ['inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6_v3','inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6_v3.1',
            #                                                         'inRLVLMF/RL-VLM-F_MiniCPM-o2.6_v3'], sigma=2)
        # python ppo_mineclip.py --mode draw --env gym --task CartPole-v1
        elif env_name=="gym":
            # draw(f'compare_gemini2.0', env_name, task, ['inRLVLMF/VLM-R3L_k8_gemini2.0', 'inRLVLMF/RL-VLM-F_gemini2.0', 'inRLVLMF/clip', 'inRLVLMF/gt_task_reward', 'inRLVLMF/sparse_task_reward'], max_timestep=100000, sigma=2)
            draw(f'ablation VLM', env_name, task, ['inRLVLMF/VLM-R3L_k8_gemini2.0', 'inRLVLMF/VLM-R3L_k16_phi3.5_v3.1', 'inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6_v3.1'], max_timestep=100000, sigma=2)

        # python ppo_mineclip.py --mode draw --env softgym --task PassWater
        elif env_name=="softgym":
            # draw(f'compare_gemini2.0', env_name, task, ['inRLVLMF/VLM-R3L_k16_pos1_neg1_gemini2.0', 'inRLVLMF/RL-VLM-F_gemini2.0',  'inRLVLMF/clip', 'inRLVLMF/gt_task_reward', 'inRLVLMF/sparse_task_reward'], sigma=1,  max_timestep=100000)
            draw(f'ablation VLM', env_name, task, ['inRLVLMF/VLM-R3L_k16_pos1_neg1_gemini2.0', 'inRLVLMF/VLM-R3L_k16_pos1_neg1_phi3.5', 'inRLVLMF/VLM-R3L_k16_pos1_neg1_MiniCPM-o2.6'], max_timestep=100000, sigma=1)
            # draw(f'gemini1.0', env_name, task, ['inRLVLMF/VLM-R3L_k1_gemini1.0_v2', 'inRLVLMF/VLM-R3L_k2_gemini1.0_v2',
            #                                 'inRLVLMF/gt_task_reward',], sigma=3)
            # draw(f'compare_no_human', env_name, task, ['inRLVLMF/VLM-R3L_k16_MiniCPM-o2.6_v3.1', 'inRLVLMF/RL-VLM-F_MiniCPM-o2.6_v3', 'inRLVLMF/clip'], sigma=3)
            # if task == "ClothFoldDiagonal":
            #     draw(f'gemini2.0', env_name, task, ['inRLVLMF/VLM-R3L_k1_gemini2.0', 'inRLVLMF/RL-VLM-F_gemini2.0', 'inRLVLMF/clip', 'inRLVLMF/gt_task_reward', 'inRLVLMF/sparse_task_reward'], sigma=3, max_timestep=15000)
    # python ppo_mineclip.py --mode gen_data --env metaworld --task sweep-into-v2 --algo sac --reward_mode phi --reward_steps 16 --reward_k 16
    # python ppo_mineclip.py --mode gen_data --env metaworld --task sweep-into-v2 --algo sac --reward_mode sparse
    elif mode == 'gen_data':
        train_and_evaluate(mode, env_name, task, algo, reward_mode, vlm, reward_steps, reward_noise, reward_k, 1)
