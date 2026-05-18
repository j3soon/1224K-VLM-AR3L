import numpy as np
import torch
from scipy.special import softmax

class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device, window=1, store_image=False, image_size=300, reward_mode="VLM-AR3L", k=16, pos_reward=1, neg_reward=-1, absolute_alpha=0.5, confidence_threshold=0.52):
        self.capacity = capacity
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)
        self.window = window
        self.store_image = store_image
        if self.store_image:
            self.images = np.empty((capacity, image_size, image_size, 3), dtype=np.uint8)

        self.reward_mode = reward_mode
        self.k = k
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward
        self.absolute_alpha = absolute_alpha
        self.confidence_threshold = confidence_threshold

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done, done_no_max, image=None):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.not_dones_no_max[self.idx], not done_no_max)
        if image is not None and self.store_image:
            np.copyto(self.images[self.idx], image)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    
    def add_batch(self, obs, action, reward, next_obs, done, done_no_max):
        
        next_index = self.idx + self.window
        if next_index >= self.capacity:
            self.full = True
            maximum_index = self.capacity - self.idx
            np.copyto(self.obses[self.idx:self.capacity], obs[:maximum_index])
            np.copyto(self.actions[self.idx:self.capacity], action[:maximum_index])
            np.copyto(self.rewards[self.idx:self.capacity], reward[:maximum_index])
            np.copyto(self.next_obses[self.idx:self.capacity], next_obs[:maximum_index])
            np.copyto(self.not_dones[self.idx:self.capacity], done[:maximum_index] <= 0)
            np.copyto(self.not_dones_no_max[self.idx:self.capacity], done_no_max[:maximum_index] <= 0)
            remain = self.window - (maximum_index)
            if remain > 0:
                np.copyto(self.obses[0:remain], obs[maximum_index:])
                np.copyto(self.actions[0:remain], action[maximum_index:])
                np.copyto(self.rewards[0:remain], reward[maximum_index:])
                np.copyto(self.next_obses[0:remain], next_obs[maximum_index:])
                np.copyto(self.not_dones[0:remain], done[maximum_index:] <= 0)
                np.copyto(self.not_dones_no_max[0:remain], done_no_max[maximum_index:] <= 0)
            self.idx = remain
        else:
            np.copyto(self.obses[self.idx:next_index], obs)
            np.copyto(self.actions[self.idx:next_index], action)
            np.copyto(self.rewards[self.idx:next_index], reward)
            np.copyto(self.next_obses[self.idx:next_index], next_obs)
            np.copyto(self.not_dones[self.idx:next_index], done <= 0)
            np.copyto(self.not_dones_no_max[self.idx:next_index], done_no_max <= 0)
            self.idx = next_index
        
    def relabel_with_predictor(self, predictor):
        if not self.store_image:
            batch_size = 200
        else:
            if self.reward_mode == "VLM-AR3L":
                # calculate the batch size based on the episode length
                batch_size = 0
                for nd in self.not_dones:
                    batch_size += 1
                    if nd == 0:
                        break
            else:
                batch_size = 32
        total_iter = int(self.idx/batch_size)
        
        if self.idx > batch_size*total_iter:
            total_iter += 1
            
        for index in range(total_iter):
            last_index = (index+1)*batch_size
            if (index+1)*batch_size > self.idx:
                last_index = self.idx
            
            if not self.store_image:
                obses = self.obses[index*batch_size:last_index]
                actions = self.actions[index*batch_size:last_index]
                inputs = np.concatenate([obses, actions], axis=-1)
            else:
                inputs = self.images[index*batch_size:last_index]
                inputs = np.transpose(inputs, (0, 3, 1, 2))
                inputs = inputs.astype(np.float32) / 255.0

                if self.reward_mode == "VLM-AR3L":
                    pre_inputs = inputs[:-self.k]
                    # the first k-1 elements are inputs[0]
                    pre_inputs = np.concatenate([[inputs[0]] * (self.k), pre_inputs], axis=0)

            pred_reward_relative = None
            pred_reward_absolute = None
            if self.reward_mode == "VLM-AR3L":
                pred_reward = np.zeros(inputs.shape[0], dtype=np.float32)
                inner_batch_size = 32
                inner_total_iter = int(inputs.shape[0]/inner_batch_size)
                if inputs.shape[0] > inner_batch_size*inner_total_iter:
                    inner_total_iter += 1
                for inner_index in range(inner_total_iter):
                    inner_last_index = (inner_index+1)*inner_batch_size
                    if (inner_index+1)*inner_batch_size > inputs.shape[0]:
                        inner_last_index = inputs.shape[0]
                        
                    inner_pre_inputs = pre_inputs[inner_index*inner_batch_size:inner_last_index]
                    inner_inputs = inputs[inner_index*inner_batch_size:inner_last_index]

                    if self.reward_mode == "VLM-AR3L":
                        logits = predictor.r_hat_batch_pair(inner_pre_inputs, inner_inputs)                
                        probs = softmax(logits, axis=1)
                        logits_inverse = predictor.r_hat_batch_pair(inner_inputs, inner_pre_inputs)
                        probs_inv = softmax(logits_inverse, axis=1)
                        p_fwd0, p_fwd1 = probs[:,0], probs[:,1]
                        p_bwd0, p_bwd1 = probs_inv[:,0], probs_inv[:,1]
                        idx  = np.arange(inner_index*inner_batch_size, inner_last_index)
                        mask_pos = (p_fwd1 > self.confidence_threshold) & (p_bwd0 > self.confidence_threshold)
                        pred_reward[idx[mask_pos]] = self.pos_reward
                        mask_neg = (p_fwd0 > self.confidence_threshold) & (p_bwd1 > self.confidence_threshold)
                        pred_reward[idx[mask_neg]] = self.neg_reward    
                pred_reward_relative = pred_reward.reshape(-1, 1)
            if self.reward_mode == "RL-VLM-F" or self.reward_mode == "VLM-AR3L":
                pred_reward_absolute = predictor.r_hat_batch(inputs)
            
            if pred_reward_relative is not None and pred_reward_absolute is not None:
                pred_reward = self.absolute_alpha * pred_reward_absolute + (1 - self.absolute_alpha) * pred_reward_relative
            elif pred_reward_relative is not None:
                pred_reward = pred_reward_relative
            elif pred_reward_absolute is not None:
                pred_reward = pred_reward_absolute

            self.rewards[index*batch_size:last_index] = pred_reward
            
        torch.cuda.empty_cache()
            
    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max
    
    def sample_state_ent(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
                                           device=self.device)
        
        if self.full:
            full_obs = self.obses
        else:
            full_obs = self.obses[: self.idx]
        full_obs = torch.as_tensor(full_obs, device=self.device)
        
        return obses, full_obs, actions, rewards, next_obses, not_dones, not_dones_no_max