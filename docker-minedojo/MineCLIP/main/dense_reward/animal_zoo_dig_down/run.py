import torch
import torchvision
import numpy as np

from tqdm import tqdm
from mineclip import HuntCowDenseRewardEnv


if __name__ == "__main__":
    env = HuntCowDenseRewardEnv(
        step_penalty=0,
        nav_reward_scale=0.1,
        attack_reward=1,
        success_reward=10,
    )

    for i in tqdm(range(2), desc="Episode"):
        frames = []
        obs = env.reset()
        done = False
        pbar = tqdm(desc="Step")
        while not done:
            action = env.action_space.no_op()
            action[3] = 24 # change camera pitch by 180 degree
            action[5] = 3  # action "attack"
            obs, reward, done, info = env.step(action)
            frames.append(obs["rgb"])
            pbar.update(1)
        print(f"{i+1}-th episode ran successful!")
        frames = np.stack(frames)
        frames = torch.from_numpy(frames)
        frames = frames.permute(0,2,3,1)
        torchvision.io.write_video(f"/workspace/video{i}.mp4", frames, fps=24)
    env.close()
