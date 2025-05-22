# Simple Reward

## Setup

Download [MineCLIP](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view) and place the `attn.pth` file in this repository.

## Build the Docker Images
- ### minedojo
    ```sh
    cd minedojo/docker-minedojo
    docker build -t minedojo .
    ```

## 4. Run task
'xvfb-run python ppo_mineclip.py --mode train --task milk_cow --reward_mode VLM-R3L --vlm phi3.5'"

