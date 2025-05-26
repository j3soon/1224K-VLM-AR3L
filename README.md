# VLM-R3L
![Framework](demo/framework.png)

## Demo
<p align="center">
  <img src="demo/cartpole.gif"  alt="CartPole"        width="120"/>
  <img src="demo/rope.gif"      alt="Straighten Rope" width="120"/>
  <img src="demo/pass_water.gif" alt="Pass Water"     width="120"/>
  <img src="demo/soccer.gif" alt="Soccer"     width="120"/>
  <img src="demo/sweep_into.gif" alt="Sweep Into"     width="120"/>
  <img src="demo/drawer_open.gif"  alt="Drawer Open"        width="120"/>
</p>

<p align="center">
  <img src="demo/combat_spider.gif"      alt="Combat Spider" width="180"/>
  <img src="demo/milk_cow.gif" alt="Milk Cow"     width="180"/>
  <img src="demo/shear_sheep.gif" alt="Shear Sheep"     width="180"/>
  <!-- <img src="demo/hunt_cow.gif" alt="Hunt Cow"     width="180"/> -->
</p>
## Setup

Download [MineCLIP](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view) and place the `attn.pth` file in this repository.

## Build the Docker Images
- ### minedojo
    ```sh
    cd minedojo/docker-minedojo
    docker build -t minedojo .
    ```

## Run task
```sh
xvfb-run python run.py --mode train --env minedojo --task combat_spider --algo ppo --reward_mode VLM-R3L --vlm phi3.5
```
