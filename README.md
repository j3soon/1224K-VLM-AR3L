# VLM-AR3L
![Framework](demo/VLM-AR3L.png)

## Demo
<p align="center">
  <img src="demo/cartpole.gif"  alt="CartPole"        width="120"/>
  <img src="demo/straighten_rope.gif"      alt="Straighten Rope" width="120"/>
  <img src="demo/pass_water.gif" alt="Pass Water"     width="120"/>
  <img src="demo/soccer.gif" alt="Soccer"     width="120"/>
  <img src="demo/sweep.gif" alt="Sweep Into"     width="120"/>
  <img src="demo/open_drawer.gif"  alt="Drawer Open"        width="120"/>
</p>

<p align="center">
  <img src="demo/combat_spider.gif"      alt="Combat Spider" width="180"/>
  <img src="demo/milk_cow.gif" alt="Milk Cow"     width="180"/>
  <img src="demo/shear_sheep.gif" alt="Shear Sheep"     width="180"/>
  <img src="demo/hunt_cow.gif" alt="Hunt Cow"     width="180"/>
</p>

## Setup

Download [MineCLIP](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view) and place the `attn.pth` file in this repository.

## Build the Docker Images
- ### minedojo
  ```sh
  cd docker-minedojo
  docker build -t minedojo .
  ```
- ### gym & softgym & metaworld
  ```sh
  cd docker-metaworld
  docker build -t metaworld .
  ```

## Run task
| Environment | Tasks |
|---|---|
| gym | CartPole-v1, RingWorld |
| softgym | PassWater, RopeFlattenEasy |
| metaworld | drawer-open-v2, sweep-into-v2, soccer-v2 |
| minedojo | combat_spider, milk_cow, shear_sheep, hunt_cow |

```sh
python run_sac.py task=CartPole-v1 --config-name gym
python run_sac.py task=PassWater --config-name softgym
python run_sac.py task=drawer-open-v2 --config-name metaworld
python run_ppo.py task=combat_spider --config-name minedojo
```