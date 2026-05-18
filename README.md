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
    cd minedojo/docker-minedojo
    docker build -t minedojo .
    ```

## Run task
```sh
python run_ppo.py --mode train --task combat_spider --reward_mode VLM-AR3L --vlm gemini2.0
```

```sh
python run_sac.py \
  env=metaworld_drawer-open-v2 \
  exp_name=VLM-AR3L \
  reward_mode=VLM-AR3L \
  reward=learn_from_preference \
  vlm_label=1 \
  vlm=gemini2.0 \
  image_reward=1 \
  reward_batch=40 \
  segment=1 \
  teacher_eps_mistake=0 \
  reward_update=10 \
  num_interact=4000 \
  max_feedback=20000 \
  agent.params.actor_lr=0.0003 agent.params.critic_lr=0.0003 gradient_update=1 activation=tanh num_unsup_steps=9000 \
  num_train_steps=300100 agent.params.batch_size=512 double_q_critic.params.hidden_dim=256 double_q_critic.params.hidden_depth=3 \
  diag_gaussian_actor.params.hidden_dim=256 diag_gaussian_actor.params.hidden_depth=3  \
  feed_type=0 teacher_beta=-1 teacher_gamma=1  teacher_eps_skip=0 teacher_eps_equal=0 \
  num_eval_episodes=1 \
  seed=1
```