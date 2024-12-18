# Simple Reward

## Setup
```sh
git clone https://github.com/1224K/minedojo.git
```
Download [MineCLIP](https://drive.google.com/file/d/1uaZM1ZLBz2dZWcn85rZmjP7LV6Sg5PZW/view) and place the `attn.pth` file in this repository.

## Build the Docker Images
```sh
cd minedojo/docker-minedojo
docker build -t 1224k/minedojo .
```

## Run on omni-farm
```sh
cd omni-farm-isaac
```
### 1. Connet VPN
Copy your .ovpn client config file to secrets/client.ovpn and install the config
```sh
scripts/vpn/install_config.sh client.ovpn
source secrets/env.sh
scripts/vpn/disconnect.sh
scripts/vpn/connect.sh
```

### 2. Upload to nuclus
```sh
cd thirdparty/omnicli
./omnicli 
auth omniverse nvidia
copy "$path_to_this_repo/ppo_mineclip.py" "omniverse://nucleus.tpe1.local/Projects/$FARM_USER/minedojo/ppo_mineclip.py"
copy "$path_to_this_repo/utils.py" "omniverse://nucleus.tpe1.local/Projects/$FARM_USER/minedojo/utils.py"
copy "$path_to_this_repo/animal_zoo" "omniverse://nucleus.tpe1.local/Projects/$FARM_USER/minedojo/animal_zoo"
copy "$path_to_this_repo/mob_combat" "omniverse://nucleus.tpe1.local/Projects/$FARM_USER/minedojo/mob_combat"
copy "$path_to_this_repo/VLM" "omniverse://nucleus.tpe1.local/Projects/$FARM_USER/minedojo/VLM"
copy "$path_to_this_repo/attn.pth" "omniverse://nucleus.tpe1.local/Projects/$FARM_USER/minedojo/attn.pth"
cd ../..
```

### 3. Nuclues to mnt
```sh
scripts/submit_task.sh k-copy \
"/run.sh \
    --download-src 'omniverse://$NUCLEUS_HOSTNAME/Projects/$FARM_USER/minedojo' \
    --download-dest '/mnt/nfs/$FARM_USER/minedojo' \
    'cd /mnt/nfs/$FARM_USER/minedojo' \
    'rm -r tensorboard' \
    'rm -r logs' \
    'rm -r model' \
    'rm -r gifs' \
    'mkdir tensorboard' \
    'mkdir logs' \
    'mkdir model' \
    'mkdir gifs' \
    'ls'" \
"nuclues -> mnt: minedojo"
```

### 4. Run task
```sh
scripts/submit_task.sh k-minedojo \
"/run.sh \
    'sudo mkdir -p /src' \
    'cd /src' \
    'sudo cp /mnt/nfs/$FARM_USER/minedojo/ppo_mineclip.py /src/ppo_mineclip.py' \
    'sudo cp /mnt/nfs/$FARM_USER/minedojo/utils.py /src/utils.py' \
    'sudo cp /mnt/nfs/$FARM_USER/minedojo/attn.pth /src/attn.pth' \
    'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/animal_zoo /src/animal_zoo' \
    'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/mob_combat /src/mob_combat' \
    'sudo cp -r /mnt/nfs/$FARM_USER/minedojo/VLM /src/VLM' \
    'sudo cp  /mnt/nfs/$FARM_USER/minedojo/setup_cuda_env.sh /src/setup_cuda_env.sh' \
    'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/tensorboard /src/tensorboard' \
    'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/model /src/model' \
    'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/gifs /src/gifs' \
    'sudo ln -s /mnt/nfs/$FARM_USER/minedojo/logs /src/logs_result' \
    'export MPLCONFIGDIR=/mnt/nfs/$FARM_USER/matplotlib_cache' \
    'export HF_HOME=/mnt/nfs/$FARM_USER/huggingface_cache' \
    'export HF_MODULES_CACHE=/mnt/nfs/$FARM_USER/hf_modules_cache' \
    'sudo chmod -R 777 /src' \
    'sudo chmod -R 777 /mnt/nfs/$FARM_USER' \
    'xvfb-run python ppo_mineclip.py --mode train --task milk_cow --reward_mode phi --reward_steps 16 --seed 3'" \
"train milk_cow_phi_n16_noise0.0 seed3"
```

