# Docker image for MineDojo

A quick reference for using MineDojo in docker container. This may be useful since MineDojo has limited examples and contain some unresolved bugs.

## Build the Docker Images

```sh
git clone https://github.com/1224K/minedojo.git
cd minedojo/docker-minedojo
docker build -t 1224k/minedojo .
```

## Run the Docker Container

```sh
# stop and remove the container if exists
docker stop minedojo && docker rm minedojo
# launch the container
docker run --name minedojo -it --gpus all -p 8080:8080 -d -v $PWD:/workspace 1224k/minedojo tail -f /dev/null
# exec into the container
docker exec -it minedojo /bin/bash
```

In the container, run the followings to test the installation:

```sh
xvfb-run -a vglrun python /workspace/MineDojo/scripts/validate_install.py
xvfb-run -a vglrun python /workspace/MineCLIP/main/mineagent/run_single.py
xvfb-run -a vglrun python /workspace/MineCLIP/main/mineagent/run_env_in_loop.py
vglrun python /workspace/MineCLIP/main/dense_reward/animal_zoo/run.py
vglrun python /workspace/K/ppo_mineclip.py
```

If you're going to use the host MineDojo and MineCLIP, run the following in the container:

```sh
pip uninstall -y minedojo mineclip
rm -r ~/MineDojo ~/MineCLIP
pip install -e /workspace/MineCLIP
pip install -e /workspace/MineDojo
```

or simply rebuild the image.

## Miscellaneous

The following commands should be ran in the container.

### Recording Video

```sh
xvfb-run -a vglrun python /workspace/MineCLIP/main/dense_reward/animal_zoo_dig_down/run.py
```

observe the video where the player is digging downwards.

https://github.com/user-attachments/assets/58249395-9df4-4b9c-835d-05f54b4c57e0

### Adventure Mode

In the animal zoo example, destroying blocks might disrupt training. To prevent this, we can switch the game mode to Adventure, which disables player from destroying the terrain:

```sh
/workspace/scripts/set_adventure_mode.sh
```

and test it with the script:

```sh
vglrun python /workspace/MineCLIP/main/dense_reward/animal_zoo_dig_down/run.py
```

observe the video where the player is digging downwards but cannot due to Adventure mode.

https://github.com/user-attachments/assets/d2225abb-f34f-419c-a459-8cda977f3cb2

The script above simply apply a quick hack by changing the line `<AgentSection mode="Survival">` to `<AgentSection mode="Adventure">` in `MineDojo/minedojo/sim/mc_meta/minedojo_mission.xml.j2`, which can also be done manually.

You can revert this change by:

```sh
/workspace/scripts/set_survival_mode.sh
```

## References

- MineDojo documentation [Installation](https://docs.minedojo.org/sections/getting_started/install.html) section
- [j3soon/MineDojo](https://github.com/j3soon/MineDojo) (commit 8950d4e) forked from [MineDojo/MineDojo](https://github.com/MineDojo/MineDojo) (commit 2731bc2)
- [MineDojo/egl-docker](https://github.com/MineDojo/egl-docker) (commit 932155a)
- [MineDojo/MineCLIP](https://github.com/MineDojo/MineCLIP) (commit e6c06a0)
