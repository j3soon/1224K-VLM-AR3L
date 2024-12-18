#!/bin/bash -e

# Get the directory of this script.
# Reference: https://stackoverflow.com/q/59895
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)"

sed -i -E 's/<AgentSection mode="\w+">/<AgentSection mode="Adventure">/' $DIR/../MineDojo/minedojo/sim/mc_meta/minedojo_mission.xml.j2

echo "Done."
