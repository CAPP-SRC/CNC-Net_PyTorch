#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install curl -y
sudo apt-get install git -y
sudo apt-get install build-essential -y
sudo apt-get install libopenblas-dev -y
sudo apt-get install libgl1 -y

curl -O https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh
bash Anaconda3-2025.12-2-Linux-x86_64.sh
