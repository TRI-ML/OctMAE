#!/bin/bash

# --gpus \"device=0,1,2,3\" \

wandb docker-run -it \
  --mount type=bind,source="$(pwd)",target=/opt/app \
  --ipc=host \
  -v $HOME/.aws:/root/.aws \
  -v $HOME/.cache:/root/.cache \
  --gpus \"device=0,1\" \
  octmae:latest /bin/bash
