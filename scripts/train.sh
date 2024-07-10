#!/bin/bash

if [ -z "$4" ]
then
  python train.py --project_name $1 --run_name $2 --config $3
else
  python train.py --project_name $1 --run_name $2 --config $3 --checkpoint $4
fi
