#!bin/bash

if [[ "$2" == "train_tiny" ]] ;then
  mkdir -p $1/train
  for i in $(seq -f "%06g" 1 10)
  do
    echo "Downloading train/shard-$i.tar.gz..." 
    aws s3 cp s3://tri-ml-public.s3.amazonaws.com/github/octmae/train/shard-$i.tar.gz $1/train/
    echo "Decompressing train/shard-$i.tar.gz..." 
    pigz -d $1/train/shard-$i.tar.gz
  done
elif [[ "$2" == "train" ]] ;then
  mkdir -p $1/train
  for i in $(seq -f "%06g" 1 9999)
  do
    echo "Downloading train/shard-$i.tar.gz..." 
    aws s3 cp s3://tri-ml-public.s3.amazonaws.com/github/octmae/train/shard-$i.tar.gz $1/train/
    echo "Decompressing train/shard-$i.tar.gz..." 
    pigz -d $1/train/shard-$i.tar.gz
  done
elif [[ "$2" == "eval" ]] ;then
  mkdir -p $1/eval
  for i in $(seq -f "%06g" 1 9)
  do
    for datasetName in synth_eval ycb_video hb hope
    do
      mkdir -p $1/eval/${datasetName}
      echo "Downloading eval/${datasetName}/shard-$i.tar.gz..." 
      aws s3 cp s3://tri-ml-public.s3.amazonaws.com/github/octmae/eval/${datasetName}/shard-$i.tar.gz $1/eval/${datasetName}
      echo "Decompressing eval/${datasetName}/shard-$i.tar.gz..." 
      pigz -d $1/eval/${datasetName}/shard-$i.tar.gz
    done
  done
else
  echo "Please specify train or eval"
fi
