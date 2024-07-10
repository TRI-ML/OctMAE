#!bin/bash

mkdir -p $1

if [[ "$2" == "train" ]] ;then
  for i in $(seq -f "%06g" 1 9999)
  do
    echo "Uploading shard-$i.tar.gz..." 
    aws s3 cp $1/train/shard-$i.tar $3/train/shard-$i.tar
  done
elif [[ "$2" == "eval" ]] ;then
  for i in $(seq -f "%06g" 1 9999)
  do
    for datasetName in synth_eval ycb_video hb hope
    do
      echo "Uploading eval/${datasetName}/shard-$i.tar.gz..." 
      aws s3 cp $1/eval/${datasetName}/shard-$i.tar $3/eval/${datasetName}/shard-$i.tar
    done
  done
else
  echo "Please specify train or eval"
fi
