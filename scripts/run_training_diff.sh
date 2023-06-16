#!/bin/bash

set -eu
set -o pipefail

HOME=$PWD
if [[ $HOME == *scripts ]]
then
  HOME=$(dirname $HOME)
fi
echo "ult-svc HOME: $HOME"

DIFF_HOME="$HOME/models/diff-svc"

echo "Activating diffusion venv.."
. $DIFF_HOME/venv/bin/activate
which python

export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$DIFF_HOME python $DIFF_HOME/run.py --config configs/diff.yaml $@
