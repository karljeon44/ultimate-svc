#!/bin/bash

set -eu
set -o pipefail

HOME=$PWD
if [[ $HOME == *scripts ]]
then
  HOME=$(dirname $HOME)
fi
echo "ult-svc HOME: $HOME"

SHALLOW_DIFF_HOME="$HOME/models/Diffusion-SVC"
DDSP_HOME="$HOME/models/DDSP-SVC"

echo "Activating shallow-diffusion venv.."
. $SHALLOW_DIFF_HOME/venv/bin/activate
which python

PYTHONPATH=$DDSP_HOME python $DDSP_HOME/train.py -c configs/ddsp.yaml
