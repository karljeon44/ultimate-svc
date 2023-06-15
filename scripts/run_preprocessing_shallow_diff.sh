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

echo "Activating shallow-diffusion venv.."
. $SHALLOW_DIFF_HOME/venv/bin/activate
which python

PYTHONPATH=$SHALLOW_DIFF_HOME python $SHALLOW_DIFF_HOME/preprocess.py -c configs/shallow-diff-config.yaml
