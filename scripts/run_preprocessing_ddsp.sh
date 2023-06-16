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

# ddsp and shallow-diff can share the virtualenv
echo "Activating ddsp (shallow-diffusion) venv.."
. $SHALLOW_DIFF_HOME/venv/bin/activate
which python

PYTHONPATH=$DDSP_HOME python $DDSP_HOME/preprocess.py -c configs/ddsp.yaml
