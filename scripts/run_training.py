#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 6/10/23 11:07 AM
import json
import logging
import os
import sys
import shutil
from datetime import datetime

import utils

logger = logging.getLogger(__name__)


## TODO (June 10): sovitz training unstable
##   UserWarning: Grad strides do not match bucket view strides. This may indicate grad
##   was not created according to the gradient layout contract, or that the param's strides changed since DDP was constructed.  This is not an error, but may impair performance.
##   grad.sizes() = [32, 1, 4], strides() = [4, 1, 1]
##   bucket_view.sizes() = [32, 1, 4], strides() = [4, 4, 1] (Triggered internally at ../torch/csrc/distributed/c10d/reducer.cpp:323.)
##   Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass


def main():
  ### setup
  config = utils.init_config(name='ult-svc Training')
  utils.init_logging(config)
  os.makedirs(config.output_dir, exist_ok=True)
  logger.info("%s started at %s", config.name, datetime.now())

  ### call training
  model = config.model
  if model == utils.DIFF_SVC:
    diff_config_fpath = utils.DIFF_CONFIG_NSF_YAML if 'nsf' in config.diff_config else utils.DIFF_CONFIG_YAML

    # diff-svc treats `work_dir` (parent dir) and `exp_name` (child dir) differently
    work_dir, exp_name = os.path.split(config.output_dir)

    # update diff config (`scripts/run_preprocessing.py` already overwrote some of these)
    update_dict = {
      'binary_data_dir': os.path.abspath(config.training_dir),
      'hubert_path': os.path.abspath(utils.HUBERT_SOFT_FPATH),
      'vocoder_ckpt': os.path.abspath(utils.NSF_HIFIGAN_MODEL_FPATH),
      'max_epochs': config.epochs,
      "max_sentences": config.batch_size,
      'speaker_id': config.speaker,
      'log_interval': config.log_interval,
      'val_check_interval': config.save_eval_interval,
      'work_dir': work_dir,
    }
    utils.update_yaml(diff_config_fpath, update_dict)

    # run diff training
    abs_diff_dir = os.path.abspath(utils.DIFF_DIR)

    try:
      cmd = f"{utils.DIFF_VENV_PYTHON} run.py --config {diff_config_fpath} --exp_name {exp_name} --reset"
      utils.run_cmd(cmd, cwd=abs_diff_dir, env={'PYTHONPATH': abs_diff_dir}, cuda_version=config.diff_cuda)
    except KeyboardInterrupt: # catch ctrl+c
      logger.info("Caught exit signal, terminating training immediately")
      pass

  elif model in [utils.DDSP_SVC, utils.SHALLOW_DIFF_SVC]:
    if model == utils.DDSP_SVC:
      config_fpath = utils.DDSP_COMBSUB_CONFIG_YAML
      cwd = utils.ABS_DDSP_DIR
    else:
      config_fpath = utils.SHALLOW_DIFF_CONFIG_YAML
      cwd = utils.ABS_SHALLOW_DIFF_DIR

    # update ddsp config
    update_dict = {
      'env/expdir': os.path.abspath(config.output_dir),
      'train/batch_size': config.batch_size,
      'train/cache_device': 'cuda',
      'train/epochs': config.epochs,
      'train/interval_log': config.log_interval,
      'train/interval_val': config.save_eval_interval
    }
    utils.update_yaml(config_fpath, update_dict)

    # run ddsp training
    cmd = f"{sys.executable} train.py -c  {config_fpath}"
    try:
      utils.run_cmd(cmd, cwd=cwd, env={'PYTHONPATH': cwd})
    except KeyboardInterrupt: # catch ctrl+c
      logger.info("Caught exit signal, terminating immediately")
      pass

  elif model == utils.SOVITZ_SVC:
    # 0. copy model checkpoints
    d_ckpt_fname = os.path.basename(utils.SOVITZ_D_CHECKPOINT_FPATH)
    g_ckpt_fname = os.path.basename(utils.SOVITZ_G_CHECKPOINT_FPATH)

    # tmp folder containing model results
    tmp_dir = os.path.join(utils.ABS_SOVITZ_DIR, 'logs/tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    d_ckpt_fpath = os.path.join(tmp_dir, d_ckpt_fname)
    if not os.path.exists(d_ckpt_fpath):
      shutil.copyfile(utils.SOVITZ_D_CHECKPOINT_FPATH, d_ckpt_fpath)

    g_ckpt_fpath = os.path.join(tmp_dir, g_ckpt_fname)
    if not os.path.exists(g_ckpt_fpath):
      shutil.copyfile(utils.SOVITZ_G_CHECKPOINT_FPATH, g_ckpt_fpath)

    sovitz_config_fpath = utils.SOVITZ_CONFIG_JSON
    with open(sovitz_config_fpath) as f:
      sovitz_config = json.load(f)

    for k,v in sovitz_config.items():
      for kk,vv in v.items():
        if kk == 'batch_size' and config.batch_size != vv:
          sovitz_config[k][kk] = config.batch_size
        if kk == 'epochs' and config.epochs != vv:
          sovitz_config[k][kk] = config.epochs
        elif kk == 'log_interval' and config.log_interval != vv:
          sovitz_config[k][kk] = config.log_interval
        elif kk == 'eval_interval' and config.save_eval_interval != vv:
          sovitz_config[k][kk] = config.save_eval_interval

    if config.encoder == 'contentvec256l9':
      sovitz_config['model']['speech_encoder'] = 'vec256l9'
      sovitz_config['model']['ssl_dim'] = 256
      sovitz_config['model']['gin_channels'] = 256
    elif config.encoder == 'contentvec768l12':
      sovitz_config['model']['speech_encoder'] = 'vec768l12'
      sovitz_config['model']['ssl_dim'] = 768
      sovitz_config['model']['gin_channels'] = 256

    with open(sovitz_config_fpath, 'w') as f:
      json.dump(sovitz_config, f, indent=4)

    # run soivitz training
    try:
      cmd = f"{sys.executable} train.py -c  {sovitz_config_fpath} -m tmp"
      utils.run_cmd(cmd, cwd=utils.ABS_SOVITZ_DIR, env={'PYTHONPATH': utils.ABS_SOVITZ_DIR})
    except KeyboardInterrupt: # catch ctrl+c
      logger.info("Caught exit signal, terminating training immediately")
      pass

    # now move `tmp` folder back to the project root level
    logger.info("Storing model outputs to `%s`", config.output_dir)
    for fname in os.listdir(tmp_dir):
      shutil.move(os.path.join(tmp_dir, fname), config.output_dir)

  else:
    raise NotImplementedError()

  logger.info("%s finished at %s", config.name, datetime.now())

if __name__ == '__main__':
  main()
