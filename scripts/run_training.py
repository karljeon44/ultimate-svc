#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 6/10/23 11:07 AM
import logging
import os

import utils

logger = logging.getLogger(__name__)


def main():
  ### setup
  config = utils.init_config(name='ult-svc Training')
  utils.init_logging(config)
  os.makedirs(config.output_dir, exist_ok=True)

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
      "max_sentences": config.batch_size,
      'speaker_id': config.speaker,
      'work_dir': work_dir,
    }
    utils.update_yaml(diff_config_fpath, update_dict)

    # train
    abs_diff_dir = os.path.abspath(utils.DIFF_DIR)

    cmd = f"{utils.DIFF_VENV_PYTHON} run.py --config {diff_config_fpath} --exp_name {exp_name} --reset"
    utils.run_cmd(cmd, cwd=abs_diff_dir, env={'PYTHONPATH': abs_diff_dir}, cuda_version=config.diff_cuda)
  else:
    raise NotImplementedError()


if __name__ == '__main__':
  main()
