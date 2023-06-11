#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 6/10/23 7:08 PM
import logging
import os
import sys
from datetime import datetime

import utils

logger = logging.getLogger(__name__)


def main():
  ### setup
  config = utils.init_config(name='ult-svc Preprocessing')
  utils.init_logging(config)
  logger.info("%s started at %s", config.name, datetime.now())

  ### call inference
  model = config.model
  if model == utils.DIFF_SVC:
    diff_config_fpath = utils.DIFF_CONFIG_NSF_YAML if 'nsf' in config.diff_config else utils.DIFF_CONFIG_YAML

    raise NotImplementedError()

  elif model == utils.DDSP_SVC:
    input_flag = f'-i {os.path.abspath(config.input_file)}'
    model_flag = f'-m {os.path.abspath(config.model_checkpoint)}'
    output_flag = f'-o {os.path.abspath(config.input_file.replace(".wav", "_svc.wav"))}'
    pitch_extractor_flag = f'-pe {config.pitch_extractor}' # should use `crepe` here??

    # run ddsp inference
    cmd = f"{sys.executable} main.py {input_flag} {model_flag} {output_flag} {pitch_extractor_flag}"
    utils.run_cmd(cmd, cwd=utils.ABS_DDSP_DIR, env={'PYTHONPATH': utils.ABS_DDSP_DIR})

  else:
    raise NotImplementedError()

  logger.info("%s finished at %s", config.name, datetime.now())

if __name__ == '__main__':
  main()
