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


## TODO (June 10): support for ddsp/sovitz inference with shallow-diff

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

    output_flag = f'-o {os.path.abspath(config.input_file.replace(".wav", "_ddsp.wav"))}'
    pitch_extractor_flag = f'-pe {config.pitch_extractor}' # should use `crepe` here??

    # run ddsp inference
    if config.diff_checkpoint is not None:
      ddsp_flag = f'-ddsp {os.path.abspath(config.model_checkpoint)}'
      diff_flag = f'-diff {os.path.abspath(config.diff_checkpoint)}'
      kstep_flag = f'-kstep {config.kstep}'
      cmd = f"{sys.executable} main_diff.py {input_flag} {ddsp_flag} {diff_flag} {output_flag} {pitch_extractor_flag} {kstep_flag}"
    else:
      model_flag = f'-m {os.path.abspath(config.model_checkpoint)}'
      cmd = f"{sys.executable} main.py {input_flag} {model_flag} {output_flag} {pitch_extractor_flag}"
    utils.run_cmd(cmd, cwd=utils.ABS_DDSP_DIR, env={'PYTHONPATH': utils.ABS_DDSP_DIR})

  elif model == utils.SHALLOW_DIFF_SVC:
    input_flag = f'-i {os.path.abspath(config.input_file)}'
    model_flag = f'-model {os.path.abspath(config.model_checkpoint)}'
    output_flag = f'-o {os.path.abspath(config.input_file.replace(".wav", "_shallow_svc.wav"))}'
    kstep_flag = f'-kstep {config.kstep}'
    pitch_extractor_flag = f'-pe {config.pitch_extractor}'  # should use `crepe` here??

    # run shallow-diff stand-alone inference
    cmd = f"{sys.executable} main.py {input_flag} {model_flag} {output_flag} {kstep_flag} {pitch_extractor_flag}"
    utils.run_cmd(cmd, cwd=utils.ABS_SHALLOW_DIFF_DIR, env={'PYTHONPATH': utils.ABS_SHALLOW_DIFF_DIR})

  elif model == utils.SOVITZ_SVC:
    raise NotImplementedError()

  else:
    raise NotImplementedError()

  logger.info("%s finished at %s", config.name, datetime.now())

if __name__ == '__main__':
  main()
