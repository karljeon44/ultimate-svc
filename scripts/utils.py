#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 6/9/23 4:31 PM
import argparse
import json
import logging
import os
import subprocess
from datetime import datetime

logger = logging.getLogger(__name__)


### constants
PROJECT_HOME = os.path.dirname(os.path.dirname(__file__))
DEFAULT_CONFIG = f'{PROJECT_HOME}/configs/default.json'

DIFF_SVC = 'diff'
DDSP_SVC = 'ddsp'
SOVITZ_SVC = 'sovitz'
RVC = 'rvc'

### default dirs
MODELS_DIR = './models'
DIFF_DIR = f'{MODELS_DIR}/diff-svc'
DDSP_DIR = f'{MODELS_DIR}/DDSP-SVC'
SOVITZ_DIR = f'{MODELS_DIR}/so-vits-svc'
RVC_DIR = f'{MODELS_DIR}/Retrieval-based-Voice-Conversion-WebUI'

### diff-svc specific paths (needs to be absolute)
DIFF_VENV_PYTHON = os.path.abspath(f'{DIFF_DIR}/venv/bin/python')
DIFF_CONFIG_YAML = os.path.abspath(f'{DIFF_DIR}/training/config.yaml')
DIFF_CONFIG_NSF_YAML = os.path.abspath(f'{DIFF_DIR}/training/config_nsf.yaml')


PRETRAIN_DIR = './pretrain'
CONTENTVEC_DIR = f'{PRETRAIN_DIR}/contentvec'
CONTENTVEC_FPATH = F'{CONTENTVEC_DIR}/checkpoint_best_legacy_500.pt'
HUBERT_SOFT_DIR = f'{PRETRAIN_DIR}/hubert-soft'
HUBERT_SOFT_FPATH = f'{HUBERT_SOFT_DIR}/hubert-soft-0d54a1f4.pt'
NSF_HIFIGAN_DIR = f'{PRETRAIN_DIR}/nsf_hifigan'
NSF_HIFIGAN_MODEL_FPATH = f'{PRETRAIN_DIR}/nsf_hifigan/model'


### config
class Config:
  def __init__(self, config, name='ult-svc'):
    assert isinstance(config, dict), "Expects dict to init Config, received %s" % type(config)
    self.__dict__.update(config)
    self.name = name

  def __contains__(self, item):
    return item in self.__dict__

  def __getitem__(self, item):
    return self.__dict__[item]

  def __setitem__(self, key, value):
    self.__dict__[key] = value

  def display(self):
    logger.info("*** Config ***")
    for k,v in self.__dict__.items():
      logger.info(" %s: %s", k, v)

  def export(self, fpath):
    save_json(vars(self), fpath)

  @classmethod
  def init_from_path(cls, fpath, name='ult-svc'):
    json_config = load_json(fpath)
    return cls(json_config, name=name)

### I/O
def load_json(fpath):
  assert os.path.exists(fpath)
  with open(fpath) as f:
    obj = json.load(f)
  return obj

def save_json(obj, fpath):
  with open(fpath, 'w') as f:
    json.dump(obj, f, indent=2)
  return fpath

### utilties
def load_wavs(dir_fpath):
  assert os.path.exists(dir_fpath)

  fpaths = []
  for fname in os.listdir(dir_fpath):
    if not fname.endswith('.wav'):
      continue
    fpaths.append(os.path.join(dir_fpath, fname))

  return fpaths

def init_config(name='ult-svc'):
  argparser = argparse.ArgumentParser(name)
  argparser.add_argument('-c', '--config', default=DEFAULT_CONFIG, help='path to config JSON to use')
  args, _ = argparser.parse_known_args()
  return Config.init_from_path(args.config)

def init_logging(config):
  logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="[%(asctime)s][%(name)s][%(levelname)s]%(message)s",
    datefmt='%Y-%m-%d %H:%M:%S'
  )
  logger.info("%s started at %s", config.name, datetime.now())
  config.display()


def update_yaml(config_fpath, update_dict):
  # update diff config
  with open(config_fpath, 'r+') as f:
    out_lines = []
    for line in f.readlines():
      lines = line.split(': ')
      if len(lines) == 1:
        out_lines.append(line)
      else:
        k, v = lines
        if k in update_dict:
          v = str(update_dict[k]) + '\n'
        out_lines.append(": ".join([k, v]))

    f.seek(0)
    f.write("".join(out_lines))
    f.truncate()

def run_cmd(cmd, cwd=None, env=None, cuda_version=None):
  if isinstance(cmd, str):
    cmd = cmd.split()

  if env is None:
    env = {}

  if cuda_version is not None:
    env['PATH'] = 'PATH=/usr/local/cuda-%s/bin${PATH:+:${PATH}}' % cuda_version
    env['LD_LIBRARY_PATH'] = '/usr/local/cuda-%s/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' % cuda_version

  logger.info("Running cmd `%s`", " ".join(cmd))
  subprocess.check_call(cmd, cwd=cwd, env=env)
