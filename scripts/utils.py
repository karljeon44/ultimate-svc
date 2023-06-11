#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 6/9/23 4:31 PM
import argparse
import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


### constants
PROJECT_HOME = os.path.dirname(os.path.dirname(__file__))
DEFAULT_CONFIG = f'{PROJECT_HOME}/configs/default.json'

DIFF_SVC = 'diff'
DDSP_SVC = 'ddsp'
SOVITZ_SVC = 'sovitz'
SHALLOW_DIFF_SVC = 'shallow-diff'

### default dirs
MODELS_DIR = './models'
DIFF_DIR = f'{MODELS_DIR}/diff-svc'
DDSP_DIR = f'{MODELS_DIR}/DDSP-SVC'
SOVITZ_DIR = f'{MODELS_DIR}/so-vits-svc'
SHALLOW_DIFF_DIR = f'{MODELS_DIR}/Diffusion-SVC'


### diff-svc specific paths (needs to be absolute)
ABS_DIFF_DIR = os.path.abspath(DIFF_DIR)
DIFF_VENV_PYTHON = os.path.join(ABS_DIFF_DIR, 'venv/bin/python')
DIFF_CONFIG_YAML = os.path.join(ABS_DIFF_DIR, 'training/config.yaml')
DIFF_CONFIG_NSF_YAML = os.path.join(ABS_DIFF_DIR, 'training/config_nsf.yaml')


### ddsp-svc specific paths (needs to be absolute)
ABS_DDSP_DIR = os.path.abspath(DDSP_DIR)
DDSP_CONFIGS_DIR = os.path.join(ABS_DDSP_DIR, 'configs')
DDSP_COMBSUB_CONFIG_YAML = os.path.join(DDSP_CONFIGS_DIR, 'combsub.yaml')
DDSP_DIFFUSION_CONFIG_YAML = os.path.join(DDSP_CONFIGS_DIR, 'diffusion.yaml')  # don't use this; just use `shallow-diff`


### sovitz-svc specific paths (needs to be absolute)
ABS_SOVITZ_DIR = os.path.abspath(SOVITZ_DIR)
SOVITZ_CONFIG_JSON = os.path.join(ABS_SOVITZ_DIR, 'configs/config.json')


### shallow-diff-svc specific paths (needs to be absolute)
ABS_SHALLOW_DIFF_DIR = os.path.abspath(SHALLOW_DIFF_DIR)
SHALLOW_DIFF_CONFIG_YAML = os.path.join(ABS_SHALLOW_DIFF_DIR, 'configs/config.yaml')


PRETRAIN_DIR = './pretrain'
CONTENTVEC_DIR = f'{PRETRAIN_DIR}/contentvec'
CONTENTVEC_FPATH = F'{CONTENTVEC_DIR}/checkpoint_best_legacy_500.pt'
HUBERT_SOFT_DIR = f'{PRETRAIN_DIR}/hubert-soft'
HUBERT_SOFT_FPATH = f'{HUBERT_SOFT_DIR}/hubert-soft-0d54a1f4.pt'
NSF_HIFIGAN_DIR = f'{PRETRAIN_DIR}/nsf_hifigan'
NSF_HIFIGAN_MODEL_FPATH = f'{PRETRAIN_DIR}/nsf_hifigan/model'
SOVITZ_CHECKPOINTS_DIR = f'{PRETRAIN_DIR}/sovitz'
# https://github.com/voicepaw/so-vits-svc-fork/blob/main/src/so_vits_svc_fork/preprocessing/config_templates/so-vits-svc-4.0v1.json
SOVITZ_D_CHECKPOINT_FPATH =f'{SOVITZ_CHECKPOINTS_DIR}/D_320000.pth'
SOVITZ_G_CHECKPOINT_FPATH =f'{SOVITZ_CHECKPOINTS_DIR}/G_320000.pth'

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
  config.display()


def update_yaml(config_fpath, update_dict):
  # update diff config
  with open(config_fpath, 'r+') as f:
    cur_scope = None

    out_lines = []
    for line in f.readlines():
      if not line.startswith(' ') and line.strip().endswith(':'):
        cur_scope = line.strip()[:-1]

      lines = line.split(': ')
      if len(lines) == 1:  # scope will fall to this case
        out_lines.append(line)
      else:
        k, v = lines
        ks = k.strip()
        ks_aug = f'{cur_scope}/{ks}'
        if ks in update_dict or (cur_scope is not None and ks_aug in update_dict):
          v = str(update_dict[ks_aug]) + '\n'
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
