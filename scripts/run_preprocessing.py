#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 6/9/23 4:30 PM
"""WAV Preprocessing Script

input audio files under `input_dir` are pre-processed into `training_dir`

Currently assumes all input data are relatively high-quality MR-removed vocals in wav.
"""
import logging
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime

import librosa
import numpy as np
import soundfile
import tqdm
from pydub import AudioSegment, effects

import utils

logger = logging.getLogger(__name__)


def preprocess(input_fpath, output_dir, split_into_clips=False, sample_rate=44100):
  """modified from https://github.com/Kor-SVS/Diff-SVC-Tools/blob/f56d626dc30cb489557d243164843d19b94511bd/sep_wav.py"""
  input_fname = os.path.basename(input_fpath).strip()
  norm_fpath = os.path.join(output_dir, input_fname)

  # 1. normalize audio volume
  raw_audio = AudioSegment.from_file(input_fpath, format='wav')

  # change sample rate
  raw_audio = raw_audio.set_frame_rate(sample_rate)

  # change channels
  if raw_audio.channels != 1 :
    raw_audio = raw_audio.set_channels(1)

  norm_audio = effects.normalize(raw_audio)
  norm_audio.export(norm_fpath, format="wav")

  # 2. split into ~15s clips
  if split_into_clips:
    duration = librosa.get_duration(filename=norm_fpath)
    max_last_seg_duration = 0
    sep_duration_final = sep_duration = 15

    while sep_duration > 4:
      last_seg_duration = duration % sep_duration
      if max_last_seg_duration < last_seg_duration:
        max_last_seg_duration = last_seg_duration
        sep_duration_final = sep_duration
      sep_duration -= 1

    input_base_fname = os.path.splitext(input_fname)[0]
    norm_clips_fpath = os.path.join(output_dir, f"{input_base_fname}-%03d.wav")
    subprocess.run(f'ffmpeg -i "{norm_fpath}" -f segment -segment_time {sep_duration_final} "{norm_clips_fpath}" -y',
                   capture_output=True, shell=True)

    # 3. remove empty clips
    tmp_ffmpeg_log_fpath = "./tmp_ffmpeg_log.txt"  # ffmpeg의 무음 감지 로그의 임시 저장 위치
    for fname in os.listdir(output_dir):
      match = re.search(re.escape(input_base_fname) + '-\d{3}.wav', fname)
      if match:
        if os.path.exists(tmp_ffmpeg_log_fpath):
          os.remove(tmp_ffmpeg_log_fpath)

        norm_clip_fpath = os.path.join(output_dir, fname)
        subprocess.run(f'ffmpeg -i "{norm_clip_fpath}" -af "silencedetect=n=-50dB:d=1.5,ametadata=print:file={tmp_ffmpeg_log_fpath}" -f null -',
                       capture_output=True, shell=True)

        start = end = None
        with open(tmp_ffmpeg_log_fpath, "r", encoding="utf-8") as f:
          for line in f.readlines():
            line = line.strip()
            if "lavfi.silence_start" in line:
              start = float(line.split("=")[1])
            if "lavfi.silence_end" in line:
              end = float(line.split("=")[1])

        if start is not None:
          if start == 0 and end is None:
            os.remove(norm_clips_fpath)

            y, sr = librosa.load(norm_clips_fpath, sr=None)
            y = np.concatenate((y[: round(sr * start)], y[round(sr * end):]), axis=None)
            soundfile.write(norm_clips_fpath, y, samplerate=sr)

    # 4. clean up
    os.remove(norm_fpath)
    if os.path.exists(tmp_ffmpeg_log_fpath):
      os.remove(tmp_ffmpeg_log_fpath)


def main():
  ### setup
  config = utils.init_config(name='ult-svc Preprocessing')
  utils.init_logging(config)
  logger.info("%s started at %s", config.name, datetime.now())

  preprocess_single_file = config.input_file is not None
  if preprocess_single_file:
    input_audio_fpaths = [config.input_file]
    output_dir = config.output_dir
    split_into_clips = False
  else:
    input_audio_fpaths = utils.load_wavs(config.input_dir)
    output_dir = config.preprocess_dir
    split_into_clips = True

  os.makedirs(output_dir, exist_ok=True)


  ### preliminary preprocessing
  if config.skip_preliminary_preprocessing:
    logger.info("Skipping preliminary preprocessing")
  else:
    with os.scandir(output_dir) as it:
      if any(it):
        logger.info("Found existing dir at `%s`, some data may be overwritten", output_dir)

    for input_audio_fpath in tqdm.tqdm(input_audio_fpaths, desc='Preliminary Preprocessing'):
      preprocess(input_audio_fpath, output_dir, split_into_clips=split_into_clips)


  ### model-specific feature extraction
  if not preprocess_single_file:
    model = config.model
    if model == utils.DIFF_SVC:
      diff_config_fpath = utils.DIFF_CONFIG_NSF_YAML if 'nsf' in config.diff_config else utils.DIFF_CONFIG_YAML

      # update diff config
      update_dict = {
        'raw_data_dir': os.path.abspath(output_dir),
        'binary_data_dir': os.path.abspath(config.training_dir),
        'hubert_path': os.path.abspath(utils.HUBERT_SOFT_FPATH),
        'vocoder_ckpt': os.path.abspath(utils.NSF_HIFIGAN_MODEL_FPATH),
        'speaker_id': config.speaker,
        # 'pitch_extractor': config.pitch_extractor
      }
      utils.update_yaml(diff_config_fpath, update_dict)

      # run diff binarizer
      cmd = f"{utils.DIFF_VENV_PYTHON} preprocessing/binarize.py --config {diff_config_fpath}"
      utils.run_cmd(cmd, cwd=utils.ABS_DIFF_DIR, env={'PYTHONPATH': utils.ABS_DIFF_DIR}, cuda_version=config.diff_cuda)

      logger.info("Preprocessed TRAINING files located at `%s`", config.training_dir)
      logger.info("Preprocessed DEV files located at `%s`", config.dev_dir)

    elif model in [utils.DDSP_SVC, utils.SHALLOW_DIFF_SVC]:
      # 0. copy from `preprocess_dir` to `training_dir`
      # ddsp always looks for `audio` sub-folder, so cater to that
      training_audio_dirpath = os.path.join(config.training_dir, 'audio')

      # needs to start fresh, so delete any existing dir
      if os.path.exists(training_audio_dirpath):
        logger.info("Found existing dir at `%s`, will replace with a new one", training_audio_dirpath)
        shutil.rmtree(training_audio_dirpath)
      shutil.copytree(output_dir, training_audio_dirpath)

      dev_audio_dirpath = os.path.join(config.dev_dir, 'audio')
      if os.path.exists(dev_audio_dirpath):
        logger.info("Found existing dir at `%s`, will replace with a new one", dev_audio_dirpath)
        shutil.rmtree(dev_audio_dirpath)
      os.makedirs(dev_audio_dirpath)

      # 1. need to split dev with a separate `draw.py` call
      abs_training_dir = os.path.abspath(training_audio_dirpath)
      abs_dev_dir = os.path.abspath(dev_audio_dirpath)
      cmd = f'{sys.executable} draw.py -t {abs_training_dir} -v {abs_dev_dir}'
      utils.run_cmd(cmd, cwd=utils.ABS_DDSP_DIR)

      # 2. main preprocessing
      # common paths
      encoder_fpath = utils.HUBERT_SOFT_FPATH if 'hubert' in config.encoder else utils.CONTENTVEC_FPATH
      encoder_ckpt = os.path.abspath(encoder_fpath)
      train_path = os.path.abspath(config.training_dir)
      valid_path = os.path.abspath(config.dev_dir)
      enhancer_ckpt = os.path.abspath(utils.NSF_HIFIGAN_MODEL_FPATH)

      if model == utils.DDSP_SVC:
        ddsp_config_fpath = utils.DDSP_DIFFUSION_CONFIG_YAML if 'diff' in config.ddsp_config else utils.DDSP_COMBSUB_CONFIG_YAML

        # update ddsp config
        update_dict = {
          'data/f0_extractor': config.pitch_extractor,
          'data/encoder': config.encoder,
          'data/encoder_ckpt': encoder_ckpt,
          'data/train_path': train_path,
          'data/valid_path': valid_path,
          'enhancer/ckpt': enhancer_ckpt,
        }
        utils.update_yaml(ddsp_config_fpath, update_dict)

        # run ddsp preprocessor
        cmd = f"{sys.executable} preprocess.py -c  {ddsp_config_fpath}"
        # env = os.environ.copy()
        # env['PYTHONPATH']  = utils.ABS_DDSP_DIR
        # env['PYTORCH_KERNEL_CACHE_PATH'] = os.path.join(os.path.expanduser('~'), '.cache/torch')
        # env = {
        #   : ,
        #   # 'PYTORCH_KERNEL_CACHE_PATH': os.path.join(os.path.expanduser('~'), '.cache/torch')
        # }
        # TODO:
        #  UserWarning: No PYTORCH_KERNEL_CACHE_PATH or HOME environment variable set! This disables kernel caching.
        #  (Triggered internally at ../aten/src/ATen/native/cuda/jit_utils.cpp:1426.)
        utils.run_cmd(cmd, cwd=utils.ABS_DDSP_DIR, env={'PYTHONPATH': utils.ABS_DDSP_DIR})

      else:
        # update shallow-diff config
        update_dict = {
          'data/f0_extractor': config.pitch_extractor,
          'data/encoder': config.encoder,
          'data/encoder_ckpt': encoder_ckpt,
          'data/train_path': train_path,
          'data/valid_path': valid_path,
          'vocoder/ckpt': enhancer_ckpt,
        }
        utils.update_yaml(utils.SHALLOW_DIFF_CONFIG_YAML, update_dict)

        # run shallow-diff preprocessor
        cmd = f"{sys.executable} preprocess.py -c  {utils.SHALLOW_DIFF_CONFIG_YAML}"
        utils.run_cmd(cmd, cwd=utils.ABS_SHALLOW_DIFF_DIR, env={'PYTHONPATH': utils.ABS_SHALLOW_DIFF_DIR})

      logger.info("Preprocessed TRAINING files located at `%s`", training_audio_dirpath)
      logger.info("Preprocessed DEV files located at `%s`", dev_audio_dirpath)


    elif model == utils.SOVITZ_SVC:
      # 0. configure speaker dirs
      # sovitz always looks for speaker dir first, so cater to that
      speaker = config.speaker
      speaker_dirpath = os.path.join(config.training_dir, speaker)
      if os.path.exists(speaker_dirpath):
        logger.info("Found existing dir at `%s`, will replace with a new one", speaker_dirpath)
        shutil.rmtree(speaker_dirpath)
      shutil.copytree(output_dir, speaker_dirpath)

      # 1. split datasets and generate config
      if config.encoder.startswith('contentvec'):
        config.encoder = config.encoder[7:]

      abs_training_dirpath = os.path.abspath(config.training_dir)
      source_dir_flag = f'--source_dir {abs_training_dirpath}'
      encoder_flag = f'--speech_encoder {config.encoder}'
      cmd = f'{sys.executable} preprocess_flist_config.py {source_dir_flag} {encoder_flag}'
      utils.run_cmd(cmd, cwd=utils.SOVITZ_DIR, env={'PYTHONPATH': utils.SOVITZ_DIR})

      # 2. run hubert encoder
      # but first, update default paths for speech encoders
      vencoder_dirpath = os.path.join(utils.SOVITZ_DIR, 'vencoder')
      for encoder_option in ['contentvec', 'hubert-soft']:
        if encoder_option == 'contentvec':
          vencoder_fname = 'ContentVec768L12.py'
          new_vec_path = os.path.abspath(utils.CONTENTVEC_FPATH)
        else:
          vencoder_fname = 'HubertSoft.py'
          new_vec_path = os.path.abspath(utils.HUBERT_SOFT_FPATH)

        vencoder_fpath = os.path.join(vencoder_dirpath, vencoder_fname)
        with open(vencoder_fpath, 'r+') as f:
          data = re.sub(r'"[^"]*.pt"', f'"{new_vec_path}"', f.read())
          f.seek(0)
          f.write(data)
          f.truncate()

      # now run hubert
      in_dir_flag = f'--in_dir {abs_training_dirpath}'
      f0_predictor_flag = f'--f0_predictor {config.pitch_extractor}'
      cmd = f'{sys.executable} preprocess_hubert_f0.py {in_dir_flag} {f0_predictor_flag}'
      utils.run_cmd(cmd, cwd=utils.SOVITZ_DIR, env={'PYTHONPATH': utils.SOVITZ_DIR})

    else:
      raise NotImplementedError()

    logger.info("[!] You DO NOT need to change data paths when training!")

  logger.info("%s finished at %s", config.name, datetime.now())

if __name__ == '__main__':
  main()
