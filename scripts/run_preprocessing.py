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
import subprocess
import sys

import librosa
import numpy as np
import soundfile
import tqdm
from pydub import AudioSegment, effects

import utils

logger = logging.getLogger(__name__)


def preprocess(input_fpath, output_dir, sample_rate=44100):
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

  if config.input_file is not None:
    input_audio_fpaths = [config.input_file]
    output_dir = config.output_dir

  else:
    input_audio_fpaths = utils.load_wavs(config.input_dir)
    output_dir = config.preprocess_dir

    with os.scandir(output_dir) as it:
      if any(it) and not config.skip_preliminary_preprocessing:
        logger.info("Preprocess Data dir (`%s`) isn't empty; some data may be overwritten")

  os.makedirs(output_dir, exist_ok=True)

  ### preliminary preprocessing
  if config.skip_preliminary_preprocessing:
    logger.info("Skipping preliminary preprocessing")
  else:
    for input_audio_fpath in tqdm.tqdm(input_audio_fpaths, desc='Preliminary Preprocessing'):
      preprocess(input_audio_fpath, output_dir)


  ### model-specific preprocessing
  model = config.model
  if model == utils.DIFF_SVC:
    diff_config_fpath = utils.DIFF_CONFIG_NSF_YAML if 'nsf' in config.diff_config else utils.DIFF_CONFIG_YAML

    # update diff config
    update_dict = {
      'raw_data_dir': os.path.abspath(output_dir),
      'binary_data_dir': os.path.abspath(config.training_dir),
      'hubert_path': os.path.abspath(utils.HUBERT_SOFT_FPATH),
      'vocoder_ckpt': os.path.abspath(utils.NSF_HIFIGAN_MODEL_FPATH),
      'speaker_id': config.speaker
    }
    utils.update_yaml(diff_config_fpath, update_dict)
    # with open(diff_config_fpath, 'r+') as f:
    #   out_lines = []
    #   for line in f.readlines():
    #     lines = line.split(': ')
    #     if len(lines) == 1:
    #       out_lines.append(line)
    #     else:
    #       k,v = lines
    #       if k == 'raw_data_dir':
    #         v = os.path.abspath(output_dir) + '\n'
    #       elif k == 'binary_data_dir':
    #         v = os.path.abspath(config.training_dir) + '\n'
    #       elif k == 'hubert_path':
    #         v = os.path.abspath(utils.HUBERT_SOFT_FPATH) + '\n'
    #       elif k == 'vocoder_ckpt':
    #         v = os.path.abspath(utils.NSF_HIFIGAN_MODEL_FPATH) + '\n'
    #       elif k == 'speaker_id':
    #         v = f'{config.speaker}\n'
    #       out_lines.append(": ".join([k,v]))
    #
    #   f.seek(0)
    #   f.write("".join(out_lines))
    #   f.truncate()

    # run binarizer
    abs_diff_dir = os.path.abspath(utils.DIFF_DIR)
    cmd = f"{utils.DIFF_VENV_PYTHON} preprocessing/binarize.py --config {diff_config_fpath}"
    utils.run_cmd(cmd, cwd=abs_diff_dir, env={'PYTHONPATH': abs_diff_dir}, cuda_version=config.diff_cuda)


  elif model == utils.DDSP_SVC:
    pass

  elif model == utils.SOVITZ_SVC:
    pass

  else:
    pass

  logger.info("ult-svc Preprocessing finished at %s", datetime.now())

if __name__ == '__main__':
  main()
