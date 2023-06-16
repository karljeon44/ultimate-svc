#! /usr/bin/python3
# -*- coding: utf-8 -*-
# Author: karljeon44
# Date: 6/15/23 12:01 AM
import argparse
import logging
import os
import re
import subprocess

import librosa
import numpy as np
import soundfile
import tqdm
from pydub import AudioSegment, effects

logger = logging.getLogger(__name__)


argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', help='path to input file or dir containing wavs to be preprocessed')
argparser.add_argument('-o', '--output', help='path to output dir')
argparser.add_argument('--normalize', action='store_true', help='whether to normalize')
argparser.add_argument('--sample_rate', type=int, default=44100, help='target sample rate')
argparser.add_argument('--split', action='store_true', help='whether to split clips into shorter segments')
argparser.add_argument('--split_duration', type=int, default=15, help='target length of split clips')


def main():
  args = argparser.parse_args()
  print("Args:", args)
  os.makedirs(args.output, exist_ok=True)

  if os.path.isfile(args.input):
    args.input, fname = os.path.split(args.input)
    fnames = [fname]
  else:
    fnames = os.listdir(args.input)

  for fname in tqdm.tqdm(fnames):
    if not fname.endswith('wav'):
      continue

    input_fpath = os.path.join(args.input, fname)

    # sample rate
    raw_audio = AudioSegment.from_file(input_fpath, format='wav')
    raw_audio = raw_audio.set_frame_rate(args.sample_rate)

    # change channels
    if raw_audio.channels != 1 :
      raw_audio = raw_audio.set_channels(1)

    norm_tmp_fpath = os.path.join(args.output, fname.replace('.wav', '_norm.wav'))
    if args.normalize:
      norm_audio = effects.normalize(raw_audio)
      norm_audio.export(norm_tmp_fpath, format="wav")

    if args.split:
      duration = librosa.get_duration(filename=norm_tmp_fpath)
      max_last_seg_duration = 0
      sep_duration_final = sep_duration = args.split_duration

      while sep_duration > 4:
        last_seg_duration = duration % sep_duration
        if max_last_seg_duration < last_seg_duration:
          max_last_seg_duration = last_seg_duration
          sep_duration_final = sep_duration
        sep_duration -= 1

      input_base_fname = os.path.splitext(fname)[0]
      clips_fpath_template = os.path.join(args.output, f"{input_base_fname}-%03d.wav")
      subprocess.run(f'ffmpeg -i "{norm_tmp_fpath}" -f segment -segment_time {sep_duration_final} "{clips_fpath_template}" -y',
                     capture_output=True, shell=True)

      # 3. remove empty clips
      tmp_ffmpeg_log_fpath = "./tmp_ffmpeg_log.txt"  # ffmpeg의 무음 감지 로그의 임시 저장 위치
      for fname in os.listdir(args.output):
        match = re.search(re.escape(input_base_fname) + '-\d{3}.wav', fname)
        if match:
          if os.path.exists(tmp_ffmpeg_log_fpath):
            os.remove(tmp_ffmpeg_log_fpath)

          clip_fpath = os.path.join(args.output, fname)
          subprocess.run(f'ffmpeg -i "{clip_fpath}" -af "silencedetect=n=-50dB:d=1.5,ametadata=print:file={tmp_ffmpeg_log_fpath}" -f null -',
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
              os.remove(clip_fpath)
              y, sr = librosa.load(clip_fpath, sr=None)
              y = np.concatenate((y[: round(sr * start)], y[round(sr * end):]), axis=None)
              soundfile.write(clip_fpath, y, samplerate=sr)

      # 4. clean up
      os.remove(norm_tmp_fpath)
      if os.path.exists(tmp_ffmpeg_log_fpath):
        os.remove(tmp_ffmpeg_log_fpath)


if __name__ == '__main__':
  main()
