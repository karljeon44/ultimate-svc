import argparse
import io
import os
import time
import yaml
from pathlib import Path

import librosa
import numpy as np
import soundfile

from infer_tools import infer_tool
from infer_tools import slicer
from infer_tools.infer_tool import Svc
from utils.hparams import hparams

chunks_dict = infer_tool.read_temp("./infer_tools/new_chunks_temp.json")


def run_clip(svc_model, key, acc, use_pe, use_crepe, thre, use_gt_mel, add_noise_step, project_name='', f_name=None,
             file_path=None, out_path=None, slice_db=-40,**kwargs):
  print(f'code version:2022-12-04')
  use_pe = use_pe if hparams['audio_sample_rate'] == 24000 else False

  assert file_path is not None
  raw_audio_path = file_path
  clean_name = str(Path(file_path).name)[:-4]

  infer_tool.format_wav(raw_audio_path)
  wav_path = Path(raw_audio_path).with_suffix('.wav')

  global chunks_dict
  audio, sr = librosa.load(wav_path, mono=True,sr=None)
  wav_hash = infer_tool.get_md5(audio)
  if wav_hash in chunks_dict.keys():
    print("load chunks from temp")
    chunks = chunks_dict[wav_hash]["chunks"]
  else:
    chunks = slicer.cut(wav_path, db_thresh=slice_db)
  chunks_dict[wav_hash] = {"chunks": chunks, "time": int(time.time())}
  infer_tool.write_temp("./infer_tools/new_chunks_temp.json", chunks_dict)
  audio_data, audio_sr = slicer.chunks2audio(wav_path, chunks)

  count = 0
  f0_tst = []
  f0_pred = []
  audio = []
  for (slice_tag, data) in audio_data:
    print(f'#=====segment start, {round(len(data) / audio_sr, 3)}s======')
    length = int(np.ceil(len(data) / audio_sr * hparams['audio_sample_rate']))
    raw_path = io.BytesIO()
    soundfile.write(raw_path, data, audio_sr, format="wav")
    if hparams['debug']:
      print(np.mean(data), np.var(data))
    raw_path.seek(0)
    if slice_tag:
      print('jump empty segment')
      _f0_tst, _f0_pred, _audio = (np.zeros(int(np.ceil(length / hparams['hop_size']))), np.zeros(int(np.ceil(length / hparams['hop_size']))), np.zeros(length))
    else:
      _f0_tst, _f0_pred, _audio = svc_model.infer(raw_path, key=key, acc=acc, use_pe=use_pe, use_crepe=use_crepe,
                                                  thre=thre, use_gt_mel=use_gt_mel, add_noise_step=add_noise_step)
    fix_audio = np.zeros(length)
    fix_audio[:] = np.mean(_audio)
    fix_audio[:len(_audio)] = _audio[0 if len(_audio)<len(fix_audio) else len(_audio)-len(fix_audio):]
    f0_tst.extend(_f0_tst)
    f0_pred.extend(_f0_pred)
    audio.extend(list(fix_audio))
    count += 1

  assert out_path is not None
  # export
  soundfile.write(out_path, audio, hparams["audio_sample_rate"], 'PCM_16',format=out_path.split('.')[-1])

  return np.array(f0_tst), np.array(f0_pred), audio


if __name__ == '__main__':
  argparser = argparse.ArgumentParser()
  argparser.add_argument('-c', '--config', help='path to config file')
  argparser.add_argument('-i', '--input', help='path to input audio dir or file')
  argparser.add_argument('-o', '--output', help='path to output audio dir or file')
  argparser.add_argument('-speedup', default=20, help='speedup (or accelerate)')
  argparser.add_argument()
  args = argparser.parse_args()
  with open(args.config, encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.Loader)

  project_name = os.path.basename(config.work_dir)
  model_path = config.load_ckpt
  config_path =  args.config

  output_fpath = output_dir = None
  if os.path.isfile(args.input):
    file_names = [args.input]
    output_fpath = args.output
    trans = [0]
  else:
    file_names = [os.path.join(args.input, x) for x in os.listdir(args.input)]
    trans = [0 for _ in file_names]
    output_dir = args.output

  accelerate = args.speedup
  hubert_gpu = True
  step = int(model_path.split("_")[-1].split(".")[0])

  model = Svc(project_name, config_path, hubert_gpu, model_path)
  for fpath, tran in zip(file_names, trans):
    if "." not in fpath:
      fpath += ".wav"

    if output_dir is not None:
      output_fpath = os.path.join(output_dir, os.path.basename(fpath))
    run_clip(model, key=tran, acc=accelerate, use_crepe=True, thre=0.05, use_pe=True, use_gt_mel=False,
             add_noise_step=500, file_path=fpath, out_path=output_fpath, project_name=project_name, format='wav')
