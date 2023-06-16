import os
import logging

import tqdm

logger = logging.getLogger(__name__)


def preprocess_flist(train_dir: str, dev_dir: str):
  sovitz_data_dir = os.path.dirname(train_dir)
  assert sovitz_data_dir == os.path.dirname(dev_dir)

  train = []
  for speaker in os.listdir(train_dir):
    speaker_dirpath = os.path.join(train_dir, speaker)
    for fpath in tqdm.tqdm(os.listdir(speaker_dirpath), desc='[Training]'):
      if fpath.endswith('.wav'):
        train.append(os.path.abspath(os.path.join(speaker_dirpath, fpath)))

  train_flist_fpath = os.path.join(sovitz_data_dir, 'train.txt')
  print("Writing train flist at", train_flist_fpath)
  with open(train_flist_fpath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(train))

  val = []
  for speaker in os.listdir(dev_dir):
    speaker_dirpath = os.path.join(dev_dir, speaker)
    for fpath in tqdm.tqdm(os.listdir(speaker_dirpath), desc='[Val]'):
      if fpath.endswith('.wav'):
        val.append(os.path.abspath(os.path.join(speaker_dirpath, fpath)))

  dev_flist_fpath = os.path.join(sovitz_data_dir, 'val.txt')
  print("Writing dev flist at", dev_flist_fpath)
  with open(dev_flist_fpath, 'w', encoding='utf-8') as f:
    f.write('\n'.join(val))
