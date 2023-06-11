# ultimate-svc
a wrapper on top of 4 popular **SVC** (**Singing Voice Conversion**) models:
1. [diff-svc](https://github.com/prophesier/diff-svc)
2. [ddsp-svc](https://github.com/yxlllc/DDSP-SVC)
3. [so-vitz-svc](https://github.com/svc-develop-team/so-vits-svc)
4. [shallow-diff-svc](https://github.com/CNChTu/Diffusion-SVC)

not yet supported:
* [rvc](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

## 1. Setup

### a) Core
* Python 3.8 
  * developed and tested with 3.8.12
* [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
* latest-stable `torch` and `torchaudio` (see [here](https://pytorch.org/get-started/locally/))
    ```console
    $ pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
* [ffmpeg](https://www.gyan.dev/ffmpeg/builds/)
    ```console
    $ sudo apt install ffmpeg
    ```
  
#### [!] additional virtualenv for `diff-svc`
if using `diff-svc`, set up a separate venv inside [diff-svc](./models/diff-svc)
* requires [CUDA 11.6](https://developer.nvidia.com/cuda-11-6-2-download-archive)
* from project root, run
  ```console
  $ virtualenv models/diff-svc/venv
  $ activate models/diff-svc/venv/bin/activate
  $ pip install torch==1.13.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  $ pip install -r models/diff-svc/requirements_short.txt
  ```

### b) Repo-specific Packages
If only interested in running one model, simply use repo-specific packages
* each repo has its own requirements under `requirements` dir
  * diff-svc: [requirements/requirements_diff.txt](./requirements/requirements_diff.txt)
  * ddsp-svc: [requirements/requirements_ddsp.txt](./requirements/requirements_ddsp.txt)
  * so-vitz-svc: [requirements/requirements_sovitz.txt](./requirements/requirements_sovitz.txt)

* Or use [requirements/requirements_all.txt](./requirements/requirements_all.txt) to be able to run all

```console
$ pip install -r requirements/[your_requirements.txt]
```

### c) Pre-trained Models & Checkpoints

Choose `ConceptVec` or `HuberSoft`
1. [ConceptVec](https://ibm.ent.box.com/s/z1wgl1stco8ffooyatzdwsqn2psd9lrr) encoder
* manually download then place under [pretrain/contentvec](./pretrain/contentvec)

2. [HubertSoft](https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt) encoder
  ```console
  $ wget https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt -P ./pretrain/hubert-soft
  ```

3. [NSF-HiFiGAN](https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip) vocoder
  ```console
  $ wget https://github.com/openvpi/vocoders/releases/download/nsf-hifigan-v1/nsf_hifigan_20221211.zip  && \
  unzip  nsf_hifigan_20221211.zip -d ./pretrain/vocoder && \
  rm nsf_hifigan_20221211.zip
  ```

4. (if using `sovitz`) [D and G weights]
  ```console
  $ wget https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_D_320000.pth -P ./pretrain/sovitz/D_320000.pth
  $ wget https://huggingface.co/datasets/ms903/sovits4.0-768vec-layer12/resolve/main/sovits_768l12_pre_large_320k/clean_G_320000.pth -P ./pretrain/sovitz/G_320000.pth
  ```
  * see [here](https://github.com/voicepaw/so-vits-svc-fork/blob/main/src/so_vits_svc_fork/preprocessing/config_templates/so-vits-svc-4.0v1.json) for urls

## 2. Preprocessing
First phase applies preliminary preprocessing such as normalization and clip segmentation

Second phase performs model-specific feature extraction

Consider updating following config fields:
* `input_dir`
* `preprocess_dir`
* `training_dir`
* `dev_dir`
* `pitch_extractor`
* `encoder`
* `skip_preliminary_preprocessing`: skips the first phase

Currently assumes all input data are relatively high-quality MR-removed vocals in wav. 

First, populate `configs/default.json` correctly, then run
```console
$ python scripts/run_preprocessing.py
```

## 3. Training

Consider updating following config fields:
* `output_dir`
* `batch_size`

```console
$ python scripts/run_training.py
```
Ctrl+C to terminate

## 4. Inference

Consider updating following config fields:
* `input_file`
* `output_dir`
* `model_checkpoint`

input data should be MR-removed wav file
1. populate `input_file` field in config json
2. run preprocessing, as above
3. run inference:
```console
$ python scripts/run_inference.py 
```

## 5. Future


