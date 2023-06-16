# ultimate-svc

## 1. DDSP-SVC
### venv
* Python 3.8.12
* cuda 11.8
* torch 2

### comments
* recommends about 1000 clips for training
* can combine with shallow-diffusion during inference
* no pre-trained models found
* loss hovers around 0.8 at lowest

### commands
* find out what `main.py` requires during infernece
```console
$ PYTHONPATH=$DDSP_HOME python $DDSP_HOME/preprocess.py -c configs/ddsp.yaml
$ PYTHONPATH=$DDSP_HOME python $DDSP_HOME/train.py -c configs/ddsp.yaml
$ PYTHONPATH=$DDSP_HOME python $DDSP_HOME/main.py $@
```


## 2. Shallow Diffusion
### venv
same as DDSP

### comments
* extremely fast to train
* stand-alone may not produce highest quality audio
  * may serve better as an enhancer for other models
* use pre-trained model at [shallow-diff](checkpoints%2Fshallow-diff)


### commands
* find out what `main.py` requires during infernece
```console
$ PYTHONPATH=$SHALLOW_DIFF_HOME python $SHALLOW_DIFF_HOME/preprocess.py -c configs/shallow-diff.yaml
$ PYTHONPATH=$SHALLOW_DIFF_HOME python $SHALLOW_DIFF_HOME/train.py -c configs/shallow-diff.yaml
$ PYTHONPATH=$SHALLOW_DIFF_HOME python $SHALLOW_DIFF_HOME/main.py $@
```


## 3. Diffusion
### venv
only one with torch v1
* python 3.9.15
* cuda 11.7
  * !!!! NEED TO SET `PATH` AND `LD_LIBRARY_PATH` CORRECTLY TO POINT TO `cuda-11.7`
* torch 1.13.1

### comments
* training used to be too slow but above venv fixed the issue
* can produce high-quality audio with sufficient training
* use pre-trained model at [diffusion](checkpoints%2Fdiffusion)

### commands
```console
$ PYTHONPATH=$DIFF_HOME python $DIFF_HOME/preprocessing/binarize.py --config configs/diff.yaml
$ CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$DIFF_HOME python $DIFF_HOME/run.py --config configs/diff.yaml $@
$ 
```


## 4. so-vits-svc
### venv
located in [venvs](venvs)
* python 3.8.12
* cuda 11.8
* torch 2

### comments
* could have used `so-vits-svc-fork` pip package but then hard to override default behavior
* will automatically download pre-trained models at the start of training
* using 4.0 not 4.1

### commands
```console
$ PYTHONPATH=models/so_vits_svc_fork python models/so_vits_svc_fork pre-flist -t ./DATA/sovits/train -d ./DATA/sovits/dev
$ PYTHONPATH=models/so_vits_svc_fork python models/so_vits_svc_fork pre-hubert -i ./DATA/sovits -c ./configs/sovits.json -fm crepe
$ PYTHONPATH=models/so_vits_svc_fork python models/so_vits_svc_fork train -c ./configs/sovits.json -m ./outputs/sovits-run-1
$ 
```


## 5. Fish-Diffusion
### venv
located in [venvs](venvs)
* python 3.10.12
* cuda 11.8
* torch 2

### comments
* well-structured codebase
* two pre-trained models available: hifigan vs contentvec
  * only tried hifigan and result was okay

### commands

```console
$ PYTHONPATH=models/fish-diffusion python models/fish-diffusion/tools/preprocessing/extract_features.py --config models/fish-diffusion/configs/svc_hifisinger_finetune.py --path ./DATA/fish/train --clean --num-workers 4
$ PYTHONPATH=models/fish-diffusion python models/fish-diffusion/tools/preprocessing/extract_features.py --config models/fish-diffusion/configs/svc_hifisinger_finetune.py --path ./DATA/fish/dev --clean --no-augmentation
$ PYTHONPATH=models/fish-diffusion python models/fish-diffusion/tools/hifisinger/train.py --config models/fish-diffusion/configs/svc_hifisinger_finetune.py --tensorboard --pretrained ./checkpoints/fish/hifisinger-pretrained-20230329-540k.ckpt --dest-path ./outputs/fish-run-1

```


## 6. RVC
### venv
located in [venvs](venvs)

### comments
* mainly web-ui but can still run cmds

### commands
```console
$ python trainset_preprocess_pipeline_print.py ../../DATA/rvc/Beberry 40000 12 ../../rvc/train False

$ python extract_f0_print.py ../../rvc/train  12 harvest
$ python extract_feature_print.py cuda:0 1 0 0 ../../rvc/train  v2

$ python train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 32 -g 0 -te 1000 -se 11 -pg pretrained_v2/f0G40k.pth -pd pretrained_v2/f0D40k.pth -l 0 -c 1 -sw 0 -v v2 
```


## 7.

## DISCLAIMER
프리아 베베리 기획의 ai 히든싱어 컨텐츠 (가제: 프든싱어) 지원용 팬메이드 프로그램입니다. 사용함에 있어 따르는 어떠한 문제에 대해서도 책임을 지지 않습니다.

This is a fan-made software for Fria Beberry's AI Hidden Singer content. Use it at your own risk.
