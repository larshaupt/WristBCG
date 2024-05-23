# WristBCG

WristBCG is part of my master's thesis, aiming at estimating the heart rate from wrist-worn accelerometers while asleep. 

For more results, please refer to [results.md](results.md)

## Installation Requirements
To install required packages, run the following code. The current Pytorch version is 2.2.1

```
conda env create -f environment.yml
```


## Experiments
To execute all the experiments, you need to run all sweep configurations from sweep_config
```
wandb login
wandb sweep --project WristBCG LINK_TO_REPO/sweep_config/supervised_train.yaml
wandb agent SWEEP_ID
```

## Training

```
# Supervised
python main.py --framework supervised --dataset appleall

# Self-Supervised Learning
python main.py --framework nnclr --aug1 t_warp --aug2 bioglass --pretrain 1 --finetune 1 --dataset appleall --pretrain_dataset capture24

# With Postprocessing
python main.py --framework supervised --finetune 1 --model_uncertainty NLE --postprocessing sumprod --dataset appleall

# With Postprocessing and Self-Supervised Learning
python main.py --framework nnclr --aug1 t_warp --aug2 bioglass --pretrain_subsample 0.2 --pretrain 1 --finetune 1 --model_uncertainty NLE --postprocessing sumprod --dataset appleall
```

## Inference
```

# Supervised
python main.py --framework supervised --dataset appleall --finetune 0

# Self-supervised
python main.py --framework nnclr --aug1 t_warp --aug2 bioglass --pretrain 0 --finetune 0 --dataset appleall --pretrain_dataset capture24

# With Postprocessing
python main.py --framework supervised --finetune 0 --model_uncertainty NLE --postprocessing sumprod --dataset appleall

# With Postprocessing and Self-Supervised Learning
python main.py --framework nnclr --aug1 t_warp --aug2 bioglass --pretrain_subsample 0.2 --pretrain 0 --finetune 0 --model_uncertainty NLE --postprocessing sumprod --dataset appleall

```
## Supported Datasets
- Apple Watch [link](https://www.physionet.org/content/sleep-accel/1.0.0/) (with labels)
- Capture24 [link](https://github.com/OxWearables/capture24) (without labels)
- In-House dataset (to be published soon)

## Data Split Cases
- subject
- time



## Encoder Networks
Refer to ```models/backbones.py```
- CorNET (adapted from Biswas et al. CorNET: Deep Learning Framework for PPG-Based Heart Rate Estimation and Biometric Identification in Ambulant Environment)
- FrequencyCorNET (stacks fft-derived spectrogram on top of signal)
- AttentionCorNET (adds channel attention)
- FCN
- DeepConvLSTM
- LSTM
- AE
- CAE
- Transformer
- HRCTPNet (ConvTransformer, see Zhang et al. A Conv -Transformer network for heart rate estimation using ballistocardiographic signals)
- ResNET (see [link](https://github.com/OxWearables/ssl-wearables))


<br>To train an encoder network under supervised setting, you can run the following code:
```angular2html
python main.py --framework supervised --backbone CorNET
python main.py --framework supervised --backbone FCN
...
```
## Contrastive Models
Refer to ```models/frameworks.py```. For sub-modules (projectors, predictors) in the frameworks, refer to ```models/backbones.py```
- TS-TCC 
- SimSiam
- BYOL
- SimCLR
- NNCLR



## Augmentations
Refer to ```augmentations.py```
- ### Time Domain
  - noise
  - scale
  - negate
  - perm
  - shuffle
  - t\_flip
  - t\_warp
  - resample
  - rotation
  - perm\_jit
  - jit\_scal
  - bioinsights

- ### Frequency Domain
  - hfc
  - lfc
  - p\_shift
  - ap\_p
  - ap\_f
  - bioinsights

## Utils
- WandB
- t-SNE


## Related Links
The framework has been adapted from 
- https://github.com/Tian0426/CL-HAR

Part of the augmentation transformation functions are adapted from
- https://github.com/emadeldeen24/TS-TCC
- https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
- https://github.com/LijieFan/AdvCL/blob/main/fr_util.py

Part of the contrastive models are adapted from 
- https://github.com/lucidrains/byol-pytorch
- https://github.com/lightly-ai/lightly
- https://github.com/emadeldeen24/TS-TCC

The ResNET model and pretrained weight have been taken from 
- https://github.com/OxWearables/ssl-wearables

The BeliefPPG and Viterbi algorithm have been adapted from
- https://github.com/eth-siplab/BeliefPPG