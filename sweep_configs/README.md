# Sweeps

To execute all experiments, I use Weights and Biases. You can find all sweep configurations in this directory. A configuration can be executed like this:

```
wandb login
wandb sweep --project WristBCG PATH_TO_CONFIG
# Returns SWEEP_ID
wandb agent SWEEP_ID
```

## Self-Supervised Learning
Before executing the sweeps ending with _finetune.yaml, you need to execute the corresponding sweep ending with _pretrain.yaml.
To use the OxWearables ResNET pretrained model, please donwnload [it](https://wearables-files.ndph.ox.ac.uk/files/ssl/mtl_best.mdl) (from [here](https://github.com/OxWearables/ssl-wearables?tab=readme-ov-file)) and place it into the model directory.
## Results
To compare the results you can use make_wandb_plots.py. Enter the respective sweep id for every experiment and generate the plots and tables containing the results