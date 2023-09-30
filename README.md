# ddpo-pytorch

This is an implementation of [Denoising Diffusion Policy Optimization (DDPO)](https://rl-diffusion.github.io/) in PyTorch with support for [low-rank adaptation (LoRA)](https://huggingface.co/docs/diffusers/training/lora). Unlike our original research code (which you can find [here](https://github.com/jannerm/ddpo)), this implementation runs on GPUs, and if LoRA is enabled, requires less than 10GB of GPU memory to finetune Stable Diffusion!

![DDPO](teaser.jpg)

## Installation
Requires Python 3.10 or newer.

```bash
git clone git@github.com:kvablack/ddpo-pytorch.git
cd ddpo-pytorch
pip install -e .
```

## Usage
```bash
accelerate launch scripts/train.py
```
This will immediately start finetuning Stable Diffusion v1.5 for compressibility on all available GPUs using the config from `config/base.py`. It should work as long as each GPU has at least 10GB of memory. If you don't want to log into wandb, you can run `wandb disabled` before the above command.

Please note that the default hyperparameters in `config/base.py` are not meant to achieve good performance, they are just to get the code up and running as fast as possible. I would not expect to get good results without using a much larger number of samples per epoch and gradient accumulation steps.

## Important Hyperparameters

A detailed explanation of all the hyperparameters can be found in `config/base.py`. Here are a few of the important ones.

### prompt_fn and reward_fn
At a high level, the problem of finetuning a diffusion model is defined by 2 things: a set of prompts to generate images, and a reward function to evaluate those images. The prompts are defined by a `prompt_fn` which takes no arguments and generates a random prompt each time it is called. The reward function is defined by a `reward_fn` which takes in a batch of images and returns a batch of rewards for those images. All of the prompt and reward functions currently implemented can be found in `ddpo_pytorch/prompts.py` and `ddpo_pytorch/rewards.py`, respectively.

### Batch Sizes and Accumulation Steps
Each DDPO epoch consists of generating a batch of images, computing their rewards, and then doing some training steps on those images. One important hyperparameter is the number of images generated per epoch; you want enough images to get a good estimate of the average reward and the policy gradient. Another important hyperparameter is the number of training steps per epoch.

However, these are not defined explicitly but are instead defined implicitly by several other hyperparameters. First note that all batch sizes are **per GPU**. Therefore, the total number of images generated per epoch is `sample.batch_size * num_gpus * sample.num_batches_per_epoch`. The effective total training batch size (if you include multi-GPU training and gradient accumulation) is `train.batch_size * num_gpus * train.gradient_accumulation_steps`. The number of training steps per epoch is the first number divided by the second number, or `(sample.batch_size * sample.num_batches_per_epoch) / (train.batch_size * train.gradient_accumulation_steps)`.

(This assumes that `train.num_inner_epochs == 1`. If this is set to a higher number, then training will loop over the same batch of images multiple times before generating a new batch of images, and the number of training steps per epoch will be multiplied accordingly.)

At the beginning of each training run, the script will print out the calculated value for the number of images generated per epoch, the effective total training batch size, and the number of training steps per epoch. Make sure to double-check these numbers!

## Reproducing Results
The image at the top of this README was generated using LoRA! However, I did use a fairly powerful DGX machine with 8xA100 GPUs, on which each experiment took about 4 hours for 100 epochs. In order to run the same experiments with a single small GPU, you would set `sample.batch_size = train.batch_size = 1` and multiply `sample.num_batches_per_epoch` and `train.gradient_accumulation_steps` accordingly.

You can find the exact configs I used for the 4 experiments in `config/dgx.py`. For example, to run the aesthetic quality experiment:
```bash
accelerate launch scripts/train.py --config config/dgx.py:aesthetic
```

If you want to run the LLaVA prompt-image alignment experiments, you need to dedicate a few GPUs to running LLaVA inference using [this repo](https://github.com/kvablack/LLaVA-server/).

## Reward Curves
<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/593c9be3-e2a7-45d8-b1ae-ca4f77197c18" width="49%">
<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/d12fef0a-68b8-4cef-a9b8-cb1b6878fcec" width="49%">
<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/669076d5-2826-4b77-835b-d82e0c18a2a6" width="49%">
<img src="https://github.com/kvablack/ddpo-pytorch/assets/12429600/393a929e-36af-46f2-8022-33384bdae1c8" width="49%">

## Training using 🤗 `trl`

🤗 `trl` provides a [`DDPOTrainer` class](https://huggingface.co/docs/trl/ddpo_trainer) which lets you fine-tune Stable Diffusion on different reward functions using DDPO. The integration supports LoRA, too.  You can check out the [supplementary blog post](https://huggingface.co/blog/trl-ddpo) for additional guidance. The DDPO integration was contributed by @metric-space to `trl`. 


