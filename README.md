# DeepSurface
ESM for deep surface classification of proteins. Uses Huggingface and pretrain

## Overview
This repo uses pre-trained ESM weights to train on the deep vs. surface protein task that Ashley is working on. `esm/train_esm.py` shows the workflow for training models in huggingface. Definitely check this out as a rough guide to using this library, and feel free to make your own tweaks on it!

## Running on ISAAC cluster
We will be using the [ISAAC cluster](https://oit.utk.edu/hpsc/isaac-open-enclave-new-kpb/) for our purposes on these projects, UT's compute cluster for research purposes. It has lots of GPUs, which are specialized processors (in contrast to CPUs) that we train neural nets on. This is a good chance to use the cluster and get aquainted with SLURM, the resource manager used to run jobs (run code) on compute clusters. It takes a bit to learn how to use these resources, but they make things so much faster once you learn. Fortunately, UT provides great resources on how to use ISAAC, so check out that link at the beginning of this paragraph for more info, and of course let me know if you have any questions! 

## Potential optimizations
There are several potential optimizations for making the prediction performance better with ESM:

1. Changing hyperparameters, such as learning rate, batch size, optimizer, etc. There are literally endless options in huggingface that you can try out, check the documentation for the Trainer and TrainingArguments as those will have a lot!
2. Changing the number of parameters. I only tried 35 million and 8 million, but ESM model weights go up to the billions! That will be difficult to fit onto one GPU, but you can try different numbers of parameters to see how it performs. In general, larger models perform better than smaller ones, especially with the huge datasets on which ESM was trained!
3. Dataset processing. Right now, I just treat each sequence as a separate sample, but you could totally pair them and ask the model to choose which is surface and which is deep. That might be hard to set up, but it would be an interesting experiment. 
4. Speeding up the training. If high-performance computing and efficient training of neural nets interests you, there is the option of speeding this training procedure up. Even with 8 million parameters, the training is still pretty slow. I use floating-point 16 training, i.e., storing the model weights as 16-bit numbers instead of 32-bit normal floats, which saves space. Also, I use gradient accumulation, which saves memory and time by waiting to perform backpropagation until a certain number of steps has passed. There are plenty of other options on huggingface, I recommend checking [this link](https://huggingface.co/docs/transformers/v4.18.0/en/performance) out!

All of these will be great chances to get practical experience with huggingface! If you have any questions, feel free to reach out to Owen on Slack!