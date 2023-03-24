# DeepSurface
ESM for deep surface classification of proteins. Uses Huggingface and pretrain

## Overview
This repo uses pre-trained ESM weights to train on the deep vs. surface protein task that Ashley is working on. `esm/train_esm.py` shows the workflow for training models in huggingface. Definitely check this out as a rough guide to using this library, and feel free to make your own tweaks on it!

## Running on ISAAC cluster
We will be using the [ISAAC cluster](https://oit.utk.edu/hpsc/isaac-open-enclave-new-kpb/) for our purposes on these projects, UT's compute cluster for research purposes. It has lots of GPUs, which are specialized processors (in contrast to CPUs) that we train neural nets on. It also hosts the data that you'll need to access for this script to run. This is a good chance to use the cluster and get aquainted with SLURM, the resource manager used to run jobs (run code) on compute clusters. It takes a bit to learn how to use these resources, but they make things so much faster once you learn. Fortunately, UT provides great resources on how to use ISAAC, so check out that link at the beginning of this paragraph for more info, and of course let me know if you have any questions! 

**NOTE**: please don't try to run this script on your local machine bc it will most likely take up all your memory and get killed or do something weird

## Changes to code + conda environment
You'll need to change the paths throughout the training script. Basically anywhere that has something like "../oqueen/..", you'll have to change to your own path. Python development also requires installing a lot of packages, which can be frustrating at first. Luckily, I have included here a `.yaml` file that contains info to load a conda environment to run this code. Anaconda is a package manager for python that helps you keep track of all the packages you need for projects. It's a pain to load, but basically you'll use the following command to load the conda environment:

```
conda env create --name <your environment name> --file=deepsurface.yaml
```

Note that you replace `<your environment name>` with whatever name you want, like `deepsurface`. You'll then need to activate the conda env with `conda activate <your environment name>`. Then you should have all the proper versions and packages that I used to run the code.

## Potential optimizations
There are several potential optimizations for making the prediction performance better with [ESM](https://huggingface.co/docs/transformers/v4.26.0/en/model_doc/esm):

1. Changing hyperparameters, such as learning rate, batch size, optimizer, etc. There are literally endless options in huggingface that you can try out, check the documentation for the Trainer and TrainingArguments as those will have a lot!
2. Changing the number of parameters. I only tried 35 million and 8 million, but ESM model weights go up to the billions! That will be difficult to fit onto one GPU, but you can try different numbers of parameters to see how it performs. In general, larger models perform better than smaller ones, especially with the huge datasets on which ESM was trained! You can find their other trained model weights [here](https://huggingface.co/facebook/esm2_t12_35M_UR50D).
3. Dataset processing. Right now, I just treat each sequence as a separate sample, but you could totally pair them and ask the model to choose which is surface and which is deep. That might be hard to set up, but it would be an interesting experiment. 
4. Speeding up the training. If high-performance computing and efficient training of neural nets interests you, there is the option of speeding this training procedure up. Even with 8 million parameters, the training is still pretty slow. I use floating-point 16 training, i.e., storing the model weights as 16-bit numbers instead of 32-bit normal floats, which saves space. Also, I use gradient accumulation, which saves memory and time by waiting to perform backpropagation until a certain number of steps has passed. There are plenty of other options on huggingface, I recommend checking [this link](https://huggingface.co/docs/transformers/v4.18.0/en/performance) out!

All of these will be great chances to get practical experience with huggingface! If you have any questions, feel free to reach out to Owen on Slack!

## Dependencies
Here is a toned-down list of dependencies to install if you're manually installing with conda:
```
biopython
datasets==2.1.0
matplotlib
numpy
pandas
scikit-learn
scipy
tokenizers==0.12.1
torch==1.13.1
tqdm
transformers==4.26.0
```
To install each of these, copy and paste one of the above lines and use it in your pip intall command, for example `pip install torch==1.13.1`. **Make sure you are first in your conda environment of choice**. You make activate your conda environment with `conda activate <name of environment>`. If you run into any issues running the code after installing the above packages, let Owen know.
