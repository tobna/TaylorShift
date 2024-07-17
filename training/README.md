# TaylorShift Experiments
The code for our experiments is based on [this repository](https://github.com/tobna/WhatTransformerToFavor).

This is the code for all experiments requiring training and analysis of full models.


## Requirements
This project heavily builds on [timm](https://github.com/huggingface/pytorch-image-models) and open source implementations of the models that are tested.
All requirements are listed in [requirements.txt](./requirements.txt).
To install those, run
```commandline
pip install -r requirements.txt
```

## Usage
After **cloning this repository**, you can train and test a lot of different models.
By default, a `srun` command is executed to run the code on a slurm cluster. 
To run on the local machine, append the `-local` flag to the command.

### Dataset Preparation
Supported datasets are CIFAR10, ImageNet-21k, and ImageNet-1k.

The CIFAR10 dataset has to be located in a subfolder of the dataset root directory called `CIFAR`.
This is the normal `CIFAR10` from `torchvision.datasets`.

To speed up the data loading, the ImageNet datasets are read using [`datadings`](https://datadings.readthedocs.io/en/stable/). 
The `.msgpack` files for **ImageNet-1k** should be located in `<dataset_root_folder>/imagenet/msgpack`, 
while the ones for **ImageNet-21k** should be in `<dataset_root_folder>/imagenet-21k`.
See the [datadings documentation](https://datadings.readthedocs.io/en/stable/) for information on how to create those files.

### Training
#### Pretraining
To pretrain a model on a given dataset, run
```commandline
./main.py -model <model_name> -epochs <epochs> -dataset_root <dataset_root_folder>/ -results_folder <folder_for_results>/ -logging_folder <logging_folder> -run_name <name_or_description_of_the_run> (-local)
```
This will save a checkpoint (`.tar` file) every `<save_epochs>` epochs (the default is 10), which contains all the model weights, along with the optimizer and scheduler state, and the current training stats.
The default pretraining dataset is ImageNet-21k.

#### Finetuning
A model (checkpoint) can be finetuned on another dataset using the following command:
```commandline
./main.py -task fine-tune -model <model_checkpoint_file.tar> -epochs <epochs> -lr <lr> -dataset_root <dataset_root_folder>/ -results_folder <folder_for_results>/ -logging_folder <logging_folder> -run_name <name_or_description_of_the_run> (-local)
```
This will also save new checkpoints during training.
The default finetuning dataset is ImageNet-1k.

### Evaluation
It is also possible to evaluate the models.
To evaluate the model's accuracy and the efficiency metrics, run
```commandline
./main.py -task eval -model <model_checkpoint_file.tar> -dataset_root <dataset_root_folder>/ -results_folder <folder_for_results>/ -logging_folder <logging_folder> -run_name <name_or_description_of_the_run> (-local)
```
The default evaluation dataset is ImageNet-1k.

To only evaluate the efficiency metrics, run
```commandline
./main.py -task eval-metrics -model <model_checkpoint_file.tar> -dataset_root <dataset_root_folder>/ -results_folder <folder_for_results>/ -logging_folder <logging_folder> -run_name <name_or_description_of_the_run> (-local)
```
This utilizes the CIFAR10 dataset by default.

### Further Arguments
There can be multiple further arguments and flags given to the scripts.
The most important ones are

| Arg                                | Description                                            |
|:-----------------------------------|:-------------------------------------------------------|
| `-model <model>`                   | Model name or checkpoint.                              |
| `-run_name <name for the run>`     | Name or description of this training run.              |
| `-dataset <dataset>`               | Specifies a dataset to use.                            |
| `-task <task>`                     | Specifies a task. The default is `pre-train`.          |
| `-local`                           | Run on the local machine, not on a slurm cluster.      |
| `-dataset_root <dataset root>`     | Root folder of the datasets.                           |
| `-results_folder <results folder>` | Folder to save results into.                           |
| `-logging_folder <logging folder>` | Folder for saving logfiles.                            |
| `-epochs <epochs>`                 | Epochs to train.                                       |
| `-lr <lr>`                         | Learning rate. Default is 3e-3.                        |
| `-batch_size <bs>`                 | Batch size. Default is 2048.                           |
| `-weight_decay <wd>`               | Weight decay. Default is 0.02.                         |
| `-imsize <image resolution>`       | Resulution of the image to train with. Default is 224. |

For a list of all arguments, run
```commandline
./main.py --help
```

## License
We release this code under the [MIT license](./LICENSE).


