# mlp_transformer_registration

## Introduction 
This repository contains scripts and functions used for STACOM2022 paper 'Unsupervised Echocardiography Registration through Patch-based MLPs and Transformers'. 
![Model](model.png "Proposed Models")


## Usage
- Model related files: mlp\_model (proposed three patch-based MLP/Transformer models), vxm\_model (the 2D VoxelMorph model), modelio (basic model class), swin\_trains\_utils (useful functions for SwinTrans model), layers (useful layer class)
- helper files: losses (loss function), utils (help functions), metrics (evaluation functions)
- dataset related files: generators (specific CAMUS data generator) 
- train\_model: script used to train different models, need to declare the different settings in config\_model.ini file 


