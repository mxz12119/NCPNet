# NCPNet
![PyPI](https://img.shields.io/pypi/v/NCPNet)![Packagist License](https://img.shields.io/packagist/l/mxz12119/NCPNet)


## 1. Brief Introduction
Neuronal Circuit Prediction Network (NCPNet), a simple and effective model for inferring neuron-level connections in a brain circuit network.
## 2. Installation
### Requirements
* Linux
* CUDA environment
* Python 3.7~3.11
* NVIDIA CUDA Compiler Driver NVCC version>=10.0. This is used for compiling the dependencies of torch_geometric: torch-cluster,torch-sparse,torch-scatter.
* [Pytorch](https://pytorch.org/)>=2.0.1
* [Pytorch Geometric](https://pyg.org/)>=2.3.1

### Step 1
1. Ensure that [nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html) is accessible from terminal and version >=10.0

```
nvcc --version
>>>nvcc: NVIDIA (R) Cuda compiler driver Copyright (c) 2005-2018 NVIDIA Corporation Built on Sat_Aug_25_21:08:01_CDT_2018 Cuda compilation tools, release 10.0, V10.0.130
```

If NVCC is not not included in your device,  it can be installed through the CUDA Toolkit. NVCC is a part of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux).

2. Ensure that CUDA is usable and version >=11.6
```
nvidia-smi |grep Version
>>>CUDA Version: >=11.6
```
### Step 2
Use pip to install NCPNet
```
pip install NCPNet
```
Note that it takes a long time to build torch-cluster,torch-sparse,and torch-scatter by pip. Don`t worry, just wait for a while.
### Our main dependencies:
```
torch==2.0.1
torch_geometric==2.3.1
torch-cluster==1.6.1
torch-sparse==0.6.17
torch-scatter==2.1.1
navis==1.3.1
neuprint-python==0.4.25
```
If you would like to reproduce our experiments and plots, please also install jupyter.
```
pip install jupyter
```

## Code structure:
```
Source Code
├── data
|   ├──Hemibrain
|   └──C.Elegans
├── example
├── runs
├── configs
├── NCPNet
|   ├── approaches
|   ├── brain_data.py
|   ├── task.py
|   ├── trainer.py
|   └── utils.py
└── requirements.txt
```
## Examples
### 1. Easy train a model on *Drosophila* HemiBrain
NCPNet uses a configuration file (yaml) to control training and test. You can run 'example/train-eval.py' with 'configs/linkpred_example.yaml':

Run 
```
python example/train-eval.py -c configs/linkpred.yaml
```
linkpred.yaml include the hyperparameters of NCPNet.

Results:
```
Task num:1
********************************************************************************************************************************************************************************************************
Begin 0 Task
{'Experiment': 'HemiBrain', 'val': 0.85, 'test': 0.1, 'train': 0.05, 'lr': 0.005, 'epoch': 400, 'weight_decay': 5e-06, 'seed': 68, 'device': 'cuda:1', 'optimizer': 'adam', 'save': './saved', 'loss': 'bce', 'eval_step': 2, 'negtive_num': 1, 'task_save': 'mlp_demo_model3', 'use_tensor_board': True, 'batch_size': 10000, 'save_dataset': True, 'Model': 'LinkPred', 'node_encoder': 'GCN', 'pair_encoder': 'NeighEnco2', 'use_type_info': False, 'in_channels': 5555, 'hidden_channels': 128, 'out_channels': 64, 'dim': 100, 'dropout': 0.5, 'score_func': 'mlptri', 'num_layer': 2, 'hop': 2}
|::Training::|Epoch:0|Iter:36 |Training loss:0.5731|epoch_time_cost:10.0 s|
|::Training::|Epoch:1|Iter:72 |Training loss:0.4414|epoch_time_cost:4.6 s|
|::Training::|Epoch:2|Iter:108 |Training loss:0.3764|epoch_time_cost:4.4 s|
|::Testing::|Epoch:2|Iter:108|Loss:0.3572|Accuracy:0.841|AUC:0.921|
|::Training::|Epoch:3|Iter:144 |Training loss:0.3377|epoch_time_cost:4.7 s|
|::Training::|Epoch:4|Iter:180 |Training loss:0.2822|epoch_time_cost:4.8 s|
|::Testing::|Epoch:4|Iter:180|Loss:0.2675|Accuracy:0.897|AUC:0.958|
|::Training::|Epoch:5|Iter:216 |Training loss:0.2565|epoch_time_cost:3.7 s|
|::Training::|Epoch:6|Iter:252 |Training loss:0.2430|epoch_time_cost:3.9 s|
|::Testing::|Epoch:6|Iter:252|Loss:0.2575|Accuracy:0.903|AUC:0.962|
|::Training::|Epoch:7|Iter:288 |Training loss:0.2348|epoch_time_cost:3.7 s|
|::Training::|Epoch:8|Iter:324 |Training loss:0.2271|epoch_time_cost:3.9 s|
|::Testing::|Epoch:8|Iter:324|Loss:0.2420|Accuracy:0.908|AUC:0.966|
|::Training::|Epoch:9|Iter:360 |Training loss:0.2209|epoch_time_cost:3.7 s|
|::Training::|Epoch:10|Iter:396 |Training loss:0.2163|epoch_time_cost:4.1 s|
|::Testing::|Epoch:10|Iter:396|Loss:0.2347|Accuracy:0.912|AUC:0.968|
|::Training::|Epoch:11|Iter:432 |Training loss:0.2123|epoch_time_cost:4.5 s|
|::Training::|Epoch:12|Iter:468 |Training loss:0.2072|epoch_time_cost:4.7 s|
|::Testing::|Epoch:12|Iter:468|Loss:0.2364|Accuracy:0.914|AUC:0.969|
|::Training::|Epoch:13|Iter:504 |Training loss:0.2042|epoch_time_cost:3.7 s|
|::Training::|Epoch:14|Iter:540 |Training loss:0.2021|epoch_time_cost:4.1 s|
|::Testing::|Epoch:14|Iter:540|Loss:0.2312|Accuracy:0.916|AUC:0.970|
|::Training::|Epoch:15|Iter:576 |Training loss:0.1997|epoch_time_cost:4.5 s|
|::Training::|Epoch:16|Iter:612 |Training loss:0.1980|epoch_time_cost:4.7 s|
|::Testing::|Epoch:16|Iter:612|Loss:0.2393|Accuracy:0.917|AUC:0.971|
```
### 2. Predict  neuronal connection
 Once you train a model, such as 'model.ncpnet', then use the following command to predict the probility between two neurons:
 ```
 python -m NCPNet -pred 1721996278 1722670151 -m model.ncpnet
 ```
 Response:
 ```
 Inferring the connection probability of (1721996278->1722670151)
The score of (1721996278->1722670151): 0.874
 ```
## Reproducibility of Our Paper
Please try to use jupyter to reproduce our experiments in ./example/

## Access Data
### Raw Data
The *Drosophila* connectome is available at <https://www.janelia.org/project-team/flyem/hemibrain>.


The *C.elegans* connectome is available at <https://wormwiring.org/>
### Preprocessed Data
The data will be released after the review process.







