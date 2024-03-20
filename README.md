# Uncertainties in Continual Learning

In continual learning, deep learning models incrementally learn more classes or tasks over time. Doing so, they should not forget previously learned knowledge. Making it even harder, we want the models to also estimate correct uncertainty. For example, they should be highly uncertain about a new object type, but not uncertain about an object that they just learned correctly. This paper seeks to understand various continual learning strategies and uncertainty quantification methods. These are combined by conducting experiments on continual learning baselines to analyse the trends and correlations of accuracy and uncertainty metrics while applying the various continual learning techniques. The full report can be found [here](./Uncertainties_in_Continual_Learning.pdf).

## Datasets
There are a few datasets used in this paper. To download the datasets, 
```
python install_datasets.py [desired data directory] [dataset name]
```
where the possible dataset names are
- mnist
- pacs
- domainnet

For the domainnet dataset, there was some preprocessing done to keep only the 50 most populated classes. This preprocessing can be done via the following command
```
python process_domainnet.py [directory of domainnet dataset] [number of classes]
```
where number of classes is an optional parameter that is defaulted to 50.

## Experiments
### Training
To train the model, the [train.py](./train.py) file is utilised with the following parameters.
```
python train.py [data_dir] [dataset] [num_experiences] [order] [train_pct] [batch_size] [num_workers] [-a] [seed] [model] [model_num] [hidden_sizes] [replay_prop] [ewc_lambda] [si_lambda] [-cwr] [-lwf] [save_dir] [epochs] [model_save_interval] [lr] 
```
where each of the parameters are explained below. Note that not every parameter is relevant for every experiment and may be redundant in some cases.

| Argument            | Default     | Comment                                                                                                                                                                                                                                                                  |
|---------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data_dir            | data        | Dataset directory                                                                                                                                                                                                                                                        |
| dataset             |             | Dataset name: {PermutedMNIST, PACS, domainnet}                                                                                                                                                                                                                           |
| num_experiences     | 5           | Number of experiences to generate - Only relevant for PermutedMNIST                                                                                                                                                                                                      |
| order               | None        | Order to train the different experiences. These are denoted by the first letter of the category (e.g. "ACPS" for PACS dataset would train Art-painting->Cartoon->Photo->Sketch, and "CIPQRS" would train Clipart->Inforgraphic->Painting->Quickdraw->Real-world->Sketch) |
| train_pct           | 0.7         | Percentage to allocate as train data                                                                                                                                                                                                                                     |
| batch_size          | 64          | Batch size                                                                                                                                                                                                                                                               |
| num_workers         | 2           | Number of workers                                                                                                                                                                                                                                                        |
| -a                  | False       | Whether to include data augmentation                                                                                                                                                                                                                                     |
| seed                | 42          | Random seed                                                                                                                                                                                                                                                              |
| model               |             | See below for valid models                                                                                                                                                                                                                                               |
| model_num           | 0           | Model number (To help identify various models in an ensemble)                                                                                                                                                                                                            |
| hidden_sizes        |             | List of hidden sizes for linear models                                                                                                                                                                                                                                   |
| replay_prop         | 0           | Replay proportion (0, 1) for replay strategy                                                                                                                                                                                                                             |
| ewc_lambda          | 0           | lambda for EWC strategy                                                                                                                                                                                                                                                  |
| si_lambda           | 0           | lambda for Synaptic Intelligence strategy                                                                                                                                                                                                                                |
| -cwr                | False       | Flag whether to employ CWR* strategy                                                                                                                                                                                                                                     |
| -lwf                | False       | Flag whether to employ LwF strategy                                                                                                                                                                                                                                      |
| save_dir            | checkpoints | Directory to save models                                                                                                                                                                                                                                                 |
| epochs              | 100         | Number of epochs                                                                                                                                                                                                                                                         |
| model_save_interval | None        | Number of epochs between model saves                                                                                                                                                                                                                                     |
| lr                  | 0.001       | Learning rate                                                                                                                                                                                                                                                            |

Valid model names
- mlp
- resnet18
- resnet34
- resnet50
- vgg11
- vgg13
- vgg16
- efficientnet-b0
- efficientnet-b1
- custom_cnn
- small_custom_cnn

### Testing
To test the model, the [test.py](./test.py) file is utilised with the following parameters.
```
python test.py [data_dir] [dataset] [num_experiences] [order] [train_pct] [batch_size] [num_workers] [-a] [seed] [ckpt_dir] [save_dir] [evaluation_method] 
```
where each of the parameters are explained below. Note that not every parameter is relevant for every experiment and may be redundant in some cases.

| Argument          | Default     | Comment                                                                                                                                                                                                                                                                  |
|-------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data_dir          | data        | Dataset directory                                                                                                                                                                                                                                                        |
| dataset           |             | Dataset name: {PermutedMNIST, PACS, domainnet}                                                                                                                                                                                                                           |
| num_experiences   | 5           | Number of experiences to generate - Only relevant for PermutedMNIST                                                                                                                                                                                                      |
| order             | None        | Order to train the different experiences. These are denoted by the first letter of the category (e.g. "ACPS" for PACS dataset would train Art-painting->Cartoon->Photo->Sketch, and "CIPQRS" would train Clipart->Inforgraphic->Painting->Quickdraw->Real-world->Sketch) |
| train_pct         | 0.7         | Percentage to allocate as train data                                                                                                                                                                                                                                     |
| batch_size        | 64          | Batch size                                                                                                                                                                                                                                                               |
| num_workers       | 2           | Number of workers                                                                                                                                                                                                                                                        |
| -a                | False       | Whether to include data augmentation                                                                                                                                                                                                                                     |
| seed              | 42          | Random seed                                                                                                                                                                                                                                                              |
| ckpt_dir          | checkpoints | Directory of saved models                                                                                                                                                                                                                                                |
| save_dir          | results     | Directory to save results                                                                                                                                                                                                                                                |
| evaluation_method |             | Evaluation method. List of possible evaluation methods are listed below                                                                                                                                                                                                  |

Valid evaluation methods
- baseline
- mc_dropout
- ensemble

where 'baseline' produces a single-point output (and hence no uncertainty bounds).


### Process results
To process the test results and plot the relevant graphs, [process_results.py](./process_results.py) file is utilised with the following parameters
```
python process_results.py [data_dir] [dataset] [order] [-ev] [-ee] [-esnr] [-eu] [-ec]
```
where each of the parameters are explained below.

| Argument | Default | Comment                                                                                                                                                                                                                                                                  |
|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| data_dir | data    | Directory of the results folder                                                                                                                                                                                                                                          |
| dataset  | pacs    | Dataset name: {PermutedMNIST, PACS, domainnet}                                                                                                                                                                                                                           |
| order    | None    | Order to train the different experiences. These are denoted by the first letter of the category (e.g. "ACPS" for PACS dataset would train Art-painting->Cartoon->Photo->Sketch, and "CIPQRS" would train Clipart->Inforgraphic->Painting->Quickdraw->Real-world->Sketch) |
| -ev      | False   | Flag whether to plot variance graph                                                                                                                                                                                                                                      |
| -ee      | False   | Flag whether to plot entropy graph                                                                                                                                                                                                                                       |
| -esnr    | False   | Flag whether to plot signal-to-noise ratio graph                                                                                                                                                                                                                         |
| -eu      | False   | Flag whether to plot uncertainty graphs (Mutual Information, Aleaetoric Uncertainty)                                                                                                                                                                                                                                  |
| -ec      | False   | Flag whether to plot calibration error graphs (ECE, SCE, UCE)                                                                                                                                                                                                                            |

By default, the accuracy and loss graphs will be generated and saved. In addition, a csv of the results will be produced at results.csv.

## Acknowledgements
This project was done under the supervision of Hermann Blum and Janis Postels in the Computer Vision and Geometry (CVG) Group in ETH Zurich. I am grateful for their valuable advice and guidance throughout this project. 
