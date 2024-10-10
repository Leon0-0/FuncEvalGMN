# Towards Database-Free Text-to-SQL Evaluation: A Graph-Based Metric for Functional Correctness

## Overview
Execution Accuracy and Exact Set Match are commonly used metrics for evaluating the correctness of SQL queries in Text-to-SQL tasks, but both have limitations. Exact Set Match struggles with queries that are functionally equivalent but differ in syntax, while Execution Accuracy may lead to false positives because of the need for complete test databases. To address these issues, we proposed a metric called FuncEvalGMN. This graph-based metric uses relational operator tree (ROT), referred to as RelNode, to capture semantic information and applies a graph neural network for contrastive graph matching. Unlike previous approaches, FuncEvalGMN only requires the database schema, avoiding the need for costly test databases, and it generalizes well to unseen datasets, making it a scalable and reliable method for evaluating functional correctness.


## Setup
If you're running Python version 3.7 on a macOS machine, you can establish a virtual environment by executing the following command:

```shell
bash setup.sh
```

If your setup doesn't meet these requirements, you'll need to manually install `torch_scatter` and `torch_sparse` from this URL `https://data.pyg.org/whl/`

Please note that the versions of `torch_scatter` and `torch_sparse` you'll need are contingent upon your specific torch version, Python version, and machine platform. After installation, you should adjust the paths of `torch_scatter` and `torch_sparse` in the `setup.sh` file.

Alternatively, you can also use the `environment.yaml` file to create a conda environment with all the necessary dependencies by executing the following command:

```shell
conda env create -f environment.yaml
conda activate FuncEvalGMN
```

## Dataset
Our datasets are stored in the path ./GMN/database, which includes four datasets: Spider-pair train, Spider-pair dev, Spider-DK-pair dev, and Bird-pair dev.

## Models
To download the best model, please check the following link: [FuncEvalGMN Google Drive folder] (https://drive.google.com/drive/folders/1KnVtwlDuIExoEY3Bq7ayKHhldhCtIydJ?usp=drive_link)

We assume that you have downloaded it into ./GMN/save_file/checkpoints

## Training
For a quick start, you can train the model using our default settings by running the following command:

```shell
python3 GMN/train_with_pretrained_model.py
```

If you prefer to train the model with your own settings, you'll need to modify the `configure.py` file. You can also adjust the `ONLINE_TRAIN_SETTINGS` in the `train_with_pretrained_model.py` file to utilize your own dataset.

The data processor in the `train_with_pretrained_model.py` file is interchangeable. If you wish to use the AST-based model, change the `data_processor` in the `train_with_pretrained_model.py` file to `ASTProcessor`. If you're aiming to use the ROT-based model, change the `data_processor` to either `PositionalEncodingProcessor` or `MatchEdgeProcessor`.

## Inference
To run inference on the model, execute the following command:

```shell
python3 GMN/test.py
```

By default, this will use the checkpoint located in `GMN/save_file/checkpoints/`. If you'd like to use your own checkpoints, simply place them in this folder and adjust the `model_path` in the `test.py` file.