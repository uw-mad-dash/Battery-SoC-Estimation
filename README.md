# Battery-SoC-Estimation 🔋

This repository contains the data and code for the paper [Estimating Battery State-of-Charge within 1% using Machine Learning and Physics-based Models (SAE'23)](https://www.sae.org/publications/technical-papers/content/2023-01-0522/). Specifically, here we included LiPo 5450 mAh cell dataset, Python framework for building neural network model that predicts battery SoC, and the best hyperparameters we found during hyperparameter tuning.

<p align="center">
  <a href="#data">Data 🗃️</a> •
  <a href="#installation">Installation 🛠️</a> •
  <a href="#quick-start">Quick Start 🚀</a> •
  <a href="#hyperparameter-tuning">Hyperparameter Tuning ⚙️</a> •
  <a href="#contributing">Contributing 🐜</a> •
  <a href="#license">License ⚖️</a>
</p>

## Data
The [LiPo 5450 mAh cell](https://maxamps.com/products/lipo-5450-1s-3-7v-battery) data can be found in <em>data</em> directory 👀. To understand data better, we provided a [Jupyter](https://jupyter.org/) notebook <em>data_visualization.ipynb</em> in <em>notebooks</em> directory. We used [Jupyter Dash](https://github.com/plotly/jupyter-dash) library which allows to create interactive plots.

<div align="center">

![](data_visualization_demo.gif)

</div>

## Installation

> __Note__
> We used Python 3.8 and versions of dependencies specified in <em>requirements.txt</em>, but it is not necessary to follow exactly the same versions.

We recommend using the package manager [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to create isolated environment for development and install necessary dependencies inside of it.

```console
conda create -n battery-soc-estimation python=3.8
conda activate battery-soc-estimation
conda install --file requirements.txt
```

## Quick Start

<strong>First of all, run data processing</strong>. This step creates training, validation, and test datasets. Besides that, it can add new features which are formed by applying filter to one of the existing features. In order to run data processing, you must specify the name of the <em>data processing configuration (dpc)</em> file which should be located in <em>src/data_processing/configs</em>:

```console
python3 main.py -dpc=I_V_T.ini
```

<details><summary>Click here to learn more about dpc</summary>
<p>

You can specify any other data processing configuration file or create your own. Each configuration file should contain the following:
<div align="center">

| Property  | Value | Example
| ------------- | :-------------: | -------------: |
| train_cycles  | Range of cycle numbers to be included in training dataset from each CRate (border values incl.) | [1, 30] |
| val_cycles  | Range of cycle numbers to be included in validation dataset from each CRate (border values incl.) | [31, 38]  |
| test_cycles  | Range of cycle numbers to be included in test dataset from each CRate (border values incl.) | [39, 48] |
| features  | Which features will be fed as input to the ML model | ['Voltage [V]', 'Current [A]', 'TempBottom [C]', 'Current_MA [A]', 'TempBottom_BWF [C]', 'Voltage_H [V'] |
| ma_window_size | If you included in <em>features</em> a feature modified with Moving Average filter, then you need to specify the window size of the filter | {'Current_MA [A]': 20} |
| bwf_cutoff_fs | If you included in <em>features</em> a feature modified with Butterworth filter, then you need to specify the cutoff frequency of the filter | {'TempBottom_BWF [C]': 0.0001} |
| bwf_order | If you included in <em>features</em> a feature modified with Butterworth filter, then you need to specify the order of the filter | {'TempBottom_BWF [C]': 1} |
| h_window_size | If you included in <em>features</em> a feature modified with Hampel filter, then you need to specify the window size of the filter | {'Voltage_H [V]': 1} |
| h_n | If you included in <em>features</em> a feature modified with Hampel filter, then you need to specify the threshold n of the filter | {'Voltage_H [V]': 3} |

</div>

</p>
</details>

Processed data will be saved in <em>src/data_processing/processed_data</em>. If the dpc name is <em>anyname.ini</em>, then there will be created three of the following files - <em>anyname_train.csv</em>, <em>anyname_val.csv</em>, and <em>anyname_test.csv</em>.

_ _ _

<strong>Second, run model training</strong>. 

```console
python3 main.py -r=model_training -dpc=I_V_T.ini -mtc=config1.ini
```

_ _ _

<strong>Third, run model inference</strong>. 

```console
python3 main.py -r=inference -dpc=I_V_T.ini -mtc=config1.ini
```

## Hyperparameter Tuning

> __Note__
> Hyperparameter tuning is a long-running process that takes hours so in <em>src/model_training/configs</em> we included some of the best hyperparameter configurations we found. <strong>You can skip this section if the model you trained in <em>Quick Start</em> satisfies your requirements.</strong> Otherwise, follow instructions in this section on how to run tuning and find good neural network architectures.

## Contributing

Feel free to create a pull request or start a discussion in issues section.

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)