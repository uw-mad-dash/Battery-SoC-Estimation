# Battery-SoC-Estimation 🔋

This repository contains the data and code for the paper [Estimating Battery State-of-Charge within 1% using Machine Learning and Physics-based Models (SAE'23)](https://www.sae.org/publications/technical-papers/content/2023-01-0522/). Specifically, here we included LiPo 5450 mAh cell dataset, Python framework for building neural network model that predicts battery SoC, and the best hyperparameters we found during hyperparameter tuning.

<p align="center">
  <a href="#data">Data 🗃️</a> •
  <a href="#installation">Installation 🛠️</a> •
  <a href="#quick-start">Quick Start 🚀</a> •
  <a href="#contributing">Contributing 🐜</a> •
  <a href="#license">License ⚖️</a>
</p>

## Data
The [LiPo 5450 mAh cell](https://maxamps.com/products/lipo-5450-1s-3-7v-battery) data can be found in <em>data</em> directory 👀. We collected 48 discharging-charging cycles for CRates 1, 2, 3, and 4. The measurements were recorded at 1 second rate. To understand data better, we provided a [Jupyter](https://jupyter.org/) notebook <em>data_visualization.ipynb</em> in <em>notebooks</em> directory. We used [Jupyter Dash](https://github.com/plotly/jupyter-dash) library which allows to create interactive plots.

<div align="center">

![](data_visualization_demo.gif)

</div>

## Installation

> __Note__
> We used Python 3.8 and versions of dependencies specified in <em>requirements.txt</em>, but it is not necessary to follow exactly the same versions.

Create [Python virtual environment](https://docs.python.org/3/library/venv.html):

```console
python3.8 -m venv battery-soc-estimation-venv
source  battery-soc-estimation-venv/bin/activate
```

[Fork the repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo) and clone forked repo:

```console
git clone https://github.com/your_github_username/Battery-SoC-Estimation
```

Install dependencies:

```console
cd Battery-SoC-Estimation
pip3 install -r requirements.txt
```

## Quick Start

<strong>First of all, run data processing</strong>. This step creates training, validation, and test datasets. Besides that, it can add new features which are formed by applying filter to one of the existing features. In order to run data processing, you must specify the name of the <em>data processing configuration (dpc)</em> file which should be located in <em>src/data_processing/configs</em>:

```console
python3 main.py -r=data_processing -dpc=I_V_T.ini
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

Processed data is saved in <em>src/data_processing/processed_data</em>. If the dpc name was <em>anyname.ini</em>, then there are created three of the following files - <em>anyname_train.csv</em>, <em>anyname_val.csv</em>, and <em>anyname_test.csv</em>.

_ _ _

<strong>Second, run model training</strong>.  The command takes for input the name of a data processing configuration file, and the name of a <em>model training configuration (mtc)</em> file which can be found in  <em>src/model_training/configs</em>:

```console
python3 main.py -r=model_training -dpc=I_V_T.ini -mtc=I_V_T.ini
```

<details><summary>Click here to learn more about mtc</summary>
<p>

You can specify any other model training configuration file or create your own. Each configuration file should contain the following:
<div align="center">

| Property  | Value | Example
| ------------- | :-------------: | -------------: |
| num_hidden_layers  | Number of hidden layers in neural network | 2 |
| units_hidden_layers  | Number of units in each hidden layer | [80, 80] |
| activations_hidden_layers  | Activations of each hidden layer | ["tanh", "leaky_relu"] |
| activation_response_layer  | Activation of response layer | "clipped_relu" |
| epochs  | Number of epochs | 30 |
| batch_size  | Batch size | 64 |
| learning_rate  | Learning rate | 0.003311 |

</div>

</p>
</details>

Trained model is saved in <em>src/model_training/trained_models</em>. If the dpc name is <em>anyname1.ini</em> and mtc name were <em>anyname2.ini</em>, then the name of the saved model is <em>anyname1_anyname2</em>.

_ _ _

<strong>Third, run model inference</strong>. The command takes for input dpc and mtc. There is NO inference configuration file. Running this command simply makes predictions on a test dataset:

```console
python3 main.py -r=inference -dpc=I_V_T.ini -mtc=I_V_T.ini
```

Test dataset with the model predictions is saved in <em>src/inference/results</em>. If the dpc name was <em>anyname1.ini</em> and mtc name was <em>anyname2.ini</em>, then the name of saved results is <em>anyname1_anyname2.csv</em>. To analyze the results we included Jupyter notebook <em>results_visualization.ipynb</em>.

## Contributing

Feel free to start a [discussion](https://github.com/uw-mad-dash/Battery-SoC-Estimation/discussions), or create a [pull request](https://github.com/uw-mad-dash/Battery-SoC-Estimation/pulls).

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)