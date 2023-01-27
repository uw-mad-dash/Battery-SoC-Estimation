import pandas as pd
import os
import json
import wandb
import logging

from wandb.keras import WandbCallback
from configparser import ConfigParser
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import (
    RootMeanSquaredError,
    MeanAbsoluteError,
)

import tensorflow as tf

from src.data_processing.filters import *
from src.model_training.activations import set_custom_activations
from src.model_training.model import build_model
from src.inference.analysis import get_stats
from src.hyperparameter_tuning.util import (
    fill_in_filters_parameters,
    add_all_possible_filtered_features,
)

logging.basicConfig()
logger = logging.getLogger(name="logger")
logger.setLevel(logging.INFO)

tf.compat.v1.disable_eager_execution()

LABEL_FEATURE = "SOC [%]"
PREDICTED_FEATURE = "SOC_PREDICTED [%]"


def get_sweep_config():
    # Specify global variables
    global DPC_VALUES, MTC_VALUES, HTC_VALUES

    # Specify what are the possible values of hyperparameters
    parameters = {"batch_size": {"values": json.loads(HTC_VALUES["batch_size"])}}
    parameters["activation_response_layer"] = {
        "values": json.loads(HTC_VALUES["activations"])
    }
    parameters["learning_rate"] = {
        "min": json.loads(HTC_VALUES["learning_rate"])["min"],
        "max": json.loads(HTC_VALUES["learning_rate"])["max"],
    }
    for i in range(int(MTC_VALUES["num_hidden_layers"])):
        parameters[f"units_hidden_layer_{i}"] = {
            "values": json.loads(HTC_VALUES["units_hidden_layer"])
        }
        parameters[f"activation_{i}"] = {
            "values": json.loads(HTC_VALUES["activations"])
        }

    # If filtered features are present in config, specify possible values of filters' hyperparameters
    parameters = fill_in_filters_parameters(parameters, DPC_VALUES, HTC_VALUES)

    sweep_config = {
        "method": json.loads(HTC_VALUES["tuning_algo"]),
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": parameters,
    }

    return sweep_config


def sweep_function():
    # Specify global variables
    global WANDB_PROJECT_NAME

    # Initialize project
    wandb.init(project=WANDB_PROJECT_NAME)

    # Config is a variable that holds and saves hyperparameters
    configs = wandb.config
    units_hidden_layers = []
    activations_hidden_layers = []
    for i in range(int(MTC_VALUES["num_hidden_layers"])):
        units_hidden_layers.append(configs[f"units_hidden_layer_{i}"])
        activations_hidden_layers.append(configs[f"activation_{i}"])

    # Get model
    features = json.loads(DPC_VALUES["features"])
    model = build_model(
        len(features),
        json.loads(MTC_VALUES["num_hidden_layers"]),
        units_hidden_layers,
        activations_hidden_layers,
        configs["activation_response_layer"],
    )

    # Select filtered features corresponding to hyperparameters
    for feature in features:
        if "_MA" in feature:
            TRAIN_DF[feature] = TRAIN_DF[
                f"{feature}_{configs[f'{feature}_window_size']}"
            ]
            VAL_DF[feature] = VAL_DF[f"{feature}_{configs[f'{feature}_window_size']}"]
        if "_BWF" in feature:
            TRAIN_DF[feature] = TRAIN_DF[
                f"{feature}_{configs[f'{feature}_cutoff_fs']}_{configs[f'{feature}_order']}"
            ]
            VAL_DF[feature] = VAL_DF[
                f"{feature}_{configs[f'{feature}_cutoff_fs']}_{configs[f'{feature}_order']}"
            ]
        if "_H" in feature:
            TRAIN_DF[feature] = TRAIN_DF[
                f"{feature}_{configs[f'{feature}_window_size']}_{configs[f'{feature}_n']}"
            ]
            VAL_DF[feature] = VAL_DF[
                f"{feature}_{configs[f'{feature}_window_size']}_{configs[f'{feature}_n']}"
            ]

    # Get data
    X_train = TRAIN_DF[features]
    y_train = TRAIN_DF[LABEL_FEATURE]
    X_val = VAL_DF[features]
    y_val = VAL_DF[LABEL_FEATURE]

    # Compile model
    optimizer = optimizers.Adam(learning_rate=configs["learning_rate"])
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=[RootMeanSquaredError(), MeanAbsoluteError()],
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs=int(MTC_VALUES["epochs"]),
        batch_size=configs["batch_size"],
        validation_data=(
            X_val,
            y_val,
        ),
        shuffle=True,
        callbacks=[WandbCallback(log_evaluation_frequency=1)],
    )

    # Run inference on validation dataset
    VAL_DF[PREDICTED_FEATURE] = model.predict(X_val).flatten()

    # Log results
    stats = get_stats(VAL_DF, LABEL_FEATURE, PREDICTED_FEATURE)
    wandb.log({f"mean_abs_error_val": stats["mae"]})
    wandb.log({f"max_abs_error_val": stats["max_abs"]})
    wandb.log({f"root_mean_sq_error_val": stats["rmse"]})
    wandb.finish()


def run_hyperparameter_tuning(dpc: str, mtc: str, htc: str, wandb_user: str):
    # Specify global variables
    global DPC_VALUES, MTC_VALUES, HTC_VALUES, TRAIN_DF, VAL_DF, WANDB_PROJECT_NAME

    # Set custom activations
    set_custom_activations()

    # Parse dpc
    config_object = ConfigParser()
    config_file_name = os.path.join("src", "data_processing", "configs", dpc)
    config_object.read(config_file_name)
    DPC_VALUES = config_object["VALUES"]

    # Parse mtc
    config_file_name = os.path.join("src", "model_training", "configs", mtc)
    config_object.read(config_file_name)
    MTC_VALUES = config_object["VALUES"]

    # Parse htc
    config_file_name = os.path.join("src", "hyperparameter_tuning", "configs", htc)
    config_object.read(config_file_name)
    HTC_VALUES = config_object["VALUES"]

    # Get data
    TRAIN_DF = pd.read_csv(
        os.path.join(
            "src",
            "data_processing",
            "processed_data",
            f"{dpc[:-4]}_train.csv",
        )
    )
    VAL_DF = pd.read_csv(
        os.path.join(
            "src",
            "data_processing",
            "processed_data",
            f"{dpc[:-4]}_val.csv",
        )
    )

    # Immediately apply filters with all possible configurations
    logger.info("Adding all possible filtered features to training dataset")
    TRAIN_DF = add_all_possible_filtered_features(TRAIN_DF, DPC_VALUES, HTC_VALUES)
    logger.info("Adding all possible filtered features to validation dataset")
    VAL_DF = add_all_possible_filtered_features(VAL_DF, DPC_VALUES, HTC_VALUES)

    # Sweep
    logger.info("Starting W&B project")
    WANDB_PROJECT_NAME = f"{dpc[:-4]}_{mtc[:-4]}_{htc[:-4]}"
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(sweep_config, entity=wandb_user, project=WANDB_PROJECT_NAME)
    wandb.agent(
        sweep_id, function=sweep_function, count=int(MTC_VALUES["tuning_count"])
    )
