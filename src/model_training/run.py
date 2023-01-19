"""
1. This module builds model based on specified configurations. 
2. Then trains the model, and saves it.
"""
import pandas as pd
import os
import json
import logging

from tensorflow.keras import optimizers
from tensorflow.keras.metrics import (
    RootMeanSquaredError,
    MeanAbsoluteError,
)
from configparser import ConfigParser

from src.data_processing.filters import *
from src.model_training.model import build_model
from src.model_training.activations import set_custom_activations

logging.basicConfig()
logger = logging.getLogger(name="logger")
logger.setLevel(logging.INFO)

LABEL_FEATURE = "SOC [%]"
PREDICTED_FEATURE = "SOC_PREDICTED [%]"


def run_model_training(dpc: str, mtc: str):
    # Parse data processing config
    config_object = ConfigParser()
    config_file_name = os.path.join("src", "data_processing", "configs", dpc)
    config_object.read(config_file_name)
    dpc_values = config_object["VALUES"]

    # Parse model training config
    config_file_name = os.path.join("src", "model_training", "configs", mtc)
    config_object.read(config_file_name)
    mtc_values = config_object["VALUES"]

    # Set custom activations
    set_custom_activations()

    # Get data
    train_df = pd.read_csv(
        os.path.join(
            "src",
            "data_processing",
            "processed_data",
            f"{dpc[:-4]}_train.csv",
        )
    )
    val_df = pd.read_csv(
        os.path.join(
            "src",
            "data_processing",
            "processed_data",
            f"{dpc[:-4]}_val.csv",
        )
    )

    # Get model
    features = json.loads(dpc_values["features"])
    model = build_model(
        len(features),
        json.loads(mtc_values["num_hidden_layers"]),
        json.loads(mtc_values["units_hidden_layers"]),
        json.loads(mtc_values["activations_hidden_layers"]),
        json.loads(mtc_values["activation_response_layer"]),
    )

    # Compile model
    optimizer = optimizers.Adam(learning_rate=json.loads(mtc_values["learning_rate"]))
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=[RootMeanSquaredError(), MeanAbsoluteError()],
    )

    # Train model
    X_train = train_df[features]
    y_train = train_df[[LABEL_FEATURE]]
    X_val = val_df[features]
    y_val = val_df[[LABEL_FEATURE]]

    logger.info(f"Training model")
    model.fit(
        X_train,
        y_train,
        epochs=json.loads(mtc_values["epochs"]),
        batch_size=json.loads(mtc_values["batch_size"]),
        validation_data=(
            X_val,
            y_val,
        ),
        shuffle=True,
    )

    # Save model
    logger.info(f"Saving trained model")
    model.save(
        os.path.join(
            "src",
            "model_training",
            "trained_models",
            f"{dpc[:-4]}_{mtc[:-4]}",
        )
    )
