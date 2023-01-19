import numpy as np
import pandas as pd
import math

from tensorflow.keras import losses
import tensorflow as tf


def get_stats(df: pd.DataFrame, label_feature: str, predicted_feature: str):

    # Calculate absolute errors
    abs_error = np.abs(df[label_feature] - df[predicted_feature])

    # Calculate mean absolute error
    mae = round(
        np.mean(abs_error),
        2,
    )

    # Calculate max absolute error
    max_abs = round(np.percentile(abs_error, 99.9, method="closest_observation"), 2)

    # Calculate root mean squared error
    with tf.compat.v1.Session() as sess:
        rmse = round(
            math.sqrt(
                losses.mean_squared_error(
                    df[label_feature], df[predicted_feature]
                ).eval()
            ),
            2,
        )

    return {"mae": mae, "max_abs": max_abs, "rmse": rmse}
