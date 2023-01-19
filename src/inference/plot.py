import pandas as pd
import matplotlib.pyplot as plt


def plot_results(
    df: pd.DataFrame,
    stats: dict,
    label_feature: str,
    predicted_feature: str,
):
    fig, axs = plt.subplots(2, figsize=(20, 20))

    # Create label feature vs predicted feature plot
    axs[0].plot(
        df[label_feature],
        color="black",
        marker=".",
        label=label_feature,
        linestyle="None",
        markersize=5,
    )
    axs[0].plot(
        df[predicted_feature],
        color="green",
        marker="*",
        label=predicted_feature,
        linestyle="None",
        markersize=3,
    )
    axs[0].set(ylabel=label_feature)
    axs[0].set_title(
        f"Mean Abs Error: {stats['mae']} %, Max Abs Error: {stats['max_abs']} %, Root Mean Sq Error {stats['rmse']} %"
    )

    # Create error plot
    error = df[predicted_feature] - df[label_feature]
    axs[1].plot(error, color="blue", marker=".", linestyle="None", markersize=5)
    axs[1].set(ylabel=f"Error {label_feature}")

    # Format plots
    plt.rc("font", size=18)
    plt.subplots_adjust(
        left=None, bottom=0.5, right=None, top=0.8, wspace=2.7, hspace=2.7
    )

    return fig
