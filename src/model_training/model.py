from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input


def build_model(
    num_selected_features: int,
    num_hidden_layers: int,
    units_hidden_layers: list,
    activations_hidden_layers: list,
    activation_response_layer: str,
):
    """
    Build neural network architecture.
    """
    model = Sequential()
    model.add(Input(shape=(num_selected_features,)))
    for i in range(num_hidden_layers):
        model.add(
            Dense(
                units=units_hidden_layers[i],
                activation=activations_hidden_layers[i],
            )
        )
    model.add(Dense(1, activation=activation_response_layer))

    return model
