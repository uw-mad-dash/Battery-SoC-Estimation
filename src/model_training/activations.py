from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.utils import get_custom_objects
from tensorflow.nn import leaky_relu


def clipped_relu(x, max_value=1):
    return relu(x, max_value)


def set_custom_activations():
    get_custom_objects().update({"clipped_relu": layers.Activation(clipped_relu)})
    get_custom_objects().update({"leaky_relu": layers.Activation(leaky_relu)})
