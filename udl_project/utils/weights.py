import torch.nn as nn


def weights_init(layer_in: nn.Module) -> None:
    """
    Initialize the weights of a given layer.

    If the layer is an instance of nn.Linear, its weights are initialized using
    Kaiming uniform initialization and its biases are set to 0.

    Args:
        layer_in (nn.Module): The layer to initialize.
    """
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)
