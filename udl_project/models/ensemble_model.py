import torch
import torch.nn as nn

from udl_project.models.res_net import ResNet
from udl_project.utils.weights import weights_init


class EnsembleModel(nn.Module):
    def __init__(self, num_classes, num_models=3):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([ResNet(num_classes) for _ in range(num_models)])

        # Apply different initializations (from Sören's weights_init)
        for i, model in enumerate(self.models):
            torch.manual_seed(i)
            model.apply(weights_init)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs).mean(dim=0)
