import torch
import torch.nn as nn

from udl_project.models.res_block import ResBlock


class EnsembleModel(nn.Module):
    def __init__(self, num_classes, num_models=3):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList([ResBlock(num_classes) for _ in range(num_models)])

        # Apply different initializations (from Sören's weights_init)
        for i, model in enumerate(self.models):
            torch.manual_seed(i)
            self._apply_original_init(model)

    def _apply_original_init(self, model):
        def weights_init(layer_in):
            if isinstance(layer_in, nn.Linear):
                nn.init.kaiming_uniform_(layer_in.weight)
                layer_in.bias.data.fill_(0.0)

        model.apply(weights_init)

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs).mean(dim=0)
