import pickle
from datetime import datetime
from typing import Any

import numpy as np
import torch
from torch import nn

from udl_project import config
from udl_project.data_handling.custom_data_loader import CustomDataLoader
from udl_project.data_handling.flower_dataset import FlowerDataset
from udl_project.data_handling.food_dataset import FoodDataset
from udl_project.models.res_net import ResNet
from udl_project.training.abstract_trainer import Trainer
from udl_project.utils.weights import weights_init


def load_tfl_model(new_numclasses, freeze_original):
    # Load pretrained model (10 classes -> food dataset)
    model = ResNet(num_classes=10)
    model.load_state_dict(torch.load(config.ARTIFACTS_DIR / "pretraining_model.pth"))
    if freeze_original:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = nn.Linear(256 * 8 * 8, new_numclasses)
    return model


class ResNetModelTFLTrainer(Trainer):
    prev_numclasses = 10
    ft_numclasses = 5
    dataset = ""
    finetune_bool = False
    freeze = False

    def __init__(self, *, epochs: int, learning_rate: float):
        super().__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self):
        print("=" * 60)
        print("PRETRAINING ORIGINAL RESNET MODEL")
        print("=" * 60)

        pre_train_data_source = FoodDataset()
        # Pre-train model
        train_accs, val_accs = self._train(
            data_source=pre_train_data_source,
        )

        print("Pretraining model training completed!")
        print(f"Final overfitting gap: {train_accs[-1] - val_accs[-1]:.4f}")
        print(f"Results saved to {config.ARTIFACTS_DIR / 'pretrain_results.pkl'}")

        print("=" * 60)
        print("FINETUNING RESNET MODEL")
        print("=" * 60)

        finetune_dataset = FlowerDataset(train_test_split=0.8)

        # Finetune model
        train_accs, val_accs = self._train(data_source=finetune_dataset)

        print("\nFinetuning model training completed!")
        print(f"Final overfitting gap: {train_accs[-1] - val_accs[-1]:.4f}")
        print(f"Results saved to {config.ARTIFACTS_DIR / 'finetune_results.pkl'}")

        print("=" * 60)
        print("FINETUNING RESNET MODEL")
        print("=" * 60)

        finetune_dataset = FlowerDataset(train_test_split=0.8)

        # Finetune model (FREEZE)
        train_accs, val_accs = self._train(data_source=finetune_dataset, freeze=True)

        print("\nFinetuning model training completed!")
        print(f"Final overfitting gap: {train_accs[-1] - val_accs[-1]:.4f}")
        print(f"Results saved to {config.ARTIFACTS_DIR / 'finetune_results.pkl'}")

    def _train(
        self,
        data_source: Any,
        freeze: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        device = torch.device("cpu")

        is_finetune_dataset = isinstance(data_source, FlowerDataset)

        data_loader = CustomDataLoader.create_dataloader(data_source=data_source)

        # create model and initialize parameters
        if is_finetune_dataset:
            model = load_tfl_model(data_loader.num_classes, freeze)
        else:
            model = ResNet(num_classes=data_loader.num_classes)
            model.apply(weights_init)

        # choose loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # initial tracking variables
        train_losses = np.zeros(self.epochs)
        val_losses = np.zeros(self.epochs)
        train_accs = np.zeros(self.epochs)
        val_accs = np.zeros(self.epochs)

        print("Training Unregularized ResNet...")
        save_str_prefix = "pretraining"
        if is_finetune_dataset:
            save_str_prefix = "finetune"

        save_str = save_str_prefix + "_model.pth"
        for epoch in range(self.epochs):
            model.train()
            t0 = datetime.now()

            train_loss = []
            val_loss = []
            n_correct_train = 0
            n_total_train = 0

            for image, label in data_loader.get_train_dataloader():
                images_device = image.to(device)
                labels_device = label.to(device)

                optimizer.zero_grad()

                y_pred = model(images_device)
                loss = criterion(y_pred, labels_device)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                # Compute training accuracy
                _, predicted_labels = torch.max(y_pred, 1)
                n_correct_train += (predicted_labels == labels_device).sum().item()
                n_total_train += labels_device.shape[0]

            train_loss = np.mean(train_loss)
            train_losses[epoch] = train_loss
            train_accs[epoch] = n_correct_train / n_total_train
            print(train_loss)

            # Validation phase
            model.eval()
            n_correct_val = 0
            n_total_val = 0
            with torch.no_grad():
                for image, label in data_loader.get_test_dataloader():
                    images_device = image.to(device)
                    labels_device = label.to(device)

                    y_pred = model(images_device)
                    loss = criterion(y_pred, labels_device)

                    # Store the validation loss
                    val_loss.append(loss.item())

                    # Compute validation accuracy
                    _, predicted_labels = torch.max(y_pred, 1)
                    n_correct_val += (predicted_labels == labels_device).sum().item()
                    n_total_val += labels_device.shape[0]

            val_loss = np.mean(val_loss)
            val_losses[epoch] = val_loss
            val_accs[epoch] = n_correct_val / n_total_val
            duration = datetime.now() - t0

            # Print the metrics for the current epoch
            print(
                f"Epoch [{epoch + 1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accs[epoch]:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accs[epoch]:.4f} | "
                f"Duration: {duration}"
            )

        torch.save(model.state_dict(), config.ARTIFACTS_DIR / save_str)

        # Save results for comparison
        original_results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "model_name": "Original ResNet",
        }

        print(save_str_prefix + "_results.pkl")
        results_path = config.ARTIFACTS_DIR / (save_str_prefix + "_results.pkl")
        with results_path.open("wb") as f:
            pickle.dump(original_results, f)

        return train_accs, val_accs


if __name__ == "__main__":
    # example usage
    trainer = ResNetModelTFLTrainer(learning_rate=0.001, epochs=25)
    trainer.train()
