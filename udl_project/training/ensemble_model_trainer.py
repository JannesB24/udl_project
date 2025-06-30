import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import pickle

from udl_project import config
from udl_project.data_loader_flowers import DataLoaderFlowers
from udl_project.models.ensemble_model import EnsembleModel

from udl_project.training.abstract_trainer import Trainer


class EnsembleModelTrainer(Trainer):
    def __init__(self, num_models: int, *, epochs: int):
        super().__init__()
        self.num_models = num_models
        self.epochs = epochs

    def train(self):
        # TODO: extract/refactor the printing.
        print("ENSEMBLE REGULARIZATION TRAINING")
        print(f"Using {self.num_models} ResNet models in ensemble.")
        print("=" * 60)

        self._train()

        print("\nENSEMBLE TRAINING COMPLETED!")
        print("Generated files:")
        print("  - ../artifacts/ensemble_model.pth")
        print("  - ../artifacts/ensemble_results.pkl")

    def _train(self) -> None:
        print("=" * 60)
        print("RUNNING ENSEMBLE RESNET")
        print("=" * 60)

        device = torch.device("cpu")

        # use standard parameters of the data loader
        dataloader = DataLoaderFlowers.create_dataloader()

        model = EnsembleModel(num_classes=dataloader.num_classes, num_models=self.num_models)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_losses = np.zeros(self.epochs)
        val_losses = np.zeros(self.epochs)
        train_accs = np.zeros(self.epochs)
        val_accs = np.zeros(self.epochs)

        print("Training Ensemble ResBlock...")

        for epoch in range(self.epochs):
            model.train()
            t0 = datetime.now()

            train_loss = []
            val_loss = []
            n_correct_train = 0
            n_total_train = 0

            # Training phase
            for images, labels in dataloader.get_train_dataloader():
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                y_pred = model(images)
                loss = criterion(y_pred, labels)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                _, predicted_labels = torch.max(y_pred, 1)
                n_correct_train += (predicted_labels == labels).sum().item()
                n_total_train += labels.shape[0]

            train_loss = np.mean(train_loss)
            train_losses[epoch] = train_loss
            train_accs[epoch] = n_correct_train / n_total_train

            # Validation phase
            model.eval()
            n_correct_val = 0
            n_total_val = 0
            with torch.no_grad():
                for images, labels in dataloader.get_test_dataloader():
                    images = images.to(device)
                    labels = labels.to(device)

                    y_pred = model(images)
                    loss = criterion(y_pred, labels)
                    val_loss.append(loss.item())

                    _, predicted_labels = torch.max(y_pred, 1)
                    n_correct_val += (predicted_labels == labels).sum().item()
                    n_total_val += labels.shape[0]

            val_loss = np.mean(val_loss)
            val_losses[epoch] = val_loss
            val_accs[epoch] = n_correct_val / n_total_val
            duration = datetime.now() - t0

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accs[epoch]:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accs[epoch]:.4f} | "
                f"Duration: {duration}"
            )

        # Save ensemble model
        torch.save(model.state_dict(), config.ARTIFACTS_DIR / "ensemble_model.pth")

        # Save ensemble results
        ensemble_results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "model_name": "Ensemble ResNet",
        }

        with open(config.ARTIFACTS_DIR / "ensemble_results.pkl", "wb") as f:
            pickle.dump(ensemble_results, f)


if __name__ == "__main__":
    # Example usage
    num_models = 5  # Example number of models in the ensemble
    epochs = 25  # Example number of training epochs
    trainer = EnsembleModelTrainer(num_models=num_models, epochs=epochs)
    trainer.train()
