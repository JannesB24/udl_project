import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import pickle

from udl_project import config
from udl_project.data_loader import CustomDataLoader
from udl_project.models.res_net import ResNet
from udl_project.training.abstract_trainer import Trainer
from udl_project.utils.weights import weights_init


class L2RegularizedModelTrainer(Trainer):
    def __init__(self, weight_decay: float, *, epochs: int):
        super().__init__()

        self.weight_decay = weight_decay
        self.epochs = epochs

    def train(self):
        print("L2 REGULARIZATION TRAINING")
        print(f"Using weight_decay={self.weight_decay}")
        print("=" * 60)

        self._train()

        print("\nL2 REGULARIZATION TRAINING COMPLETED!")
        print("Generated files:")
        # TODO: create single point of reference for path names and so on
        print("  - ../artifacts/l2_model_wd_0.01.pth")
        print("  - ../artifacts/l2_results.pkl")

    def _train(self):
        print("=" * 60)
        print(f"TRAINING L2 REGULARIZED RESNET (weight_decay={self.weight_decay})")
        print("=" * 60)

        device = torch.device("cpu")

        # call with standard parameters
        data_loader = CustomDataLoader.create_dataloader()

        # Create model exactly the unregularized
        model = ResNet(num_classes=data_loader.num_classes)
        model.apply(weights_init)

        # MAIN DIFFERENCE: Adding weight_decay parameter to optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=self.weight_decay,  # Explicit L2 regularization
        )

        train_losses = np.zeros(self.epochs)
        val_losses = np.zeros(self.epochs)
        train_accs = np.zeros(self.epochs)
        val_accs = np.zeros(self.epochs)

        print("Training L2 Regularized ResNet...")

        for epoch in range(self.epochs):
            model.train()
            t0 = datetime.now()

            train_loss = []
            val_loss = []
            n_correct_train = 0
            n_total_train = 0

            # Training phase
            for images, labels in data_loader.get_train_dataloader():
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
                for images, labels in data_loader.get_test_dataloader():
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

        torch.save(
            model.state_dict(), config.ARTIFACTS_DIR / f"l2_model_wd_{self.weight_decay}.pth"
        )

        l2_results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "weight_decay": self.weight_decay,
            "model_name": f"L2 Regularized (wd={self.weight_decay})",
        }

        with open(config.ARTIFACTS_DIR / "l2_results.pkl", "wb") as f:
            pickle.dump(l2_results, f)

        # Print summary for this configuration
        overfitting_gap = train_accs[-1] - val_accs[-1]
        print("\nL2 Regularized model training completed!")
        print(f"Final overfitting gap: {overfitting_gap:.4f}")
        print("Results saved to ../artifacts/l2_results.pkl")


if __name__ == "__main__":
    # Example usage
    trainer = L2RegularizedModelTrainer(weight_decay=0.01, epochs=25)
    trainer.train()
    print("Training complete.")
