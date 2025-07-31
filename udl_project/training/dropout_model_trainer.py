from udl_project.training.resnet_model_trainer import ResNetModelTrainer
import pickle
from datetime import datetime

import numpy as np
import torch
from torch import nn

from udl_project import config
from udl_project.data_loader_flowers import DataLoaderFlowers
from udl_project.models.res_net import ResNet

from udl_project.utils.weights import weights_init
from udl_project.models.dropout_res_net import DPResNet

class DropoutModelTrainer(ResNetModelTrainer):
    def __init__(self, *, epochs, learning_rate, dp_rate):
        self.dp = dp_rate
        super().__init__(epochs=epochs, learning_rate=learning_rate)
    
    def _train(self) -> tuple[np.ndarray, np.ndarray]:
        device = torch.device("cpu")

        # call with standard parameters
        data_loader = DataLoaderFlowers.create_dataloader()

        # create model and initialize parameters
        model = DPResNet(num_classes=data_loader.num_classes,dp_rate=self.dp)
        model.apply(weights_init)

        # choose loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # initial tracking variables
        train_losses = np.zeros(self.epochs)
        val_losses = np.zeros(self.epochs)
        train_accs = np.zeros(self.epochs)
        val_accs = np.zeros(self.epochs)

        print("Training Dropout ResNet...")

        for epoch in range(self.epochs):
            model.train()
            t0 = datetime.now()

            train_loss = []
            val_loss = []
            n_correct_train = 0
            n_total_train = 0

            for images, labels in data_loader.get_train_dataloader():
                images_device = images.to(device)
                labels_device = labels.to(device)

                optimizer.zero_grad()

                y_pred = model(images_device)
                loss = criterion(y_pred, labels_device)

                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

                # Compute training accuracy
                _, predicted_labels = torch.max(y_pred, 1)
                n_correct_train += (predicted_labels == labels).sum().item()
                n_total_train += labels.shape[0]

            train_loss = np.mean(train_loss)
            train_losses[epoch] = train_loss
            train_accs[epoch] = n_correct_train / n_total_train
            print(train_loss)

            # Validation phase
            model.eval()
            n_correct_val = 0
            n_total_val = 0
            with torch.no_grad():
                for images, labels in data_loader.get_test_dataloader():
                    images_device = images.to(device)
                    labels_device = labels.to(device)

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

        # Save the model
        torch.save(model.state_dict(), config.ARTIFACTS_DIR / "flower_classification_dropout_model.pth")

        # Save results for comparison
        original_results = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "model_name": "Original ResNet",
        }

        results_path = config.ARTIFACTS_DIR / "dropout_results.pkl"
        with results_path.open("wb") as f:
            pickle.dump(original_results, f)

        return train_accs, val_accs