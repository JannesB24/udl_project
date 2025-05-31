import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import DataLoaderFFSet
from models.ensemble_model import EnsembleModel


def train_ensemble_model():
    print("=" * 60)
    print("RUNNING ENSEMBLE RESNET")
    print("=" * 60)

    device = torch.device("cpu")

    model = EnsembleModel(5, num_models=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    train_accs = np.zeros(num_epochs)
    val_accs = np.zeros(num_epochs)

    print("Training Ensemble ResBlock...")

    for epoch in range(num_epochs):
        model.train()
        t0 = datetime.now()

        train_loss = []
        val_loss = []
        n_correct_train = 0
        n_total_train = 0

        # Training phase
        for images, labels in DataLoaderFFSet.train_dataloader_simple:
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
            for images, labels in DataLoaderFFSet.test_dataloader_simple:
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

        # Print in same format as Sören
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accs[epoch]:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accs[epoch]:.4f} | "
            f"Duration: {duration}"
        )

    # Save ensemble model
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(artifacts_dir, "ensemble_model.pth"))

    # Save ensemble results
    ensemble_results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "model_name": "Ensemble ResNet",
    }

    with open(os.path.join(artifacts_dir, "ensemble_results.pkl"), "wb") as f:
        pickle.dump(ensemble_results, f)

    return ensemble_results


def main():
    print("ENSEMBLE REGULARIZATION TRAINING")
    print("Using 3 ResNet models in ensemble")
    print("=" * 60)

    # Train ensemble model
    ensemble_results = train_ensemble_model()

    print("\nENSEMBLE TRAINING COMPLETED!")
    print("Generated files:")
    print("  - ../artifacts/ensemble_model.pth")
    print("  - ../artifacts/ensemble_results.pkl")


if __name__ == "__main__":
    main()
