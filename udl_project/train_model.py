import numpy as np
import torch
import torch.nn as nn
from udl_project import DataLoaderFFSet
from datetime import datetime

from udl_project.models.res_block import ResBlock

# Number of epochs for training
EPOCHS = 10


def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)


def main():
    device = torch.device("cpu")

    # create model and initialize parameters
    model = ResBlock(5)
    model.apply(weights_init)

    # choose loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # initial tracking variables
    train_losses = np.zeros(EPOCHS)
    val_losses = np.zeros(EPOCHS)
    train_accs = np.zeros(EPOCHS)
    val_accs = np.zeros(EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        t0 = datetime.now()

        train_loss = []
        val_loss = []
        n_correct_train = 0
        n_total_train = 0

        for images, labels in DataLoaderFFSet.train_dataloader_simple:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            y_pred = model(images)
            loss = criterion(y_pred, labels)

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
            for images, labels in DataLoaderFFSet.test_dataloader_simple:
                images = images.to(device)
                labels = labels.to(device)

                y_pred = model(images)
                loss = criterion(y_pred, labels)

                # Store the validation loss
                val_loss.append(loss.item())

                # Compute validation accuracy
                _, predicted_labels = torch.max(y_pred, 1)
                n_correct_val += (predicted_labels == labels).sum().item()
                n_total_val += labels.shape[0]

        val_loss = np.mean(val_loss)
        val_losses[epoch] = val_loss
        val_accs[epoch] = n_correct_val / n_total_val
        duration = datetime.now() - t0

        # Print the metrics for the current epoch
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] - "
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accs[epoch]:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accs[epoch]:.4f} | "
            f"Duration: {duration}"
        )

    # Optionally, save the model after training
    torch.save(model.state_dict(), "artifacts/flower_classification_model.pth")


if __name__ == "__main__":
    main()
