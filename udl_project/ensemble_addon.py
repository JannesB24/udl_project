"""
Ensemble Add-on for Your Original resNetEx.py
This file ONLY handles the ensemble training and comparison
Your original resNetEx.py remains unchanged and is used as-is
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

import DataLoaderFFSet
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


def load_original_results():
    try:
        # Try to load saved results from original resNetEx.py
        with open("original_results.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def run_ensemble_only():
    print("=" * 60)
    print("RUNNING ENSEMBLE RESNET")
    print("=" * 60)

    device = torch.device("cpu")

    # Create ensemble model
    model = EnsembleModel(5, num_models=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Same parameters as Sören
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

        # Training phase (same structure as Sören)
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

        # Validation phase (same structure as Sören)
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
    torch.save(model.state_dict(), "ensemble_flower_classification_model.pth")
    print("Ensemble model saved as ensemble_flower_classification_model.pth")

    # Save ensemble results
    ensemble_results = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
    }

    with open("ensemble_results.pkl", "wb") as f:
        pickle.dump(ensemble_results, f)

    return ensemble_results


def compare_results(original_results, ensemble_results):
    """
    Compare original vs ensemble results
    """
    print("\n" + "=" * 60)
    print("COMPARING ORIGINAL vs ENSEMBLE")
    print("=" * 60)

    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    results = {"Original ResBlock": original_results, "Ensemble ResBlock": ensemble_results}

    # Plot 1: Training Loss
    axes[0, 0].set_title("Training Loss Comparison", fontweight="bold")
    for name, data in results.items():
        axes[0, 0].plot(data["train_losses"], label=name, linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[0, 1].set_title("Validation Loss Comparison", fontweight="bold")
    for name, data in results.items():
        axes[0, 1].plot(data["val_losses"], label=name, linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Training Accuracy
    axes[1, 0].set_title("Training Accuracy Comparison", fontweight="bold")
    for name, data in results.items():
        axes[1, 0].plot(data["train_accs"], label=name, linewidth=2)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Validation Accuracy
    axes[1, 1].set_title("Validation Accuracy Comparison", fontweight="bold")
    for name, data in results.items():
        axes[1, 1].plot(data["val_accs"], label=name, linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("original_vs_ensemble_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Overfitting?
    plt.figure(figsize=(10, 6))
    plt.title("Overfitting Analysis: Train - Validation Accuracy Gap", fontweight="bold")

    for name, data in results.items():
        overfitting_gap = np.array(data["train_accs"]) - np.array(data["val_accs"])
        plt.plot(overfitting_gap, label=f"{name} Gap", linewidth=3)

    plt.xlabel("Epoch")
    plt.ylabel("Train Accuracy - Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    plt.savefig("overfitting_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print comparison summary
    orig_gap = original_results["train_accs"][-1] - original_results["val_accs"][-1]
    ens_gap = ensemble_results["train_accs"][-1] - ensemble_results["val_accs"][-1]

    print("\nCOMPARISON SUMMARY:")
    print(f"   Original Overfitting Gap:  {orig_gap:.4f}")
    print(f"   Ensemble Overfitting Gap:  {ens_gap:.4f}")
    print(f"   Gap Reduction:             {orig_gap - ens_gap:.4f}")

    if ens_gap < orig_gap:
        improvement = (orig_gap - ens_gap) / orig_gap * 100
        print(f"Ensemble reduced overfitting by {improvement:.1f}%")
    else:
        print("Ensemble did not improve overfitting")


def main():
    print("ENSEMBLE REGULARIZATION EXPERIMENT")
    print("=" * 60)

    # Check if original results exist
    original_results = load_original_results()

    if original_results is None:
        print("\n Original results not found!")
        print("Please run resNetEx.py first, then run this script.")
        print("\nWorkflow:")
        print("1. python resNetEx.py          # Run your original code")
        print("2. python ensemble_addon.py    # Run this ensemble comparison")
        return

    # Run ensemble training
    ensemble_results = run_ensemble_only()

    # Compare if we have original results
    if original_results is not None:
        compare_results(original_results, ensemble_results)

    print("\n ENSEMBLE EXPERIMENT COMPLETED!")
    if original_results is not None:
        print("Generated files:")
        print("  - ensemble_flower_classification_model.pth")
        print("  - original_vs_ensemble_comparison.png")
        print("  - overfitting_comparison.png")


if __name__ == "__main__":
    main()
