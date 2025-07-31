import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import pickle
import copy

from udl_project import config
from udl_project.data_loader_flowers import DataLoaderFlowers
from udl_project.models.res_net import ResNet
from udl_project.training.abstract_trainer import Trainer
from udl_project.utils.weights import weights_init


class EarlyStoppingModelTrainer(Trainer):
    def __init__(self, patience: int, min_delta: float = 0.0, monitor: str = 'val_loss', *, epochs: int):
        """
        Early stopping trainer for ResNet model.
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in monitored quantity to qualify as improvement
            monitor: Metric to monitor ('val_loss' or 'val_acc')
            epochs: Maximum number of epochs to train
        """
        super().__init__()
        
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.epochs = epochs
        
        # Early stopping state
        self.best_score = None
        self.epochs_without_improvement = 0
        self.best_model_state = None
        self.stopped_epoch = 0

    def train(self):
        print("EARLY STOPPING TRAINING")
        print(f"Using patience={self.patience}, min_delta={self.min_delta}, monitor='{self.monitor}'")
        print("=" * 60)

        self._train()

        print("\nEARLY STOPPING TRAINING COMPLETED!")
        print("Generated files:")
        print(f"  - ../artifacts/early_stopping_model_p{self.patience}_d{self.min_delta}.pth")
        print("  - ../artifacts/early_stopping_results.pkl")

    def _train(self):
        print("=" * 60)
        print(f"TRAINING RESNET WITH EARLY STOPPING (patience={self.patience}, monitor={self.monitor})")
        print("=" * 60)

        device = torch.device("cpu")

        # Call with standard parameters
        data_loader = DataLoaderFlowers.create_dataloader()

        # Create model
        model = ResNet(num_classes=data_loader.num_classes)
        model.apply(weights_init)

        # Standard optimizer without weight decay for pure early stopping
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []

        print("Training ResNet with Early Stopping...")

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

            train_loss_avg = np.mean(train_loss)
            train_acc = n_correct_train / n_total_train
            
            train_losses.append(train_loss_avg)
            train_accs.append(train_acc)

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

            val_loss_avg = np.mean(val_loss)
            val_acc = n_correct_val / n_total_val
            
            val_losses.append(val_loss_avg)
            val_accs.append(val_acc)
            
            duration = datetime.now() - t0

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] - "
                f"Train Loss: {train_loss_avg:.4f}, Train Accuracy: {train_acc:.4f} | "
                f"Val Loss: {val_loss_avg:.4f}, Val Accuracy: {val_acc:.4f} | "
                f"Duration: {duration}"
            )

            # Early stopping logic
            if self._check_early_stopping(val_loss_avg, val_acc, model, epoch):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                print(f"Best {self.monitor}: {self.best_score:.4f}")
                print(f"Epochs without improvement: {self.epochs_without_improvement}")
                self.stopped_epoch = epoch + 1
                break

        # Load best model if early stopping occurred
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print("\nLoaded best model weights from early stopping.")

        # Save final model
        model_filename = f"early_stopping_model_p{self.patience}_d{self.min_delta}.pth"
        torch.save(model.state_dict(), config.ARTIFACTS_DIR / model_filename)

        early_stopping_results = {
            "train_losses": np.array(train_losses),
            "val_losses": np.array(val_losses),
            "train_accs": np.array(train_accs),
            "val_accs": np.array(val_accs),
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": self.monitor,
            "stopped_epoch": self.stopped_epoch,
            "total_epochs": len(train_losses),
            "best_score": self.best_score,
            "model_name": f"Early Stopping (p={self.patience}, δ={self.min_delta})",
        }

        with open(config.ARTIFACTS_DIR / "early_stopping_results.pkl", "wb") as f:
            pickle.dump(early_stopping_results, f)

        # Print summary
        if len(train_accs) > 0 and len(val_accs) > 0:
            final_overfitting_gap = train_accs[-1] - val_accs[-1]
            print(f"\nEarly stopping model training completed!")
            print(f"Training stopped at epoch: {self.stopped_epoch}/{self.epochs}")
            print(f"Final overfitting gap: {final_overfitting_gap:.4f}")
            print(f"Best {self.monitor}: {self.best_score:.4f}")
            print("Results saved to ../artifacts/early_stopping_results.pkl")

    def _check_early_stopping(self, val_loss: float, val_acc: float, model, epoch: int) -> bool:
        """
        Check if early stopping should be triggered.
        
        Returns:
            True if training should stop, False otherwise
        """
        # Determine current score based on monitoring metric
        if self.monitor == 'val_loss':
            current_score = val_loss
            is_better = lambda current, best: current < best - self.min_delta
        elif self.monitor == 'val_acc':
            current_score = val_acc
            is_better = lambda current, best: current > best + self.min_delta
        else:
            raise ValueError(f"Unknown monitor metric: {self.monitor}")

        # Initialize best score on first epoch
        if self.best_score is None:
            self.best_score = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False

        # Check if current score is better than best
        if is_better(current_score, self.best_score):
            self.best_score = current_score
            self.epochs_without_improvement = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            print(f"    → New best {self.monitor}: {self.best_score:.4f}")
        else:
            self.epochs_without_improvement += 1
            print(f"    → No improvement for {self.epochs_without_improvement} epochs")

        # Check if patience exceeded
        if self.epochs_without_improvement >= self.patience:
            return True

        return False
