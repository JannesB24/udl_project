import pickle
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from udl_project import config
from udl_project.data_loader_flowers import DataLoaderFlowers
from udl_project.models.res_net import ResNet
from udl_project.training.abstract_trainer import Trainer
from udl_project.utils.weights import weights_init

def loadTfL_ResNetModel(prev_numclasses, new_numclasses, freezeOriginal):
    model = ResNet(num_classes=prev_numclasses)
    model.load_state_dict(torch.load(config.ARTIFACTS_DIR / "pretraining_model.pth"))
    if freezeOriginal:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier=nn.Linear(256 * 8 * 8, new_numclasses)
    return model


class ResNetModelTFLTrainer(Trainer):
    prev_numclasses= 10
    ft_numclasses= 5
    dataset= ""
    finetune_bool=False
    freeze = False
    def __init__(self, *, epochs: int, learning_rate: float):
        super().__init__()
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self,numclasses,dataset):
        print("=" * 60)
        print("PRETRAINING ORIGINAL RESNET MODEL")
        print("=" * 60)
        self.prev_numclasses=numclasses
        self.dataset=dataset
        self.finetune_bool=False
        print(self.dataset)
        train_accs, val_accs = self._train()

        print("\pretraining model training completed!")
        print(f"Final overfitting gap: {train_accs[-1] - val_accs[-1]:.4f}")
        print(f"Results saved to {config.ARTIFACTS_DIR / 'pretrain_results.pkl'}")


    def finetune_model(self,numclasses, dataset,freeze):
        print("=" * 60)
        print("FINETUNING RESNET MODEL")
        print("=" * 60)
        self.ft_numclasses=numclasses
        self.dataset=dataset
        self.finetune_bool=True
        self.freeze=freeze
        train_accs, val_accs = self._train()

        print("\nFinetuning model training completed!")
        print(f"Final overfitting gap: {train_accs[-1] - val_accs[-1]:.4f}")
        print(f"Results saved to {config.ARTIFACTS_DIR / 'finetune_results.pkl'}")

    def _train(self) -> tuple[np.ndarray, np.ndarray]:
        device = torch.device("cpu")

        # call with standard parameters

        data_loader = DataLoaderFlowers.create_dataloader_Pathchoice( kagglePath=self.dataset)
        # create model and initialize parameters
        if self.finetune_bool:
            model = loadTfL_ResNetModel(self.prev_numclasses,self.ft_numclasses, self.freeze)
        else:
            model = ResNet(num_classes=self.prev_numclasses)
            
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
        save_str_prefix="pretraining"
        if self.finetune_bool:
            save_str_prefix="finetune"

        save_str= save_str_prefix + "_model.pth"
        for epoch in range(self.epochs):
            model.train()
            t0 = datetime.now()

            train_loss = []
            val_loss = []
            n_correct_train = 0
            n_total_train = 0

            for images, labels in data_loader.get_train_dataloader():
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
                for images, labels in data_loader.get_test_dataloader():
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
                f"Epoch [{epoch + 1}/{self.epochs}] - "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accs[epoch]:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accs[epoch]:.4f} | "
                f"Duration: {duration}"
            )

        # Save the model
#        torch.save(model.state_dict(), config.ARTIFACTS_DIR / "flower_classification_model.pth")
       
       
       
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
        with open(config.ARTIFACTS_DIR / (save_str_prefix + "_results.pkl"), "wb") as f:
            pickle.dump(original_results, f)

        return train_accs, val_accs




if __name__ == "__main__":
    # example usage
    trainer = ResNetModelTFLTrainer(learning_rate=0.001, epochs=25)
    trainer.train()
