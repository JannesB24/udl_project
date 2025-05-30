import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import pickle
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import DataLoaderFFSet
from models.res_block import ResBlock


def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

def train_l2_model(weight_decay=0.01):
    print("="*60)
    print(f"TRAINING L2 REGULARIZED RESNET (weight_decay={weight_decay})")
    print("="*60)
    
    device = torch.device('cpu')
    
    # Create model exactly the unregularized
    model = ResBlock(5)
    model.apply(weights_init)
    
    # MAIN DIFFERENCE: Adding weight_decay parameter to optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001,
        weight_decay=weight_decay  # Explicit L2 regularization
    )
    
    # Same parameters as original
    num_epochs = 10
    
    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)
    train_accs = np.zeros(num_epochs)
    val_accs = np.zeros(num_epochs)
    
    print("Training L2 Regularized ResNet...")
    
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

        # Print results
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accs[epoch]:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accs[epoch]:.4f} | '
              f'Duration: {duration}')

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_dir = os.path.join(script_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), os.path.join(artifacts_dir, f"l2_model_wd_{weight_decay}.pth"))

    # Store results
    l2_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'weight_decay': weight_decay,
        'model_name': f'L2 Regularized (wd={weight_decay})'
    }

    with open(os.path.join(artifacts_dir, 'l2_results.pkl'), 'wb') as f:
        pickle.dump(l2_results, f)
    
    # Print summary for this configuration
    overfitting_gap = train_accs[-1] - val_accs[-1]
    print(f"\nL2 Regularized model training completed!")
    print(f"Final overfitting gap: {overfitting_gap:.4f}")
    print("Results saved to ../artifacts/l2_results.pkl")
    
    return l2_results

def main():
    print("L2 REGULARIZATION TRAINING")
    # Weight decay = 0.001, this can be changed to different values that yield different results 
    print("Using weight_decay=0.001")
    print("="*60)
    
    # Train with medium L2 regularization
    l2_results = train_l2_model(weight_decay=0.01)
    
    print("\nL2 REGULARIZATION TRAINING COMPLETED!")
    print("Generated files:")
    print("  - ../artifacts/l2_model_wd_0.01.pth")
    print("  - ../artifacts/l2_results.pkl")

if __name__ == "__main__":
    main()