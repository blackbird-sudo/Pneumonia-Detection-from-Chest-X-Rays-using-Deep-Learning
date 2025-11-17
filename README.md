# Pneumonia-Detection-from-Chest-X-Rays-using-Deep-Learning
Build an AI system that can automatically detect pneumonia from chest X-ray images with high accuracy, assisting radiologists in medical diagnosis

import torch

import torch.nn as nn

import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, datasets

import matplotlib.pyplot as plt

import numpy as np

import os

from sklearn.metrics import classification_report, confusion_matrix

import seaborn as sns

from tqdm import tqdm

from model import create_model

print("Medical Image Classification with PyTorch - Chest X-ray Analysis")
print("=" * 60)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations

train_transform = transforms.Compose([

    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((150, 150)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_transform = transforms.Compose([
  
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_data(data_dir, batch_size=32):

    """Load medical image dataset"""
    print("Loading medical images...")
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, train_dataset.classes

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15):

    """Train the model"""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct / total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with accuracy: {val_acc:.2f}%')
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)
    
    return train_losses, val_losses, train_accuracies, val_accuracies, all_preds, all_labels

def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, class_names, all_preds, all_labels):

    """Plot training results and metrics"""
    # Plot loss and accuracy
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy plot
    axes[0, 1].plot(train_accuracies, label='Training Accuracy')
    axes[0, 1].plot(val_accuracies, label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                  target_names=class_names, output_dict=True)
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.9, 'Classification Report:', fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.7, f"Accuracy: {report['accuracy']:.3f}", fontsize=10)
    axes[1, 1].text(0.1, 0.6, f"Precision: {report['weighted avg']['precision']:.3f}", fontsize=10)
    axes[1, 1].text(0.1, 0.5, f"Recall: {report['weighted avg']['recall']:.3f}", fontsize=10)
    axes[1, 1].text(0.1, 0.4, f"F1-Score: {report['weighted avg']['f1-score']:.3f}", fontsize=10)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():

    # Configuration
    DATA_DIR = "data"
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found!")
        print("Please create the following structure:")
        print("data/")
        print("├── train/")
        print("│   ├── NORMAL/")
        print("│   └── PNEUMONIA/")
        print("└── val/")
        print("    ├── NORMAL/")
        print("    └── PNEUMONIA/")
        return
    
    # Load data
    train_loader, val_loader, class_names = load_data(DATA_DIR, BATCH_SIZE)
    
    print(f"\nClasses: {class_names}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(device)
    print("\nModel created:")
    print(model)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    results = train_model(model, train_loader, val_loader, criterion, 
                         optimizer, scheduler, NUM_EPOCHS)
    
    train_losses, val_losses, train_accuracies, val_accuracies, all_preds, all_labels = results
    
    # Plot results
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, 
                class_names, all_preds, all_labels)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'input_size': (150, 150)
    }, 'medical_model.pth')
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {max(val_accuracies):.2f}%")
    print("Model saved as 'medical_model.pth'")

if __name__ == "__main__":
    main()
