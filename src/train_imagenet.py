import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from ffcv.loader import Loader,OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, Squeeze
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from tqdm import tqdm
import numpy as np
from src.dataset import get_imagenet_loaders
import random


# Hyperparameters
batch_size = 32
num_epochs = 100
learning_rate = 0.01
lr_momenutum = 0.9
criterion = nn.CrossEntropyLoss(reduction='sum')
device_type='cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
# Gradient Accumulation Hyperparameters
accumulation_steps = 4  # Accumulate gradients over 4 mini-batches

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

def calculate_accuracy(model, data_loader, desc="Accuracy"):
    """
    Calculate accuracy and loss for a given model and data loader.
    
    Args:
        model: The neural network model
        data_loader: DataLoader containing the data
        device: Device to run the computation on
        desc: Description for the progress bar
    
    Returns:
        tuple: (accuracy, loss, correct_count, total_count)
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    
    pbar = tqdm(data_loader, desc=desc, 
                total=len(data_loader),
                bar_format='{l_bar}{bar:30}{r_bar}')
    
    with torch.no_grad():
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            running_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            current_correct = pred.eq(target.view_as(pred)).sum().item()
            correct += current_correct
            total += len(data)
            
            # Update progress bar
            current_acc = 100 * correct / total
            current_loss = running_loss / total
            pbar.set_description(
                f'{desc}: {current_acc:.2f}% | Loss: {current_loss:.4f}'
            )
    
    return current_acc, current_loss, correct, total


def print_training_conclusion(history, model):
    """Print final training conclusions and best model details"""
    print("\n" + "="*50)
    print("TRAINING COMPLETED - FINAL RESULTS")
    print("="*50)
        
    best_info = history['best_model_info']
    
    print(f"\nBest Model achieved at Epoch {best_info['epoch']}:")
    print(f"├── Test Accuracy: {best_info['test_acc']:.2f}%")
    print(f"└── Test Loss: {best_info['test_loss']:.4f}")
    
    print("\nModel saved as: 'best_model.pth'")
    print("="*50)

def train(model, train_loader, test_loader, optimizer, scheduler):
    """
    Train the model for multiple epochs
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer instance
        scheduler: Learning rate scheduler
    
    Returns:
        dict: Training history containing accuracies and losses
    """
    history = {
        'test_acc': [],
        'test_loss': [],
        'learning_rates': [],
        'best_model_info': {
            'epoch': 0,
            'test_acc': 0,
            'test_loss': 0
        }
    }
    
    best_acc = 0.0
    total_steps = num_epochs * len(train_loader)
    current_step = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("=" * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        desc =f"Realtime Epoch Training"

        pbar = tqdm(train_loader, desc=desc, 
                total=len(train_loader),
                bar_format='{l_bar}{bar:30}{r_bar}')
        
        optimizer.zero_grad()
        # Training loop
        for i, (data, target) in enumerate(pbar):
            current_step += 1
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            
            loss.backward()
            # Perform gradient accumulation
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
               optimizer.step()
               # Reset gradients
               optimizer.zero_grad()
               if scheduler is not None:
                  scheduler.step()

            running_loss += loss.item()
            # Calculate batch accuracy
            pred = output.argmax(dim=1, keepdim=True)
            total += target.size(0)
            correct += pred.eq(target.view_as(pred)).sum().item()     
            epoch_acc = 100. * correct / total
            
            pbar.set_description(
                f'{desc} Accuracy: {epoch_acc:.2f}% | Loss: {running_loss / (total + 1):.4f}'
            )
                   
        # Calculate test accuracy
        print("\nCalculating test accuracy...")
        test_acc, test_loss, _, _ = calculate_accuracy(
            model, test_loader, desc=f"Epoch {epoch} Testing"
        )
        
        # Save metrics
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Testing  - Accuracy: {test_acc:.2f}%, Loss: {test_loss:.4f}")
        
        # Save best model and its details
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best accuracy! Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')
            
            # Store best model information
            history['best_model_info'] = {
                'epoch': epoch,
                'test_acc': test_acc,
                'test_loss': test_loss
            }
    
    # Print conclusion after training
    print_training_conclusion(history, model)
    return history


def main():
    print(f"Using device: {device}")
    # Set seeds first
    set_seed(42)
    # Get data loaders
    train_loader, test_loader = get_imagenet_loaders(batch_size=batch_size)

    # Model, loss, optimizer, and scheduler
    model = resnet18(weights=None).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=lr_momenutum, weight_decay=1e-4)
   
    # Scheduler for the entire training duration
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        div_factor=10,
        final_div_factor=1000, # final LR = initial_lr / final_div_factor
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Train the model
    history = train(model, train_loader, test_loader, optimizer, scheduler)
    


if __name__ == "__main__":
    main()