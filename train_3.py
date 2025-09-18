import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision import transforms
from tqdm import tqdm
import os
from utils import check_set_gpu
from datareader import MakananIndo
from utils import check_set_gpu
from datareader import MakananIndo

def create_label_encoder(dataset):
    """Create a mapping from string labels to numeric indices"""
    all_labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        all_labels.append(label)
    
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    return label_to_idx, idx_to_label, unique_labels

def train_one_epoch(model, dataloader, criterion, optimizer, device, label_to_idx):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (inputs, labels_tuple, _) in enumerate(pbar):  # MakananIndo returns (image, label, filepath)
        inputs = inputs.to(device)
        
        # Convert string labels to numeric indices
        if isinstance(labels_tuple, (tuple, list)):
            if isinstance(labels_tuple[0], str):
                label_indices = [label_to_idx[label] for label in labels_tuple]
            else:
                label_indices = labels_tuple
            targets = torch.tensor(label_indices, dtype=torch.long).to(device)
        else:
            targets = torch.tensor(labels_tuple, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': total_loss/(batch_idx+1),
            'acc': 100.*correct/total
        })
    
    return total_loss/len(dataloader), 100.*correct/total

def validate(model, dataloader, criterion, device, label_to_idx):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (inputs, labels_tuple, _) in enumerate(pbar):  # MakananIndo returns (image, label, filepath)
            inputs = inputs.to(device)
            
            # Convert string labels to numeric indices
            if isinstance(labels_tuple, (tuple, list)):
                if isinstance(labels_tuple[0], str):
                    label_indices = [label_to_idx[label] for label in labels_tuple]
                else:
                    label_indices = labels_tuple
                targets = torch.tensor(label_indices, dtype=torch.long).to(device)
            else:
                targets = torch.tensor(labels_tuple, dtype=torch.long).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
    
    return total_loss/len(dataloader), 100.*correct/total

def main():
    # Set device
    device = check_set_gpu()
    
    # Hyperparameters
    num_epochs = 30
    batch_size = 32
    learning_rate = 0.001
    num_classes = 10  # adjust based on your dataset
    
    # Create datasets using MakananIndo class
    print("Loading datasets...")
    train_dataset = MakananIndo(
        data_dir='IF25-4041-dataset/train',
        img_size=(300, 300),  # EfficientNet-B3 input size
        split='train'
    )
    
    val_dataset = MakananIndo(
        data_dir='IF25-4041-dataset/train',  # Use same dir as it handles splitting internally
        img_size=(300, 300),
        split='val'
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create label encoder
    print("Creating label encoder...")
    label_to_idx, idx_to_label, unique_labels = create_label_encoder(train_dataset)
    num_classes = len(unique_labels)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {unique_labels}")
    print(f"Label to index mapping: {label_to_idx}")
    
    # Adjust num_workers based on CPU count
    cpu_count = os.cpu_count()
    nworkers = cpu_count - 4 if cpu_count > 4 else 2
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=nworkers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=nworkers,
        pin_memory=True
    )
    
    # Load model with pretrained weights
    print(f"\nInitializing EfficientNet-B3 model for {num_classes} classes...")
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    print("Froze all backbone parameters")
    
    # Replace classifier with new untrained layers
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes)
    )
    
    print(f"Replaced classifier with new layers for {num_classes} classes")
    
    # Only train the classifier parameters
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=learning_rate)
    
    model = model.to(device)
    
    # Loss and scheduler
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # DataLoaders are already created above
    
    # Training loop
    best_val_acc = 0
    for epoch in range(num_epochs):
        print(f'\nEpoch: {epoch+1}/{num_epochs}')
        
        # Training
        model.train()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, label_to_idx
        )
        
        # Validation
        model.eval()
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, label_to_idx
        )
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, 'best_model.pth')
            
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

if __name__ == '__main__':
    main()