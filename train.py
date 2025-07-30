#!/usr/bin/env python
# coding: utf-8

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
from config import Config

def main():
    # Setup project directories
    Config.create_directories()
    
    # Get data directory
    data_dir = Config.get_data_path()
    if data_dir is None:
        print("Please set up your data directory first!")
        return
    
    print(f"Using training data from: {data_dir}")
    
    # Enhanced transforms for medical imaging
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),           
        transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
    ])

    # Load the dataset
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    print(f"Detected brain tumor classes: {full_dataset.classes}")
    print(f"Number of classes: {len(full_dataset.classes)}")
    print(f"Total images: {len(full_dataset)}")

    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply validation transform to validation set
    val_dataset.dataset.transform = val_transform

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Enhanced CNN model for medical imaging
    class BrainTumorCNN(nn.Module):
        def __init__(self, num_classes):
            super(BrainTumorCNN, self).__init__()
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            
            # Unfreeze more layers for medical imaging fine-tuning
            for param in self.base_model.parameters():
                param.requires_grad = False
                
            for param in self.base_model.layer4.parameters():
                param.requires_grad = True
            for param in self.base_model.layer3.parameters():
                param.requires_grad = True
                
            # Enhanced classifier with dropout
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            return self.base_model(x)

    # Initialize model
    num_classes = len(full_dataset.classes)
    model = BrainTumorCNN(num_classes)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    # Training loop
    best_val_accuracy = 0.0

    print("Starting training...")
    for epoch in range(Config.NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        epoch_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        val_loss_avg = val_loss / len(val_loader)
        
        scheduler.step(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")

        print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}]")
        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.2f}%")
        print(f"Best Val Acc: {best_val_accuracy:.2f}%")
        print("-" * 50)

    # Save final model
    torch.save(model.state_dict(), Config.FINAL_MODEL_PATH)
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Models saved to: {Config.MODEL_DIR}")

if __name__ == "__main__":
    main()