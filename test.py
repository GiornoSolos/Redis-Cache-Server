#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        # Match the enhanced model architecture from training
        self.base_model = models.resnet50(weights=None)  # No pretrained weights needed for inference
        
        # Enhanced classifier matching the training model
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

def load_brain_tumor_model(model_path, num_classes, device):
    """Load the trained brain tumor detection model"""
    model = BrainTumorCNN(num_classes)
    
    # Load model weights with proper device mapping
    if device.type == 'cuda':
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    return model

def run_brain_tumor_inference(model, test_loader, class_names, device):
    """Run inference with detailed medical metrics"""
    model.eval()
    results = []
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (images, image_paths) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)  # Faster GPU transfer
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            # Store results for each image
            for i in range(len(predicted)):
                pred_class = class_names[predicted[i].item()]
                confidence = probabilities[i][predicted[i].item()].item()
                results.append((image_paths[i], pred_class, confidence))
                all_predictions.append(predicted[i].item())
                all_probabilities.append(probabilities[i].cpu().numpy())
            
            # Calculate accuracy using folder names as ground truth
            labels = []
            for path in image_paths:
                folder_name = os.path.basename(os.path.dirname(path))
                if folder_name in class_names:
                    labels.append(class_names.index(folder_name))
                else:
                    # Handle unknown folder names
                    labels.append(0)  # Default to first class
                    
            labels = torch.tensor(labels).to(device, non_blocking=True)
            all_labels.extend(labels.cpu().numpy())
            
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Progress indicator for longer inference
            if batch_idx % 10 == 0 and batch_idx > 0:
                print(f"Processed {batch_idx * len(images)} images...", end='\r')

    accuracy = 100 * correct / total
    
    # Generate detailed classification report
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return results, accuracy, report, cm, all_labels, all_predictions

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and save confusion matrix"""
    # Use config path if available, otherwise use default
    if save_path is None:
        try:
            from config import Config
            save_path = Config.CONFUSION_MATRIX_PATH
        except ImportError:
            save_path = 'confusion_matrix.png'
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Brain Tumor Classification Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.show()

def print_medical_metrics(report, class_names):
    """Print detailed medical classification metrics"""
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION METRICS")
    print("="*60)
    
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"\n{class_name.upper()}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall:    {metrics['recall']:.3f}")
            print(f"  F1-Score:  {metrics['f1-score']:.3f}")
            print(f"  Support:   {metrics['support']}")
    
    print(f"\nOVERALL METRICS:")
    print(f"  Macro Avg F1:    {report['macro avg']['f1-score']:.3f}")
    print(f"  Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")
    print(f"  Overall Accuracy: {report['accuracy']:.3f}")

# Custom Dataset class for brain tumor images
class BrainTumorImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(BrainTumorImageFolder, self).__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple[0], path

def main():
    # Import config for portable paths
    try:
        from config import Config
        Config.create_directories()
        
        # Use config paths
        model_path = Config.BEST_MODEL_PATH
        test_data_dir = Config.TEST_DATA_DIR
        
        print(f"Looking for test data at: {test_data_dir}")
        
        # Check if test data exists
        if not os.path.exists(test_data_dir):
            print(f"Test data directory not found: {test_data_dir}")
            print("Please ensure your dataset is properly set up.")
            print("Expected structure:")
            print("  data/")
            print("  └── Testing/")
            print("      ├── glioma/")
            print("      ├── meningioma/")
            print("      ├── notumor/")
            print("      └── pituitary/")
            return
            
    except ImportError:
        # Fallback to relative paths if config.py doesn't exist
        print("Config file not found, using relative paths...")
        model_path = "best_brain_tumor_model.pth"
        test_data_dir = "./data/Testing"
        
        if not os.path.exists(test_data_dir):
            print(f"Test data directory not found: {test_data_dir}")
            print("Please create a 'data' folder with your dataset or update the path.")
            return

    # Medical imaging optimized transforms
    try:
        from config import Config
        transform = transforms.Compose([
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(Config.IMAGENET_MEAN, Config.IMAGENET_STD)
        ])
    except ImportError:
        # Fallback transforms if config.py doesn't exist
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # Load test dataset with optimized DataLoader for GPU
    test_dataset = BrainTumorImageFolder(root=test_data_dir, transform=transform)
    
    # Optimize batch size and workers based on device
    if torch.cuda.is_available():
        batch_size = 32  # Larger batch for GPU
        num_workers = 4  # More workers for GPU
    else:
        batch_size = 16  # Smaller batch for CPU
        num_workers = 2  # Fewer workers for CPU
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=torch.cuda.is_available())
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    print(f"Detected brain tumor classes: {class_names}")
    print(f"Number of test images: {len(test_dataset)}")
    
    # Setup device with better GPU detection
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
        print("Note: CUDA not available, using CPU (this will be slower)")
    
    # Load model
    try:
        model = load_brain_tumor_model(model_path, num_classes, device)
        model.to(device)
        
        if torch.cuda.is_available():
            print(f"Successfully loaded model from {model_path} (GPU accelerated)")
        else:
            print(f"Successfully loaded model from {model_path} (CPU only)")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model file exists and matches the number of classes")
        print(f"Looking for model at: {model_path}")
        return

    # Run inference
    print("\nRunning brain tumor classification inference...")
    results, accuracy, report, cm, true_labels, predictions = run_brain_tumor_inference(
        model, test_loader, class_names, device
    )

    # Print results
    print(f"\nCLASSIFICATION RESULTS")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Print detailed medical metrics
    print_medical_metrics(report, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)
    
    # Print some individual predictions with confidence
    print(f"\nSAMPLE PREDICTIONS (showing first 10):")
    print("-" * 80)
    for i, (image_path, pred_class, confidence) in enumerate(results[:10]):
        filename = os.path.basename(image_path)
        true_class = os.path.basename(os.path.dirname(image_path))
        status = "CORRECT" if pred_class == true_class else "INCORRECT"
        print(f"{filename:30} | True: {true_class:12} | Pred: {pred_class:12} | Conf: {confidence:.3f} | {status}")
    
    # Performance interpretation
    print(f"\nPERFORMANCE ASSESSMENT:")
    if accuracy >= 95:
        print("EXCELLENT: Model shows high accuracy suitable for medical assistance")
    elif accuracy >= 90:
        print("GOOD: Model shows strong performance, suitable for screening")
    elif accuracy >= 85:
        print("MODERATE: Model needs improvement before clinical application")
    else:
        print("POOR: Model requires significant improvement for medical use")
    
    print(f"\nDetailed results saved and confusion matrix plotted.")
    print("This model could assist in medical screening applications.")

if __name__ == "__main__":
    main()