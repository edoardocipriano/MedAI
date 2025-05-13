import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import seaborn as sns
from data_utils import load_data
from model import create_model
import os

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler=None, num_epochs=10, threshold=0.35):
    """Train the model and evaluate after each epoch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_recall = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > threshold).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix(loss=loss.item(), acc=correct/total)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Evaluation phase
        test_loss, test_acc, test_recall = evaluate_model(model, test_loader, criterion, device, threshold)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Update learning rate if scheduler is provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_recall)  # Pass the recall metric to the scheduler
            else:
                scheduler.step()
        
        # Save best model based on recall
        if test_recall > best_recall:
            best_recall = test_recall
            best_model_state = model.state_dict().copy()
            
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Recall: {test_recall:.4f}")
    
    # Load best model state if it was saved
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, train_losses, test_losses, train_accuracies, test_accuracies

def evaluate_model(model, data_loader, criterion, device, threshold=0.65):
    """Evaluate the model on the given data loader."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > threshold).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    total_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score([l[0] for l in all_labels], [p[0] for p in all_preds], zero_division=0)
    
    return total_loss, accuracy, recall

def get_metrics(model, data_loader, threshold=0.65):
    """Calculate various metrics for model evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            predicted = (probs > threshold).float()
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to flat lists
    all_preds = [p[0] for p in all_preds]
    all_probs = [p[0] for p in all_probs]
    all_labels = [l[0] for l in all_labels]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }

def plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies):
    """Plot training and validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(labels, predictions):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_roc_curve(labels, probabilities):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

def main():
    # Load data
    train_loader, test_loader = load_data()
    
    # Determine input size from the first batch
    inputs, _ = next(iter(train_loader))
    input_size = inputs.shape[1]
    
    # Calculate positive weight for loss function (inverse of class frequency)
    # The dataset has 91.5% class 0 and 8.5% class 1, so weight is 91.5/8.5 ≈ 10.76
    # Reduce the weight slightly to decrease false positives
    pos_weight = torch.tensor([6.0])  # Further reduced from 8.0 to 6.0 to decrease false positives
    
    # Create model
    model = create_model(input_size)
    
    # Define loss function with class weighting to improve recall but avoid too many false positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with better parameters
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Define prediction threshold (higher threshold decreases false positives)
    prediction_threshold = 0.65  # Higher threshold to reduce false positives
    
    # Train model with more epochs
    print("Starting training...")
    model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        scheduler=scheduler, num_epochs=30, threshold=prediction_threshold)
    
    # Plot training history
    plot_training_history(train_losses, test_losses, train_accuracies, test_accuracies)
    
    # Evaluate model and get metrics with the threshold
    metrics = get_metrics(model, test_loader, threshold=prediction_threshold)
    
    # Print metrics
    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['labels'], metrics['predictions'])
    
    # Plot ROC curve
    plot_roc_curve(metrics['labels'], metrics['probabilities'])
    
    # Create directory for saved models if it doesn't exist
    os.makedirs('model/saved', exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), 'model_training/diabetes_model.pth')
    print("Model saved to model_training/diabetes_model.pth")

if __name__ == "__main__":
    main()