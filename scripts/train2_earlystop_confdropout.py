import torch
import torchvision.transforms as T
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig
from PIL import Image
import numpy as np
import os
import json
import time
from collections import Counter

# Step 2: Load and preprocess the dataset
class SHHSDataset(Dataset):
    def __init__(self, dataset_folder, json_folder, transforms=None):
        self.transforms = transforms
        self.img_folder = dataset_folder
        self.json_folder = json_folder
        self.img_files = [f for f in os.listdir(dataset_folder) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_file)
        json_path = os.path.join(self.json_folder, os.path.splitext(img_file)[0] + '.json')

        img = Image.open(img_path).convert("RGB")

        # Load annotations from JSON file
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {json_path}: {e}")
                raise

        # Extract image info from JSON
        image_info = next((img for img in data['images'] if img['file_name'] == img_file), None)
        if image_info is None:
            print(f"Error: No image info found for {img_file}")
            raise ValueError(f"No image info found for {img_file}")

        img_width = image_info['width']
        img_height = image_info['height']

        if self.transforms:
            img = self.transforms(img)

        # Extract annotations for the specific image
        image_id = image_info['id']
        anns = [ann for ann in data['annotations'] if ann['image_id'] == image_id]

        # Ensure anns is a list of dictionaries
        if not isinstance(anns, list):
            print(f"Error: Annotations for {img_file} are not a list.")
            raise TypeError(f"Annotations for {img_file} are not a list.")

        # Prepare target
        try:
            boxes = [ann['bbox'] for ann in anns]
            labels = [ann['category_id'] for ann in anns]
        except KeyError as e:
            print(f"Error: Missing key in annotations for {img_file}: {e}")
            raise

        target = {
            'image_id': torch.tensor([idx]),
            'class_labels': torch.tensor(labels, dtype=torch.long),  # Ensure dtype is torch.long
            'boxes': torch.tensor(boxes, dtype=torch.float32)  # Ensure the boxes are float tensors
        }

        return img, target

# Define transformations
transform = T.Compose([
    T.ToTensor(),
])

# Paths for datasets and annotations
train_images_path = r"D:\DETR_PROJECT\dataset\YUsplit_dataset\train\Images"
train_json_path = r"D:\DETR_PROJECT\dataset\YUsplit_dataset\train\Annotations"
val_images_path = r"D:\DETR_PROJECT\dataset\YUsplit_dataset\val\Images"
val_json_path = r"D:\DETR_PROJECT\dataset\YUsplit_dataset\val\Annotations"
test_images_path = r"D:\DETR_PROJECT\dataset\YUsplit_dataset\test\Images"
test_json_path = r"D:\DETR_PROJECT\dataset\YUsplit_dataset\test\Annotations"

# Initialize datasets and dataloaders
train_dataset = SHHSDataset(train_images_path, train_json_path, transforms=transform)
val_dataset = SHHSDataset(val_images_path, val_json_path, transforms=transform)
test_dataset = SHHSDataset(test_images_path, test_json_path, transforms=transform)

# Function to calculate and print class distribution
def print_class_distribution(dataset):
    labels = []
    for _, target in dataset:
        if target is not None:
            labels.extend(target['class_labels'].tolist())
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    print("Class Distribution:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} events, {count / total_samples:.2%} of total")

# Print class distribution
print_class_distribution(train_dataset)

# Function to calculate class weights based on the number of events
def calculate_class_weights(dataset):
    labels = []
    for _, target in dataset:
        if target is not None:
            labels.extend(target['class_labels'].tolist())
    
    # Count occurrences of each class
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    # Calculate weights inversely proportional to class frequency
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # Normalize class weights (if needed)
    max_weight = max(class_weights.values())
    class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}
    
    # Convert to tensor
    class_weights_list = [class_weights.get(i, 0) for i in range(len(class_weights))]
    return torch.tensor(class_weights_list, dtype=torch.float)

# Calculate class weights based on events
class_weights_tensor = calculate_class_weights(train_dataset)
print("Class Weights:", class_weights_tensor)

# Define your loss function with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Initialize dataloaders
def collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

config = DetrConfig.from_pretrained('facebook/detr-resnet-50')
config.dropout = 0.5  
config.classifier_dropout = 0.2

config.num_labels = 2
config.id2label = {0: "normal", 1: "apnea"}
config.label2id = {"normal": 0, "apnea": 1}

# Step 3: Initialize the DETR model
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', config=config, ignore_mismatched_sizes=True)
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', do_rescale=False)  # Avoid rescaling images

# Check if GPU is available and move model to GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

def validate(model, dataloader):
    model.eval()
    validation_losses = []
    with torch.no_grad():  # No need to compute gradients during validation
        for batch in dataloader:
            images, targets = batch
            images = [image.to(device) for image in images]

            # Convert targets to the required format
            processed_targets = []

            for target in targets:
                processed_target = {
                    'image_id': target['image_id'].to(device),
                    'class_labels': target['class_labels'].to(device),
                    'boxes': target['boxes'].to(device)
                }
                processed_targets.append(processed_target)

            if len(processed_targets) == 0:
                continue

            # Preprocess images
            encoding = processor(images=images, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}  # Move encoding to GPU

            # Forward pass
            outputs = model(**encoding, labels=processed_targets)

            if hasattr(outputs, 'loss') and outputs.loss is not None:
                validation_losses.append(outputs.loss.item())

    avg_validation_loss = np.mean(validation_losses) if validation_losses else float('inf')
    return avg_validation_loss

# Step 4: Define the training loop without gradient accumulation
def train(model, train_dataloader, val_dataloader, epochs=100, patience=5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    training_losses = []
    validation_losses = []
    best_validation_loss = float('inf')  # Initialize best validation loss
    patience_counter = 0  # Counter for early stopping

    for epoch in range(epochs):
        epoch_start_time = time.time()
        num_batches = len(train_dataloader)
        print(f"Epoch {epoch} started with {num_batches} batches.")
        
        # Initialize event counter
        total_events = 0

        optimizer.zero_grad()
        epoch_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            batch_start_time = time.time()
            if batch is None:
                print(f"Warning: Skipping empty batch {batch_idx + 1}")
                continue

            images, targets = batch
            images = [image.to(device) for image in images]

            processed_targets = []
            for target in targets:
                if target is None:
                    print(f"Warning: Skipping empty target in batch {batch_idx + 1}")
                    continue
                processed_target = {
                    'image_id': target['image_id'].to(device),
                    'class_labels': target['class_labels'].to(device),
                    'boxes': target['boxes'].to(device)
                }
                processed_targets.append(processed_target)

                # Update event counter
                total_events += len(target['class_labels'])

            if len(processed_targets) == 0:
                print(f"Warning: No valid targets in batch {batch_idx + 1}")
                continue

            # Preprocess images
            encoding = processor(images=images, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Forward pass
            outputs = model(**encoding, labels=processed_targets)
            
            # Compute loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time

            # scheduler.step()

            print(f"Epoch {epoch}, Batch {batch_idx + 1}/{num_batches}, "
                  f"Total Loss: {loss.item():.4f}, "
                  f"Time: {batch_duration:.2f} seconds")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_epoch_loss = epoch_loss / num_batches
        training_losses.append(avg_epoch_loss)

        # Validate after each epoch
        avg_validation_loss = validate(model, val_dataloader)
        validation_losses.append(avg_validation_loss)

        print(f"Epoch {epoch} completed in {epoch_duration:.2f} seconds")
        print(f"Epoch {epoch} Loss: {avg_epoch_loss:.4f}")
        print(f"Epoch {epoch} Validation Loss: {avg_validation_loss:.4f}")
        print(f"Total events processed in epoch {epoch}: {total_events}")

        # Save the best model
        if avg_validation_loss < best_validation_loss:
            best_validation_loss = avg_validation_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model2.pth')
            print(f"New best model saved with validation loss: {best_validation_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

        # Check if patience has been exceeded
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save training and validation losses to files
    np.save('model2_training_losses.npy', np.array(training_losses))
    np.save('model2_validation_losses.npy', np.array(validation_losses))

# Step 5: Train the model
train(model, train_dataloader, val_dataloader)

# Step 6: Save the last model (move to CPU first)
model.to('cpu')
torch.save(model.state_dict(), 'last_model2.pth')