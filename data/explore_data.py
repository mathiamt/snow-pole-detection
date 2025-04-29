import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def explore_dataset(data_path):
    """Explore the dataset structure and sample images."""
    data_path = Path(data_path)
    print(f"Exploring dataset at {data_path}")
    
    # Check for LiDAR and RGB datasets
    lidar_path = data_path / "lidar"
    rgb_path = data_path / "rgb"
    
    dataset_paths = []
    if lidar_path.exists():
        dataset_paths.append(("LiDAR", lidar_path))
    if rgb_path.exists():
        dataset_paths.append(("RGB", rgb_path))
    
    for name, path in dataset_paths:
        print(f"\nExploring {name} dataset:")
        _explore_subset(name, path)

def _explore_subset(name, path):
    # Check structure based on README
    if name == "LiDAR":
        modality_dir = path / "combined_color"
    else:  # RGB
        modality_dir = path / "images"
    
    labels_dir = path / "labels"
    
    if not modality_dir.exists():
        print(f"Missing modality directory in {name} dataset")
        # List what's actually there to help debug
        print(f"Contents of {path}:")
        for item in path.iterdir():
            print(f"  - {item.name}")
        return
    
    print(f"Found modality directory: {modality_dir}")
    
    if not labels_dir.exists():
        print(f"Missing labels directory in {name} dataset")
        return
    
    print(f"Found labels directory: {labels_dir}")
    print(f"Contents of labels directory:")
    for item in labels_dir.iterdir():
        print(f"  - {item.name}")
    
    # Check splits for images
    train_dir = modality_dir / "train"
    valid_dir = modality_dir / "valid"
    test_dir = modality_dir / "test"
    
    splits = []
    if train_dir.exists():
        splits.append(("train", train_dir))
    if valid_dir.exists():
        splits.append(("valid", valid_dir))
    if test_dir.exists():
        splits.append(("test", test_dir))
    
    if not splits:
        print(f"No train/valid/test splits found in {name} dataset")
        # List what's actually there
        print(f"Contents of {modality_dir}:")
        for item in modality_dir.iterdir():
            print(f"  - {item.name}")
        return
    
    # Count files in each split
    total_images = 0
    for split_name, split_dir in splits:
        image_files = list(split_dir.glob("*.PNG")) + list(split_dir.glob("*.png"))
        print(f"{split_name}: {len(image_files)} images")
        total_images += len(image_files)
    
    # Check if labels are in subdirectories matching the splits
    labels_splits = {}
    for split_name, _ in splits:
        # Skip test split since it doesn't have labels
        if split_name == "test":
            print(f"Skipping labels check for test split (test images don't have labels)")
            continue
            
        split_label_dir = labels_dir / split_name
        if split_label_dir.exists():
            label_files = list(split_label_dir.glob("*.txt"))
            labels_splits[split_name] = (split_label_dir, len(label_files))
            print(f"Labels for {split_name}: {len(label_files)} files in {split_label_dir}")
    
    if not labels_splits and any(split_name != "test" for split_name, _ in splits):
        # Fall back to checking for labels directly in the labels directory
        # (but only if we have non-test splits)
        label_files = list(labels_dir.glob("*.txt"))
        print(f"Labels (in main directory): {len(label_files)} files")
    
    print(f"Total: {total_images} images across all splits")
    
    # Debug - sample filenames
    if total_images > 0:
        sample_split = splits[0][0]
        sample_dir = splits[0][1]
        sample_images = list(sample_dir.glob("*.PNG")) + list(sample_dir.glob("*.png"))
        if sample_images:
            print(f"Sample image filename format: {sample_images[0].name}")
    
    # Visualize some examples from each split
    for split_name, split_dir in splits:
        # Skip test split visualization since it doesn't have labels
        if split_name == "test":
            print(f"Skipping visualization for test split (no labels)")
            continue
            
        # Check if we have a dedicated labels directory for this split
        if split_name in labels_splits:
            split_label_dir, _ = labels_splits[split_name]
            visualize_samples(name, split_name, split_dir, split_label_dir)
        else:
            # Fall back to using the main labels directory
            visualize_samples(name, split_name, split_dir, labels_dir)

def visualize_samples(name, split_name, images_dir, labels_dir, num_samples=2):
    """Visualize sample images with bounding boxes."""
    # Find image files
    image_files = list(images_dir.glob("*.PNG")) + list(images_dir.glob("*.png"))
    if not image_files:
        print(f"No images found in {split_name} split")
        return
    
    print(f"Found {len(image_files)} images in {split_name} split")
    
    # Debug - show some image filenames
    if image_files:
        print(f"Image examples: {image_files[0].name}")
        if len(image_files) > 1:
            print(f"              {image_files[1].name}")
    
    # Find images with corresponding labels
    valid_pairs = []
    for img_path in image_files:
        # Try to find the matching label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Debug output for label search
        if len(valid_pairs) == 0:  # Just for the first few to avoid clutter
            print(f"Looking for label: {label_path}")
            if label_path.exists():
                print(f"Found matching label!")
            else:
                print(f"Label not found")
        
        if label_path.exists():
            valid_pairs.append((img_path, label_path))
    
    if not valid_pairs:
        print(f"No valid image-label pairs found in {split_name} split")
        return
    
    print(f"Found {len(valid_pairs)} valid image-label pairs in {split_name} split")
    
    # Select random samples
    import random
    random.seed(42)  # For reproducibility
    samples = random.sample(valid_pairs, min(num_samples, len(valid_pairs)))
    
    # Create figure
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    if len(samples) == 1:
        axes = [axes]
    
    for i, (img_path, label_path) in enumerate(samples):
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Load labels
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # YOLO format: class x_center y_center width height
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * w
                    y_center = float(parts[2]) * h
                    width = float(parts[3]) * w
                    height = float(parts[4]) * h
                    
                    # Calculate box coordinates
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    boxes.append((x1, y1, x2, y2))
                    
                    # Draw bounding box
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{name} - {split_name}\n{len(boxes)} poles")
        axes[i].axis('off')
    
    plt.tight_layout()
    os.makedirs("./outputs", exist_ok=True)
    output_path = f"./outputs/{name.lower()}_{split_name}_samples.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    # Update this path to match your environment
    DATA_PATH = "/datasets/tdt4265/ad/open/Poles"  # For Cybele
    
    explore_dataset(DATA_PATH)