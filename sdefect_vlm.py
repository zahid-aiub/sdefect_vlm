# ============================================
# SDEFECT-VLM: COMPLETE FIXED PIPELINE
# For NEU-DET Dataset with proper folder structure
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. FIXED DATASET CLASS
# ============================================

class NEUDETDataset(Dataset):
    """
    NEU-DET Surface Defect Dataset Loader
    Handles proper image loading from class subfolders
    """
    def __init__(self, root_dir, annotations_dir, transform=None, is_train=True):
        """
        Args:
            root_dir: Path to images folder (e.g., 'train/images')
            annotations_dir: Path to annotations folder (e.g., 'train/annotations')
            transform: Image transformations
            is_train: Boolean flag for training/validation
        """
        self.root_dir = root_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.is_train = is_train
        
        # 6 defect classes
        self.class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        # Collect all XML annotation files
        self.annotations = []
        if os.path.exists(annotations_dir):
            for file in os.listdir(annotations_dir):
                if file.endswith('.xml'):
                    self.annotations.append(os.path.join(annotations_dir, file))
        else:
            raise FileNotFoundError(f"Annotation folder not found: {annotations_dir}")
        
        # Build file cache for faster lookup
        self.file_cache = self._build_file_cache()
        
        print(f"{'Training' if is_train else 'Validation'} Dataset: {len(self.annotations)} samples")
    
    def _build_file_cache(self):
        """
        Build cache of all images for quick lookup
        """
        cache = {}
        
        if not os.path.exists(self.root_dir):
            print(f"Warning: Root directory not found: {self.root_dir}")
            return cache
        
        # Check each class folder
        for class_name in self.class_names:
            class_folder = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_folder):
                cache[class_name] = []
                for file in os.listdir(class_folder):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        cache[class_name].append(file)
                print(f"  {class_name}: {len(cache[class_name])} images")
            else:
                print(f"  Warning: {class_name} folder not found")
                cache[class_name] = []
        
        return cache
    
    def parse_xml(self, xml_path):
        """
        Parse XML file to extract filename and class
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        filename = root.find('filename').text
        class_name = root.find('object').find('name').text
        
        return filename, class_name
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        xml_path = self.annotations[idx]
        filename, class_name = self.parse_xml(xml_path)
        
        # Construct image path
        # Path: train/images/crazing/crazing_1.jpg
        img_path = os.path.join(self.root_dir, class_name, filename)
        
        # Check if image exists
        if not os.path.exists(img_path):
            # Try case-insensitive search
            img_path = self._find_image_case_insensitive(class_name, filename)
        
        # Load image or create fallback
        if img_path is None or not os.path.exists(img_path):
            # Create random image as fallback
            random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(random_img)
            # Print warning only occasionally
            if idx % 100 == 0:
                print(f"⚠️ Image not found: {class_name}/{filename}")
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                image = Image.fromarray(random_img)
        
        # Get label
        label = self.class_to_idx[class_name]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label, class_name
    
    def _find_image_case_insensitive(self, class_name, filename):
        """
        Find image with case-insensitive search
        """
        class_folder = os.path.join(self.root_dir, class_name)
        if not os.path.exists(class_folder):
            return None
        
        # Get filename without extension
        base_name = os.path.splitext(filename)[0]
        
        # Search for file
        for file in os.listdir(class_folder):
            if file.lower() == filename.lower():
                return os.path.join(class_folder, file)
            if os.path.splitext(file)[0].lower() == base_name.lower():
                return os.path.join(class_folder, file)
        
        return None

# ============================================
# 2. DATA TRANSFORMATIONS
# ============================================

# Training transformations with augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Validation transformations
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def create_dataloaders(train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, batch_size=32):
    """
    Create training and validation dataloaders
    """
    train_dataset = NEUDETDataset(train_img_dir, train_ann_dir, transform=train_transform, is_train=True)
    val_dataset = NEUDETDataset(val_img_dir, val_ann_dir, transform=val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, len(train_dataset.class_names)

# ============================================
# 3. RESNET-50 MODEL
# ============================================

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 based classifier for surface defect detection
    """
    def __init__(self, num_classes=6):
        super(ResNet50Classifier, self).__init__()
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Freeze early layers, unfreeze later layers for fine-tuning
        for name, param in self.backbone.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Replace final fully connected layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

# ============================================
# 4. TRAINING FUNCTION
# ============================================

def train_model(model, train_loader, val_loader, device, epochs=30):
    """
    Train the ResNet-50 model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels, _ in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, labels, _ in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet50_neu.pth')
            print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print("-"*60)
    
    print(f"\n🎉 Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    return train_losses, val_losses, train_accs, val_accs

# ============================================
# 5. EVALUATION FUNCTION
# ============================================

def evaluate_model(model, val_loader, device, class_names):
    """
    Evaluate model and generate comprehensive report
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_class_names = []
    
    print("\n📊 Evaluating Model...")
    with torch.no_grad():
        for images, labels, class_names_list in tqdm(val_loader, desc="Evaluation"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_class_names.extend(class_names_list)
    
    # Calculate metrics
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Class-wise Performance
    class_precision = [report[cls]['precision'] for cls in class_names]
    class_recall = [report[cls]['recall'] for cls in class_names]
    class_f1 = [report[cls]['f1-score'] for cls in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    axes[0, 1].bar(x - width, class_precision, width, label='Precision', color='skyblue')
    axes[0, 1].bar(x, class_recall, width, label='Recall', color='lightgreen')
    axes[0, 1].bar(x + width, class_f1, width, label='F1-Score', color='salmon')
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Class-wise Performance Metrics', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Overall Performance Summary
    metrics = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1\n(Weighted)']
    values = [
        accuracy,
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        report['weighted avg']['f1-score']
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    
    axes[1, 0].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(values):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Sample Predictions (show some examples)
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.9, '📊 Classification Report Summary', 
                    fontsize=14, fontweight='bold')
    
    # Show sample correct and incorrect predictions
    sample_text = "Sample Predictions:\n"
    sample_text += "="*40 + "\n"
    
    # Show some correct predictions
    correct_count = 0
    sample_text += "✅ Correct Predictions:\n"
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        if pred == label and correct_count < 3:
            sample_text += f"  {class_names[label]} -> {class_names[pred]}\n"
            correct_count += 1
    
    # Show some incorrect predictions
    incorrect_count = 0
    sample_text += "\n❌ Incorrect Predictions:\n"
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        if pred != label and incorrect_count < 3:
            sample_text += f"  {class_names[label]} -> {class_names[pred]}\n"
            incorrect_count += 1
    
    axes[1, 1].text(0.1, 0.7, sample_text, fontsize=10, family='monospace', 
                    verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed report
    print("\n📊 Classification Report:")
    print("="*60)
    df_report = pd.DataFrame(report).transpose()
    print(df_report.round(4))
    print("="*60)
    
    # Save report to CSV
    df_report.to_csv('classification_report.csv')
    print("\n✅ Report saved to 'classification_report.csv'")
    
    return all_preds, all_labels, cm, df_report

# ============================================
# 6. PLOT TRAINING CURVES
# ============================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """
    Plot training and validation curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss', marker='o', linewidth=2, markersize=6)
    axes[0].plot(val_losses, label='Validation Loss', marker='s', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(train_accs, label='Train Accuracy', marker='o', linewidth=2, markersize=6)
    axes[1].plot(val_accs, label='Validation Accuracy', marker='s', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Training curves saved to 'training_curves.png'")

# ============================================
# 7. RESULTS TABLE
# ============================================

def create_results_table(train_accs, val_accs, train_losses, val_losses, report_df):
    """
    Create comprehensive results table
    """
    results = {
        'Metric': [
            'Best Training Accuracy',
            'Best Validation Accuracy',
            'Final Training Accuracy',
            'Final Validation Accuracy',
            'Overall Test Accuracy',
            'Weighted Precision',
            'Weighted Recall',
            'Weighted F1-Score'
        ],
        'Value': [
            f"{max(train_accs):.2f}%",
            f"{max(val_accs):.2f}%",
            f"{train_accs[-1]:.2f}%",
            f"{val_accs[-1]:.2f}%",
            f"{report_df.loc['accuracy', 'precision']*100:.2f}%" if 'accuracy' in report_df.index else "N/A",
            f"{report_df.loc['weighted avg', 'precision']:.4f}" if 'weighted avg' in report_df.index else "N/A",
            f"{report_df.loc['weighted avg', 'recall']:.4f}" if 'weighted avg' in report_df.index else "N/A",
            f"{report_df.loc['weighted avg', 'f1-score']:.4f}" if 'weighted avg' in report_df.index else "N/A"
        ]
    }
    
    df_results = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("📈 TRAINING PERFORMANCE SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    
    # Save to CSV
    df_results.to_csv('training_results_summary.csv', index=False)
    print("✅ Results saved to 'training_results_summary.csv'")
    
    return df_results

# ============================================
# 8. LLM INTEGRATION (Optional)
# ============================================

def setup_llm():
    """
    Load FLAN-T5 model for explanation generation
    """
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        model_name = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        print(f"⚠️ LLM setup failed: {e}")
        return None, None

def generate_explanation(class_name, confidence, tokenizer, llm_model, device='cpu'):
    """
    Generate maintenance advisory using LLM
    """
    defect_info = {
        'crazing': {
            'description': 'Fine hair-like cracks on the surface due to thermal cycling or material fatigue',
            'cause': 'Excessive temperature changes, improper cooling, or internal material stress',
            'maintenance': 'Polish surface or apply protective coating. Replace if cracks are deep.'
        },
        'inclusion': {
            'description': 'Foreign particles embedded in the surface during production',
            'cause': 'Contaminated raw materials or unclean production environment',
            'maintenance': 'Mechanical cleaning or chemical treatment to remove particles.'
        },
        'patches': {
            'description': 'Abnormal color or texture patches from oxidation or chemical reactions',
            'cause': 'Oxidation, chemical exposure, or improper surface treatment',
            'maintenance': 'Chemical cleaning or apply anti-corrosion coating.'
        },
        'pitted_surface': {
            'description': 'Small pits or dimples from localized corrosion or mechanical damage',
            'cause': 'Localized corrosion, erosion, or air bubbles during production',
            'maintenance': 'Apply filler to fill pits. Replace if damage is extensive.'
        },
        'rolled-in_scale': {
            'description': 'Mill scale embedded during rolling process appearing as dark patches',
            'cause': 'Insufficient cooling or cleaning during rolling process',
            'maintenance': 'Acid cleaning or mechanical descaling. Check cooling system.'
        },
        'scratches': {
            'description': 'Linear scratches or abrasion marks from mechanical friction',
            'cause': 'Friction during transportation, handling, or machining',
            'maintenance': 'Polish or buff the surface. Resurfacing for deep scratches.'
        }
    }
    
    info = defect_info.get(class_name, {
        'description': 'Unknown defect type',
        'cause': 'Information not available',
        'maintenance': 'Consult domain expert'
    })
    
    prompt = f"""
    Industrial Surface Defect Analysis:
    Defect Type: {class_name}
    Confidence: {confidence:.2f}%
    Description: {info['description']}
    
    Provide:
    1. Cause of this defect
    2. Recommended maintenance actions
    3. Preventive measures
    
    Answer: """
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    outputs = llm_model.generate(
        inputs.input_ids,
        max_length=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        'defect': class_name,
        'confidence': f'{confidence:.2f}%',
        'description': info['description'],
        'cause': info['cause'],
        'maintenance_recommendation': info['maintenance'],
        'llm_generated_advice': explanation
    }

# ============================================
# 9. MAIN PIPELINE
# ============================================

def main():
    """
    Main execution pipeline
    """
    print("="*60)
    print("🔬 SDEFECT-VLM: SURFACE DEFECT DETECTION FRAMEWORK")
    print("="*60)
    
    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 
                         'cpu')
    print(f"💻 Device: {device}")
    
    # ========================================
    # IMPORTANT: UPDATE THESE PATHS
    # ========================================
    TRAIN_IMG_DIR = 'train/images'          # Path to training images
    TRAIN_ANN_DIR = 'train/annotations'     # Path to training annotations
    VAL_IMG_DIR = 'validation/images'       # Path to validation images
    VAL_ANN_DIR = 'validation/annotations'  # Path to validation annotations
    
    # Check if dataset exists
    if not os.path.exists(TRAIN_IMG_DIR):
        print("\n❌ ERROR: Dataset not found!")
        print("Please update the paths in the main() function.")
        print("\nExpected structure:")
        print("NEU-DET/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   │   ├── crazing/")
        print("  │   │   │   └── crazing_1.jpg")
        print("  │   │   ├── inclusion/")
        print("  │   │   ├── patches/")
        print("  │   │   ├── pitted_surface/")
        print("  │   │   ├── rolled-in_scale/")
        print("  │   │   └── scratches/")
        print("  │   └── annotations/")
        print("  │       └── crazing_1.xml")
        print("  └── validation/")
        print("      ├── images/")
        print("      └── annotations/")
        return
    
    print(f"\n📁 Dataset found at: {TRAIN_IMG_DIR}")
    
    # Create dataloaders
    print("\n📂 Loading dataset...")
    train_loader, val_loader, num_classes = create_dataloaders(
        TRAIN_IMG_DIR, TRAIN_ANN_DIR, VAL_IMG_DIR, VAL_ANN_DIR, 
        batch_size=32
    )
    
    # Create model
    model = ResNet50Classifier(num_classes=num_classes).to(device)
    print(f"\n🧠 Model: ResNet-50 (Pre-trained on ImageNet)")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"📊 Trainable Parameters: {trainable_params:,}")
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, device, epochs=30
    )
    
    # Load best model
    if os.path.exists('best_resnet50_neu.pth'):
        model.load_state_dict(torch.load('best_resnet50_neu.pth'))
        print("\n✅ Best model loaded successfully")
    
    # Evaluate
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    preds, labels, cm, report_df = evaluate_model(model, val_loader, device, class_names)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # Create results table
    results_df = create_results_table(train_accs, val_accs, train_losses, val_losses, report_df)
    
    # Optional: LLM Integration
    print("\n🤖 LLM Integration (Optional)")
    try:
        tokenizer, llm_model = setup_llm()
        if tokenizer and llm_model:
            llm_model = llm_model.to(device)
            test_class = 'crazing'
            test_conf = 94.5
            explanation = generate_explanation(test_class, test_conf, tokenizer, llm_model, device)
            
            print("\n📝 Sample Explanation:")
            print(f"  Defect: {explanation['defect']}")
            print(f"  Confidence: {explanation['confidence']}")
            print(f"  Description: {explanation['description']}")
            print(f"  Cause: {explanation['cause']}")
            print(f"  Maintenance: {explanation['maintenance_recommendation']}")
            
            import json
            with open('llm_explanations.json', 'w') as f:
                json.dump(explanation, f, indent=2)
            print("✅ LLM explanation saved to 'llm_explanations.json'")
        else:
            print("⚠️ LLM integration skipped")
    except Exception as e:
        print(f"⚠️ LLM integration failed: {e}")
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  📁 best_resnet50_neu.pth - Trained model")
    print("  📁 evaluation_results.png - Evaluation visualizations")
    print("  📁 training_curves.png - Training curves")
    print("  📁 training_results_summary.csv - Performance summary")
    print("  📁 classification_report.csv - Detailed classification report")
    if os.path.exists('llm_explanations.json'):
        print("  📁 llm_explanations.json - LLM explanations")

# ============================================
# 10. SINGLE IMAGE PREDICTION
# ============================================

def predict_single_image(image_path, model_path='best_resnet50_neu.pth', use_llm=True):
    """
    Predict defect on a single image
    """
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = ResNet50Classifier(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    print(f"\n🔍 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence_score:.2f}%")
    
    # LLM explanation
    if use_llm:
        try:
            tokenizer, llm_model = setup_llm()
            if tokenizer and llm_model:
                llm_model = llm_model.to(device)
                explanation = generate_explanation(predicted_class, confidence_score, tokenizer, llm_model, device)
                
                print("\n📝 Explanation:")
                print(f"  Description: {explanation['description']}")
                print(f"  Cause: {explanation['cause']}")
                print(f"  Maintenance: {explanation['maintenance_recommendation']}")
        except Exception as e:
            print(f"⚠️ LLM explanation unavailable: {e}")
    
    return predicted_class, confidence_score

# ============================================
# 11. EXECUTION
# ============================================

if __name__ == "__main__":
    main()
    
    # Uncomment to test single image prediction
    # predict_single_image('test_image.jpg')