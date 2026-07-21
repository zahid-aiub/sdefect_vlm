# ============================================
# SDEFECT-VLM: COMPLETE PIPELINE FOR MAC M4
# Surface Defect Detection + LLM Maintenance Advisory
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
import gc
import json
warnings.filterwarnings('ignore')

# ============================================
# 1. DEVICE CONFIGURATION FOR MAC M4
# ============================================

def setup_device():
    """Configure device for Mac M4 with MPS support"""
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍏 Using Apple M4 with MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("🚀 Using CUDA GPU")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    torch.set_default_dtype(torch.float32)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    
    return device

# ============================================
# 2. DATASET CLASS
# ============================================

class NEUDETDataset(Dataset):
    """NEU-DET Surface Defect Dataset Loader"""
    def __init__(self, root_dir, annotations_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.is_train = is_train
        
        self.class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
        
        self.annotations = []
        if os.path.exists(annotations_dir):
            for file in os.listdir(annotations_dir):
                if file.endswith('.xml'):
                    self.annotations.append(os.path.join(annotations_dir, file))
        else:
            raise FileNotFoundError(f"Annotation folder not found: {annotations_dir}")
        
        self.file_cache = self._build_file_cache()
        print(f"{'Training' if is_train else 'Validation'} Dataset: {len(self.annotations)} samples")
    
    def _build_file_cache(self):
        cache = {}
        if not os.path.exists(self.root_dir):
            return cache
        
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
        
        img_path = os.path.join(self.root_dir, class_name, filename)
        if not os.path.exists(img_path):
            img_path = self._find_image_case_insensitive(class_name, filename)
        
        if img_path is None or not os.path.exists(img_path):
            random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            image = Image.fromarray(random_img)
            if idx % 100 == 0:
                print(f"⚠️ Image not found: {class_name}/{filename}")
        else:
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                random_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                image = Image.fromarray(random_img)
        
        label = self.class_to_idx[class_name]
        if self.transform:
            image = self.transform(image)
        
        return image, label, class_name
    
    def _find_image_case_insensitive(self, class_name, filename):
        class_folder = os.path.join(self.root_dir, class_name)
        if not os.path.exists(class_folder):
            return None
        base_name = os.path.splitext(filename)[0]
        for file in os.listdir(class_folder):
            if file.lower() == filename.lower():
                return os.path.join(class_folder, file)
            if os.path.splitext(file)[0].lower() == base_name.lower():
                return os.path.join(class_folder, file)
        return None

# ============================================
# 3. DATA TRANSFORMATIONS
# ============================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def create_dataloaders(train_img_dir, train_ann_dir, val_img_dir, val_ann_dir, batch_size=32):
    train_dataset = NEUDETDataset(train_img_dir, train_ann_dir, transform=train_transform, is_train=True)
    val_dataset = NEUDETDataset(val_img_dir, val_ann_dir, transform=val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, len(train_dataset.class_names)

# ============================================
# 4. RESNET-50 MODEL
# ============================================

class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        for name, param in self.backbone.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
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
# 5. TRAINING FUNCTION (MAC M4 OPTIMIZED)
# ============================================

def train_model_mac(model, train_loader, val_loader, device, epochs=30):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING ON MAC M4 (MPS)")
    print("="*60)
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, labels, _ in train_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
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
        
        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
            for images, labels, _ in val_pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet50_neu_mac.pth')
            print(f"✅ New best model saved! Accuracy: {val_acc:.2f}%")
        
        print(f'Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print("-"*60)
        
        if device.type == 'mps' and epoch % 5 == 0:
            torch.mps.empty_cache()
            gc.collect()
    
    print(f"\n🎉 Training Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    return train_losses, val_losses, train_accs, val_accs

# ============================================
# 6. EVALUATION FUNCTION
# ============================================

def evaluate_model(model, val_loader, device, class_names):
    model.eval()
    all_preds, all_labels = [], []
    
    print("\n📊 Evaluating Model...")
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc="Evaluation"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
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
    
    metrics = ['Accuracy', 'Precision\n(Weighted)', 'Recall\n(Weighted)', 'F1\n(Weighted)']
    values = [accuracy, report['weighted avg']['precision'], report['weighted avg']['recall'], report['weighted avg']['f1-score']]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
    axes[1, 0].bar(metrics, values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(values):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.1, 0.9, '📊 Classification Report Summary', fontsize=14, fontweight='bold')
    sample_text = "Sample Predictions:\n" + "="*40 + "\n"
    correct_count, incorrect_count = 0, 0
    sample_text += "✅ Correct Predictions:\n"
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        if pred == label and correct_count < 3:
            sample_text += f"  {class_names[label]} -> {class_names[pred]}\n"
            correct_count += 1
    sample_text += "\n❌ Incorrect Predictions:\n"
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        if pred != label and incorrect_count < 3:
            sample_text += f"  {class_names[label]} -> {class_names[pred]}\n"
            incorrect_count += 1
    axes[1, 1].text(0.1, 0.7, sample_text, fontsize=10, family='monospace', verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Classification Report:")
    print("="*60)
    df_report = pd.DataFrame(report).transpose()
    print(df_report.round(4))
    df_report.to_csv('classification_report.csv')
    print("\n✅ Report saved to 'classification_report.csv'")
    
    return all_preds, all_labels, cm, df_report

# ============================================
# 7. PLOT TRAINING CURVES
# ============================================

def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].plot(train_losses, label='Train Loss', marker='o', linewidth=2, markersize=6)
    axes[0].plot(val_losses, label='Validation Loss', marker='s', linewidth=2, markersize=6)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, label='Train Accuracy', marker='o', linewidth=2, markersize=6)
    axes[1].plot(val_accs, label='Validation Accuracy', marker='s', linewidth=2, markersize=6)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Validation Accuracy', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training curves saved to 'training_curves.png'")

# ============================================
# 8. RESULTS TABLE
# ============================================

def create_results_table(train_accs, val_accs, train_losses, val_losses, report_df):
    results = {
        'Metric': ['Best Training Accuracy', 'Best Validation Accuracy', 'Final Training Accuracy', 
                   'Final Validation Accuracy', 'Overall Test Accuracy', 'Weighted Precision', 
                   'Weighted Recall', 'Weighted F1-Score'],
        'Value': [
            f"{max(train_accs):.2f}%", f"{max(val_accs):.2f}%", f"{train_accs[-1]:.2f}%",
            f"{val_accs[-1]:.2f}%", f"{report_df.loc['accuracy', 'precision']*100:.2f}%" if 'accuracy' in report_df.index else "N/A",
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
    df_results.to_csv('training_results_summary.csv', index=False)
    print("✅ Results saved to 'training_results_summary.csv'")
    return df_results

# ============================================
# 9. LLM INTEGRATION (FROM test_llm_only.py)
# ============================================

def setup_llm_mac(device):
    """Load FLAN-T5 model optimized for Mac M4"""
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        
        print("🤖 Loading FLAN-T5 model for Mac M4...")
        model_name = "google/flan-t5-base"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        print(f"✅ FLAN-T5 loaded successfully on {device}")
        return tokenizer, model
    except ImportError as e:
        print(f"⚠️ Transformers library not found: {e}")
        print("   Run: pip install transformers sentencepiece protobuf")
        return None, None
    except Exception as e:
        print(f"⚠️ LLM setup failed: {e}")
        return None, None

def generate_explanation(class_name, confidence, tokenizer, llm_model, device='cpu'):
    """Generate maintenance advisory using LLM"""
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
    
    prompt = f"""Question: What causes {class_name} defects on steel surfaces and how can they be repaired?

Context: {info['description']}

Please provide a complete answer with these 3 sections:
1. Main Cause:
2. Repair Method:
3. Prevention:"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True, padding=True).to(device)
    
    with torch.no_grad():
        outputs = llm_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=150,
            min_new_tokens=50,
            temperature=0.4,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.8,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=3,
            no_repeat_ngram_size=3,
            early_stopping=True
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
# 10. SINGLE IMAGE PREDICTION (WITH LLM)
# ============================================

def predict_single_image_mac(image_path, model_path='best_resnet50_neu_mac.pth', use_llm=True):
    """Predict defect on a single image with LLM explanation"""
    device = setup_device()
    
    model = ResNet50Classifier(num_classes=6).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    
    print(f"\n🔍 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence_score:.2f}%")
    
    if use_llm:
        tokenizer, llm_model = setup_llm_mac(device)
        if tokenizer and llm_model:
            try:
                explanation = generate_explanation(predicted_class, confidence_score, tokenizer, llm_model, device)
                print("\n📝 Explanation:")
                print(f"  Description: {explanation['description']}")
                print(f"  Cause: {explanation['cause']}")
                print(f"  Maintenance: {explanation['maintenance_recommendation']}")
                print(f"  LLM Advice: {explanation['llm_generated_advice']}")
                with open('single_prediction_explanation.json', 'w') as f:
                    json.dump(explanation, f, indent=2)
                print("✅ Explanation saved to 'single_prediction_explanation.json'")
            except Exception as e:
                print(f"⚠️ LLM explanation unavailable: {e}")
    
    return predicted_class, confidence_score

# ============================================
# 11. MAIN PIPELINE (MAC M4 OPTIMIZED + LLM)
# ============================================

def main_mac():
    """Main pipeline with training, evaluation, and LLM integration"""
    print("="*60)
    print("🔬 SDEFECT-VLM ON APPLE M4 (WITH LLM)")
    print("="*60)
    
    device = setup_device()
    
    # ========================================
    # IMPORTANT: UPDATE THESE PATHS
    # ========================================
    TRAIN_IMG_DIR = 'train/images'
    TRAIN_ANN_DIR = 'train/annotations'
    VAL_IMG_DIR = 'validation/images'
    VAL_ANN_DIR = 'validation/annotations'
    
    if not os.path.exists(TRAIN_IMG_DIR):
        print("\n❌ Dataset not found! Please update paths in main_mac()")
        print("\nExpected structure:")
        print("NEU-DET/")
        print("  ├── train/")
        print("  │   ├── images/")
        print("  │   │   ├── crazing/")
        print("  │   │   │   └── crazing_1.jpg")
        print("  │   │   ├── inclusion/")
        print("  │   │   └── ...")
        print("  │   └── annotations/")
        print("  │       └── crazing_1.xml")
        print("  └── validation/")
        print("      ├── images/")
        print("      └── annotations/")
        return
    
    print(f"\n📁 Dataset found at: {TRAIN_IMG_DIR}")
    
    print("\n📂 Loading dataset...")
    train_loader, val_loader, num_classes = create_dataloaders(
        TRAIN_IMG_DIR, TRAIN_ANN_DIR, VAL_IMG_DIR, VAL_ANN_DIR, batch_size=32
    )
    
    model = ResNet50Classifier(num_classes=num_classes).to(device)
    print(f"\n🧠 Model: ResNet-50 (Pre-trained on ImageNet)")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"📊 Trainable Parameters: {trainable_params:,}")
    
    # Train model (or skip if already trained)
    train_losses, val_losses, train_accs, val_accs = train_model_mac(
        model, train_loader, val_loader, device, epochs=30
    )
    
    if os.path.exists('best_resnet50_neu_mac.pth'):
        model.load_state_dict(torch.load('best_resnet50_neu_mac.pth', map_location=device))
        print("\n✅ Best model loaded successfully")
    
    class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    preds, labels, cm, report_df = evaluate_model(model, val_loader, device, class_names)
    
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    results_df = create_results_table(train_accs, val_accs, train_losses, val_losses, report_df)
    
    # ========================================
    # LLM INTEGRATION (FROM test_llm_only.py)
    # ========================================
    print("\n" + "="*60)
    print("🤖 LLM INTEGRATION FOR MAINTENANCE ADVISORY")
    print("="*60)
    
    tokenizer, llm_model = setup_llm_mac(device)
    
    if tokenizer and llm_model:
        try:
            # Test with sample defect
            test_class = 'crazing'
            test_conf = 99.72
            explanation = generate_explanation(test_class, test_conf, tokenizer, llm_model, device)
            
            print(f"\n📝 Sample Explanation for '{test_class}':")
            print(f"  Description: {explanation['description']}")
            print(f"  Cause: {explanation['cause']}")
            print(f"  Maintenance: {explanation['maintenance_recommendation']}")
            print(f"\n💡 LLM Generated Advice:")
            print("-"*50)
            print(explanation['llm_generated_advice'])
            print("-"*50)
            
            with open('llm_explanations_mac.json', 'w') as f:
                json.dump(explanation, f, indent=2)
            print("✅ LLM explanation saved to 'llm_explanations_mac.json'")
            
            # Test with all defect types
            print("\n📊 Generating explanations for all defect types...")
            all_explanations = {}
            for defect in class_names:
                exp = generate_explanation(defect, 95.0, tokenizer, llm_model, device)
                all_explanations[defect] = exp
                print(f"  ✅ {defect}: {exp['llm_generated_advice'][:100]}...")
            
            with open('all_defect_explanations.json', 'w') as f:
                json.dump(all_explanations, f, indent=2)
            print("✅ All explanations saved to 'all_defect_explanations.json'")
            
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
    else:
        print("⚠️ LLM integration skipped - install transformers and sentencepiece")
    
    print("\n" + "="*60)
    print("🎉 MAC M4 PIPELINE COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  📁 best_resnet50_neu_mac.pth - Trained model")
    print("  📁 evaluation_results.png - Evaluation visualizations")
    print("  📁 training_curves.png - Training curves")
    print("  📁 training_results_summary.csv - Performance summary")
    print("  📁 classification_report.csv - Detailed classification report")
    print("  📁 llm_explanations_mac.json - LLM explanations")
    print("  📁 all_defect_explanations.json - All defect explanations")

# ============================================
# 12. EXECUTION
# ============================================

if __name__ == "__main__":
    main_mac()
    
    # Uncomment to test single image prediction
    # predict_single_image_mac('test_image.jpg')