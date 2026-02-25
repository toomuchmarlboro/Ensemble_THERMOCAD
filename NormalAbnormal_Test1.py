import torch
import torch.nn as nn
import torchvision.models as models
import timm 
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from skrebate import ReliefF 

# --- CONFIGURATION (Workstation Optimized) ---
INPUT_ROOT = r"C:\Users\User\Documents\Skripsi Faiz_DL\Ensemble_THERMOCAD\DatasetDMR-IR_Watershed"  # Folder with Normal/Abnormal subfolders
BATCH_SIZE = 32                     # Increased for GPU efficiency
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42

print(f"--- Running on {DEVICE} with Batch Size {BATCH_SIZE} ---")

# --- PART 1: DATA LOADER ---
class ThermographyDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.label_map = {'Normal': 0, 'Abnormal': 1}
        
        # Preprocessing for ImageNet Models (299x299 for Inception compatibility)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Walk through directory
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    folder_name = os.path.basename(root)
                    if folder_name in self.label_map:
                        self.samples.append((
                            os.path.join(root, file),
                            self.label_map[folder_name]
                        ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 299, 299), label # Fail safe

# --- PART 2: THE PARALLEL DEEP EXTRACTOR ---
class ParallelFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. ResNet101 (Texture)
        self.resnet = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity() # Remove classifier
        
        # 2. Inception V3 (Multi-Scale)
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        
        # 3. ConvNeXt V2 (Hierarchical Shape)
        self.convnext = timm.create_model('convnextv2_base', pretrained=True, num_classes=0)

    def forward(self, x):
        # Extract in parallel
        f1 = self.resnet(x)
        f2 = self.inception(x)
        f3 = self.convnext(x)
        
        # Flatten everything
        f1 = torch.flatten(f1, 1)
        f2 = torch.flatten(f2, 1)
        f3 = torch.flatten(f3, 1)
        
        # Concatenate: [Batch, 2048 + 2048 + 1024] = [Batch, ~5120]
        return torch.cat([f1, f2, f3], dim=1)

# --- PART 3: EXECUTION LOGIC ---
def run_unified_pipeline():
    # A. Setup Data
    dataset = ThermographyDataset(INPUT_ROOT)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Dataset Size: {len(dataset)} images")

    # B. Setup Model
    model = ParallelFeatureExtractor().to(DEVICE)
    model.eval()

    # C. Feature Extraction Loop
    print("\n--- Phase 1: Deep Feature Extraction (Parallel Fusion) ---")
    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(DEVICE)
            
            # Forward pass through all 3 networks
            features = model(imgs)
            
            # Move to CPU and store
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            print(f"Batch {batch_idx+1}/{len(dataloader)} processed...", end='\r')

    # Concatenate all batches
    X = np.concatenate(all_features, axis=0)
    y = np.array(all_labels)
    print(f"\nExtraction Done. Feature Matrix Shape: {X.shape}")

    # D. The "Brain" (ML Pipeline)
    print("\n--- Phase 2: Hybrid Classification (NNMF -> ReliefF -> SVM) ---")
    
    # 1. Scaling (Critical for NNMF)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    # 3. Component 1: NNMF (Dimensionality Reduction)
    print("  -> Step 2a: NNMF (Compressing 5000+ features to 50 latent patterns)")
    nmf = NMF(n_components=50, init='nndsvd', random_state=RANDOM_STATE, max_iter=500)
    X_train_nmf = nmf.fit_transform(X_train)
    X_test_nmf = nmf.transform(X_test)

    # 4. Component 2: ReliefF (Feature Selection)
    print("  -> Step 2b: ReliefF (Selecting top 20 best patterns)")
    relief = ReliefF(n_features_to_select=20, n_neighbors=100)
    X_train_sel = relief.fit_transform(X_train_nmf, y_train)
    X_test_sel = relief.transform(X_test_nmf)

    # 5. Component 3: SVM (Classification)
    print("  -> Step 2c: SVM Training")
    svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=RANDOM_STATE)
    svm.fit(X_train_sel, y_train)

    # E. Evaluation
    print("\n--- Phase 3: Evaluation ---")
    preds = svm.predict(X_test_sel)
    acc = accuracy_score(y_test, preds)
    
    print(f"Global Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['Normal', 'Abnormal']))

    # F. Plotting
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Abnormal'], 
                yticklabels=['Normal', 'Abnormal'])
    plt.title(f'Unified Hybrid Pipeline\nResNet+Inception+ConvNeXt > NNMF > ReliefF > SVM\nAccuracy: {acc:.2%}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_unified_pipeline()