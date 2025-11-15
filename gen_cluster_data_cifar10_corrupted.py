import os
import numpy as np
import torch
from model.resnet import resnet18
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

# ==================== CONFIG ====================
CHECKPOINT_PATH = './weights/epoch_200.pth'
CORRUPTED_DATA_DIR = './corrupted_data/cifar10'
SAVE_DIR = './cluster_data/ResNet32_cifar10_corrupted'
os.makedirs(SAVE_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== LOAD MODEL ====================
print("[INFO] Loading trained model...")
model = resnet18(num_classes=10)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()

state = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(state['net'] if 'net' in state else state)
model.to(device)
model.eval()

# ==================== LOAD CORRUPTED CIFAR-10 ====================
print("[INFO] Loading corrupted CIFAR-10 data...")
x_test = np.load(os.path.join(CORRUPTED_DATA_DIR, 'data_corrupted.npy'))
y_test = np.load(os.path.join(CORRUPTED_DATA_DIR, 'label_corrupted.npy')).reshape(-1)
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0

print(f"[INFO] Loaded corrupted test data: {x_test.shape}, labels: {y_test.shape}")

# ==================== PREDICT & FIND MISCLASSIFIED ====================
print("[INFO] Finding misclassified samples...")
BATCH_SIZE = 128
preds, labels = [], []
with torch.no_grad():
    for i in range(0, len(x_test), BATCH_SIZE):
        xb = torch.tensor(x_test[i:i+BATCH_SIZE]).permute(0, 3, 1, 2).to(device)
        logits = model(xb)
        batch_preds = logits.argmax(dim=1).cpu().numpy()
        preds.extend(batch_preds)
        labels.extend(y_test[i:i+BATCH_SIZE])

preds, labels = np.array(preds), np.array(labels)
mis_test_ind = np.where(preds != labels)[0]
print(f"[INFO] Misclassified samples: {len(mis_test_ind)}")

# ==================== EXTRACT FEATURES ====================
def get_features(model, x, device, batch_size=128):
    feats = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(x), batch_size)):
            xb = torch.tensor(x[i:i+batch_size]).permute(0, 3, 1, 2).to(device)
            out = model.representation(xb)
            feats.append(out.cpu())
    return torch.cat(feats).numpy()

print("[INFO] Extracting features for misclassified samples...")
mis_x = x_test[mis_test_ind]
mis_features = get_features(model, mis_x, device)
print(f"[INFO] Feature shape: {mis_features.shape}")

# ==================== CLUSTER MISCLASSIFIED FEATURES ====================
n_clusters = 30  # adjustable based on variety of corruptions
print(f"[INFO] Clustering {len(mis_test_ind)} samples into {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(mis_features)
cluster_labels = kmeans.labels_

# ==================== SAVE ====================
np.save(os.path.join(SAVE_DIR, 'mis_test_ind.npy'), mis_test_ind)
np.save(os.path.join(SAVE_DIR, 'cluster1.npy'), cluster_labels)
print(f"[INFO] Saved cluster data for CIFAR-10 corrupted → {SAVE_DIR}")
