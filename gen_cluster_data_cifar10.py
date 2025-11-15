import os
import torch, numpy as np
from model.resnet import resnet18
import torchvision, torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.cluster import KMeans

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = './weights/epoch_200.pth'

# Model
model = resnet18(num_classes=10)
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()
state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state['net'] if 'net' in state else state)
model.to(device)
model.eval()

# CIFAR-10 data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# Compute predictions
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in testloader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds, all_labels = np.array(all_preds), np.array(all_labels)
mis_test_ind = np.where(all_preds != all_labels)[0]
print(f"Misclassified samples: {len(mis_test_ind)}")


def get_features(model, x, batch_size=128):
    feats = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(x), batch_size)):
            xb = torch.tensor(x[i:i+batch_size]).permute(0,3,1,2).to(device)
            out = model.representation(xb)
            feats.append(out.cpu())
    return torch.cat(feats).numpy()

x_test = testset.data.astype('float32') / 255.0
x_test = x_test.transpose(0,2,3,1) if x_test.shape[1]==3 else x_test  # ensure (N,32,32,3)
mis_x = x_test[mis_test_ind]
mis_features = get_features(model, mis_x)
print(mis_features.shape)

n_clusters = 20  # you can adjust based on your CIFAR-10 results
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(mis_features)
cluster_labels = kmeans.labels_

os.makedirs('./cluster_data/ResNet32_cifar10_nominal', exist_ok=True)

np.save('./cluster_data/ResNet32_cifar10_nominal/mis_test_ind.npy', mis_test_ind)
np.save('./cluster_data/ResNet32_cifar10_nominal/cluster1.npy', cluster_labels)
print("Saved cluster_data for CIFAR-10 nominal")
