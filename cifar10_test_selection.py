import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch
from model.resnet import resnet18  # <-- Use your custom CIFAR version
import torch.nn.functional as F
import ssl
from pathlib import Path
import copy
from DATIS.DATIS import DATIS_test_input_selection, DATIS_redundancy_elimination
import torchvision
import torchvision.transforms as transforms


# ========================== FEATURE UTILITIES ==========================

def get_features(model, x, device, batch_size=128):
    # Extract penultimate layer features in batches.
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i:i+batch_size]).permute(0, 3, 1, 2).to(device)
            out = model.representation(xb)
            feats.append(out.cpu())
    return torch.cat(feats).numpy()


def get_softmax(model, x, device, batch_size=128):
    # Extract softmax probabilities in batches.
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i:i+batch_size]).permute(0, 3, 1, 2).to(device)
            logits = model(xb)
            out = F.softmax(logits, dim=1)
            probs.append(out.cpu())
    return torch.cat(probs).numpy()


def get_faults(sample, mis_ind_test, Clustering_labels):
    # Fault detection rate calculation helper.
    neg = 0
    cluster_lab = []
    nn = -1
    for l in sample:
        if l in mis_ind_test:
            neg += 1
            ind = list(mis_ind_test).index(l)
            if Clustering_labels[ind] > -1:
                cluster_lab.append(Clustering_labels[ind])
            else:
                cluster_lab.append(nn)
                nn -= 1

    faults_n = len(list(set(cluster_lab)))

    cluster_1noisy = copy.deepcopy(cluster_lab)
    for i in range(len(cluster_1noisy)):
        if cluster_1noisy[i] <= -1:
            cluster_1noisy[i] = -1
    faults_1noisy = len(list(set(cluster_1noisy)))
    return faults_n, faults_1noisy, neg


def calculate_rate(budget_ratio_list, test_support_output, x_test, rank_lst, ans, cluster_path):
    # Compute and print fault detection rate at multiple budgets.
    top_list = [int(len(x_test) * r) for r in budget_ratio_list]
    result_fault_rate = []

    clustering_labels = np.load(Path(cluster_path) / "cluster1.npy")
    mis_test_ind = np.load(Path(cluster_path) / "mis_test_ind.npy")

    fault_sum_all = np.max(clustering_labels) + 1 + np.count_nonzero(clustering_labels == -1)
    print(f"Total test cases: {len(x_test)}")

    for i_, n in enumerate(top_list):
        n_indices = ans[i_] if len(ans) != 0 else rank_lst[:n]
        n_fault, n_noisy, n_neg = get_faults(n_indices, mis_test_ind, clustering_labels)
        faults_rate = n_fault / min(n, fault_sum_all)
        print(f"The Fault Detection Rate of Top {n} cases: {faults_rate:.4f}")
        result_fault_rate.append(faults_rate)

    return result_fault_rate


# ========================== DATA LOADERS ==========================

def load_data():
    """Load CIFAR-10 nominal dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    x_train = trainset.data.astype('float32') / 255.0  # (50000,32,32,3)
    y_train = np.array(trainset.targets)
    x_test = testset.data.astype('float32') / 255.0
    y_test = np.array(testset.targets)
    return (x_train, y_train), (x_test, y_test)


def load_data_corrupted():
    """Load pre-created corrupted CIFAR-10 dataset (CIFAR-10-C formatted)."""
    data_corrupted_file = "/content/drive/MyDrive/DATIS/corrupted_data/cifar10/data_corrupted.npy"
    label_corrupted_file = "/content/drive/MyDrive/DATIS/corrupted_data/cifar10/label_corrupted.npy"

    x_test_ood = np.load(data_corrupted_file)
    y_test_ood = np.load(label_corrupted_file)
    y_test_ood = y_test_ood.reshape(-1)
    x_test_ood = x_test_ood.reshape(-1, 32, 32, 3).astype('float32') / 255.0
    return x_test_ood, y_test_ood


# ========================== MAIN PIPELINE ==========================

def demo(data_type):
    """Run DATIS prioritization pipeline with feature caching."""
    if data_type == 'nominal':
        (x_train, y_train), (x_test, y_test) = load_data()
        cluster_path = '/content/drive/MyDrive/DATIS/cluster_data/ResNet32_cifar10_nominal'
        cache_dir = Path("/content/drive/MyDrive/DATIS/saved_features/cifar10_nominal")
    elif data_type == 'corrupted':
        (x_train, y_train), _ = load_data()
        x_test, y_test = load_data_corrupted()
        cluster_path = '/content/drive/MyDrive/DATIS/cluster_data/ResNet32_cifar10_corrupted'
        cache_dir = Path("/content/drive/MyDrive/DATIS/saved_features/cifar10_corrupted")
    else:
        raise ValueError("data_type must be 'nominal' or 'corrupted'")

    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Model setup ===
    checkpoint_path = '/content/drive/MyDrive/DATIS/weights/epoch_200.pth'
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'] if 'net' in checkpoint else checkpoint)
    model.eval()

    # === Load cached features if available ===
    train_feat_path = cache_dir / "train_features.npy"
    test_feat_path = cache_dir / "test_features.npy"
    softmax_path = cache_dir / "test_softmax.npy"
    y_train_path = cache_dir / "y_train.npy"
    y_test_path = cache_dir / "y_test.npy"

    if all(p.exists() for p in [train_feat_path, test_feat_path, softmax_path, y_train_path, y_test_path]):
        print(f"[INFO] Loading cached features for {data_type} dataset...")
        train_support_output = np.load(train_feat_path)
        test_support_output = np.load(test_feat_path)
        softmax_test_prob = np.load(softmax_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
    else:
        print(f"[INFO] Extracting features for {data_type} dataset (first run)...")
        train_support_output = get_features(model, x_train, device)
        test_support_output = get_features(model, x_test, device)
        softmax_test_prob = get_softmax(model, x_test, device)

        np.save(train_feat_path, train_support_output)
        np.save(test_feat_path, test_support_output)
        np.save(softmax_path, softmax_test_prob)
        np.save(y_train_path, y_train)
        np.save(y_test_path, y_test)
        print(f"[INFO] Features cached to {cache_dir}")

    # === DATIS Pipeline ===
    rank_lst = DATIS_test_input_selection(
        softmax_test_prob, train_support_output, y_train,
        test_support_output, y_test, 10
    )

    budget_ratio_list = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]

    ans = DATIS_redundancy_elimination(
        budget_ratio_list, rank_lst, test_support_output, y_test
    )

    calculate_rate(budget_ratio_list, test_support_output, x_test, rank_lst, ans, cluster_path)


# ========================== ENTRY POINT ==========================

if __name__ == '__main__':
    demo('nominal')
    print("         =====================================           ")
    demo('corrupted')
