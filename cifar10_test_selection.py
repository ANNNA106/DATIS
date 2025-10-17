import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch
from model.resnet import resnet18  # <-- Use your custom CIFAR version
import copy
from DATIS.DATIS import DATIS_test_input_selection,DATIS_redundancy_elimination
from keras.datasets import mnist
import torch.nn.functional as F

def get_features(model, x, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x).permute(0, 3, 1, 2).to(device)  # (N,H,W,C) → (N,C,H,W)
        feats = model.representation(x)
        return feats.cpu().numpy()

def get_softmax(model, x, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x).permute(0, 3, 1, 2).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()


def get_faults(sample, mis_ind_test, Clustering_labels):
    i=0
    pos=0
    neg=0
    i=0
    cluster_lab=[]
    nn=-1
    for l in sample:
        if l in mis_ind_test:
            neg=neg+1 
            ind=list(mis_ind_test).index(l)
            if (Clustering_labels[ind]>-1):
                cluster_lab.append(Clustering_labels[ind])
            if (Clustering_labels[ind]==-1):
                cluster_lab.append(nn)
                nn=nn-1
        else:
            pos=pos+1

    faults_n=len(list(set(cluster_lab)))
    #All noisy mispredicted inputs are considered as one specific fault

    cluster_1noisy=copy.deepcopy(cluster_lab)
    for i in range(len(cluster_1noisy)):
        if cluster_1noisy[i] <=-1:
            cluster_1noisy[i]=-1
    faults_1noisy=len(list(set(cluster_1noisy)))
    return faults_n,faults_1noisy, neg


def calculate_rate(budget_ratio_list,test_support_output,x_test,rank_lst,ans,cluster_path):
   
    
    top_list =[]
    for ratio_ in budget_ratio_list:
        top_list.append(int(len(x_test)*ratio_))
    result_fault_rate= []
    clustering_labels = np.load(cluster_path+'/cluster1.npy')
    fault_sum_all = np.max(clustering_labels)+1+np.count_nonzero(clustering_labels == -1)
    mis_test_ind= np.load(cluster_path+'/mis_test_ind.npy')
    

    print('total test case:{len}')
   
    for i_, n in enumerate(top_list):

        if len(ans)!=0:
            n_indices =ans[i_] 
        else :
            n_indices = rank_lst[:n] 

        n_fault,n_noisy,n_neg = get_faults(n_indices,mis_test_ind,clustering_labels)

        faults_rate = n_fault/min(n,fault_sum_all)
        
        print(f"The Fault Detection Rate of Top: {n} cases :{faults_rate}")
        result_fault_rate.append(faults_rate)

    return 



import torchvision
import torchvision.transforms as transforms

def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    x_train = trainset.data.astype('float32') / 255.0  # shape (50000,32,32,3)
    y_train = np.array(trainset.targets)
    x_test  = testset.data.astype('float32') / 255.0
    y_test  = np.array(testset.targets)
    return (x_train, y_train), (x_test, y_test)



def load_data_corrupted():

    data_corrupted_file = "./corrupted_data/mnist/data_corrupted.npy"
    label_corrupted_file = "./corrupted_data/mnist/label_corrupted.npy"
    data_corrupted_file = "./corrupted_data/mnist/data_corrupted.npy"
    label_corrupted_file = "./corrupted_data/mnist/label_corrupted.npy"
    x_test_ood = np.load(data_corrupted_file)
    y_test_ood = np.load(label_corrupted_file)
    y_test_ood = y_test_ood.reshape(-1)
    x_test_ood = x_test_ood.reshape(-1, 28, 28, 1)
    x_test_ood = x_test_ood.astype('float32')
    x_test_ood /= 255
    return x_test_ood,y_test_ood
   
     


def demo(data_type):
   
    if data_type =='nominal':
        (x_train, y_train), (x_test, y_test) = load_data()
        cluster_path ='./cluster_data/ResNet32_cifar100_nominal'
    elif data_type == 'corrupted':
        (x_train, y_train), (x_test, y_test)= load_data()
        x_test, y_test = load_data_corrupted()
        cluster_path ='./cluster_data/ResNet32_cifar100_corrupted'
        
    checkpoint_path = '/content/drive/MyDrive/cka_models/resnet18_CIFAR10/noisy/random_0/epochs_200_tune_hp8/epoch_200.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model setup
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = torch.nn.Identity()
    model = model.to(device)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # === Feature extraction ===
    train_support_output = get_features(model, x_train, device)
    test_support_output  = get_features(model, x_test, device)
    softmax_test_prob    = get_softmax(model, x_test, device)

    rank_lst = DATIS_test_input_selection(softmax_test_prob,train_support_output,y_train,test_support_output,y_test,10)
    

    budget_ratio_list =[0.001,0.005,0.01,0.02,0.03,0.05,0.1]

    ans= DATIS_redundancy_elimination(budget_ratio_list,rank_lst,test_support_output,y_test)

    calculate_rate(budget_ratio_list,test_support_output,x_test,rank_lst,ans,cluster_path)

    
    

if __name__ == '__main__':
    print("Testing on CIFAR10 dataset")
    demo('nominal')
    print("         =====================================           ")
    demo('corrupted')
    

   
