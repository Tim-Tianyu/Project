from torchvision import datasets, transforms
import numpy as np


CIFAR10_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize()])
MNIST_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
Distribution_names = ["linear_unbalanced_1", "linear_unbalanced_2", "balanced", "expo_unbalance"]
Datasets_name = ["MNIST", "CIFAR10"]

class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=CIFAR10_transform, target_transform=None, download=False, indexs_name = "index_test2.npy"):
        super().__init__(root, train, transform, target_transform, download)
        result_idxs = np.load(root+'/'+indexs_name)
        if (train):
            self.data = self.data[result_idxs,:,:]
            self.targets = list(np.array(self.targets)[result_idxs])
            
        
class CustomMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=MNIST_transform, target_transform=None, download=False, indexs_name = "index_test.npy"):
        super().__init__(root, train, transform, target_transform, download)
        result_idxs = np.load(root+'/'+indexs_name)
        if (train):
            self.data = self.data[result_idxs,:,:]
            self.targets = self.targets[result_idxs]

class DataSetNotFound(Exception):
    pass

class DistributionNotFound(Exception):
    pass

# def load_all_datasets(data_folder="./data", distribution_names = Distribution_names):
#     datasets_dic = {}
#     for name in distribution_names:
#         if not (name in Distribution_names):
#             continue
#         datasets_dic["MNIST_"+name] = CustomMNIST(data_folder, train=True, download=True, indexs_name = "MNIST_"+name+".npy")
#         datasets_dic["CIFAR10_"+name] = CustomCIFAR10(data_folder, train=True, download=True, indexs_name="CIFAR10_"+name+".npy")
#     return datasets_dic

def load_dataset(dataset_name, distribution_name, transform, data_folder="./data"):
    if not ((distribution_name in Distribution_names)):
        raise DistributionNotFound
    if (dataset_name == "MNIST"):
        return CustomMNIST(data_folder, train=True, transform=transform, download=True, indexs_name = "MNIST_"+distribution_name+".npy")
    elif (dataset_name == "CIFAR10"):
        return CustomCIFAR10(data_folder, train=True, transform=transform, download=True, indexs_name="CIFAR10_"+distribution_name+".npy")
    else:
        raise DataSetNotFound

def load_testset(dataset_name, transform, data_folder="./data"):
    if (dataset_name == "MNIST"):
        return datasets.MNIST(data_folder, transform=transform, train=False, download=True)
    elif (dataset_name == "CIFAR10"): 
        return datasets.CIFAR10(data_folder, transform =transform, train=False, download=True)
    else:
        raise DataSetNotFound

def random_select(idx_dic, num_list = [5000]*10, randomState=np.random.RandomState(np.random.seed(12345))):
    idxs = []
    for i in range(0,10):
        idxs_for_i = idx_dic[i].copy()
        randomState.shuffle(idxs_for_i)
        idxs = idxs + idxs_for_i[0:num_list[i]]
    return np.sort(np.array(idxs))

def produce_idx_dic(targets):
    # initialize the index dic
    idx_dic = {}
    for i in set(np.array(targets)):
        idx_dic[i] = []
    
    # put indexs in
    i=0
    for target in np.array(targets):
        idx_dic.get(target).append(i)
        i=i+1
    return idx_dic
    