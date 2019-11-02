from torchvision import datasets, transforms
import numpy as np

class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, indexs_name = "index_test2.npy"):
        super().__init__(root, train, transform, target_transform, download)
        result_idxs = np.load(root+'/'+indexs_name)
        if (train):
            self.data = self.data[result_idxs,:,:]
            self.targets = list(np.array(self.targets)[result_idxs])
            
        
class CustomMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, indexs_name = "index_test.npy"):
        super().__init__(root, train, transform, target_transform, download)
        result_idxs = np.load(root+'/'+indexs_name)
        if (train):
            self.data = self.data[result_idxs,:,:]
            self.targets = self.targets[result_idxs]

def load_all_datasets(data_folder="./data", distribution_names = ["linear_unbalanced_1", "linear_unbalanced_2", "balanced", "expo_unbalance"]):
    datasets_dic = {}
    for name in distribution_names:
        
        datasets_dic["MNIST_"+name] = CustomMNIST(data_folder, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]), indexs_name = "MNIST_"+name+".npy")
    
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        datasets_dic["CIFAR10_"+name] = CustomCIFAR10(data_folder, train=True, download=True, transform=transform, indexs_name="CIFAR10_"+name+".npy")
    return datasets_dic

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
    