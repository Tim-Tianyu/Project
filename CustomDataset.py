from torchvision import datasets, transforms
import numpy as np


CIFAR10_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.247, 0.243, 0.261])])
MNIST_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
Distribution_names = ["linear_unbalanced_1", 
         "linear_unbalanced_2", 
         "balanced", 
         "balanced_5000", 
         "balanced_1600", 
         "balanced_800", 
         "balanced_400", 
         "balanced_200",
         "balanced_100", 
         "balanced_50", 
         "balanced_25", 
         "balanced_10", 
         "balanced_5", 
         "expo_unbalance", 
         "expo_unbalance_2560to5", 
         "expo_unbalance_5000to1",
         "cut_unbalance_4890or25", 
         "linear_unbalance_1020to3",
         "2560to5_rescaled",
         "balanced_2560"]
Datasets_name = ["MNIST", "CIFAR10"]

partitions_names = [
    "partition_object_test",
    "???"
]

class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=CIFAR10_transform, target_transform=None, download=False, indexs_name = "index_test2.npy"):
        super().__init__(root, train, transform, target_transform, download)
        result_idxs = np.load(root+'/'+indexs_name)
        self.data = self.data[result_idxs,:,:]
        self.targets = list(np.array(self.targets)[result_idxs])
            
        
class CustomMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=MNIST_transform, target_transform=None, download=False, indexs_name = "index_test.npy"):
        super().__init__(root, train, transform, target_transform, download)
        result_idxs = np.load(root+'/'+indexs_name)
        self.data = self.data[result_idxs,:,:]
        self.targets = self.targets[result_idxs]
            
class CustomMNIST_partition(datasets.MNIST):
    def __init__(self, root, indexs_name, classes, train=True, transform=MNIST_transform, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)
        result_idxs = np.load(root+'/'+indexs_name)
        self.data = self.data[result_idxs,:,:]
        self.targets = classes[self.targets][result_idxs]
        
        # filter out unused classes (samples that belongs to -1 class)
        samples_used = (self.targets != -1)
        self.data = self.data[samples_used, :, :]
        self.targets = self.targets[samples_used]
            
class dataset_partition():
    def __init__(self, **kwargs):
        old_class_number = kwargs.get("old_class_number")
        if old_class_number is not None:
            self.old_class_number = old_class_number
            self.has_children = False
        else:
            self.classes = kwargs.get("classes")
            self.has_children = True
            self.children = kwargs.get("children")
            if (self.classes is None) or (self.children is None):
                raise ValueError("wrong arguments")

class binary_target_transformer():
    def _init_(self, classes):
        self.classes = classes
    
    def target_transform(self, target):
        if target in self.clasees:
            return 1
        return 0

class muliclass_target_transformer():
    def _init_(self, classes_list):
        # Check if partistion of classes is right
        temp = []
        for classes in classes_list:
            temp += classes
        if len(temp) != len(set(temp)):
            raise InvalidPartition
        
        self.classes_list = classes_list
    
    def target_transform(self, target):
        for i in range(0,len(self.classes_list)):
            if target in self.classes_list[i]:
                return i
        raise InvalidTarget

class InvalidPartition(Exception):        
    pass
    
class InvalidTarget(Exception):
    pass
    
class DataSetNotFound(Exception):
    pass

class DistributionNotFound(Exception):
    pass

class PartitionNotFound(Exception):
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
        train_set = CustomMNIST(data_folder, train=True, transform=transform, download=True, indexs_name = "MNIST_"+distribution_name+".npy")
        eval_set = CustomMNIST(data_folder, train=False, transform=transform, download=True, indexs_name = "MNIST_"+distribution_name+"_eval.npy")
        print(len(eval_set.targets))
        test_set = CustomMNIST(data_folder, train=False, transform=transform, download=True, indexs_name = "MNIST_"+distribution_name+"_test.npy")
    elif (dataset_name == "CIFAR10"):
        train_set = CustomCIFAR10(data_folder, train=True, transform=transform, download=True, indexs_name="CIFAR10_"+distribution_name+".npy")
        eval_set = CustomCIFAR10(data_folder, train=False, transform=transform, download=True, indexs_name = "CIFAR10_"+distribution_name+"_eval.npy")
        test_set = CustomCIFAR10(data_folder, train=False, transform=transform, download=True, indexs_name = "CIFAR10_"+distribution_name+"_test.npy")
    else:
        raise DataSetNotFound
    return train_set, eval_set, test_set

def load_partition_dataset(dataset_name, distribution_name, classes, transform, data_folder="./data"):
    if not ((distribution_name in Distribution_names)):
        raise DistributionNotFound
    if (dataset_name == "MNIST"):
        train_set = CustomMNIST_partition(data_folder, indexs_name=dataset_name + "_" + distribution_name + ".npy", classes=classes, train=True, transform=transform, download=True)
        eval_set = CustomMNIST_partition(data_folder, indexs_name=dataset_name + "_" + distribution_name + "_eval.npy", classes=classes, train=False, transform=transform, download=True)
        test_set = CustomMNIST_partition(data_folder, indexs_name=dataset_name + "_" + distribution_name + "_test.npy", classes=classes, train=False, transform=transform, download=True)
    elif (dataset_name == "CIFAR10"):
        raise DataSetNotFound
    else:
        raise DataSetNotFound
    return train_set, eval_set, test_set

# def load_testset(dataset_name, transform, data_folder="./data"):
#     if (dataset_name == "MNIST"):
#         return datasets.MNIST(data_folder, transform=transform, train=False, download=True)
#     elif (dataset_name == "CIFAR10"): 
#         return datasets.CIFAR10(data_folder, transform =transform, train=False, download=True)
#     else:
#         raise DataSetNotFound

def random_select(idx_dic, num_list = [5000]*10, randomState=np.random.RandomState(np.random.seed(12345))):
    idxs = []
    for i in range(0,10):
        idxs_for_i = idx_dic[i].copy()
        randomState.shuffle(idxs_for_i)
        idxs = idxs + idxs_for_i[0:num_list[i]]
    return np.sort(np.array(idxs))

def duplicate_idxs(idx_dic, target_num, randomState=np.random.RandomState(np.random.seed(12345))):
    idxs = []
    isList = isinstance(target_num,list)
    for (k,v) in idx_dic.items():
        if isList:
            number = target_num[k]
        else:
            number = target_num    
        idxs = idxs + int(number / len(v)) * v
        idxs = idxs + v[0: (number%len(v))]
    randomState.shuffle(idxs)
    return idxs

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
    