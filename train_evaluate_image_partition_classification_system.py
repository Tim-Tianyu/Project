import os
import torch
import random
import CustomDataset
import CustomModels
import numpy as np
import pickle
import hierarchical_method_whole_system_test
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from arg_extractor_for_partition import get_args
from experiment_builder import ExperimentBuilder
from storage_utils import save_list
from CustomDataset import dataset_partition

args = get_args()  # get arguments from command line

def run(args):
    rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
    torch.manual_seed(seed=args.seed)  # sets pytorch's seed

    # set up data augmentation transforms for training and testing
    transform_CIFAR10 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.247, 0.243, 0.261])
        ])


    transform_MNIST = transforms.Compose([
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    with open(args.partition_path, 'rb') as f: 
        partitions = pickle.load(f)

    frontier = [partitions]
    count = 0
    total_size = 0

    experiment_folder = os.path.abspath(args.experiment_name)
    if not os.path.exists(experiment_folder):  # If experiment directory does not exist
        os.mkdir(experiment_folder)

    models = []
    classes = []

    if args.dataset_name == 'CIFAR10':
        global_test_data = CustomDataset.load_testset(dataset_name = "CIFAR10", transform = transform_CIFAR10)
    elif args.dataset_name == 'MNIST':
        global_test_data = CustomDataset.load_testset(dataset_name = "MNIST", transform = transform_MNIST)
    else:
        raise CustomDataset.DataSetNotFound
    global_test_loader = DataLoader(global_test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    while len(frontier) != 0:
        partition = frontier.pop(0)
        if (not partition.has_children):
            continue
        frontier = frontier+partition.children[:]
        classes.append(partition.classes)
        experiment_folder = args.experiment_name+"/partition_"+str(count)
        count = count + 1
        
        if args.dataset_name == 'CIFAR10':
            orginal_size = 4000 * 10
            train_data = CustomDataset.load_partition_dataset(dataset_name = "CIFAR10", index_path = args.train_index_path, classes=partition.classes, transform = transform_CIFAR10)
            val_data = CustomDataset.load_partition_dataset(dataset_name = "CIFAR10", index_path = args.eval_index_path, classes=partition.classes, transform = transform_CIFAR10)
            test_data = CustomDataset.load_partition_testset(dataset_name = "CIFAR10", classes=partition.classes, transform = transform_CIFAR10)
        elif args.dataset_name == 'MNIST':
            orginal_size = 4000 * 10
            train_data = CustomDataset.load_partition_dataset(dataset_name = "MNIST", index_path = args.train_index_path, classes=partition.classes, transform = transform_MNIST)
            val_data = CustomDataset.load_partition_dataset(dataset_name = "MNIST", index_path = args.eval_index_path, classes=partition.classes, transform = transform_MNIST)
            test_data = CustomDataset.load_partition_testset(dataset_name = "MNIST", classes=partition.classes, transform = transform_MNIST)
        
        current_size = len(train_data.targets)
        num_epochs = int(np.ceil(orginal_size * args.num_epochs / max(args.batch_size, current_size)))
        
        drop_last = False
        if (len(train_data.targets) % args.batch_size == 1):
            drop_last = True
        train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=drop_last)
        
        drop_last = False
        if (len(val_data.targets) % args.batch_size == 1):
            drop_last = True
        val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=drop_last)
        
        test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        custom_conv_net = CustomModels.load_model(model_name=args.model_name, num_of_channels=args.image_num_channels, num_classes=2)
        conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                            experiment_name=experiment_folder,
                                            num_epochs=num_epochs,
                                            use_gpu=args.use_gpu,
                                            continue_from_epoch= -1,
                                            train_data=train_data_loader, val_data=val_data_loader,
                                            test_data=test_data_loader)  # build an experiment object
        save_list(experiment_folder, "classes.txt", partition.classes)
        experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
        conv_experiment.train()
        
        # models.append(conv_experiment.model)
        pass
    
    hierarchical_method_whole_system_test.test(args.experiment_name, partitions, args.model_name, args.image_num_channels, 2, global_test_loader)
    #hierarchical_method_whole_system_test.test2(args.experiment_name, models, classes, test_loader)

if __name__ == "__main__":
    run(args) 
