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

if args.partition_name not in CustomDataset.partitions_names:
    raise CustomDataset.PartitionNotFound

with open('./data/'+args.partition_name+'.txt', 'rb') as f: 
    partitions = pickle.load(f)

frontier = [partitions]
count = 0
total_size = len(partitions.idxs)

experiment_folder = os.path.abspath(args.experiment_name)
if not os.path.exists(experiment_folder):  # If experiment directory does not exist
    os.mkdir(experiment_folder)

models = []
classes = []

while len(frontier) != 0:
    partition = frontier.pop(0)
    if (not partition.has_children):
        continue
    #continue
    frontier = frontier+partition.children[:]
    classes.append(partition.classes)
    experiment_name = args.experiment_name+"/partition_"+str(count)
    count = count + 1
    current_size = len(partition.idxs)


    if args.dataset_name == 'CIFAR10':
        raise CustomDataset.DataSetNotFound
        #train_data, val_data, test_data = CustomDataset.load_dataset(dataset_name = "CIFAR10", distribution_name = args.distribution_name, transform = transform_CIFAR10)
    elif args.dataset_name == 'MNIST':
        train_data = CustomDataset.load_partition_dataset(dataset_name="MNIST", partition=partition, transform=transform_train_MNIST, train=True)
        val_data = CustomDataset.load_partition_dataset(dataset_name="MNIST", partition=partition, transform=transform_test_MNIST, train=False)
        test_data = CustomDataset.load_partition_dataset(dataset_name="MNIST", partition=partition, transform=transform_test_MNIST, train=False)

        #train_data, val_data, test_data = CustomDataset.load_dataset(dataset_name = "MNIST", distribution_name = args.distribution_name, transform = transform_MNIST)
    else:
        raise CustomDataset.DataSetNotFound
        
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

    custom_conv_net = CustomModels.load_model(model_name=args.model_name, num_of_channels=args.image_num_channels, num_classes=2)
    
    conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                        experiment_name=experiment_name,
                                        num_epochs=min(int(args.num_epochs * total_size / current_size), 200),
                                        use_gpu=args.use_gpu,
                                        continue_from_epoch=min(int(args.num_epochs * total_size / current_size), 200)-1,
                                        train_data=train_data_loader, val_data=val_data_loader,
                                        test_data=test_data_loader)  # build an experiment object
    
    experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
    conv_experiment.train()
    models.append(conv_experiment.model)
    save_list(experiment_name, "classes.txt", partition.classes)
    pass
test_data = CustomDataset.load_testset(dataset_name = "MNIST", transform= transform_MNIST)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
hierarchical_method_whole_system_test.test(args.experiment_name, partitions, args.model_name, args.image_num_channels, 2, test_loader)
#hierarchical_method_whole_system_test.test2(args.experiment_name, models, classes, test_loader)
