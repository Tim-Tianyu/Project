import torch
import random
import CustomDataset
import CustomModels
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from arg_extractor import get_args
from experiment_builder import ExperimentBuilder

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

if args.dataset_name == 'CIFAR10':
    train_data, val_data, test_data = CustomDataset.load_dataset(dataset_name = "CIFAR10", distribution_name = args.distribution_name, transform = transform_CIFAR10)
elif args.dataset_name == 'MNIST':
    train_data, val_data, test_data = CustomDataset.load_dataset(dataset_name = "MNIST", distribution_name = args.distribution_name, transform = transform_MNIST)
else:
    raise CustomDataset.DataSetNotFound

train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)

custom_conv_net = CustomModels.load_model(model_name=args.model_name, num_of_channels=args.image_num_channels, num_classes=args.num_classes)

conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=-1,
                                    train_data=train_data_loader, val_data=val_data_loader,
                                    test_data=test_data_loader)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
