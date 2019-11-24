import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import imp
import random
import CustomDataset
import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
from experiment_builder import ExperimentBuilder

from pytorch_mlp_framework.arg_extractor import get_args
from pytorch_mlp_framework.experiment_builder import ExperimentBuilder

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

# set up data augmentation transforms for training and testing
transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(1.0,1.0), ratio=(1.0,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

transform_test = transforms.Compose([
    transforms.Resize(224)
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

if args.datasets_type = 'CIFAR10_balanced':
    train_data = CustomDataset.load_dataset(dataset = "CIFAR10", distribution_name = "balanced", transform = transform_train)
    val_data = CustomDataset.load_testset(dataset = "CIFAR10", transform= transform_test)
    test_data = CustomDataset.load_testset(dataset = "CIFAR10", transform= transform_test)
else:
    raise CustomDataset.DataSetNotFound

train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=4)


custom_conv_net = ConvolutionalNetwork(  # initialize our network object, in this case a ConvNet
    input_shape=(args.batch_size, args.image_num_channels, args.image_height, args.image_width),
    num_output_classes=args.num_classes, num_filters=args.num_filters, use_bias=False,
    num_blocks_per_stage=args.num_blocks_per_stage, num_stages=args.num_stages,
    processing_block_type=processing_block_type,
    dimensionality_reduction_block_type=dim_reduction_block_type)

conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data_loader, val_data=val_data_loader,
                                    test_data=test_data_loader)  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
