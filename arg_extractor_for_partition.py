import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description='Welcome to the MLP course\'s Pytorch training and inference helper script')

    parser.add_argument('--batch_size', nargs="?", type=int, default=100, help='Batch_size for experiment')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--image_num_channels', nargs="?", type=int, default=3,
                        help='The channel dimensionality of our image-data')
    parser.add_argument('--num_epochs', nargs="?", type=int, default=100, help='The experiment\'s epoch budget, for the root classifer, child classifer will have more epochs to make sure total number of batch is same')
    parser.add_argument('--num_classes', nargs="?", type=int, default=10, help='The number of classes')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    parser.add_argument('--use_gpu', nargs="?", type=str2bool, default=True,
                        help='A flag indicating whether we will use GPU acceleration or not')
    parser.add_argument('--model_name', type=str, default='Custom_07',
                        help='Type of model used')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Type of dataset used')                    
    parser.add_argument('--partition_name', type=str, default='partition_object_test',
                        help='Type of partition used')
    parser.add_argument('--distribution_name', type=str, default='balanced',
                        help='Type of distribution used')
    args = parser.parse_args()
    print(args)
    return args
