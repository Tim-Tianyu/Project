import os
from setting import *
from arg_extractor import get_args
from train_evaluate_image_classification_system import run

def create(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        
def set_args(experiment_path, train_index_path, eval_index_path, args):
    create(experiment_path)
    args.experiment_name = experiment_path
    args.train_index_path = train_index_path
    args.eval_index_path = eval_index_path

args = get_args()
index_dir = "Indexs/MNIST"
experiment_dir = "experiments/MNIST"
args.num_epochs = 5
args.model_name = "Custom_05"
args.dataset_name = "MNIST"
args.image_num_channels = 1 

experiment_path = os.path.join(experiment_dir, BALANCE) 
train_index_path = os.path.join("Indexs", "MNIST_train.npy") 
eval_index_path = os.path.join("Indexs", "MNIST_eval.npy") 
set_args(experiment_path, train_index_path, eval_index_path, args)
run(args)

# for p in p_MNIST:
#     experiment_path = os.path.join(experiment_dir, "p_" + str(p), BALANCE) 
#     train_index_path = os.path.join(index_dir, "p_" + str(p), BALANCE +"_train.npy") 
#     eval_index_path = os.path.join(index_dir, "p_" + str(p), BALANCE +"_eval.npy") 
#     set_args(experiment_path, train_index_path, eval_index_path, args)
#     run(args)
        
index_dir = "Indexs/CIFAR10"
experiment_dir = "experiments/CIFAR10"
args.num_epochs = 50
args.model_name = "Custom_05"
args.dataset_name = "CIFAR10"
args.image_num_channels = 3

experiment_path = os.path.join(experiment_dir, BALANCE) 
train_index_path = os.path.join("Indexs", "CIFAR10_train.npy") 
eval_index_path = os.path.join("Indexs", "CIFAR10_eval.npy") 
set_args(experiment_path, train_index_path, eval_index_path, args)
run(args)

for p in p_CIFAR10:
    experiment_path = os.path.join(experiment_dir, "p_" + str(p), BALANCE) 
    train_index_path = os.path.join(index_dir, "p_" + str(p), BALANCE +"_train.npy") 
    eval_index_path = os.path.join(index_dir, "p_" + str(p), BALANCE +"_eval.npy") 
    set_args(experiment_path, train_index_path, eval_index_path, args)
    run(args)