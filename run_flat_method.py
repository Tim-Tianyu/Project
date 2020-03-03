import os
from setting import *
from arg_extractor import get_args
from train_evaluate_image_classification_system import run

def create(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

args = get_args()

index_dir = "Indexs/MNIST"
experiment_dir = "experiments/MNIST"
args.num_epochs = 5
args.model_name = "Custom_05"
args.dataset_name = "MNIST"
args.image_num_channels = 1 

for p in p_MNIST:
    for mu in mu_MNIST:
        for i in range(len(permutations)):
            experiment_path = os.path.join(experiment_dir, "p_" + str(p), "mu_" + str(mu), FLAT, str(i+1)) 
            train_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1) +".npy") 
            eval_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1)+"_eval.npy") 
            create(experiment_path)
            args.experiment_name = experiment_path
            args.train_index_path = train_index_path
            args.eval_index_path = eval_index_path
            run(args)