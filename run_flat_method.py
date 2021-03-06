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

# for p in [1000, 2000, 4000]:
#     for i in range(len(permutations)):
#         for mu in mu_MNIST:
#             experiment_path = os.path.join(experiment_dir, "p_" + str(p), "mu_" + str(mu), FLAT, str(i+1)) 
#             train_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1) +".npy") 
#             eval_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1)+"_eval.npy") 
#             set_args(experiment_path, train_index_path, eval_index_path, args)
#             run(args)
#         
#         experiment_path = os.path.join(experiment_dir, "p_" + str(p), EXPONENTIAL, FLAT, str(i+1)) 
#         train_index_path = os.path.join(index_dir, "p_" + str(p), EXPONENTIAL, str(i+1) +".npy") 
#         eval_index_path = os.path.join(index_dir, "p_" + str(p), EXPONENTIAL, str(i+1)+"_eval.npy") 
#         set_args(experiment_path, train_index_path, eval_index_path, args)
#         run(args)
#         
#         experiment_path = os.path.join(experiment_dir, "p_" + str(p), LINEAR, FLAT, str(i+1)) 
#         train_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1) +".npy") 
#         eval_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1)+"_eval.npy") 
#         set_args(experiment_path, train_index_path, eval_index_path, args)
#         run(args)

index_dir = "Indexs/CIFAR10"
experiment_dir = "experiments/CIFAR10"
args.num_epochs = 50
args.model_name = "Custom_05"
args.dataset_name = "CIFAR10"
args.image_num_channels = 3
        
for p in [40]:
    i = 1
    for mu in [0.8]:
        experiment_path = os.path.join(experiment_dir, "p_" + str(p), "mu_" + str(mu), FLAT, str(i+1)) 
        train_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1) +".npy") 
        eval_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1)+"_eval.npy") 
        set_args(experiment_path, train_index_path, eval_index_path, args)
        run(args)
    
    experiment_path = os.path.join(experiment_dir, "p_" + str(p), EXPONENTIAL, FLAT, str(i+1)) 
    train_index_path = os.path.join(index_dir, "p_" + str(p), EXPONENTIAL, str(i+1) +".npy") 
    eval_index_path = os.path.join(index_dir, "p_" + str(p), EXPONENTIAL, str(i+1)+"_eval.npy") 
    set_args(experiment_path, train_index_path, eval_index_path, args)
    run(args)
    
    experiment_path = os.path.join(experiment_dir, "p_" + str(p), LINEAR, FLAT, str(i+1)) 
    train_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1) +".npy") 
    eval_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1)+"_eval.npy") 
    set_args(experiment_path, train_index_path, eval_index_path, args)
    run(args)
    for i in [2,3,4]:
        for mu in mu_MNIST:
            experiment_path = os.path.join(experiment_dir, "p_" + str(p), "mu_" + str(mu), FLAT, str(i+1)) 
            train_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1) +".npy") 
            eval_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1)+"_eval.npy") 
            set_args(experiment_path, train_index_path, eval_index_path, args)
            run(args)
        
        experiment_path = os.path.join(experiment_dir, "p_" + str(p), EXPONENTIAL, FLAT, str(i+1)) 
        train_index_path = os.path.join(index_dir, "p_" + str(p), EXPONENTIAL, str(i+1) +".npy") 
        eval_index_path = os.path.join(index_dir, "p_" + str(p), EXPONENTIAL, str(i+1)+"_eval.npy") 
        set_args(experiment_path, train_index_path, eval_index_path, args)
        run(args)
        
        experiment_path = os.path.join(experiment_dir, "p_" + str(p), LINEAR, FLAT, str(i+1)) 
        train_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1) +".npy") 
        eval_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1)+"_eval.npy") 
        set_args(experiment_path, train_index_path, eval_index_path, args)
        run(args)
        
p = 10
i = 0
experiment_path = os.path.join(experiment_dir, "p_" + str(p), LINEAR, FLAT, str(i+1)) 
train_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1) +".npy") 
eval_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1)+"_eval.npy") 
set_args(experiment_path, train_index_path, eval_index_path, args)
run(args)
for i in [1,2,3,4]:
    for mu in mu_MNIST:
        experiment_path = os.path.join(experiment_dir, "p_" + str(p), "mu_" + str(mu), FLAT, str(i+1)) 
        train_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1) +".npy") 
        eval_index_path = os.path.join(index_dir, "p_" + str(p), "mu_" + str(mu), str(i+1)+"_eval.npy") 
        set_args(experiment_path, train_index_path, eval_index_path, args)
        run(args)
    
    experiment_path = os.path.join(experiment_dir, "p_" + str(p), EXPONENTIAL, FLAT, str(i+1)) 
    train_index_path = os.path.join(index_dir, "p_" + str(p), EXPONENTIAL, str(i+1) +".npy") 
    eval_index_path = os.path.join(index_dir, "p_" + str(p), EXPONENTIAL, str(i+1)+"_eval.npy") 
    set_args(experiment_path, train_index_path, eval_index_path, args)
    run(args)
    
    experiment_path = os.path.join(experiment_dir, "p_" + str(p), LINEAR, FLAT, str(i+1)) 
    train_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1) +".npy") 
    eval_index_path = os.path.join(index_dir, "p_" + str(p), LINEAR, str(i+1)+"_eval.npy") 
    set_args(experiment_path, train_index_path, eval_index_path, args)
    run(args)
