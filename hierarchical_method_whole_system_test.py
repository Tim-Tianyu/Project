import os
import tqdm
import torch
import CustomModels
from CustomModels import hierarchical_model
from experiment_builder import ExperimentBuilder
from CustomDataset import dataset_partition

def test(experiment_name, partitions, model_name, num_of_channels, num_classes, test_loader):
    frontier = [partitions]
    models = []
    classes = []
    count = 0
    while len(frontier) != 0:
        partition = frontier.pop(0)
        if (partition.has_children):
            frontier = frontier+partition.children[:]
        else:
            continue
        
        partition_folder = experiment_name+"/partition_"+str(count)
        state = torch.load(f=os.path.join(os.path.abspath(partition_folder), "saved_models/train_model_latest"))
        model = CustomModels.load_model(model_name, num_of_channels, num_classes)
        model.load_state_dict(state_dict=state['network'])
        models.append(model)
        classes.append(partition.classes)
        count = count+1
        
    model = hierarchical_model(classes, models)
    conv_experiment = ExperimentBuilder(network_model=model,
                                        experiment_name=experiment_name + "_whole_system_test",
                                        num_epochs=0,
                                        weight_decay_coefficient=0,
                                        use_gpu=args.use_gpu,
                                        continue_from_epoch=-1,
                                        train_data=test_loader, val_data=test_loader,
                                        test_data=test_loader)
    experiment_metrics, test_metrics = conv_experiment.run_experiment()
        
