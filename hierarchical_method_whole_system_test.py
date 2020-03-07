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
        
        partition_folder = experiment_name+"/partition_"+str(count)+"/saved_models"
        model = CustomModels.load_model(model_name, num_of_channels, num_classes)
        experiment = ExperimentBuilder(model, "", -1, None, None, None)
        _, best_index, _ = experiment.load_model(model_save_dir=partition_folder, model_idx="latest",model_save_name="train_model")
        experiment.load_model(model_save_dir=partition_folder, model_idx=best_index ,model_save_name="train_model")
        models.append(experiment.model)
        classes.append(partition.classes)
        count = count+1
        
    model = hierarchical_model(classes, models)
    conv_experiment = ExperimentBuilder(network_model=model,
                                        experiment_name=experiment_name + "/whole_system_test",
                                        num_epochs=-1,
                                        weight_decay_coefficient=0,
                                        use_gpu=True,
                                        continue_from_epoch=-1,
                                        train_data=test_loader, val_data=test_loader,
                                        test_data=test_loader)
    conv_experiment.save_model(model_save_dir=conv_experiment.experiment_saved_models,
                    # save model and best val idx and best val acc, using the model dir, model name and model idx
                    model_save_name="train_model", model_idx="best",
                    best_validation_model_idx=0,
                    best_validation_model_sensitivity=-1)
    experiment_metrics, test_metrics = conv_experiment.run_experiment()
        
def test2(experiment_name, models, classes, test_loader):
        model = hierarchical_model(classes, models)
        conv_experiment = ExperimentBuilder(network_model=model,
                                            experiment_name=experiment_name + "/whole_system_test",
                                            num_epochs=-1,
                                            weight_decay_coefficient=0,
                                            use_gpu=True,
                                            continue_from_epoch=-1,
                                            train_data=test_loader, val_data=test_loader,
                                            test_data=test_loader)
        conv_experiment.save_model(model_save_dir=conv_experiment.experiment_saved_models,
                        # save model and best val idx and best val acc, using the model dir, model name and model idx
                        model_save_name="train_model", model_idx="best",
                        best_validation_model_idx=0,
                        best_validation_model_sensitivity=-1)
        experiment_metrics, test_metrics = conv_experiment.run_experiment()