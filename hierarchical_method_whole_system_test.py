def test(experiment_name, experiment_builder, partitions, model_name, num_of_channels, num_classes):
    frontier = [partitions]
    models = []
    count = 0
    while len(frontier) != 0:
        partition = frontier.pop(0)
        if (partition.has_children):
            frontier = frontier+partition.children[:]
            partition.num = count
        else:
            continue
        
        partition_folder = experiment_name+"/partition_"+str(count)
        
        state = torch.load(f=os.path.join(os.path.abspath(partition_folder), "saved_models/train_model_latest"))
        model = load_model(model_name, num_of_channels, num_classes)
        model.load_state_dict(state_dict=state['network'])
        models.append(model)
        
        count = count+1
        
class hierarchical_model(nn.Module):
    def __init__(self, partitions, models):
        super(hierarchical_model, self).__init__()
        self.classifier = nn.Linear(4, 2)
        self.partitions = partitions
    
    def forward(self, x):
        partition = partitions
        
        while (partition.has_children):
            x = models[partition.num](x)
            partition = partition.children[x]
            
        return partition.