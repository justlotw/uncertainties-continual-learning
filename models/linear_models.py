import torch.nn as nn 

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout=0.5):
        super().__init__()
        hidden_lst = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                hidden_lst.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                hidden_lst.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        self.hidden = nn.ModuleList(hidden_lst)
        self.fc_out = nn.Linear(hidden_sizes[-1], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hidden_sizes = hidden_sizes
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.hidden:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.fc_out(x)
        return x
    
def get_linear_model(model_name, num_classes, **kwargs):
    if model_name not in VALID_MODEL_NAMES:
        raise ValueError('model_name must be one of {}'.format(VALID_MODEL_NAMES.keys()))
    model = VALID_MODEL_NAMES[model_name]
    return model(num_classes=num_classes, **kwargs)

VALID_MODEL_NAMES = {
    'mlp': MLP,
}