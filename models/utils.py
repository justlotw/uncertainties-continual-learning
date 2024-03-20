
import torch.nn as nn 

## UTILS ## 
def get_num_params(net):
    nb_param = 0
    nb_trainable_param = 0
    for param in net.parameters():
        if param.requires_grad:
            nb_trainable_param += param.numel()
        nb_param += param.numel()
    print(f'There are {nb_param} ({nb_param/1e6:.2f} million) parameters in this neural network')
    print(f'There are {nb_trainable_param} ({nb_trainable_param/1e6:.2f} million) trainable parameters in this neural network')
    return nb_param, nb_trainable_param
    

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.Linear):
        stdv = 1. / m.weight.size(1)**0.5
        nn.init.uniform_(m.weight.data, -stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)