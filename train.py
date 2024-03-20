import os
import sys 
from datetime import datetime
from argparse import ArgumentParser

import torch

from models.utils import get_num_params
from models.convolution_models import VALID_MODEL_NAMES as VALID_CONV_MODEL_NAMES
from models.linear_models import VALID_MODEL_NAMES as VALID_LINEAR_MODEL_NAMES
from models.convolution_models import get_conv_model
from models.linear_models import get_linear_model

from datasets import AvalancheDataset, TorchDataset, FolderDataset, VALID_TORCH_DATASETS, VALID_FOLDER_DATASETS, VALID_AVALANCHE_DATASETS
from experiments import TrainExperiment, TrainAvalancheExperiment

from avalanche.training.plugins import ReplayPlugin, EWCPlugin, SynapticIntelligencePlugin, CWRStarPlugin, LwFPlugin


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--num_experiences', type=int, default=5, help='number of experiences')
    parser.add_argument('--order', type=str, default=None, help='order of experiences')
    parser.add_argument('--train_pct', type=float, default=0.7, help='train percentage')   
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('-a', '--augmentation', default=False, action='store_true', help='use data augmentation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--model', type=str, help='model name')
    parser.add_argument('--model_num', type=int, default=0, help='model number')
    parser.add_argument('--hidden_sizes', type=int, nargs='*', default=None, help='hidden sizes for linear models')
    parser.add_argument('--replay_prop', type=float, default=0, help='replay proportion (0, 1) for replay strategy')
    parser.add_argument('--ewc_lambda', type=float, default=0, help='lambda for ewc strategy')
    parser.add_argument('--si_lambda', type=float, default=0, help='lambda for synaptic intelligence strategy')
    parser.add_argument('-cwr', '--cwr_star', default=False, action='store_true', help='use cwr star strategy')
    parser.add_argument('-lwf', default=False, action='store_true', help='use lwf strategy')
    # parser.add_argument('--n_unfreeze', '--layers_to_unfreeze', type=int, default=0, help='number of blocks of layers to unfreeze')
    # parser.add_argument('-r', '--reset_weights', '--reset_unfrozen_weights', default=False, action='store_true', help='reset unfrozen weights')

    parser.add_argument('--save_dir', type=str, default='checkpoints', help='directory to save models')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--model_save_interval', type=int, default=None, help='number of epochs between model saves')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()

    original_stdout = sys.stdout
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    with open(os.path.join(args.save_dir, f'training_output_{args.model_num}.txt'), 'a') as f:
        sys.stdout = f
        print(f'Training started at: {datetime.now()}', end = '\n\n')
        
        print(args, end = '\n\n')
        
        use_avalanche = False
        if args.dataset.upper() in VALID_AVALANCHE_DATASETS:
            dset = AvalancheDataset(args.data_dir, args.dataset, args.num_experiences, args.batch_size, args.num_workers, args.seed, args.order)
            use_avalanche = True
        elif args.dataset.upper() in VALID_TORCH_DATASETS:
            dset = TorchDataset(args.data_dir, args.dataset, args.num_experiences, batch_size=args.batch_size, num_workers=args.num_workers, augmentation=args.augmentation, seed=args.seed)
        elif args.dataset.upper() in VALID_FOLDER_DATASETS:
            dset = FolderDataset(args.data_dir, args.dataset, order=args.order, train_pct=args.train_pct, batch_size=args.batch_size, num_workers=args.num_workers, augmentation=args.augmentation, seed=args.seed)
        else:
            raise ValueError(f'dataset must be one of {VALID_TORCH_DATASETS} or {VALID_FOLDER_DATASETS} or {VALID_AVALANCHE_DATASETS}')
        print('Dataset details:')
        dset.print_dataset_details()
        print()

        if args.model in VALID_CONV_MODEL_NAMES:
            model = get_conv_model(args.model, 
                                   dset.get_num_classes(), 
                                #    layers_to_freeze=args.n_unfreeze, 
                                #    reset_weights=args.reset_weights, 
                                #    pretrained=False
                                   )
        elif args.model in VALID_LINEAR_MODEL_NAMES:
            input_size = dset.get_input_shape() 
            model = get_linear_model(args.model, dset.get_num_classes(), 
                                     input_size=torch.Size(input_size).numel(), 
                                     hidden_sizes=args.hidden_sizes)
        else:
            raise ValueError(f'model must be one of {VALID_CONV_MODEL_NAMES.keys()} or {VALID_LINEAR_MODEL_NAMES.keys()}')

        print('Model details:')
        print(model)
        print()
        get_num_params(model)
        print()

        print('Commencing training...', flush=True)
        if args.model_save_interval is None:
            model_save_interval = args.epochs // 5
        else:
            model_save_interval = args.model_save_interval

        if use_avalanche:
            sys.stdout = original_stdout
            scenario = dset.get_scenario()
            train_expt = TrainAvalancheExperiment(scenario.train_stream)
            
            plugins = []
            if args.replay_prop > 0:
                plugins.append(ReplayPlugin(mem_size=int(len(scenario.original_train_dataset) // scenario.n_experiences * args.replay_prop), batch_size=args.batch_size//2))
            if args.ewc_lambda > 0:
                plugins.append(EWCPlugin(args.ewc_lambda))
            if args.si_lambda > 0:
                plugins.append(SynapticIntelligencePlugin(args.si_lambda, eps=0.1))
            if args.cwr_star:
                plugins.append(CWRStarPlugin(model, freeze_remaining_model=False))
            if args.lwf:
                plugins.append(LwFPlugin())
            train_expt.train(model, args.model_num, plugins, args.save_dir, args.epochs, args.batch_size, model_save_interval, args.lr, args.seed, args.num_workers)
            sys.stdout = f
        else:
            train_expt = TrainExperiment(dset.get_train_dataloaders())
            train_expt.train(model, args.model_num, args.save_dir, n_epochs=args.epochs, model_save_interval=args.model_save_interval, lr=args.lr, seed=args.seed)
        print()
        print(f'Training completed at: {datetime.now()}', file=f, flush=True)
        sys.stdout = original_stdout    
