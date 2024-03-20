import os
import sys 
from shutil import copy2 
from datetime import datetime

from argparse import ArgumentParser

from datasets import TorchDataset, FolderDataset, AvalancheDataset, VALID_TORCH_DATASETS, VALID_FOLDER_DATASETS, VALID_AVALANCHE_DATASETS
from experiments import TestExperiment

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--num_experiences', type=int, default=5, help='number of experiences')
    parser.add_argument('--order', type=str, default=None, help='order of experiences')
    parser.add_argument('--train_pct', type=float, default=0.7, help='train percentage')   
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
    parser.add_argument('-a', '--augmentation', default=True, action='store_true', help='use data augmentation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--ckpt_dir', type=str, default='checkpoints', help='directory of saved models')
    parser.add_argument('--save_dir', type=str, default='results', help='directory to save results')
    parser.add_argument('-e', '--evaluation_method', type=str, help='evaluation method')

    args = parser.parse_args()

    if os.path.exists(args.save_dir):   
        raise Exception("Output folder already exists")
    os.makedirs(args.save_dir)

    for file in os.listdir(args.ckpt_dir):
        if file.startswith('training_output'):
            copy2(os.path.join(args.ckpt_dir, file), args.save_dir)

    original_stdout = sys.stdout
    with open(os.path.join(args.save_dir, 'testing_output.txt'), 'w') as f:
        sys.stdout = f
        print(f"Evaluation started at {datetime.now()}", end = '\n\n')
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
        
        test_expt = TestExperiment(dset.get_test_dataloaders())
        test_expt.test(args.ckpt_dir, args.save_dir, args.evaluation_method, verbose=True, overwrite=True)
        print()
        print(f"Evaluation ended at {datetime.now()}", end = '\n\n')
        sys.stdout = original_stdout
