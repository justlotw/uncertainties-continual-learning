import os

import splitfolders

from argparse import ArgumentParser

if __name__ == '__main__':  
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='directory of the data folder')   
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--ratio', type=float, default=0.7, help='train percentage')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise Exception("Data directory does not exist")
    
    data_folder = os.path.join(args.data_dir)
    domains = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]

    out_folder = data_folder + '_split'

    train_percent = args.ratio
    test_percent = 1 - train_percent

    for domain in domains:
        splitfolders.ratio(os.path.join(data_folder, domain), output=os.path.join(out_folder, domain), seed=args.seed, ratio=(train_percent, test_percent), move=False)