import os 

from argparse import ArgumentParser

import gdown
import zipfile
from wget import download

from torchvision.datasets import MNIST

VALID_DATASETS = ['domainnet', 'mnist', 'pacs']

def download_mnist(data_dir):
    MNIST(data_dir, download=True)

def download_pacs(data_dir):
    if os.path.exists(os.path.join(data_dir, 'pacs')):
        print('PACS dataset already downloaded.')
        return
    url = 'https://drive.google.com/file/d/19GH4W-o2QCvsce_33cFov8oJsnUz5zU8/view?usp=sharing'
    output_path = os.path.join(data_dir, 'pacs.zip')
    gdown.download(url, output_path, quiet=False, fuzzy=True)
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(output_path)
    os.rename(os.path.join(data_dir, 'kfold'), os.path.join(data_dir, 'pacs'))

def download_domainnet(data_dir):
    if os.path.exists(os.path.join(data_dir, 'domainnet')):
        print('DomainNet dataset already downloaded.')
        return
    os.makedirs(os.path.join(data_dir, 'domainnet'))    

    domains = {
        'clipart': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
        'infograph': 'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
        'painting': 'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
        'quickdraw': 'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
        'real': 'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
        'sketch': 'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip'
    }
    
    for domain, url in domains.items():
        download(url, os.path.join(data_dir, 'domainnet', f'{domain}.zip'))
        with zipfile.ZipFile(os.path.join(data_dir, 'domainnet', f'{domain}.zip'), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(data_dir, 'domainnet'))
        os.remove(os.path.join(data_dir, 'domainnet', f'{domain}.zip'))


if __name__ == '__main__':  
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='directory of the data folder')
    parser.add_argument('--dataset', type=str, help='dataset to download')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    if args.dataset.lower() == 'mnist':
        print(f'Downloading {args.dataset} dataset to {args.data_dir}...')
        download_mnist(args.data_dir)
    elif args.dataset.lower() == 'pacs':
        print(f'Downloading {args.dataset} dataset to {args.data_dir}...')
        download_pacs(args.data_dir)
    elif args.dataset.lower() == 'domainnet':
        print(f'Downloading {args.dataset} dataset to {args.data_dir}...')
        download_domainnet(args.data_dir)
    else:
        raise ValueError(f'dataset must be one of {VALID_DATASETS}')
    print('Done!')
    




