import os
from shutil import rmtree

from argparse import ArgumentParser

if __name__ == '__main__':  
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='directory of the data folder')   
    parser.add_argument('--num_classes', type=int, default=50, help='number of classes')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise Exception("Data directory does not exist")
    
    data_folder = os.path.join(args.data_dir)
    domains = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]

    class_counter = {}
    for domain in domains:
        for class_name in os.listdir(os.path.join(data_folder, domain)):
            class_counter[class_name] = class_counter.get(class_name, 0) + len(os.listdir(os.path.join(data_folder, domain, class_name)))

    top_k_classes = sorted(class_counter, key=class_counter.get, reverse=True)[:args.num_classes]
    

    print(sorted(class_counter.items(), key = lambda x: x[1], reverse=True)[:args.num_classes])
    
    for domain in domains:
        for class_name in os.listdir(os.path.join(data_folder, domain)):
            if class_name not in top_k_classes:
                rmtree(os.path.join(data_folder, domain, class_name))

