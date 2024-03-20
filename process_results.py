import os
import sys 

from argparse import ArgumentParser

import pandas as pd
import matplotlib.pyplot as plt

EXPERIENCES_NAMES = {
    'pacs': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'small_pacs': ['photo', 'art_painting', 'cartoon', 'sketch'],
    'permutedmnist': [],
    'pmnist': [],
    'domainnet': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
}

def process_results(file_path):
    df = pd.read_csv(file_path, dtype={'Train_Experience': int, "Test_Experience": int, "Epoch": int})
    df['MI_percent'] = df['Mutual_Info'] / df['Entropy']
    df['Forgetting'] = df.groupby(['Test_Experience'])['Accuracy'].transform(lambda x: x.max() - x)
    df['Total_Epoch'] = df['Train_Experience'] * (df['Epoch'].max()+1) + df['Epoch'] + 1
    df.loc[df.Train_Experience == -1, 'Total_Epoch'] = 0
    df = df.sort_values(by=["Test_Experience", "Train_Experience", "Epoch", "Step"]).reset_index(drop=True)
    return df

def get_experience_start(df):
    return df[df.Epoch==0].Total_Epoch.unique()
    
def plot_graph(df, metric, dataset, order=None,
               experience_start=[], save=False, save_path=None):
    num_experiences = df.Test_Experience.max() + 1
    name_exp = EXPERIENCES_NAMES[dataset.lower()]
    if name_exp == []:
        name_exp = [f"{i}" for i in range(num_experiences)]
    if num_experiences != len(name_exp):
        raise Exception("Number of experiences does not match number of experience names")
    if order is not None:
        order = order.upper()
        if set(order) == set('PACS'):
            order = [order.index(i) for i in 'PACS']
        elif set(order) == set('CIPQRS'):
            order = [order.index(i) for i in 'CIPQRS']
        name_exp = [name_exp[i] for i in order]

    plt.style.use(['seaborn-v0_8-colorblind'])
    for i in experience_start:
        plt.axvline(x=i, color="gray", linestyle="--", alpha=0.5)
    for exp in range(num_experiences):
        df_exp = df[df.Test_Experience == exp].reset_index(drop=True)
        plt.plot(df_exp.Total_Epoch, df_exp[metric], label=f"{exp}: {name_exp[exp]}", marker='.')
        if metric == "SNR_Softmax":
            plt.yscale('log')
    plt.legend()
    plt.title(metric)
    plt.ylabel(metric)
    if save:
        if save_path is None:
            raise Exception("Save path not specified")
        file_name = os.path.join(save_path, f"{metric.lower()}.png")
        plt.savefig(file_name)
        plt.close()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='results', help='directory of the results folder')
    parser.add_argument('--dataset', type=str, default='pacs', help='dataset to plot')
    parser.add_argument('--order', type=str, default=None, help='order of experiences')
    parser.add_argument('-ev', '--variance', default=False, action='store_true', help='plot variance')
    parser.add_argument('-ee', '--entropy', default=False, action='store_true', help='plot entropy')
    parser.add_argument('-esnr', '--snr', default=False, action='store_true', help='plot signal to noise ratio')
    parser.add_argument('-eu', '--uncertainty', default=False, action='store_true', help='plot uncertainty')
    parser.add_argument('-ec', '--calibration', default=False, action='store_true', help='plot calibration error')

    args = parser.parse_args()

    df = process_results(os.path.join(args.data_dir, 'results.csv'))
    experience_start = get_experience_start(df)
    metrics_lst = ['Accuracy', 'Loss',]
    if args.variance:
        metrics_lst.append('Variance')
    if args.entropy:
        metrics_lst.append('Entropy')
    if args.uncertainty:
        metrics_lst.append('Mutual_Info')
        metrics_lst.append('Aleatoric_Uncertainty')
    if args.snr:
        metrics_lst.append('SNR_Logit')
        metrics_lst.append('SNR_Softmax')

    if args.calibration:
        metrics_lst.append('ECE')
        metrics_lst.append('SCE')
        metrics_lst.append('UCE')

    for metric in metrics_lst:
        try:
            plot_graph(df, metric, args.dataset, args.order, experience_start, save=True, save_path=args.data_dir)
        except Exception as e:   
            print(f"Error plotting {metric}: {e}")

