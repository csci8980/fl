import argparse

import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':
    # read argument
    parser = argparse.ArgumentParser()
    parser.add_argument("title")
    args = parser.parse_args()
    title = args.title

    # read csv
    df = pd.read_csv(f'{title}.csv')

    # plot
    epoch = df.shape[1] - 5
    client = len(df)
    for c in range(client):
        c_label = f'{df.at[c, "label_dist"]}_{df.at[c, "data_count"]}'
        dist, count = c_label.split('_')
        if dist == 'zipf' or count == 'less':
            plt.plot(range(epoch), df.loc[c, [str(i) for i in range(epoch)]], 'x-r', alpha=1, label=c_label)
        else:
            plt.plot(range(epoch), df.loc[c, [str(i) for i in range(epoch)]], '.-', alpha=0.6, label=c_label)

    plt.grid(alpha=0.4)
    plt.legend(loc='lower right', ncols=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks([i for i in range(epoch)])
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.title(title.split('_')[0])
    plt.savefig(f'{title}.png', dpi=360)
