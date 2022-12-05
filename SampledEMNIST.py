from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import zeta
from torchvision import datasets


# train_data = datasets.EMNIST(root='data', split='byclass', train=True, download=True)
# test_data = datasets.EMNIST(root='data', split='byclass', train=False, download=True)

def get_zipf(a=2.0, n_label=62):
    k = np.arange(1, n_label + 1)
    d = (k ** -a) / zeta(a)
    sum = d.sum()
    nd = d / sum
    return nd


def plot_zipf(a=2.0, n_label=62):
    k = np.arange(1, n_label + 1)
    nd = get_zipf(a=a, n_label=n_label)
    plt.plot(k - 1, nd, '.-', label='Zipf PDF')
    plt.plot(k - 1, np.repeat(1 / n_label, n_label), '.-', label='Even PDF')
    plt.semilogy()
    plt.grid(alpha=0.4)
    plt.legend()
    plt.title(f'Zipf PDF, a={a}')
    plt.savefig('zipf.png', dpi=360)


def get_zipf_label_count(n_label, n_total):
    zipf = get_zipf(n_label=n_label)
    label_count = zipf * n_total
    label_count = list(map(int, label_count))
    label_count_dict = {i: label_count[i] for i in range(n_label)}
    return label_count_dict


class SampledEMNIST(datasets.EMNIST):
    def __init__(self,
                 root: str,
                 split: str = 'byclass',
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = True,
                 iid: bool = True,
                 n_total: int = 1000,
                 n_label: int = 62) -> None:
        super().__init__(
            root=root,
            split=split,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )

        # generate label count dict
        if iid:
            n_per_label = int(n_total / n_label)
            print('Generate iid dataset')
            label_count_dict = {i: n_per_label for i in range(n_label)}
        else:
            print('Generate niid dataset')
            label_count_dict = get_zipf_label_count(n_label=n_label, n_total=n_total)

        print(label_count_dict)

        # sample dataset based on label count dict
        data_list = []
        targets_list = []
        for label in label_count_dict:  # for each label
            label_count = label_count_dict[label]
            idx_label = self.targets == label  # get index of the label
            data_label, targets_label = self.data[idx_label], self.targets[idx_label]  # get data and targets of the label
            idx_rand = torch.randperm(len(targets_label))[:label_count]  # get random index of 'label_count'
            data_rand, targets_rand = data_label[idx_rand], targets_label[idx_rand]  # get random data and label
            data_list.append(data_rand)
            targets_list.append(targets_rand)

            # # get only first n data
            # first_n_data = data_label[:label_count]
            # first_n_target = targets_label[:label_count]
            # data_list.append(first_n_data)
            # targets_list.append(first_n_target)

        # concat data and targets for all labels
        self.data = torch.cat(data_list)
        self.targets = torch.cat(targets_list)


if __name__ == '__main__':
    # 10 clients

    # train, test
    # distribution: even, zipf
    # count: train: 6000, 600
    # count: test : 1000, 100
    # name: <tt>_<dist>_<count>_<#>.pkl
    from torchvision.transforms import ToTensor
    import pickle

    for dist in ['even', 'zipf']:
        if dist == 'even':
            iid = True
        else:
            iid = False
        for total in [1000, 100]:
            for i in range(10):
                a = SampledEMNIST(root='data', train=False, transform=ToTensor(), download=True, iid=iid, n_total=total)
                name = f'test_{dist}_{total}_{i}.pkl'
                print(name)
                with open(f'data/SampledEMNIST/pickle/{name}', 'wb') as file:
                    pickle.dump(a, file)
