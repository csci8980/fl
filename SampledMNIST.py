from typing import Callable, Optional

import torch
from torchvision import datasets


def split_count(count, div=9):
    """
    Split count to near equal
    :param count:15
    :param div: 4
    :return: [4, 4, 4, 3]
    """
    return [count // div + (1 if x < count % div else 0) for x in range(div)]


class SampledMNIST(datasets.MNIST):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 n_total: int = 100,
                 skew_label=None,
                 skew_prop=None) -> None:
        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform, download=download)

        # sample data
        n_per_label = int(n_total / 10)
        label_count_dict = {i: 0 for i in range(10)}
        if skew_label is None or skew_prop is None:
            print('Uniform sample dataset')
            for i in range(10):
                label_count_dict[i] = n_per_label  # every label has same count if uniform sample
        else:
            print('Skewed sample dataset')
            if skew_label not in range(10) or skew_prop not in range(11):
                ValueError('skew_label must in [0, .., 9] and skew_prop must in [0, .., 10]')
            else:
                skew_count = n_per_label * skew_prop
                rest_count = n_total - skew_count
                division = split_count(rest_count)
                for i in range(10):
                    if i == skew_label:
                        label_count_dict[i] = skew_count
                    else:
                        label_count_dict[i] = division.pop()

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

        # concat data and targets for all labels
        self.data = torch.cat(data_list)
        self.targets = torch.cat(targets_list)
