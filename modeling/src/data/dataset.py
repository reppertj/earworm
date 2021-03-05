import os
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

from src.data.preprocessing import TensorPreprocesser  # type: ignore


def prepare_tensors_from_data(
    in_dir: str,
    file_txt: str,
    out_dir: str,
    n_workers: int=2,
    suc_file="successes.txt",
    fail_file="failures.txt",
    suffix=".pt",
):
    """Output paths will mirror input paths"""
    with open(file_txt, "r") as listing_file:
        in_paths_short = listing_file.readlines()
    in_paths = list(map(lambda p: os.path.join(in_dir, p.strip()), in_paths_short))
    out_paths = map(lambda p: os.path.join(out_dir, p.strip() + suffix), in_paths_short)
    processer = TensorPreprocesser()
    successes, failures = processer(in_paths, out_paths, n_workers=n_workers)
    with open(suc_file, "w") as out_file:
        out_file.writelines([p + '\n' for p in successes])
    with open(fail_file, "w") as out_file:
        out_file.writelines([p + '\n' for p in failures])


class SgramDataset(Dataset):
    def __init__(
        self, paths, split, transform, val_size, test_size, random_state=42
    ):
        """val_size, test_size can be int or float between 0 and 1
        directory is root directory of audio samples
        dataframe should contain a `path` column pointing to preprocessed tensors in dataset
        """
        self.paths = paths
        self.split = split
        self.transform = transform
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.subset(paths)

    def subset(self, paths):
        idxs = np.arange(len(paths))
        if self.split not in {"train", "test", "val"}:
            raise ValueError
        train, test = train_test_split(
            idxs, test_size=self.test_size, random_state=self.random_state
        )
        if self.split == "test":
            self.idxs = test
            return
        train, val = train_test_split(
            train, test_size=self.val_size, random_state=self.random_state
        )
        if self.split == "train":
            self.idxs = train
        elif self.split == "val":
            self.idxs = val

    def __len__(self):
        return self.idxs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        path = self.paths[self.idxs[idx]]
        tensor = torch.load(path)

        if self.transform:
            with torch.no_grad():
                tensor = self.transform(tensor)

        return tensor


class SgramDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, files, test_size, val_size, random_state=42):
        """[summary]

        Arguments:
            data_dir {str} -- root directory of dataset
            files {str} -- txt file containing one path to a .pt file per line
        """
        super().__init__()
        self.data_dir = data_dir
        self.files = files
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def setup(self):
        # Make assignments here
        
        # TODO: Pretrain and setup normalizer
        
        return

    def train_dataloader(self):
        train_split

    def val_dataloader(self):
        return

    def test_dataloader(self):
        pass
