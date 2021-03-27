import os
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch._C import ClassType
from torch.utils.data.sampler import Sampler
from music_metric_learning.data.stats import MEANS as MEANS_LIST, STDS as STDS_LIST
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from music_metric_learning.samplers.per_class import ClassThenInstanceSampler


def make_np(x):
    return np.array(x).copy().astype("float32")


MEANS = torch.tensor(MEANS_LIST)
STDS = torch.tensor(STDS_LIST)


class MELNormalize(nn.Module):
    def __init__(self, mus: torch.Tensor = MEANS, stds: torch.Tensor = STDS):
        """Normalizes by mel band (i.e., along the H dimesion)
        of a mel spectrogram.

        Arguments:
            mus {torch.Tensor} -- tensor of means per mel band
            stds {torch.Tensor} -- tensor of stds per mel band
        """
        super().__init__()
        self.mus = mus
        self.stds = stds

    def forward(self, sgrams: torch.Tensor):
        return sgrams.transpose(-1, -2).sub(self.mus).div(self.stds).transpose(-1, -2)


class InverseMELNormalize(MELNormalize):
    def __init__(self, mus: torch.Tensor = MEANS, stds: torch.Tensor = STDS):
        """Inverse of MELNormalize

        Keyword Arguments:
            mus {torch.Tensor} -- tensor of means per mel band (default: {MEANS})
            stds {torch.Tensor} -- tensor of stds per mel band (default: {STDS})
        """
        stds_inv = 1 / (stds + 1e-7)
        mus_inv = -1 * mus * stds_inv
        super().__init__(mus=mus_inv, stds=stds_inv)

    def forward(self, sgrams: torch.Tensor):
        return super().forward(sgrams.clone())


class Identity(nn.Module):
    def __init__(self) -> None:
        """For a do nothing transform."""
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


def prepare_tensors_from_data(
    model: nn.Module,
    in_dir: str,
    file_txt: str,
    out_dir: str,
    n_workers: int = 2,
    suc_file="successes.txt",
    fail_file="failures.txt",
    suffix=".pt",
):
    from music_metric_learningdata.preprocessing import TensorPreprocesser  # type: ignore

    """Output paths will mirror input paths"""
    with open(file_txt, "r") as listing_file:
        in_paths_short = listing_file.readlines()
    in_paths = list(map(lambda p: os.path.join(in_dir, p.strip()), in_paths_short))
    out_paths = map(lambda p: os.path.join(out_dir, p.strip() + suffix), in_paths_short)
    processer = TensorPreprocesser(model, return_two=True)
    successes, failures = processer(in_paths, out_paths, n_workers=n_workers)
    with open(suc_file, "w") as out_file:
        out_file.writelines([p + "\n" for p in successes])
    with open(fail_file, "w") as out_file:
        out_file.writelines([p + "\n" for p in failures])


def tensor_files_in_dir(directory: str):
    result = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f[-3:] == ".pt":
                result.append(os.path.join(root, f))
    return result


class CategorySpecificDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        split: Literal["train", "val", "test"],
        transform: Optional[nn.Module],
        val_size: Union[int, float],
        test_size: Union[int, float],
        random_state: Union[np.random.Generator, int],
    ) -> None:
        """Dataset containing tuples of
        (tensors, class_label, track_label)

        Tensors are (2, H, W), containing two spectrograms per sample. In training mode,
        the order of these spectrograms is random. The class label is category-specific
        (class labels are not unique across categories), but the track label is globally unique
        across all datasets.

        Arguments:
            df {pd.DataFrame} -- DataFrame containing data paths and labels, created by
            `make_combined_tensor_df` with only a single category and with full paths to tensors
            in the `path` column
            split -- one of 'train, 'test', or 'val'
            transform {Optional[nn.Module]} -- transform to apply to tensor
            val_size {Union[int, float]} -- Size of validation set
            test_size {Union[int, float]} -- Size of test set
            random_state {Union[np.random.Generator, int]} -- Bit generator or seed for generator
        """
        super().__init__()
        self.df = df
        self.split = split
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.bg = np.random.default_rng(random_state)
        self.transform = transform
        self.class_label_map = (
            self.df[["label_n", "label"]].set_index("label_n").to_dict()["label"]
        )
        self._subset()

    def _subset(self):
        idxs = np.arange(len(self.df))
        train, test = train_test_split(
            idxs, test_size=self.test_size, random_state=self.random_state
        )
        train, val = train_test_split(
            train, test_size=self.val_size, random_state=self.random_state
        )
        if self.split == "train":
            self.df = self.df.iloc[train, :]
        elif self.split == "val":
            self.df = self.df.iloc[val, :]
        elif self.split == "test":
            self.df = self.df.iloc[test, :]
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self, index: Union[torch.Tensor, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(index, torch.Tensor):
            idx = int(index.item())
        else:
            idx = index

        item = self.df.iloc[idx]
        left: torch.Tensor
        right: torch.Tensor
        left, right = torch.load(item.path)

        class_label = torch.tensor(item.label_n, dtype=torch.long)
        track_label = torch.tensor(item.item_n, dtype=torch.long)

        if self.transform:
            with torch.no_grad():
                left = self.transform(left)
                right = self.transform(right)

        if self.split == "train":
            selector = self.bg.integers(2)
        else:
            selector = 0
        images = torch.stack(((left, right)[selector], (left, right)[1 - selector]))
        images = images.unsqueeze(1)  # Add channel dimension

        return images, class_label, track_label


class MusicMetricDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_csv: str,
        tensor_dir: str,
        m_per_class: int,
        batch_size: int = 128,
        test_size: Union[float, int] = 1024,
        val_size: Union[float, int] = 1024,
        transforms: Optional[Union[Type[MELNormalize], Type[nn.Module]]] = MELNormalize,
        random_state: Union[int, np.random.Generator] = 42,
        num_workers: int = 2,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.csv = dataset_csv
        self.m_per_class = m_per_class
        self.tensor_dir = tensor_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        if transforms is None:
            self.transform = Identity
        else:
            self.transforms = transforms
        self.random_state = random_state
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.is_setup = False

    def setup(self, stage=None):
        self.df = pd.read_csv(self.csv)
        self.df.path = self.tensor_dir + self.df.path
        self.bg = np.random.default_rng(self.random_state)
        self.is_setup = True
    
    def train_dataloader(self, dataset_idx, batch_size=None):
        batch_size = batch_size or self.batch_size

        dataset = CategorySpecificDataset(
            self.df[self.df.category_n == dataset_idx],
            "train",
            self.transforms(),
            self.val_size,
            self.test_size,
            self.random_state,
        )

        sampler = ClassThenInstanceSampler(
            labels=dataset.df.label_n,
            m_per_class=self.m_per_class,
            batch_size=batch_size,
            generator=self.bg,
            use_length=100_000,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, dataset_idx, batch_size=None):
        batch_size = batch_size or self.batch_size

        dataset = CategorySpecificDataset(
            self.df[self.df.category_n == dataset_idx],
            "train",
            self.transforms(),
            self.val_size,
            self.test_size,
            self.random_state,
        )

        sampler = ClassThenInstanceSampler(
            labels=dataset.df.label_n,
            m_per_class=self.m_per_class,
            batch_size=batch_size,
            generator=self.bg,
            use_length=100_000,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, dataset_idx, batch_size=None):
        batch_size = batch_size or self.batch_size

        dataset = CategorySpecificDataset(
            self.df[self.df.category_n == dataset_idx],
            "train",
            self.transforms(),
            self.val_size,
            self.test_size,
            self.random_state,
        )

        sampler = ClassThenInstanceSampler(
            labels=dataset.df.label_n,
            m_per_class=self.m_per_class,
            batch_size=batch_size,
            generator=self.bg,
            use_length=100_000,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class TripletDataset(Dataset):
    def __init__(self, df, split, transform, val_size, test_size, random_state: int):
        """
        df should have columns 'path' and 'label_n'
        """
        super().__init__()
        self.df = df
        self.label_map = (
            self.df[["label_n", "label"]].set_index("label_n").to_dict()["label"]
        )
        self.n_classes = max(self.label_map.keys()) + 1
        self.split = split
        self.transform = transform
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.bg = np.random.default_rng(random_state)
        self.subset()

    def subset(self):
        idxs = np.arange(len(self.df))
        # To prevent leakage, these splits must be the same every time, so we apply the random seed
        # directly here instead of using a global bit generator. Ideally, we would siphon off test
        # data entirely into a different directory.
        train, test = train_test_split(
            idxs, test_size=self.test_size, random_state=self.random_state
        )
        train, val = train_test_split(
            train, test_size=self.val_size, random_state=self.random_state
        )
        if self.split == "train":
            self.idxs = train
        elif self.split == "val":
            self.idxs = val
        elif self.split == "test":
            self.idxs = test

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(
        self, idx: Union[torch.Tensor, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.item()  # type: ignore

        item = self.df.iloc[idx]
        left, right = torch.load(item.path)
        label = torch.tensor(item.label_n, dtype=torch.long)  # type: ignore

        if self.transform:
            with torch.no_grad():
                left = self.transform(left)
                right = self.transform(right)

        if self.split == "train":
            selector = self.bg.integers(2)
        else:
            selector = 0
        stacked = torch.stack(((left, right)[selector], (left, right)[1 - selector]))

        return ((left, right)[selector].unsqueeze(0), label, stacked.unsqueeze(1))


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class DummyDataModule(pl.LightningDataModule):
    def __init__(self):
        pass

    def setup(self, stage=None):
        return None


class ConditionalTripletDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        category_csvs: List[str],
        dataset_dir: str,
        batch_size=128,  # TODO: Change defaults outside dev set
        test_size=1024,
        val_size=1024,
        transforms=MELNormalize,
        random_state=42,
        num_workers=2,
        pin_memory=True,
    ):
        """Primary entry point to create train/validate dataloaders for multi-category datasets.

        Arguments:
            category_dfs {List[pd.DataFrame]} -- one DataFrame per dataset, each with "path" and
            "label_n" columns.
        """
        super().__init__()
        self.csv_paths = category_csvs
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.transforms = transforms
        self.random_state = random_state
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.transforms = MELNormalize
        dfs = [pd.read_csv(path) for path in self.csv_paths]
        for i in range(len(dfs)):
            dfs[i]["path"] = self.dataset_dir + dfs[i]["path"]
        self.dfs = dfs

    def train_dataloader(self, dataset_idx, batch_size=None):
        batch_size = batch_size or self.batch_size

        dataset = TripletDataset(
            self.dfs[dataset_idx],
            "train",
            self.transforms(),
            self.val_size,
            self.test_size,
            self.random_state,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, dataset_idx, batch_size=None):
        batch_size = batch_size or self.batch_size
        dataset = TripletDataset(
            self.dfs[dataset_idx],
            "val",
            self.transforms(),
            self.val_size,
            self.test_size,
            self.random_state,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, dataset_idx, batch_size=None):
        batch_size = batch_size or self.batch_size
        dataset = TripletDataset(
            self.dfs[dataset_idx],
            "test",
            self.transforms(),
            self.val_size,
            self.random_state,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


class SgramDataset(Dataset):
    def __init__(self, paths, split, transform, val_size, test_size, random_state=42):
        """val_size, test_size can be int or float between 0 and 1
        directory is root directory of audio samples
        dataframe should contain a `path` column pointing to preprocessed tensors in dataset
        """
        super().__init__()
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

        return tensor.unsqueeze(0)


class RunningStats(object):
    """Computes running mean and standard deviation
    Url: https://gist.github.com/wassname/a9502f562d4d3e73729dc5b184db2501
    Adapted from:
        *
        <http://stackoverflow.com/questions/1174984/how-to-efficiently-\
calculate-a-running-standard-deviation>
        * <http://mathcentral.uregina.ca/QQ/database/QQ.09.02/carlos1.html>
        * <https://gist.github.com/fvisin/5a10066258e43cf6acfa0a474fcdb59f>
        
    Usage:
        rs = RunningStats()
        for i in range(10):
            rs += np.random.randn()
            print(rs)
        print(rs.mean, rs.std)
    """

    def __init__(self, n=0.0, m=None, s=None):
        self.n = n
        self.m = m
        self.s = s

    def clear(self):
        self.n = 0.0

    def push(self, x, per_dim=False):
        x = make_np(x)
        # process input
        if per_dim:
            self.update_params(x)
        else:
            for el in x.flatten():
                self.update_params(el)

    def update_params(self, x):
        self.n += 1
        if self.n == 1:
            self.m = x
            self.s = 0.0
        else:
            prev_m = self.m.copy()
            self.m += (x - self.m) / self.n
            self.s += (x - prev_m) * (x - self.m)

    def __add__(self, other):
        if isinstance(other, RunningStats):
            sum_ns = self.n + other.n
            prod_ns = self.n * other.n
            delta2 = (other.m - self.m) ** 2.0
            return RunningStats(
                sum_ns,
                (self.m * self.n + other.m * other.n) / sum_ns,
                self.s + other.s + delta2 * prod_ns / sum_ns,
            )
        else:
            self.push(other)
            return self

    @property
    def mean(self):
        return self.m if self.n else 0.0

    def variance(self):
        return self.s / (self.n) if self.n else 0.0

    @property
    def std(self):
        return np.sqrt(self.variance())

    def __repr__(self):
        return "<RunningMean(mean={: 2.4f}, std={: 2.4f}, n={: 2f}, m={: 2.4f}, s={: 2.4f})>".format(
            self.mean, self.std, self.n, self.m, self.s
        )

    def __str__(self):
        return "mean={: 2.4f}, std={: 2.4f}".format(self.mean, self.std)


class Transpose(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: torch.Tensor):
        return tensor.t()


class MinMaxScale(nn.Module):
    """Scale input to a given range. Input should not be batched."""

    def __init__(self, range=(-1, 1)):
        super().__init__()
        self.min = range[0]
        self.max = range[1]

    def forward(self, sgrams: torch.Tensor):
        sgrams_std = (sgrams - sgrams.min()) / (sgrams.max() - sgrams.min())
        return sgrams_std * (self.max - self.min) + self.min


class SgramDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=64,
        test_size=2048,
        val_size=2048,
        random_state=42,
        num_workers=2,
        pin_memory=True,
    ):
        """Primary entry point to create train/test/validate dataloaders for spectrogram datasets.

        Arguments:
            data_dir {str} -- path to directory containing db-scaled mel spectrograms with ".pt"
            extensions (from `torch.save`), possibly in subdirectories. No other ".pt" files
            should be in this directory.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        paths = tensor_files_in_dir(self.data_dir)
        idxs = np.arange(len(paths))
        np.random.seed(self.random_state)
        np.random.shuffle(idxs)
        self.paths = list(np.array(paths, dtype="object")[idxs])
        transforms = Compose([Transpose(), MELNormalize()])
        if stage == "fit" or stage is None:
            self.train = SgramDataset(
                self.paths,
                "train",
                transforms,
                self.val_size,
                self.test_size,
                self.random_state,
            )
            self.val = SgramDataset(
                self.paths,
                "val",
                transforms,
                self.val_size,
                self.test_size,
                self.random_state,
            )
        if stage == "test" or stage is None:
            self.test = SgramDataset(
                self.paths,
                "test",
                transforms,
                self.val_size,
                self.test_size,
                self.random_state,
            )

    def train_dataloader(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
