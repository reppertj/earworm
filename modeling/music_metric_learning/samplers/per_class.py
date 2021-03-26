from collections import Counter, defaultdict
from typing import DefaultDict, Dict, Iterator, List, Optional, Union

import numpy as np
import torch
from torch._C import Value
from torch.utils.data.sampler import Sampler, T_co


def label_index_mapping(labels: np.ndarray) -> DefaultDict[int, np.ndarray]:
    """Given an ndarray of labels, return a dict mapping labels to ndarrays of their indices"""
    mapping = defaultdict(list)
    for idx, label in enumerate(labels):
        mapping[label].append(idx)
    for k, v in mapping.items():
        mapping[k] = np.array(v)
    return mapping


class ClassThenInstanceSampler(Sampler):
    def __init__(
        self,
        labels: Union[torch.Tensor, List[int], np.ndarray],
        m_per_class: int,
        batch_size: int,
        generator: Optional[Union[int, np.random.Generator]] = None,
        use_length: Optional[int] = None,
    ) -> None:
        """Samples classes randomly then m samples per class for every iteration. If the batch size
        is larger than m * num_classes, classes will be sampled with
        replacement, taking care to avoid returning identical samples in a batch

        Arguments:
            labels {Union[torch.Tensor, List[int], np.ndarray]} -- 1d array-like containing labels
             for the dataset
            m_per_class {int} -- The number of samples per class
            batch_size {int}

        Keyword Arguments:
            generator {Optional[Union[int, np.random.Generator]]} -- If int, used as random seed
            use_length {Optional[int]} -- If None, the sampler will have the same length as the
             dataset (default: {None})

        Raises:
            ValueError: If batch_size is not evenly divisible by m_per_class or if batch_size is
            larger than dataset
        """
        if isinstance(labels, torch.Tensor):
            self.labels: np.ndarray = labels.numpy()
        elif isinstance(labels, list):
            self.labels = np.array(labels)
        else:
            self.labels = labels
        super().__init__(self.labels)
        self.m_per_class = m_per_class
        self.batch_size = batch_size
        self.mapping = label_index_mapping(self.labels)
        self.classes: np.ndarray = np.array(list(self.mapping.keys()))
        if use_length is None:
            self.use_length: int = self.labels.shape[0]
        else:
            self.use_length = use_length
        self.use_length -= (
            self.use_length % self.batch_size
        )  # Ensure batches divide length evenly
        if self.batch_size > self.use_length:
            raise ValueError("batch_size cannot be larger than epoch length")
        if self.batch_size % self.m_per_class != 0:
            raise ValueError("batch_size must be divisible by m_per_class")

        self.classes_per_batch = self.batch_size // self.m_per_class
        self.replace_classes = self.classes_per_batch > len(self.classes)
        self.generator: np.random.Generator = np.random.default_rng(generator)

    def __len__(self) -> int:
        return self.use_length

    @property
    def num_batches(self) -> int:
        return (
            self.use_length // self.batch_size
            if self.batch_size < self.use_length
            else 1
        )

    def __iter__(self) -> Iterator[int]:
        idxs = [0] * len(self)
        i = 0
        for _ in range(self.num_batches):
            classes = self.generator.choice(
                self.classes, self.classes_per_batch, replace=self.replace_classes
            )
            class_counts: Dict[int, int] = Counter(classes)
            for label, multiple in class_counts.items():
                items_in_class = self.mapping[label]
                m_to_sample = multiple * self.m_per_class
                replace = len(items_in_class) < m_to_sample
                choices: np.ndarray = self.generator.choice(
                    items_in_class, m_to_sample, replace=replace
                )
                idxs[i : i + m_to_sample] = choices
                i += m_to_sample
            # breakpoint()
        return iter(idxs)
