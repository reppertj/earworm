from music_metric_learning.samplers.per_class import ClassThenInstanceSampler
import torch
import numpy as np
import pytest


def test_class_then_instance_sampler():
    torch_gen = torch.manual_seed(42)
    np_gen = np.random.default_rng(42)
    batch_size = 128
    m_per_class = 8
    labels = torch.randint(0, 10, size=(1000,), generator=torch_gen)
    sampler = ClassThenInstanceSampler(
        labels=labels, m_per_class=m_per_class, batch_size=batch_size, generator=np_gen
    )
    assert sampler.use_length == (1000 - (1000 % 128))
    assert sampler.classes_per_batch == 128 // 8
    assert len(sampler.mapping[0]) == 1000 - labels.count_nonzero()
    assert len(sampler.mapping) == 10
    assert sampler.replace_classes  # 128 // 8 == 16 classes per batch > 10 classes

    no_replace_labels = torch.randint(0, 16, size=(1000,), generator=torch_gen)
    sampler_without_replacement = ClassThenInstanceSampler(
        labels=no_replace_labels,
        m_per_class=m_per_class,
        batch_size=batch_size,
        generator=np_gen,
    )
    assert sampler_without_replacement.replace_classes == False

    iterable = iter(sampler_without_replacement)
    for _ in range(7):
        batch = [next(iterable) for _ in range(batch_size)]
        batch_labels = no_replace_labels[batch]
        unique_labels, counts = torch.unique(batch_labels, return_counts=True)
        assert len(unique_labels) == batch_size // m_per_class
        assert torch.all(counts == m_per_class)

    with pytest.raises(StopIteration):
        batch = next(iterable)
