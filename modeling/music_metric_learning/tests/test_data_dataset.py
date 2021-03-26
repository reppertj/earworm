import os
import torch
import pandas as pd
from math import floor
from music_metric_learning.data.dataset import CategorySpecificDataset, InverseMELNormalize, MELNormalize

def test_normalize_inverse_roundtrip():
    normalizer = MELNormalize()
    inverse_normalizer = InverseMELNormalize()
    torch.manual_seed(42)
    sgram = torch.randn(1, 128, 130)
    normalized = normalizer(sgram)
    inverse_normalized = inverse_normalizer(normalized)
    assert torch.allclose(sgram, inverse_normalized, atol=1e-6, rtol=1e-3)


def test_dataset(tensor_dir: str, train_csv: os.PathLike):
    df: pd.DataFrame = pd.read_csv(train_csv)
    df.path = tensor_dir + df.path
    
    def category_specific_closure(df, category_n):
        df = df[df.category_n == category_n]
        dataset = CategorySpecificDataset(df, split="train", transform=None, val_size=0.1, test_size=0.1, random_state=42)
        n_items = int(int(len(df) * 0.9) * 0.9)
        assert len(dataset) == n_items
        image, c_lab, t_lab = dataset[0]
        assert len(image.shape) == 3
        assert image.shape[0] == 2
        assert c_lab.shape == tuple()
        assert t_lab.shape == tuple()

    for category_n in range(4):
        category_specific_closure(df, category_n)
