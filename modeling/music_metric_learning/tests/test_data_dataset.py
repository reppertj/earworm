import os
import torch
import pandas as pd
from music_metric_learning.data.dataset import (
    CategorySpecificDataset,
    InverseMELNormalize,
    MELNormalize, MusicMetricDatamodule,
)


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
        dataset = CategorySpecificDataset(
            df,
            split="train",
            transform=None,
            val_size=0.1,
            test_size=0.1,
            random_state=42,
        )
        n_items = int(int(len(df) * 0.9) * 0.9)
        assert len(dataset) == n_items
        image, c_lab, t_lab = dataset[0]
        assert len(image.shape) == 4
        assert image.shape[0:2] == (2, 1)
        assert c_lab.shape == tuple()
        assert t_lab.shape == tuple()

    for category_n in range(4):
        category_specific_closure(df, category_n)

def test_datamodule(tensor_dir: str, train_csv: os.PathLike):
    dm = MusicMetricDatamodule(dataset_csv=train_csv, tensor_dir=tensor_dir, batch_size=64, m_per_class=8)
    dm.setup()
    def category_closure(category_n):
        loader = dm.train_dataloader(category_n)
        tensors, labels, track_labels = next(iter(loader))
        assert tensors.shape[:3] == (64, 2, 1)
        assert labels.shape == (64,)
        assert track_labels.shape == (64,)
        original_df = pd.read_csv(train_csv).set_index('item_n')
        for label, track_label in zip(labels, track_labels):
            gt = original_df.loc[track_label.item(), :]
            if isinstance(gt, pd.DataFrame):
                gt = gt[gt['category_n'] == category_n]
                assert label.item() in gt.label_n.to_numpy()
            else:
                assert label.item() == gt.label_n
    
    for category_n in range(4):
        category_closure(category_n)