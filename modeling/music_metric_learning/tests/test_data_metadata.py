import sys

from ..data.metadata import (
    make_dataset_df,
    make_ontology_df,
    make_combined_tensor_df,
)
import os


def test_still_do_it(tensor_dir):
    print(tensor_dir)
    assert tensor_dir is not None


def test_make_dataset_df(audioset_csv):
    df = make_dataset_df(audioset_csv)
    assert all(
        df.columns == ["# YTID", "start_seconds", "end_seconds", "positive_labels"]
    )


def test_make_tensor_dfs(tensor_dir, audioset_csv, audioset_json):
    dataset = make_dataset_df(audioset_csv)
    ontology = make_ontology_df(audioset_json)
    print(tensor_dir)
    df = make_combined_tensor_df(tensor_dir, dataset, ontology)
    assert all(
        df.columns
        == [
            "ytid",
            "path",
            "label_n",
            "category_n",
            "label_ontology",
            "label",
            "category",
            "item_n",
        ]
    )
    for category_n in df.category_n.unique():
        labels = df[df.category_n == category_n].label_n.unique()
        n_labels = len(labels)
        # Labels should increase monotonically from 0
        assert sorted(labels) == list(range(n_labels))
