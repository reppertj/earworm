import os
from typing import Dict, List, Optional, Generator, Tuple
import pandas as pd
import numpy as np

CATEGORIES = {
    "Music genre": 0,
    "Musical instrument": 1,
    "Music mood": 2,
    "Music role": 3,
}


def make_ontology_df(ontology_path: str) -> pd.DataFrame:
    ontology = pd.read_json(ontology_path)
    return ontology.set_index("id")


def make_dataset_df(dataset_path: str) -> pd.DataFrame:
    return pd.read_csv(
        dataset_path, header=2, sep=",", skipinitialspace=True, quotechar='"'
    )


def yid_from_path(path: str) -> str:
    _, yid = os.path.split(path)
    tail = "ext"
    while tail:
        yid, tail = os.path.splitext(yid)
    return yid


def path_from_yid(root: str, yid: str, suffix: str) -> Optional[str]:
    path = os.path.join(root, yid, suffix)
    return path if os.path.isfile(path) else None


def paths_in_dir(dirname: str) -> Generator[str, None, None]:
    for root, _, files in os.walk(dirname):
        for name in files:
            yield os.path.join(root, name)


def yids_in_dir(dirname: str) -> Generator[Tuple[str, str], None, None]:
    for path in paths_in_dir(dirname):
        if path[-3:] == ".pt":
            yield yid_from_path(path), os.path.split(path)[1]


def oid_from_label(label, ontology: pd.DataFrame):
    return ontology[ontology.name == label].index.values[0]


def label_from_oid(ont_id: str, ontology: pd.DataFrame) -> str:
    return ontology.loc[ont_id, "name"]


def ontology_leaves(root_id, ontology: pd.DataFrame):
    children = []

    def dfs(current):
        child_ids = ontology.loc[current, "child_ids"]
        if not child_ids:
            children.append(current)
        else:
            for child in child_ids:
                dfs(child)

    dfs(root_id)
    return children


def make_tensor_df(
    dirname: str,
    category: str,
    category_id: int,
    source_df: pd.DataFrame,
    ontology: pd.DataFrame,
) -> pd.DataFrame:
    yids = {yid: path for yid, path in yids_in_dir(dirname)}
    common_df = source_df[source_df["# YTID"].isin(yids)]
    labels = ontology_leaves(oid_from_label(category, ontology), ontology)
    label_ints = {labels[n]: n for n in range(len(labels))}
    dfs = []
    for label in labels:
        matched_df = common_df[common_df.positive_labels.map(lambda pos: label in pos)][
            ["# YTID"]
        ]
        matched_df = matched_df.rename(columns={"# YTID": "ytid"})
        matched_df["path"] = matched_df["ytid"].map(yids)
        matched_df["label_n"] = label_ints[label]
        matched_df["category_n"] = category_id
        matched_df["label_ontology"] = label
        matched_df["label"] = label_from_oid(label, ontology)
        matched_df["category"] = category
        dfs.append(matched_df.reset_index(drop=True))
    ret = pd.concat(dfs).reset_index(drop=True)
    old_labels = sorted(ret["label_n"].unique())
    label_map = {old_labels[new]: int(new) for new in range(len(old_labels))}
    ret["label_n"] = ret["label_n"].map(label_map).astype("int")
    return ret


def make_combined_tensor_df(
    tensor_path: str, youtube_dataset: pd.DataFrame, ontology: pd.DataFrame
) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for CAT in CATEGORIES:
        dfs.append(
            make_tensor_df(tensor_path, CAT, CATEGORIES[CAT], youtube_dataset, ontology)
        )
    combined_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    ytids_unique: np.ndarray[str] = combined_df.ytid.unique()
    ytid_item_n_mapper: Dict[str, int] = {
        ytids_unique[i]: i for i in range(len(ytids_unique))
    }
    combined_df["item_n"] = combined_df.ytid.map(ytid_item_n_mapper)
    return combined_df


def stack_tensor_dfs(dfs):
    return pd.concat(dfs)


def save_tensor_df(path: str, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)