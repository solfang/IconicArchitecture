"""
Provides a central interface for reading files and datasets
"""

import pandas as pd
import os
from ast import literal_eval
from pathlib import Path

root_folder = Path("../data")

visitor_folder = root_folder / "visitor_numbers"
city_datasets_folder = root_folder / "city_datasets"
image_folder = root_folder / "images"
train_test_folder = root_folder / "images" / "_train_test"
events_programme_folder = root_folder / "events_programme"
events_external_folder = root_folder / "events_external"


file_paths = {
    "posts": root_folder / "posts.csv",
    "posts_city": root_folder / "posts_city.csv",
    "posts_feather": root_folder / "posts.feather",
    "image_labels": root_folder / "image_labels.csv",
    "projects": root_folder / "projects.csv",
}


# -------Util-------#
def read_data_cascade(csv_file_path, feather_file_path, converters={}):
    """
    # Try to read from feather file cause it's much faster (create feather file if it doesn't exist)
    :param converters: dict of column: function, which is used to transform the data
    :return:
    """
    if not os.path.exists(feather_file_path):
        df = pd.read_csv(csv_file_path)
        # do not use in-built pandas converters param as it's not vectorized
        for column, func in converters:
            df[column] = df[column].apply(func)
        print("Creating feather file")
        df.to_feather(feather_file_path)
    df = pd.read_feather(feather_file_path)
    return df


# -------File reading-------#
def get_posts():
    # Reading the file from csv is slow, prefer to read from feather
    df = read_data_cascade(file_paths["posts"], file_paths["posts_feather"],
                           converters={"timestamp": pd.to_datetime, "hashtags": literal_eval})
    # undo conversion to numpy array done by feather
    df["hashtags"] = df["hashtags"].apply(lambda x: list(x))
    df.set_index("timestamp", inplace=True)
    return df


def get_posts_city():
    df = pd.read_csv(file_paths["posts_city"])
    df['Post Created Date'] = pd.to_datetime(df['Post Created Date'])
    df.set_index("Post Created Date", inplace=True)
    return df


def get_image_labels():
    df = pd.read_csv(file_paths["image_labels"], parse_dates=["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


def get_projects():
    return pd.read_csv(file_paths["projects"])


def get_visitors(dataset):
    fpath = visitor_folder / (dataset + ".csv")
    if fpath.is_file():
        return pd.read_csv(fpath, converters={"Year": pd.to_datetime})


def get_events(dataset, external=True):
    event_folder = events_external_folder if external else events_programme_folder
    fpath = event_folder / (dataset + ".csv")
    if fpath.is_file():
        dtfunc = lambda x: pd.to_datetime(x, dayfirst=True)
        df = pd.read_csv(fpath, converters={"Date": dtfunc}, index_col="Date")
        if "End" in df.columns:
            df["End"] = df["End"].apply(dtfunc)
        return df[df.index.year <= 2019]
