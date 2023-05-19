import os
import pandas as pd
from pandas.errors import EmptyDataError
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
from hashtag_extractor import get_hashtags_counts
from ast import literal_eval


def read_feed_df(fpath):
    df = pd.read_csv(fpath)
    df["hashtags"] = df["hashtags"].apply(literal_eval)
    df["hashtags"] = df["hashtags"].apply(lambda l: [el.lower() for el in l])
    return df


def merge_dataframes(files):
    dfs = []
    for file in files:
        try:
            df_new = read_feed_df(file)
            dfs.append(df_new)
        except EmptyDataError:
            pass
    df_all = pd.concat(dfs, axis=0)
    return df_all


# def get_hashtag_filter(series, hashtags):
#     # hastags: list of strings
#     return series.apply(lambda htag_list: any([el in hashtags for el in htag_list]))


def create_masterlist(scrape_results, photo_only=True):
    """"
    Creates a masterlist of unique posts from different instagram feed scrapes
    scrape_results: list of dfs with at least those columns: [id, type, hashtags, search_term]
    """
    # merge scrapes
    df = merge_dataframes(scrape_results)

    print("All posts:", len(df))

    # filter by photo/video
    if photo_only:
        df = df[df["is_video"] == False]
        print("Photos only:", len(df))

    print("\nSearch term by count (non-unique):")
    print(df["search_term"].value_counts())

    # drop duplicates
    df = df.drop_duplicates(subset=["id"], keep="first")  # last keep location id entries for boijmans
    print("Unique posts:", len(df))

    print("\nSearch term by count (unique):")
    print(df["search_term"].value_counts())

    # Fill NAs
    # columns with empty values:
    #   location (intended)
    #   caption (not intended. We want no caption="" instead of NA)
    df["caption"] = df["caption"].fillna(value="")

    return df


def get_hashtag_count(df):
    all_hashtags = np.hstack(df["hashtags"])
    hashtags_grouped = pd.DataFrame(pd.Series(all_hashtags).value_counts(), columns=["count"])
    return hashtags_grouped


def get_result_path(scrape_parent_folder, scrape_name):
    return os.path.join(scrape_parent_folder, scrape_name, "result/result.csv")


def limit_to_year_range(df, min_year, max_year, do_print=True):
    if do_print:
        print("all:", len(df))

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # ["2020-01-01":"2022-01-01"] not working for some reason
    res = df[df["timestamp"].apply(lambda t: min_year <= t.year < max_year)]
    if do_print:
        print("20/21:", len(res))
    return res


if __name__ == "__main__":
    # Boijmans masterlist
    out_file_hashtags = "../../data/Instagram-API/Feed/boijmans-masterlist-hashtags.csv"
    out_file_hashtags_depot = "../../data/Instagram-API/Feed/boijmans-masterlist-hashtags_depot.csv"
    out_file_location = "../../data/Instagram-API/Feed/boijmans-masterlist-location.csv"

    scrape_parent_folder = "../../data/Instagram-API/Feed"
    hashtag_scrapes = ["depotboijmans", "depotboijmansvanbeuningen", "depotrotterdam", "boijmansdepot", "boymansdepot",
                       "boijmans", "boijmansvanbeuningen", "boymansvanbeuningen", "boymans",
                       "boijmansmuseum", "boijmansvanbeuningenmuseum", "museumpark", "museumparkrotterdam"]
    # not:
    # "hetdepot" -  a concert hall in Belgium dominates this hashtag

    scrapes = ["Boijmans-hashtag-" + scrape for scrape in hashtag_scrapes]

    location_scrapes = ["Boijmans-location-371724260325738"]
    location_result = get_result_path(scrape_parent_folder, location_scrapes[0])

    scrape_results = [get_result_path(scrape_parent_folder, scrape_name) for scrape_name in scrapes]

    df = create_masterlist(scrape_results, photo_only=True)
    df_depot = df[df["caption"].str.lower().str.contains("depot")]

    df_location = read_feed_df(location_result)

    min_year = 2020
    max_year = 2022

    df_location = df_location[df_location["is_video"] == False]

    df = limit_to_year_range(df, min_year, max_year)

    df_depot = limit_to_year_range(df_depot, min_year, max_year)

    df_location = limit_to_year_range(df_location, min_year, max_year)

    print("\ncontains depot")
    print(df_depot["search_term"].value_counts())

    # print("location", len(df_location))
    # print("hashtags", len(df))
    # print("hashtagsdepot", len(df_depot))
    # df_location_hashtags = df_location[df_location["id"].isin(df["id"])]
    # df_location_hashtagsdepot = df_location[df_location["id"].isin(df_depot["id"])]
    # print("location-hashtags", len(df_location_hashtags))
    # print("location-hashtagsdepot", len(df_location_hashtagsdepot))

    df.to_csv(out_file_hashtags, index=False)
    df_depot.to_csv(out_file_hashtags_depot, index=False)
    df_location.to_csv(out_file_location, index=False)

    # hashtag counts only make sense for location dataset cause the hashtag dataset is pretty damn biased

    # hashtags_grouped = get_hashtag_count(df)
    # print("\nhashtags by count (top 50)")
    # print(hashtags_grouped.head(50))
    # hashtags_grouped.to_csv("hashtags_count_masterlist.csv")
