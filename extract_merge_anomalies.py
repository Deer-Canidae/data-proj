#!/usr/bin/env -S uv run

from pandas import pandas
import dfio
from merge_stats import (
    NerPositionMatchSourced,
    TextToNerPositionsSourced,
)
from sys import exit, argv


def extract_merge_anomalies(
    merged_data: TextToNerPositionsSourced,
) -> TextToNerPositionsSourced:
    result: TextToNerPositionsSourced = dict()
    for text_id, tag_dict in merged_data.items():
        result_tag_dict = result.setdefault(text_id, dict())
        for tag_name, match_list in tag_dict.items():
            result_match_list = result_tag_dict.setdefault(tag_name, [])
            for tag_match in filter(lambda m: len(m["src"]) < 2, match_list):
                result_match_list.append(tag_match)

    return result


def colapse_empty_tags(data: dict[str, list[NerPositionMatchSourced]]):
    pair_list = list(data.items())
    for k, v in pair_list:
        if len(v) == 0:
            del data[k]


def collapse_empty_texts(merge_data: TextToNerPositionsSourced):
    text_id_list = list(merge_data.items())
    for k, v in text_id_list:
        colapse_empty_tags(v)
        if len(v) == 0:
            del merge_data[k]


if __name__ == "__main__":
    if len(argv) != 3:
        print(f"usage: {argv[0]} input.csv output.csv")
        exit(1)

    input_df = pandas.read_csv(argv[1])

    input_dict = dfio.df_to_dict(input_df)

    extracted = extract_merge_anomalies(input_dict)

    collapse_empty_texts(extracted)

    output_df = dfio.dict_to_df(extracted)

    output_df.to_csv(argv[2], index=False)
