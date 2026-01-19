#!/usr/bin/env python3

import pandas
import ast
import sys
import itertools
import typing

Pos = tuple[int, int, str]

PosSrcDict = dict[Pos, set[str]]  # Position to source material
LabelPosDict = dict[str, PosSrcDict]
TextLabelDict = dict[str, LabelPosDict]

T = typing.TypeVar("T")
U = typing.TypeVar("U")


def df_to_dict(df: pandas.DataFrame, src_name: str) -> TextLabelDict:
    """
    Extracts data from a DataFrame to a dictionary for further processing
    """
    text_dict: TextLabelDict = dict()
    for _, row in df.iterrows():
        text_key = str(row["sha512"])
        label_dict: LabelPosDict = text_dict.setdefault(text_key, dict())
        try:
            df_labels: dict[str, list[dict]] = ast.literal_eval(
                str(row["ner_positions"])
            )
            df_labels_flattened = flatten_df_labels(df_labels, src_name)
        except:  # noqa: E722 permet d'ignorer les lognes malformatees
            continue

        merged = merge_tag_set(label_dict, df_labels_flattened)

        text_dict[text_key] = merged

    return text_dict


def flatten_df_labels(df_labels: dict[str, list[dict]], src_name: str) -> LabelPosDict:
    """
    Formats a dict compatible with the `ner_positions` field to a label ditionary including position and source traceability
    """
    result = dict()
    for k, v in df_labels.items():
        pos_list: PosSrcDict = dict(
            map(
                lambda x: (
                    (int(x["char_start"]), int(x["char_end"]), str(x["word"])),
                    {src_name},
                ),
                v,
            )
        )
        result[k] = pos_list
    return result


def get_or(input_dict: dict[T, U], key: T, default: U) -> U:
    """
    Returns the value corresponding to `key` in `input_dict` or `default` if key is not present in `input_dict`.
    `input_dict` is not modified
    """
    if key in input_dict:
        return input_dict[key]
    else:
        return default


def merge_tag_set(tag_set_a: LabelPosDict, tag_set_b: LabelPosDict) -> LabelPosDict:
    """
    Merges two LabelPosDict together such that the returned LabelPosDict contains the merged position of every label with their respective source traceability
    """
    merged: LabelPosDict = dict()

    for tag in set(itertools.chain(tag_set_a.keys(), tag_set_b.keys())):
        pos_source_a = get_or(tag_set_a, tag, dict())
        pos_source_b = get_or(tag_set_b, tag, dict())

        def merge_ab_src(pos: Pos) -> tuple[Pos, set[str]]:
            return (
                pos,
                get_or(pos_source_a, pos, set()) | get_or(pos_source_b, pos, set()),
            )

        pos = set(itertools.chain(pos_source_a.keys(), pos_source_b.keys()))
        merged_pos: PosSrcDict = dict(map(merge_ab_src, pos))
        merged[tag] = merged_pos

    return merged


def merge_datasource(
    df1: pandas.DataFrame,
    df2: pandas.DataFrame,
    source_name_1: str = "A",
    source_name_2: str = "B",
) -> TextLabelDict:
    """
    merges two tables of texts and ner_positions such that the result is one deduplicated set of texts and ner_positions.
    A traceability `src` tag will be added for each tag match to indicate which source table it came from.
    This tag will be a list of the sources' name
    """
    dict1 = df_to_dict(df1, source_name_1)
    dict2 = df_to_dict(df2, source_name_2)

    merged: TextLabelDict = dict()

    for text in set(itertools.chain(dict1.keys(), dict2.keys())):
        tag_set_a = get_or(dict1, text, dict())
        tag_set_b = get_or(dict2, text, dict())

        merged_tags = merge_tag_set(tag_set_a, tag_set_b)
        merged[text] = merged_tags

    return merged


def map_pos_dict(pos_dict: PosSrcDict) -> typing.Iterable[dict]:
    """
    Maps position to source dictionary to a match dict format expected as a component of the `ner_positions` field.

    This helper function is nto meant to be called directly be the end user.
    """
    return map(
        lambda x: {
            "char_start": x[0][0],
            "char_end": x[0][1],
            "word": x[0][2],
            "src": list(x[1]),
        },
        pos_dict.items(),
    )


def map_label_pos_dict(label_dict: LabelPosDict) -> dict[str, list[dict]]:
    """
    Maps the label to position & source dictionary to the dict format expected by the `ner_positions` field.
    """
    return dict(map(lambda x: (x[0], list(map_pos_dict(x[1]))), label_dict.items()))


def dict_to_df(
    dict_data: TextLabelDict,
) -> pandas.DataFrame:
    """
    Morphs the resulting TextLabelDict to a DataFrame suitable as output.
    """
    raw_items_1, raw_items_2 = itertools.tee(dict_data.items())

    sha_data = map(lambda x: x[0], raw_items_1)
    tag_data = map(lambda x: x[1], raw_items_2)

    clean_data_tag = map(lambda x: str(map_label_pos_dict(x)), tag_data)

    return pandas.DataFrame({"sha512": sha_data, "ner_positions": clean_data_tag})


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"usage: {sys.argv[0]} ner.csv onet.csv outfile.csv")
        sys.exit(1)

    file1: str = sys.argv[1]
    file2: str = sys.argv[2]
    outfile: str = sys.argv[3]

    df1 = pandas.read_csv(file1)
    df2 = pandas.read_csv(file2)

    merged_dict = merge_datasource(df1, df2, "ner", "onet")
    merged_df = dict_to_df(merged_dict)
    merged_df.to_csv(outfile, index=False)
