#!/usr/bin/env -S uv run

import dfio
from merge_stats import NerPositionMatchSourced, TextToNerPositionsSourced
from typing import TypeVar, cast

_T = TypeVar("_T")


def _get_or(input_dict: dict[str, _T], key: str, default_val: _T) -> _T:
    try:
        return input_dict[key]
    except KeyError:
        return default_val


def _mk_tuple(pos_match: dfio.NerPositionsMatch) -> tuple[int, int]:
    return (pos_match["char_start"], pos_match["char_end"])


def _mk_pos_match(
    pos_tuple: tuple[int, int], src_list: list[str]
) -> NerPositionMatchSourced:
    return {
        "char_start": pos_tuple[0],
        "char_end": pos_tuple[1],
        "word": "",
        "src": src_list,
    }


def _merge_match_list(
    match_list_a: list[dfio.NerPositionsMatch],
    match_list_b: list[dfio.NerPositionsMatch],
    src_name_a: str,
    src_name_b: str,
) -> list[NerPositionMatchSourced]:
    match_set_a = set(_mk_tuple(m) for m in match_list_a)
    match_set_b = set(_mk_tuple(m) for m in match_list_b)

    set_intersect = match_set_a & match_set_b

    result: list[NerPositionMatchSourced] = [
        _mk_pos_match(m, [src_name_a, src_name_b]) for m in set_intersect
    ]

    for match_in_a in (
        _mk_pos_match(m, [src_name_a]) for m in (match_set_a - set_intersect)
    ):
        result.append(match_in_a)

    for match_in_b in (
        _mk_pos_match(m, [src_name_b]) for m in (match_set_b - set_intersect)
    ):
        result.append(match_in_b)

    return result


def merge_text_bodies(
    text_body_a: dfio.TextToNerPositions,
    text_body_b: dfio.TextToNerPositions,
    body_name_a: str,
    body_name_b: str,
) -> TextToNerPositionsSourced:
    """
    Merges two bodies of texts with tags into  single body of text with tags.

    Each tag is given a list of sources where it wa found
    """
    result: TextToNerPositionsSourced = dict()
    all_text_ids: set[str] = set(text_body_a.keys()) | set(text_body_b.keys())
    for text_id in all_text_ids:
        tag_dict_result = result.setdefault(text_id, dict())
        tag_dict_a = _get_or(text_body_a, text_id, dict())
        tag_dict_b = _get_or(text_body_b, text_id, dict())
        all_tags: set[str] = set(tag_dict_a.keys()) | set(tag_dict_b.keys())
        for tag_name in all_tags:
            match_list_a = _get_or(tag_dict_a, tag_name, [])
            match_list_b = _get_or(tag_dict_b, tag_name, [])
            tag_dict_result[tag_name] = _merge_match_list(
                match_list_a, match_list_b, body_name_a, body_name_b
            )

    return result


__all__ = ["merge_text_bodies"]

if __name__ == "__main__":
    import sys
    import pandas

    if len(sys.argv) != 4:
        print(
            f"usage: {sys.argv[0]} ner_input.csv tagger_input.csv output.csv",
            file=sys.stderr,
        )
        sys.exit(1)

    input_a_path = sys.argv[1]
    input_b_path = sys.argv[2]
    output_path = sys.argv[3]

    # reading and formating ner input
    input_a_df = pandas.read_csv(input_a_path)
    input_a_dict = dfio.df_to_dict(input_a_df)
    input_a_df = None  # marked as ready for GC

    # reading and formating tagger input
    input_b_df = pandas.read_csv(input_b_path)
    input_b_dict = dfio.df_to_dict(input_b_df)
    input_b_df = None

    # processin merge
    merged_dict = merge_text_bodies(input_a_dict, input_b_dict, "ner", "tagger")
    input_a_dict = None
    input_b_dict = None
    # casting strict superset to subset
    merged_dict = cast(dfio.TextToNerPositions, merged_dict)

    # writting output
    output_df = dfio.dict_to_df(merged_dict)
    merged_dict = None
    output_df.to_csv(output_path, index=False)
