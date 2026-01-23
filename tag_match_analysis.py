#!/usr/bin/env -S uv run

from itertools import islice
from typing import TypedDict, cast

import pandas

import dfio
from merge_stats import (
    NerPositionMatchSourced,
    TextToNerPositionsSourced,
    cast_text_to_ner_position_sourced,
)


class TagMatchStandalone(NerPositionMatchSourced):
    tag: str


TagOverlap = tuple[TagMatchStandalone, TagMatchStandalone]


class TagOverlapWrapper:
    def __init__(self, tag_overlap: TagOverlap):
        self._tag_overlap = sort_overlap(tag_overlap)
        self._cached_hash: int | None = None

    def __hash__(self) -> int:
        if self._cached_hash is None:
            self._cached_hash = hash(
                (
                    self._tag_overlap[0]["char_start"],
                    self._tag_overlap[0]["char_end"],
                    self._tag_overlap[0]["tag"],
                    self._tag_overlap[1]["char_start"],
                    self._tag_overlap[1]["char_end"],
                    self._tag_overlap[1]["tag"],
                )
            )
        return self._cached_hash

    def __eq__(self, rhs: object, /) -> bool:
        if not isinstance(rhs, TagOverlapWrapper):
            raise ValueError()

        match cmp_tag_match(self._tag_overlap[0], rhs._tag_overlap[0]):
            case 0:
                return cmp_tag_match(self._tag_overlap[1], rhs._tag_overlap[1]) == 0
            case _:
                return False

    def __lt__(self, rhs: object) -> bool:
        if not isinstance(rhs, TagOverlapWrapper):
            raise ValueError()

        match cmp_tag_match(self._tag_overlap[0], rhs._tag_overlap[0]):
            case 0:
                return cmp_tag_match(self._tag_overlap[1], rhs._tag_overlap[1]) < 0
            case x if x < 0:
                return True
            case _:
                return False

    def unwrap(self) -> TagOverlap:
        return self._tag_overlap


def match_spread_len(tag_match: TagMatchStandalone) -> int:
    return tag_match["char_end"] - tag_match["char_start"]


def cmp_tag_match(match_a: TagMatchStandalone, match_b: TagMatchStandalone) -> int:
    A_LESSER_THAN_B = -1
    A_GREATER_THAN_B = 1
    A_EQUAL_B = 0

    match (match_spread_len(match_a), match_spread_len(match_b)):
        case (l_a, l_b) if l_a < l_b:
            return A_LESSER_THAN_B
        case (l_a, l_b) if l_a > l_b:
            return A_GREATER_THAN_B

    match (match_a["char_start"], match_b["char_start"]):
        case (s_a, s_b) if s_a < s_b:
            return A_LESSER_THAN_B
        case (s_a, s_b) if s_a > s_b:
            return A_GREATER_THAN_B

    match (match_a["tag"], match_b["tag"]):
        case (t_a, t_b) if t_a < t_b:
            return A_LESSER_THAN_B
        case (t_a, t_b) if t_a > t_b:
            return A_GREATER_THAN_B

    return A_EQUAL_B


def sort_overlap(tag_overlap: TagOverlap) -> TagOverlap:
    (match_a, match_b) = tag_overlap
    if cmp_tag_match(match_a, match_b) > 0:
        return (match_b, match_a)
    return tag_overlap


def are_matches_overlaping(
    match_a: TagMatchStandalone, match_b: TagMatchStandalone
) -> bool:
    end_of_a_in_b = (
        match_a["char_end"] > match_b["char_start"]
        and match_a["char_end"] <= match_b["char_end"]
    )
    end_of_b_in_a = (
        match_b["char_end"] > match_a["char_start"]
        and match_b["char_end"] <= match_a["char_end"]
    )
    return end_of_a_in_b or end_of_b_in_a


def df_to_dict_sourced(df: pandas.DataFrame) -> TextToNerPositionsSourced:
    source_dict = dfio.df_to_dict(df)
    match cast_text_to_ner_position_sourced(source_dict):
        case None:
            raise ValueError("data source is not sourced (no src property in matches)")
        case sourced:
            return sourced


def sourced_dict_to_df(sourced_dict: TextToNerPositionsSourced) -> pandas.DataFrame:
    return dfio.dict_to_df(cast(dfio.TextToNerPositions, sourced_dict))


def get_cross_tag_overlap_per_text(
    match_list: list[TagMatchStandalone],
) -> list[TagOverlap]:
    result: list[TagOverlap] = []

    match_list.sort(key=lambda m: m["char_start"])
    for idx, match_instance_a in enumerate(match_list):
        for match_instance_b in islice(match_list, idx + 1, None, None):
            if are_matches_overlaping(match_instance_a, match_instance_b):
                result.append((match_instance_a, match_instance_b))
    return result


def get_cross_tag_overlap_all_texts(
    sourced_dict: dict[str, list[TagMatchStandalone]],
) -> dict[str, list[TagOverlap]]:
    return dict(
        (text_id, get_cross_tag_overlap_per_text(match_list))
        for (text_id, match_list) in sourced_dict.items()
    )


def get_all_tag_match(
    sourced_dict: TextToNerPositionsSourced,
) -> dict[str, list[TagMatchStandalone]]:
    result: dict[str, list[TagMatchStandalone]] = dict()

    for text_id, tag_dict in sourced_dict.items():
        text_tag_list = result.setdefault(text_id, [])
        for tag_name, match_list in tag_dict.items():
            for match_instance in match_list:
                standalone_tag = cast(
                    TagMatchStandalone,
                    match_instance | {"tag": tag_name},
                )
                text_tag_list.append(standalone_tag)

    return result


def dedup_overlaps(overlap_list: list[TagOverlap]) -> list[TagOverlap]:
    deduped = set((TagOverlapWrapper(tag_overlap) for tag_overlap in overlap_list))
    return list((d.unwrap() for d in sorted(deduped)))


def dedup_all_overlaps(
    text_to_overlap_dict: dict[str, list[TagOverlap]],
) -> dict[str, list[TagOverlap]]:
    return dict(
        (
            (text_id, dedup_overlaps(overlaps))
            for (text_id, overlaps) in text_to_overlap_dict.items()
        )
    )


class OverlapDataset(TypedDict):
    text_id: list[str]
    tag_a: list[str]
    char_start_a: list[int]
    char_end_a: list[int]
    src_a: list[str]
    tag_b: list[str]
    char_start_b: list[int]
    char_end_b: list[int]
    src_b: list[str]


def text_overlaps_to_dataset(
    text_overlaps: dict[str, list[TagOverlap]],
) -> OverlapDataset:
    text_id: list[str] = []
    tag_a: list[str] = []
    char_start_a: list[int] = []
    char_end_a: list[int] = []
    src_a: list[str] = []
    tag_b: list[str] = []
    char_start_b: list[int] = []
    char_end_b: list[int] = []
    src_b: list[str] = []

    for text_id_input, overlap_list in text_overlaps.items():
        for overlap in overlap_list:
            text_id.append(text_id_input)
            (match_a, match_b) = overlap

            tag_a.append(match_a["tag"])
            char_start_a.append(match_a["char_start"])
            char_end_a.append(match_a["char_end"])
            src_a.append(repr(match_a["src"]))

            tag_b.append(match_b["tag"])
            char_start_b.append(match_b["char_start"])
            char_end_b.append(match_b["char_end"])
            src_b.append(repr(match_b["src"]))

    return {
        "text_id": text_id,
        "tag_a": tag_a,
        "char_start_a": char_start_a,
        "char_end_a": char_end_a,
        "src_a": src_a,
        "tag_b": tag_b,
        "char_start_b": char_start_b,
        "char_end_b": char_end_b,
        "src_b": src_b,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input.csv output.csv", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    input_df = pandas.read_csv(input_path)
    input_dict = dfio.df_to_dict(input_df)
    input_df = None
    match cast_text_to_ner_position_sourced(input_dict):
        case None:
            raise ValueError()
        case sourced_dict:
            input_dict = sourced_dict

    tag_matches_all = get_all_tag_match(input_dict)
    overlaps = get_cross_tag_overlap_all_texts(tag_matches_all)
    tag_matches_all = None
    overlaps = dedup_all_overlaps(overlaps)

    overlap_dataset = text_overlaps_to_dataset(overlaps)
    overlaps = None
    output_df = pandas.DataFrame(overlap_dataset)
    overlap_dataset = None
    output_df.to_csv(output_path, index=False)
