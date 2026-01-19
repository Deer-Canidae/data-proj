#!/usr/bin/env -S uv run

import pandas
import sys
from dfio import (
    NerPositions,
    TextToNerPositions,
    df_to_dict,
    NerPositionsMatch,
    NerPositionsMatchT,
)
from typing import Any, Optional, Required, Self, TextIO, TypedDict, cast, TypeVar

T = TypeVar("T")


class NerPositionMatchSourced(NerPositionsMatch):
    src: Required[list[str]]


NerPositionsSourced = dict[str, list[NerPositionMatchSourced]]
TextToNerPositionsSourced = dict[str, NerPositionsSourced]


def cast_ner_position_match_sourced(
    ner_position_match: NerPositionsMatch,
) -> Optional[NerPositionMatchSourced]:
    match ner_position_match.get("src"):
        case src_list if isinstance(src_list, list):
            for src in src_list:
                if not isinstance(src, str):
                    return None
        case _:
            return None
    return cast(NerPositionMatchSourced, ner_position_match)


def cast_ner_position_sourced(
    ner_position: NerPositions,
) -> Optional[NerPositionsSourced]:
    for _, location_list in ner_position.items():
        for location in location_list:
            match cast_ner_position_match_sourced(location):
                case None:
                    return None
                case _:
                    pass
    return cast(NerPositionsSourced, ner_position)


def cast_text_to_ner_position_sourced(
    text_to_ner: TextToNerPositions,
) -> Optional[TextToNerPositionsSourced]:
    for _, ner_positions in text_to_ner.items():
        match cast_ner_position_sourced(ner_positions):
            case None:
                return None
            case _:
                pass
    return cast(TextToNerPositionsSourced, text_to_ner)


def sort_dict_items(input_dict: dict[str, T]) -> list[tuple[str, T]]:
    result = list(input_dict.items())
    result.sort(key=lambda i: i[0])
    return result


class TagSourceStatistic:
    def __init__(self):
        self.total_tags_count: int = 0
        self.src_tag_count: dict[str, int] = dict()
        self.src_tag_type_count: dict[str, dict[str, int]] = dict()
        self.__acounted: bool = False

    def isaccounted(self):
        return self.__acounted

    def account_stats(self, data_src: TextToNerPositionsSourced) -> Self:
        if self.__acounted:
            raise RuntimeError("Stat already acounted")

        for text_id, tag_dict in data_src.items():
            for tag, match_list in tag_dict.items():
                tag_prefix = tag[0:1]
                for tag_match in match_list:
                    self.total_tags_count += 1
                    for src in tag_match.get("src"):
                        src_count = self.src_tag_count.setdefault(src, 0) + 1
                        self.src_tag_count[src] = src_count

                        tag_type_count = self.src_tag_type_count.setdefault(src, dict())
                        tag_type_count[tag_prefix] = (
                            tag_type_count.setdefault(tag_prefix, 0) + 1
                        )

        self.__acounted = True
        return self

    def print_stats(self, file: TextIO = sys.stdout):
        if not self.__acounted:
            raise RuntimeError("cannot print statistics that have not been acounted")
        print(f"total tag: {self.total_tags_count}", file=file)
        print("tag per source:", file=file)
        for src, src_count in sort_dict_items(self.src_tag_count):
            print(f'\t"{src}": {src_count}', file=file)
        print("tag prefix per source:")
        for src, src_prefix_dict in sort_dict_items(self.src_tag_type_count):
            print(f'\t"{src}":', file=file)
            for src_prefix, src_prefix_count in sort_dict_items(src_prefix_dict):
                print(f'\t\t"{src_prefix}": {src_prefix_count}')


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} merged_file.csv")
        sys.exit(1)

    merged_file_path = sys.argv[1]

    df = pandas.read_csv(merged_file_path)

    work_data = df_to_dict(df)
    work_data_validated = cast_text_to_ner_position_sourced(work_data)
    if work_data_validated is None:
        raise ValueError("data is not merge data")
    stats = TagSourceStatistic()
    stats.account_stats(work_data_validated)
    stats.print_stats()
