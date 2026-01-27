#!/usr/bin/env -S uv run

# import onet
from tag_match_analysis import TagMatchStandalone, TagOverlap
import pandas as pd
from ast import literal_eval
from typing import NamedTuple, TypedDict


class TagMatchTuple(NamedTuple):
    tag: str
    char_start: int
    char_end: int
    src: str


class ConflictError(NamedTuple):
    total_error: int
    different_tag: int
    position: int
    offender_list: frozenset[tuple[str, int]]


OverlapTuple = tuple[TagMatchTuple, TagMatchTuple]


def standalone_to_tuple_match(input_match: TagMatchStandalone) -> TagMatchTuple:
    return TagMatchTuple(
        input_match["tag"],
        input_match["char_start"],
        input_match["char_end"],
        input_match["src"][0],
    )


def overlap_standalone_to_overlap_tuple(standalone: TagOverlap) -> OverlapTuple:
    (a, b) = standalone
    return (standalone_to_tuple_match(a), standalone_to_tuple_match(b))


def load_text_overlaps(df: pd.DataFrame) -> dict[str, list[TagOverlap]]:
    result: dict[str, list[TagOverlap]] = dict()

    for (
        text_id,
        tag_a,
        char_start_a,
        char_end_a,
        src_a,
        tag_b,
        char_start_b,
        char_end_b,
        src_b,
    ) in df[
        [
            "text_id",
            "tag_a",
            "char_start_a",
            "char_end_a",
            "src_a",
            "tag_b",
            "char_start_b",
            "char_end_b",
            "src_b",
        ]
    ].itertuples(index=False, name=None):
        overlap_list = result.setdefault(text_id, [])
        match_a: TagMatchStandalone = {
            "tag": tag_a,
            "char_start": char_start_a,
            "char_end": char_end_a,
            "src": literal_eval(src_a),
            "word": "",
        }
        match_b: TagMatchStandalone = {
            "tag": tag_b,
            "char_start": char_start_b,
            "char_end": char_end_b,
            "src": literal_eval(src_b),
            "word": "",
        }
        overlap_list.append((match_a, match_b))

    return result


def text_standalone_to_text_tuple(
    text_data: dict[str, list[TagOverlap]],
) -> dict[str, list[OverlapTuple]]:
    return dict(
        (text_id, list(map(overlap_standalone_to_overlap_tuple, overlaps)))
        for (text_id, overlaps) in text_data.items()
    )


def resolve_per_tag_conflict(
    text_data: dict[str, list[OverlapTuple]],
) -> dict[str, list[TagMatchTuple]]:
    result: dict[str, list[TagMatchTuple]] = dict()

    for _, overlap_list in text_data.items():
        for match_a, match_b in overlap_list:
            tag_a_conflic_list = result.setdefault(match_a.tag, [])
            tag_a_conflic_list.append(match_b)

            tag_b_conflic_list = result.setdefault(match_b.tag, [])
            tag_b_conflic_list.append(match_a)

    return result


def conflict_error_helper(
    conflict_list: list[TagMatchTuple], tag_name: str
) -> ConflictError:
    position_error = 0
    tag_error = 0
    offender_list: dict[str, int] = dict()

    for conflict in conflict_list:
        if conflict.tag == tag_name:
            position_error += 1
        else:
            tag_error += 1
        current_count = offender_list.setdefault(conflict.tag, 0)
        offender_list[conflict.tag] = current_count + 1

    return ConflictError(
        len(conflict_list), tag_error, position_error, frozenset(offender_list.items())
    )


def account_conflict_error(
    tag_conflict: dict[str, list[TagMatchTuple]],
) -> dict[str, ConflictError]:
    return dict(
        (
            (tag, conflict_error_helper(conflicts, tag))
            for (tag, conflicts) in tag_conflict.items()
        )
    )


def tag_conflict_error_to_df(tag_conflict: dict[str, ConflictError]) -> pd.DataFrame:
    tag: list[str] = []
    position_error: list[int] = []
    different_tag_error: list[int] = []
    total_error: list[int] = []
    offender_list: list[str] = []

    for tag_in_conflict, conflict in tag_conflict.items():
        tag.append(tag_in_conflict)
        position_error.append(conflict.position)
        different_tag_error.append(conflict.different_tag)
        offender_list.append(repr(dict(iter(conflict.offender_list))))
        total_error.append(conflict.total_error)

    return pd.DataFrame(
        {
            "tag": tag,
            "position_error": position_error,
            "different_tag_error": different_tag_error,
            "total_error": total_error,
            "offender_list": offender_list,
        }
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input.csv output.csv", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    df = pd.read_csv(input_path)

    input = load_text_overlaps(df)
    df = None

    prepared_data = text_standalone_to_text_tuple(input)
    input = None

    per_tag_conflict = resolve_per_tag_conflict(prepared_data)
    prepared_data = None

    accounted_error = account_conflict_error(per_tag_conflict)
    per_tag_conflict = None

    df = tag_conflict_error_to_df(accounted_error)
    accounted_error = None

    df.to_csv(output_path, index=False, sep=";", encoding="utf-8")
