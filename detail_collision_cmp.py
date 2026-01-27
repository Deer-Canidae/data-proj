#!/usr/bin/env -S uv run

import pandas as pd
from dfio import NerPositionsMatch, TextToNerPositions, df_to_dict
from merge_stats import NerPositionMatchSourced, TextToNerPositionsSourced
from overlap_categorization import load_text_overlaps
from tag_match_analysis import TagOverlap
from onet import load_onet_reference
from typing import Optional, cast, NamedTuple
from ast import literal_eval

# inclusion types
A_IN_B = 1
B_IN_A = 2
PERFECT_MATCH = 3
SIMPLE_OVERLAP = 0


class ExtendedConflictInfo(NamedTuple):
    text_sha: str

    error_included: int
    error_includes: int
    error_simple_overlap: int
    error_perfect_match: int

    error_position: int
    error_tag_mismatch: int
    total_error: int
    error_rate: float

    tag_a: str
    start_a: int
    end_a: int
    word_a: str
    src_a: str

    tag_b: str
    start_b: int
    end_b: int
    word_b: str
    src_b: str


class TagStat(NamedTuple):
    error_included: int
    error_includes: int
    error_simple_overlap: int
    error_perfect_match: int

    error_position: int
    error_tag_mismatch: int

    total_error: int
    error_rate: float


# class TagStat(NamedTuple):
#     tag: str
#     error_included: int
#     error_includes: int
#     error_rate: float


class TagWithWord(NamedTuple):
    tag: str
    start: int
    end: int
    word: str


class PerTagData(NamedTuple):
    text_id: str
    start: int
    end: int


class TagErrorRate(NamedTuple):
    position_error: int
    tag_mismatch_error: int
    total_error: int
    offenders: dict[str, int]


TagList = list[NerPositionsMatch]
TagWithWordList = list[TagWithWord]


def main(
    colision_path: str,
    merged_matches_path: str,
    reference_text_path: str,
    output_path: str,
) -> None:
    # Load colisions from csv
    input_df = pd.read_csv(colision_path, low_memory=False)
    text_overlaps: dict[str, list[TagOverlap]] = load_text_overlaps(input_df)

    # Load refrence texts from csv
    input_df = pd.read_csv(
        reference_text_path, low_memory=False, sep=";", encoding="utf-8-sig"
    )
    reference_texts = load_reference_texts(input_df)

    input_df = pd.read_csv(merged_matches_path, low_memory=False)

    all_tags = df_to_dict(input_df)
    input_df = None  # allows GC to free memory

    per_tag_data = get_per_tag_data(all_tags)
    all_tags = None
    per_tag_counts = get_per_tag_count(per_tag_data)
    per_tag_data = None

    result = colision_to_extended_collision(text_overlaps)
    result = retrieve_word_match(result, reference_texts)
    result = denormalize_eci_list(result)
    result = compute_eci_stats(result, per_tag_counts)

    output_df = eci_to_df(result)
    output_df = post_add_onet_ref(output_df)
    output_df.to_csv(output_path, sep=";", index=False, encoding="utf-8-sig")


def alter_eci(
    template_eci: ExtendedConflictInfo,
    *,
    text_sha: Optional[str] = None,
    error_included: Optional[int] = None,
    error_includes: Optional[int] = None,
    error_simple_overlap: Optional[int] = None,
    error_perfect_match: Optional[int] = None,
    error_position: Optional[int] = None,
    error_tag_mismatch: Optional[int] = None,
    total_error: Optional[int] = None,
    error_rate: Optional[float] = None,
    tag_a: Optional[str] = None,
    start_a: Optional[int] = None,
    end_a: Optional[int] = None,
    word_a: Optional[str] = None,
    src_a: Optional[str] = None,
    tag_b: Optional[str] = None,
    start_b: Optional[int] = None,
    end_b: Optional[int] = None,
    word_b: Optional[str] = None,
    src_b: Optional[str] = None,
) -> ExtendedConflictInfo:
    """
    Creates a copy of the input ExtendedConflictInfo and return a version with the specified modifications

    If a keyword argument is specified then the coresponding key in the resultin named tuple will be modified.
    All other keys will be identical to the template_eci.
    """
    (
        text_sha_old,
        error_included_old,
        error_includes_old,
        error_simple_overlap_old,
        error_perfect_match_old,
        error_position_old,
        error_tag_mismatch_old,
        total_error_old,
        error_rate_old,
        tag_a_old,
        start_a_old,
        end_a_old,
        word_a_old,
        src_a_old,
        tag_b_old,
        start_b_old,
        end_b_old,
        word_b_old,
        src_b_old,
    ) = template_eci

    text_sha = text_sha_old if text_sha is None else text_sha
    error_included = error_included_old if error_included is None else error_included
    error_includes = error_includes_old if error_includes is None else error_includes
    error_simple_overlap = (
        error_simple_overlap_old
        if error_simple_overlap is None
        else error_simple_overlap
    )
    error_perfect_match = (
        error_perfect_match_old if error_perfect_match is None else error_perfect_match
    )
    error_position = error_position_old if error_position is None else error_position
    error_tag_mismatch = (
        error_tag_mismatch_old if error_tag_mismatch is None else error_tag_mismatch
    )
    total_error = total_error_old if total_error is None else total_error
    error_rate = error_rate_old if error_rate is None else error_rate
    tag_a = tag_a_old if tag_a is None else tag_a
    start_a = start_a_old if start_a is None else start_a
    end_a = end_a_old if end_a is None else end_a
    word_a = word_a_old if word_a is None else word_a
    src_a = src_a_old if src_a is None else src_a
    tag_b = tag_b_old if tag_b is None else tag_b
    start_b = start_b_old if start_b is None else start_b
    end_b = end_b_old if end_b is None else end_b
    word_b = word_b_old if word_b is None else word_b
    src_b = src_b_old if src_b is None else src_b

    return ExtendedConflictInfo(
        text_sha,
        error_included,
        error_includes,
        error_simple_overlap,
        error_perfect_match,
        error_position,
        error_tag_mismatch,
        total_error,
        error_rate,
        tag_a,
        start_a,
        end_a,
        word_a,
        src_a,
        tag_b,
        start_b,
        end_b,
        word_b,
        src_b,
    )


def colision_to_extended_collision(
    colisions: dict[str, list[TagOverlap]],
) -> list[ExtendedConflictInfo]:
    result: list[ExtendedConflictInfo] = []
    for text_id, conflict_list in colisions.items():
        for match_a, match_b in conflict_list:
            eci = ExtendedConflictInfo(
                text_id,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.0,
                match_a["tag"],
                match_a["char_start"],
                match_a["char_end"],
                "",
                match_a["src"][0],
                match_b["tag"],
                match_b["char_start"],
                match_b["char_end"],
                "",
                match_b["src"][0],
            )
            result.append(eci)
    return result


def retrieve_word_match(
    eci_list: list[ExtendedConflictInfo], ref_texts: dict[str, str]
) -> list[ExtendedConflictInfo]:
    result: list[ExtendedConflictInfo] = []
    for eci in eci_list:
        text = ref_texts[eci.text_sha]
        word_a = text[eci.start_a : eci.end_a]
        word_b = text[eci.start_b : eci.end_b]
        new_eci = alter_eci(eci, word_a=word_a, word_b=word_b)
        result.append(new_eci)
    return result


def get_inclusion_type(eci: ExtendedConflictInfo) -> int:
    if eci.start_a == eci.start_b and eci.end_a == eci.end_b:
        return PERFECT_MATCH
    if eci.start_a >= eci.start_b and eci.end_a <= eci.end_b:
        return A_IN_B
    if eci.start_a <= eci.start_b and eci.end_a >= eci.end_b:
        return B_IN_A
    return SIMPLE_OVERLAP


def denormalize_eci_list(
    eci_list: list[ExtendedConflictInfo],
) -> list[ExtendedConflictInfo]:
    result: list[ExtendedConflictInfo] = []
    for eci in eci_list:
        result.append(eci)
        tag_a = eci.tag_a
        start_a = eci.start_a
        end_a = eci.end_a
        word_a = eci.word_a
        src_a = eci.src_a
        tag_b = eci.tag_b
        start_b = eci.start_b
        end_b = eci.end_b
        word_b = eci.word_b
        src_b = eci.src_b
        result.append(
            alter_eci(
                eci,
                tag_a=tag_b,
                start_a=start_b,
                end_a=end_b,
                word_a=word_b,
                src_a=src_b,
                tag_b=tag_a,
                start_b=start_a,
                end_b=end_a,
                word_b=word_a,
                src_b=src_a,
            )
        )
    return result


def post_add_onet_ref(df: pd.DataFrame) -> pd.DataFrame:
    tag_ref = load_onet_reference()
    df["onet_a"] = df["tag_a"].apply(lambda t: tag_ref.setdefault(t, ""))
    df["onet_b"] = df["tag_b"].apply(lambda t: tag_ref.setdefault(t, ""))
    return cast(
        pd.DataFrame,
        df[
            [
                "text_sha",
                "error_included",
                "error_includes",
                "error_simple_overlap",
                "error_perfect_match",
                "error_position",
                "error_tag_mismatch",
                "total_error",
                "error_rate",
                "tag_a",
                "onet_a",
                "start_a",
                "end_a",
                "word_a",
                "src_a",
                "tag_b",
                "onet_b",
                "start_b",
                "end_b",
                "word_b",
                "src_b",
            ]
        ],
    )


def compute_eci_stats(
    eci_list: list[ExtendedConflictInfo],
    all_tags_count: dict[str, int],
) -> list[ExtendedConflictInfo]:
    global A_IN_B, B_IN_A, PERFECT_MATCH, SIMPLE_OVERLAP
    cached_tag_stats: dict[str, TagStat] = dict()
    work_dict: dict[str, list[ExtendedConflictInfo]] = dict()
    for eci in eci_list:
        per_tag_conflict_list = work_dict.setdefault(eci.tag_a, [])
        per_tag_conflict_list.append(eci)

    for tag, per_tag_conflict_list in work_dict.items():
        error_included: int = 0
        error_includes: int = 0
        error_simple_overlap: int = 0
        error_perfect_match: int = 0
        error_position: int = 0
        error_tag_mismatch: int = 0
        total_error = len(per_tag_conflict_list)
        error_rate: float = total_error / all_tags_count[tag]
        for eci in per_tag_conflict_list:
            match get_inclusion_type(eci):
                case i if i == A_IN_B:
                    error_included += 1
                case i if i == B_IN_A:
                    error_includes += 1
                case i if i == PERFECT_MATCH:
                    error_perfect_match += 1
                case i if i == SIMPLE_OVERLAP:
                    error_simple_overlap += 1

            if eci.tag_a == eci.tag_b:
                error_position += 1
            else:
                error_tag_mismatch += 1

        cached_tag_stats[tag] = TagStat(
            error_included,
            error_includes,
            error_simple_overlap,
            error_perfect_match,
            error_position,
            error_tag_mismatch,
            total_error,
            error_rate,
        )

    return list((add_stat_to_eci(eci, cached_tag_stats[eci.tag_a]) for eci in eci_list))


def eci_to_df(eci_list: list[ExtendedConflictInfo]) -> pd.DataFrame:
    text_sha_list = []
    error_included_list = []
    error_includes_list = []
    error_simple_overlap_list = []
    error_perfect_match_list = []
    error_position_list = []
    error_tag_mismatch_list = []
    total_error_list = []
    error_rate_list = []
    tag_a_list = []
    start_a_list = []
    end_a_list = []
    word_a_list = []
    src_a_list = []
    tag_b_list = []
    start_b_list = []
    end_b_list = []
    word_b_list = []
    src_b_list = []

    for (
        text_sha,
        error_included,
        error_includes,
        error_simple_overlap,
        error_perfect_match,
        error_position,
        error_tag_mismatch,
        total_error,
        error_rate,
        tag_a,
        start_a,
        end_a,
        word_a,
        src_a,
        tag_b,
        start_b,
        end_b,
        word_b,
        src_b,
    ) in eci_list:
        text_sha_list.append(text_sha)
        error_included_list.append(error_included)
        error_includes_list.append(error_includes)
        error_simple_overlap_list.append(error_simple_overlap)
        error_perfect_match_list.append(error_perfect_match)
        error_position_list.append(error_position)
        error_tag_mismatch_list.append(error_tag_mismatch)
        total_error_list.append(total_error)
        error_rate_list.append(error_rate)
        tag_a_list.append(tag_a)
        start_a_list.append(start_a)
        end_a_list.append(end_a)
        word_a_list.append(word_a)
        src_a_list.append(src_a)
        tag_b_list.append(tag_b)
        start_b_list.append(start_b)
        end_b_list.append(end_b)
        word_b_list.append(word_b)
        src_b_list.append(src_b)

    return pd.DataFrame(
        {
            "text_sha": text_sha_list,
            "error_included": error_included_list,
            "error_includes": error_includes_list,
            "error_simple_overlap": error_simple_overlap_list,
            "error_perfect_match": error_perfect_match_list,
            "error_position": error_position_list,
            "error_tag_mismatch": error_tag_mismatch_list,
            "total_error": total_error_list,
            "error_rate": error_rate_list,
            "tag_a": tag_a_list,
            "start_a": start_a_list,
            "end_a": end_a_list,
            "word_a": word_a_list,
            "src_a": src_a_list,
            "tag_b": tag_b_list,
            "start_b": start_b_list,
            "end_b": end_b_list,
            "word_b": word_b_list,
            "src_b": src_b_list,
        }
    )


def add_stat_to_eci(eci: ExtendedConflictInfo, stat: TagStat) -> ExtendedConflictInfo:
    (
        error_included,
        error_includes,
        error_simple_overlap,
        error_perfect_match,
        error_position,
        error_tag_mismatch,
        total_error,
        error_rate,
    ) = stat

    return alter_eci(
        eci,
        error_included=error_included,
        error_includes=error_includes,
        error_simple_overlap=error_simple_overlap,
        error_perfect_match=error_perfect_match,
        error_position=error_position,
        error_tag_mismatch=error_tag_mismatch,
        total_error=total_error,
        error_rate=error_rate,
    )


def get_per_tag_data(all_tag: TextToNerPositions) -> dict[str, list[PerTagData]]:
    result: dict[str, list[PerTagData]] = dict()

    for text_id, tag_dict in all_tag.items():
        for tag_name, match_list in tag_dict.items():
            result_match_list = result.setdefault(tag_name, [])
            for match_instance in match_list:
                result_match_list.append(
                    PerTagData(
                        text_id,
                        match_instance["char_start"],
                        match_instance["char_end"],
                    )
                )

    return result


def get_per_tag_count(per_tag_data: dict[str, list[PerTagData]]) -> dict[str, int]:
    return dict(
        ((tag_name, len(match_list)) for (tag_name, match_list) in per_tag_data.items())
    )


def load_collision_rates(input_df: pd.DataFrame) -> dict[str, TagErrorRate]:
    df = cast(
        pd.DataFrame,
        input_df[
            [
                "tag",
                "position_error",
                "different_tag_error",
                "total_error",
                "offender_list",
            ]
        ],
    )

    df = cast(
        pd.DataFrame,
        df.astype(
            {
                "tag": str,
                "position_error": int,
                "different_tag_error": int,
                "total_error": int,
                "offender_list": str,
            }
        ),
    )

    result: dict[str, TagErrorRate] = dict()
    for tag, position, mismatch, total, offenders in df.itertuples(
        index=False, name=None
    ):
        result[tag] = TagErrorRate(position, mismatch, total, literal_eval(offenders))

    return result


def find_tagged_word(
    text_id: str, refrence_texts: dict[str, str], tag_match: NerPositionsMatch
) -> str:
    ref_text = refrence_texts[text_id]
    begin = tag_match["char_start"]
    end = tag_match["char_end"]
    return ref_text[begin:end]


def load_match_data(df: pd.DataFrame) -> TextToNerPositionsSourced:
    result = df_to_dict(df)
    return cast(TextToNerPositionsSourced, result)


def load_reference_texts(df: pd.DataFrame) -> dict[str, str]:
    df = cast(pd.DataFrame, df[["sha512", "description"]])
    df = df.astype({"sha512": str, "description": str})
    df = cast(pd.DataFrame, df[df.notna().all(axis=1)])

    return dict(df.itertuples(index=False, name=None))


def extract_src_match(src_text: str, pos_match: NerPositionsMatch) -> str:
    return src_text[pos_match["char_start"] : pos_match["char_end"]]


def find_tag_word(src_text: str, tag_list: TagList, tag_name: str) -> TagWithWordList:
    return list(
        (
            TagWithWord(
                tag_name,
                tag["char_start"],
                tag["char_end"],
                extract_src_match(src_text, tag),
            )
            for tag in tag_list
        )
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print(
            f"usage: {sys.argv[0]} colisions.csv merged_matches.csv reference_texts.csv output.csv",
            file=sys.stderr,
        )
        sys.exit(0)

    colision_path = sys.argv[1]
    merged_matches_path = sys.argv[2]
    reference_text_path = sys.argv[3]
    output_path = sys.argv[4]

    main(colision_path, merged_matches_path, reference_text_path, output_path)
