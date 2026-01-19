#!/usr/bin/env -S uv run

import sys
from itertools import dropwhile, islice
from typing import Callable, Optional, cast, Iterable

import pandas

import dfio

# PosAndWord = tuple[int, int, str]
# TagDict = dict[str, list[PosAndWord]]
# TextDict = dict[str, TagDict]

# T = TypeVar("T")

#                      text -> tag_suffix -> tag_prefix -> match
TextTagHierarchized = dict[str, dict[str, dict[str, list[dfio.NerPositionsMatch]]]]


def hierarchize_texts(text_dict: dfio.TextToNerPositions) -> TextTagHierarchized:
    result: TextTagHierarchized = dict()
    for text_id, raw_tag_dict in text_dict.items():
        result_text_dict = result.setdefault(text_id, dict())
        for raw_tag, match_list in raw_tag_dict.items():
            tag_prefix = raw_tag[:1]
            tag_sufix = raw_tag[2:]
            tag_prefix_dict = result_text_dict.setdefault(tag_sufix, dict())
            result_match_list = tag_prefix_dict.setdefault(tag_prefix, [])
            for tag_match in match_list:
                result_match_list.append(tag_match)

    return result


def sort_text_hierarchy_by_pos(text_tag_hierarchy: TextTagHierarchized):
    for _, suffix_dict in text_tag_hierarchy.items():
        for _, prefix_dict in suffix_dict.items():
            for _, match_list in prefix_dict.items():
                new_match_list = sorted(
                    filter(lambda m: (m["char_start"] >= 0), match_list),
                    key=lambda m: m["char_start"],
                )
                match_list.clear()
                for e in new_match_list:
                    match_list.append(e)


def smart_tag_merge(
    tag_front: dfio.NerPositionsMatch, tag_rear: dfio.NerPositionsMatch
) -> Optional[dfio.NerPositionsMatch]:
    dist = tag_rear["char_start"] - tag_front["char_end"]
    if dist < 0 or dist > 1:
        return None

    if (tag_rear["char_end"] - tag_rear["char_start"]) < len(tag_rear["word"]):
        tag_rear_fixed = cast(
            dfio.NerPositionsMatch, tag_rear | {"word": tag_rear["word"][2:]}
        )
    else:
        tag_rear_fixed = tag_rear

    return {
        "char_start": tag_front["char_start"],
        "char_end": tag_rear_fixed["char_end"],
        "word": tag_front["word"] + tag_rear_fixed["word"],
    }


def _before_match(
    ref: dfio.NerPositionsMatchT,
) -> Callable[[dfio.NerPositionsMatchT], bool]:
    return lambda m: m["char_start"] < ref["char_start"]


def merge_tag_hierarchy(
    text_tag_hierarchy: TextTagHierarchized,
) -> dfio.TextToNerPositions:
    result: dfio.TextToNerPositions = dict()

    sort_text_hierarchy_by_pos(text_tag_hierarchy)

    # Per text per tag suffix
    for text_id, suffix_dict in text_tag_hierarchy.items():
        result_tag_dict = result.setdefault(text_id, dict())
        for suffix, prefix_dict in suffix_dict.items():
            result_match_list = result_tag_dict.setdefault(suffix, [])

            # merging B tags with I and then L tags
            for begin_match in prefix_dict.setdefault("B", []):
                for intermediate_match in dropwhile(
                    _before_match(begin_match), prefix_dict.setdefault("I", [])
                ):
                    match smart_tag_merge(begin_match, intermediate_match):
                        case None:
                            continue
                        case new_begin_match:
                            begin_match = new_begin_match

                for last_match in dropwhile(
                    _before_match(begin_match), prefix_dict.setdefault("L", [])
                ):
                    match smart_tag_merge(begin_match, last_match):
                        case None:
                            continue
                        case new_begin_match:
                            begin_match = new_begin_match

                result_match_list.append(begin_match)

            # Merging U tags with I, L and then other U tags
            for unique_match in prefix_dict.setdefault("U", []):
                for intermediate_match in dropwhile(
                    _before_match(unique_match), prefix_dict.setdefault("I", [])
                ):
                    match smart_tag_merge(unique_match, intermediate_match):
                        case None:
                            continue
                        case new_unique_match:
                            unique_match = new_unique_match

                for last_match in prefix_dict.setdefault("L", []):
                    match smart_tag_merge(unique_match, last_match):
                        case None:
                            continue
                        case new_unique_match:
                            unique_match = new_unique_match

                for unique_match_2 in prefix_dict.setdefault("U", []):
                    match smart_tag_merge(unique_match, unique_match_2):
                        case None:
                            continue
                        case new_unique_match:
                            unique_match = new_unique_match

                result_match_list.append(unique_match)

            # Scooping up remaining unmerged tags without duplication
            #
            result_match_list.sort(key=lambda m: m["char_start"])
            additional_match_list: list[dfio.NerPositionsMatch] = []
            for intermediate_match in prefix_dict.setdefault("I", []):
                isintergrated = False
                for result_match in result_match_list:
                    if (
                        intermediate_match["char_start"] >= result_match["char_start"]
                        and intermediate_match["char_end"] <= result_match["char_end"]
                    ):
                        isintergrated = True
                        break
                if not isintergrated:
                    additional_match_list.append(intermediate_match)
            for last_match in prefix_dict.setdefault("L", []):
                isintergrated = False
                for result_match in result_match_list:
                    if (
                        last_match["char_start"] >= result_match["char_start"]
                        and last_match["char_end"] <= result_match["char_end"]
                    ):
                        isintergrated = True
                        break
                if not isintergrated:
                    additional_match_list.append(last_match)
            for additional_match in additional_match_list:
                result_match_list.append(additional_match)

            result_match_list.sort(key=lambda m: m["char_start"])
            new_match_list = dedupe_merged_tags(result_match_list)
            result_match_list.clear()
            for e in new_match_list:
                result_match_list.append(e)

    return result


def dedupe_merged_tags(
    match_itt: Iterable[dfio.NerPositionsMatch],
) -> list[dfio.NerPositionsMatch]:
    match_list: list[dfio.NerPositionsMatch] = sorted(
        match_itt, key=lambda m: m["char_end"] - m["char_start"]
    )

    result: list[dfio.NerPositionsMatch] = []

    for idx, word_match in enumerate(match_list):
        ignore = False
        for word_compare in islice(match_list, idx + 1, None, None):
            if (
                word_match["char_start"] >= word_compare["char_start"]
                and word_match["char_end"] <= word_compare["char_end"]
            ):
                ignore = True
                break
        if not ignore:
            result.append(word_match)

    return result


# def dual_iterator(itt: Iterable[T]) -> Iterable[tuple[T, T]]:
#     element_it = iter(itt)
#     last = next(element_it)
#     try:
#         while True:
#             current = next(element_it)
#             yield (last, current)
#             last = current
#     except StopIteration:
#         return


def process_composite_tag(
    tag: dfio.NerPositionsMatch,
) -> Optional[dfio.NerPositionsMatch]:
    begin, end, word = tag["char_start"], tag["char_end"], tag["word"]
    if (end - begin) == len(word):
        return None
    return {"char_start": begin, "char_end": end, "word": word[2:]}


def merge_tag(
    tag_a: dfio.NerPositionsMatch, tag_b: dfio.NerPositionsMatch
) -> Optional[dfio.NerPositionsMatch]:
    begin, merge_point_a, word_a = tag_a["char_start"], tag_a["char_end"], tag_a["word"]
    merge_point_b, end, word_b = tag_b["char_start"], tag_b["char_end"], tag_b["word"]

    if abs(merge_point_b - merge_point_a) < 2:
        return {"char_start": begin, "char_end": end, "word": word_a + word_b}
    else:
        return None


def stript_tag_prefix(tag: str) -> str:
    return tag[2:]


# OLD
# def merge_tagged_words(
#     input_tagged_words: Iterable[dfio.NerPositionsMatch],
# ) -> list[dfio.NerPositionsMatch]:
#     tagged_words = list(filter(lambda x: x["char_start"] >= 0, input_tagged_words))
#     if len(tagged_words) < 2:
#         return tagged_words
#
#     tagged_words.sort(key=lambda x: x["char_start"])
#
#     result_words: list[dfio.NerPositionsMatch] = []
#
#     word_itt = iter(tagged_words)
#     last_fixed_tag: dfio.NerPositionsMatch = next(word_itt)
#
#     for tag in word_itt:
#         match process_composite_tag(tag):
#             case None:
#                 result_words.append(last_fixed_tag)
#                 last_fixed_tag = tag
#             case t:
#                 match merge_tag(last_fixed_tag, t):
#                     case None:
#                         result_words.append(last_fixed_tag)
#                         last_fixed_tag = tag
#                     case merged:
#                         last_fixed_tag = merged
#
#     result_words.append(last_fixed_tag)
#
#     return result_words
#

# OBSOLETE
# def df_to_working_data(df: pandas.DataFrame) -> TextDict:
#     work_data: TextDict = dict()
#
#     for _, row in df[
#         df["sha512"].notnull()
#         & df["sha512"].notna()
#         & df["ner_positions"].notnull()
#         & df["ner_positions"].notna()
#     ].iterrows():
#         text_id = str(row["sha512"])
#
#         ner_positions_raw = str(row["ner_positions"])
#
#         try:
#             ner_positions: dict[str, list[dict]] = ast.literal_eval(ner_positions_raw)
#         except (ValueError, SyntaxError):
#             continue
#
#         tag_dict: TagDict = work_data.setdefault(text_id, dict())
#
#         for tag_name, prop_list in ner_positions.items():
#             tag_name = stript_tag_prefix(tag_name)
#             tag_list: list[PosAndWord] = tag_dict.setdefault(tag_name, [])
#             for prop in prop_list:
#                 tag_list.append((prop["char_start"], prop["char_end"], prop["word"]))
#
#     return work_data


# OBSOLETE
# def posandword_to_dict(pos_and_word: PosAndWord) -> dict:
#     return {
#         "char_start": pos_and_word[0],
#         "char_end": pos_and_word[1],
#         "word": pos_and_word[2],
#     }

# OBSOLETE
# def tag_dict_to_ner_positions(tag_dict: TagDict) -> str:
#     result: dict[str, list[dict]] = dict(
#         (
#             (key, list(map(posandword_to_dict, pos_list)))
#             for (key, pos_list) in tag_dict.items()
#         )
#     )
#     return str(result)


def process_text_dict(text_dict: dfio.TextToNerPositions) -> dfio.TextToNerPositions:
    # OLD
    # for text_id, tag_dict in text_dict.items():
    #     result_tag_dict = result.setdefault(text_id, dict())
    #     for tag, pos_list in tag_dict.items():
    #         result_tag_dict[tag] = cast(
    #             list[dfio.NerPositionsMatch], merge_tagged_words(pos_list)
    #         )
    intermediate_texts = hierarchize_texts(text_dict)
    return merge_tag_hierarchy(intermediate_texts)


# OBSOLETE
# def working_data_to_df(work_data: TextDict) -> pandas.DataFrame:
#     itt1, itt2 = itertools.tee(work_data.items())
#     sha512 = (text_id for (text_id, _) in itt1)
#     ner_positions = (tag_dict_to_ner_positions(tag_dict) for (_, tag_dict) in itt2)
#
#     return pandas.DataFrame({"sha512": sha512, "ner_positions": ner_positions})


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input.csv output.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    input_df = pandas.read_csv(input_path)

    input_work_data = dfio.df_to_dict(input_df)

    output_work_data = process_text_dict(input_work_data)

    output_df = dfio.dict_to_df(output_work_data)

    output_df.to_csv(output_path, index=False)
