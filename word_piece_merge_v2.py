#!/usr/bin/env -S uv run

import sys
from itertools import islice
from typing import Optional

import pandas

import dfio


def smart_tag_merge(
    tag_front: dfio.NerPositionsMatch, tag_rear: dfio.NerPositionsMatch
) -> Optional[dfio.NerPositionsMatch]:
    dist = tag_rear["char_start"] - tag_front["char_end"]
    if dist < 0 or dist > 1:
        return None

    # if (tag_rear["char_end"] - tag_rear["char_start"]) < len(tag_rear["word"]):
    #     tag_rear_fixed = cast(
    #         dfio.NerPositionsMatch, tag_rear | {"word": tag_rear["word"][2:]}
    #     )
    # else:
    #     tag_rear_fixed = tag_rear

    return {
        "char_start": tag_front["char_start"],
        "char_end": tag_rear["char_end"],
        "word": "",  # tag_front["word"] + tag_rear_fixed["word"],
    }


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


def strip_all_prefix(text_dict: dfio.TextToNerPositions) -> dfio.TextToNerPositions:
    result: dfio.TextToNerPositions = dict()
    for text_id, tag_dict in text_dict.items():
        result_tag_dict = result.setdefault(text_id, dict())
        for tag, match_list in tag_dict.items():
            result_match_list = result_tag_dict.setdefault(tag[2:], [])
            for tag_match in match_list:
                result_match_list.append(tag_match)
    return result


def aglomerate_contiguous_tags(
    text_dict: dfio.TextToNerPositions,
) -> dfio.TextToNerPositions:
    result: dfio.TextToNerPositions = dict()
    for text_id, tag_dict in text_dict.items():
        result_tag_dict = result.setdefault(text_id, dict())
        for tag, match_list in tag_dict.items():
            result_match_list = result_tag_dict.setdefault(tag, [])
            sorted_match_list = sorted(match_list, key=lambda m: m["char_start"])
            for idx, match_a in enumerate(sorted_match_list):
                match_result: dfio.NerPositionsMatch = match_a
                for match_b in islice(sorted_match_list, idx + 1, None, None):
                    match smart_tag_merge(match_result, match_b):
                        case None:
                            continue
                        case new_match_result:
                            match_result = new_match_result
                result_match_list.append(match_result)
    return result


def is_match_a_in_match_b(
    match_a: dfio.NerPositionsMatch, match_b: dfio.NerPositionsMatch
) -> bool:
    return (
        match_a["char_start"] >= match_b["char_start"]
        and match_a["char_end"] <= match_b["char_end"]
    )


def dedup_aglomerated_tags(
    text_dict: dfio.TextToNerPositions,
) -> dfio.TextToNerPositions:
    result: dfio.TextToNerPositions = dict()
    for text_id, tag_dict in text_dict.items():
        result_tag_dict = result.setdefault(text_id, dict())
        for tag, match_list in tag_dict.items():
            result_match_list = result_tag_dict.setdefault(tag, [])
            sorted_match_list = sorted(
                match_list, key=lambda m: (m["char_end"] - m["char_start"])
            )
            for idx, match_a in enumerate(sorted_match_list):
                is_included = False
                for match_b in islice(sorted_match_list, idx + 1, None, None):
                    if is_match_a_in_match_b(match_a, match_b):
                        is_included = True
                        break
                if not is_included:
                    result_match_list.append(match_a)
    return result


def process_text_dict(text_dict: dfio.TextToNerPositions) -> dfio.TextToNerPositions:
    aglomerated = aglomerate_contiguous_tags(text_dict)
    return dedup_aglomerated_tags(aglomerated)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} input.csv output.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    input_df = pandas.read_csv(input_path)

    input_work_data = dfio.df_to_dict(input_df)
    input_df = None
    input_work_data = strip_all_prefix(input_work_data)

    output_work_data = process_text_dict(input_work_data)
    input_work_data = None

    output_df = dfio.dict_to_df(output_work_data)
    output_work_data = None

    output_df.to_csv(output_path, index=False)
