#!/usr/bin/env python3
"""
This module aims to find the position of matches within a body of texts
"""

import pandas as pd
from typing import Iterable, Optional, TypeVar
from itertools import tee
from tqdm import tqdm

T = TypeVar("T")
U = TypeVar("U")

MatchData = tuple[str, str, tuple[int, int]]  # tag, word, position
TextMatch = dict[str, list[MatchData]]  # sha512 -> list[MatchData]


def get_or(search_dict: dict[T, U], key: T, default: U) -> U:
    """
    Returns the value in the dictionary corresponding to `key` or the `default` value if the key hasno match in the dictionary.
    The dict is not mutated.
    """
    if key in search_dict:
        return search_dict[key]
    return default


def find_next_match(
    ref_text: str, match_value: str, offset: int
) -> Optional[tuple[int, int]]:
    """
    Returns the begining and end position of `match_value` within `ref_text` after `offset`.
    If no instance exists, returns None.
    """
    index = ref_text.find(match_value, offset)
    if index < 0:
        return None
    else:
        return (index, index + len(match_value))


def find_match_position(ner_matches: pd.DataFrame, reference_texts) -> TextMatch:
    """
    Finds the position of every match of `ner_matches` within the texts of `reference_texts`.

    Args:
        ner_matches (pd.DataFrame): Containing the following columns : sha512, word, ner.

        reference_texts (pd.DataFrame): Containing the following columns : sha512, description.

    Returns:
        TextMatch: a dictionary of text id (sha512) to a list of tag information
    """
    result: TextMatch = dict()
    # the reference texts are expected to have unique sha512s
    for ref_text_row in tqdm(
        reference_texts.itertuples(),
        desc="Finding match location",
        unit="texts",
        total=len(reference_texts),
    ):
        text_id = str(ref_text_row.sha512)
        text_content = str(ref_text_row.description)

        matches_in_text = ner_matches[ner_matches["sha512"] == text_id]

        match_list: list[MatchData] = result.setdefault(text_id, [])

        current_index = 0
        for _, match_row in matches_in_text.iterrows():
            word = str(match_row["word"])
            match_position = find_next_match(text_content, word, current_index)

            if match_position is None:
                continue
            (_, current_index) = match_position

            tag = str(match_row["ner"])

            match tag:
                case "" | "0" | "O":
                    continue
                case _:
                    pass

            match_list.append((tag, word, match_position))

    return result


def matchdata_iterable_to_ner_position(
    data_list: Iterable[MatchData],
) -> dict[str, list[dict]]:
    """
    Morphs a series of MatchData to a dictionary suitable for the `ner_position` field of the output DataFrame
    """
    result: dict[str, list[dict]] = dict()
    for match_data in data_list:
        match_list = result.setdefault(match_data[0], [])
        match_list.append(
            {
                "word": match_data[1],
                "char_start": match_data[2][0],
                "char_end": match_data[2][1],
            }
        )
    return result


def textmatch_to_df(text_match: TextMatch) -> pd.DataFrame:
    """
    Formats text matches to DataFrame suitable for output
    """
    items = text_match.items()

    text_id, ner_position = tee(items)

    return pd.DataFrame(
        {
            "sha512": (t[0] for t in text_id),
            "ner_positions": (
                str(matchdata_iterable_to_ner_position(n[1])) for n in ner_position
            ),
        }
    )


if __name__ == "__main__":
    ## Aquiring data sources
    reference_texts = pd.read_csv(
        "./data/offre_all_v2_ner_commodity_title_concatenated.csv",
        sep=";",
        encoding="utf-8",
    )

    ner_matches = pd.read_csv(
        "./data/offre_all_v2_ner_commodity_title.csv", sep=";", encoding="utf-8"
    )

    ## Data cleanup

    del reference_texts["label"]
    del reference_texts["crc32"]

    reference_texts = reference_texts[
        reference_texts["description"].notnull()
        | reference_texts["description"].notna()
    ]
    # expected columns: sha512, description

    del ner_matches["pos"]
    del ner_matches["chunk"]
    # expected columns: sha512, word, ner

    ## Processing

    output_dict = find_match_position(ner_matches, reference_texts)

    ## Output

    df_out = textmatch_to_df(output_dict)
    df_out.to_csv("./data/out.csv", sep=",", encoding="utf-8", index=False)
