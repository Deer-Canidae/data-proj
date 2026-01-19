"""
This module allows the extraction/insertion of ner_positions data per text from/to a suitable input/output dataframe
"""

if __name__ == "__main__":
    raise RuntimeError("This module is not intended to be executed directly")

from ast import literal_eval
from typing import Any, Optional, Required, TypeVar, TypedDict, cast
from itertools import tee

import pandas


class NerPositionsMatch(TypedDict):
    word: Required[str]
    char_start: Required[int]
    char_end: Required[int]


NerPositionsMatchT = TypeVar("NerPositionsMatchT", bound=NerPositionsMatch)


# class NerPositionMatchExtra(NerPositionsMatch):
#     __extra_items__: Any


NerPositions = dict[str, list[NerPositionsMatchT]]
TextToNerPositions = dict[str, NerPositions]


def _validate_ner_possition_format(maybe_ner: Any) -> Optional[NerPositions]:
    if not isinstance(maybe_ner, dict):
        return None
    for tag, match_list in maybe_ner.items():
        if not isinstance(tag, str):
            return None
        if not isinstance(match_list, list):
            return None
        for match_position in match_list:
            if not isinstance(match_position, dict):
                return None
            if not isinstance(match_position.get("word"), str):
                return None
            if not isinstance(match_position.get("char_start"), int):
                return None
            if not isinstance(match_position.get("char_end"), int):
                return None
    return cast(NerPositions, maybe_ner)


def df_to_dict(
    input_df: pandas.DataFrame,
    *,
    do_not_throw: bool = True,
    err_idx: Optional[list[int]] = None,
) -> TextToNerPositions:
    """
    Morphs a loaded DataFrame into a processable data structure.

    The DataFrame is expected to contain the following columns: sha512, ner_positions.

    :param input_df: the dataframe to process.
    :param do_not_throw: should error be silently ignored instead of raising an exception.
    :err_idx: add indexes of eroneous rows to this list
    :raises ValueError: on malformed input if ignore_error is set to False
    """

    if ("sha512" not in input_df) or ("ner_positions" not in input_df):
        raise ValueError(
            'The dataframe doe  not contain suitable columns to be processed\n\tExpected columns: "sha512", "ner_positions"'
        )

    df = input_df[["sha512", "ner_positions"]]

    result: TextToNerPositions = dict()
    for idx, text_id, ner_positions_raw in df.itertuples(index=True, name=None):
        if not isinstance(text_id, str):
            if err_idx is not None:
                err_idx.append(idx)
            if do_not_throw:
                continue
            else:
                raise ValueError(f"malformed sha512 at row {idx}")

        if not isinstance(ner_positions_raw, str):
            if err_idx is not None:
                err_idx.append(idx)
            if do_not_throw:
                continue
            else:
                raise ValueError(f"malformed ner_positions at row {idx}")

        try:
            ner_positions_maybe: Any = literal_eval(ner_positions_raw)
        except (ValueError, SyntaxError):
            if err_idx is not None:
                err_idx.append(idx)
            if do_not_throw:
                continue
            else:
                raise ValueError(f"unparseable ner_position at row {idx}")

        match _validate_ner_possition_format(ner_positions_maybe):
            case None:
                if err_idx is not None:
                    err_idx.append(idx)
                if do_not_throw:
                    continue
                else:
                    raise ValueError(f"ner_position has wrong shape at row {idx}")

            case ner_positions:
                tag_dict = ner_positions  # _ner_positions_into_tag_to_match_dict(ner_positions)
                result[text_id] = tag_dict

    return result


def dict_to_df(text_dict: TextToNerPositions) -> pandas.DataFrame:
    itt1, itt2 = tee(text_dict.items())
    return pandas.DataFrame(
        {
            "sha512": (text_id for (text_id, _) in itt1),
            "ner_positions": (str(ner_positions) for (_, ner_positions) in itt2),
        }
    )


# __all__ = ["df_to_dict", 'dict_to_df']
