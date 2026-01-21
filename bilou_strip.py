#!/usr/bin/env -S uv run

import dfio


def _merge_match_list(
    match_list_a: list[dfio.NerPositionsMatch],
    match_list_b: list[dfio.NerPositionsMatch],
) -> list[dfio.NerPositionsMatch]:
    match_dict_a = dict(((m["char_start"], m) for m in match_list_a))
    match_dict_b = dict(((m["char_start"], m) for m in match_list_b))
    merged_dict = match_dict_b | match_dict_a
    return list((m for (_, m) in merged_dict.items()))


def remove_bilou_prefixes(
    text_dict: dfio.TextToNerPositions,
) -> dfio.TextToNerPositions:
    """
    Renames tags in a body of texts by removing BILOU prefix.

    Example: "U-OCDSW_2" -> "OCDSW_2"
    """
    result: dfio.TextToNerPositions = dict()

    for text_id, tag_dict in text_dict.items():
        result_tag_dict = result.setdefault(text_id, dict())
        for raw_tag_name, tag_match in tag_dict.items():
            stripped_tag_name = raw_tag_name[2:]
            result_tag_match = result_tag_dict.setdefault(stripped_tag_name, [])
            merged_matches = _merge_match_list(tag_match, result_tag_match)
            result_tag_dict[stripped_tag_name] = merged_matches

    return result


__all__ = ["remove_bilou_prefixes"]

if __name__ == "__main__":
    from sys import exit, argv, stderr
    import pandas

    if len(argv) != 3:
        print(f"usage: {argv[0]} input.csv output.csv", file=stderr)
        exit(1)

    input_file_path = argv[1]
    output_file_path = argv[2]

    input_df = pandas.read_csv(input_file_path)
    input_dict = dfio.df_to_dict(input_df)
    input_df = None  # allows GC to free unused old data if necessary

    output_dict = remove_bilou_prefixes(input_dict)
    input_dict = None

    output_df = dfio.dict_to_df(output_dict)
    output_dict = None
    output_df.to_csv(output_file_path, index=False)
