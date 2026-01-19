#!/usr/bin/env python3

import hashlib
import pandas
import sys
# import mimetypes
# import typing
# import pathlib


def sha512(text: str) -> str:
    return hashlib.sha512(text.encode("utf-8")).hexdigest()


def fix_df(df: pandas.DataFrame) -> pandas.DataFrame:
    df["sha512"] = df["description"].apply(lambda x: sha512(str(x)))
    return df


def reduce_df(df: pandas.DataFrame) -> pandas.DataFrame:
    output_df = pandas.DataFrame()
    output_df["sha512"] = df["sha512"]
    output_df["description"] = df["description"]
    return output_df


# def open_df(file_path:pathlib.Path) -> typing.Optional[pandas.DataFrame]:
#     EXCEL_MIME_TYPE = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
#     if not file_path.is_file():
#         return None
#     file_mimetype = mimetypes.guess_file_type(file_path)
#     try:
#         match file_mimetype:
#             case EXCEL_MIME_TYPE:
#                 return pandas.read_excel(file_path)
#             case _:
#                 return pandas.read_csv(file_path)
#     except:
#         return None


if __name__ != "__main__":
    print("this script is not meant to be imported")
    sys.exit(1)

if len(sys.argv) != 3:
    print(f"usage: {sys.argv[0]} input.csv output.csv")
    sys.exit(1)

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

input_df = pandas.read_csv(input_file_path, sep=";", encoding="utf-8-sig")

output_df = fix_df(input_df)

output_df.to_csv(output_file_path, index=False)
