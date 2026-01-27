import pathlib


def main():
    project_path = pathlib.Path(__file__).resolve().parent
    data_dir_path = project_path / "data"
    data_dir_path.mkdir(exist_ok=True)

    for subdir in [
        "onet_classification",
        "tagger_data_raw",
        "tagger_data_processed",
        "ner_data_raw",
        "colision_data",
        "ner_data_processed",
        "merge_output",
    ]:
        data_subdir = data_dir_path / subdir
        data_subdir.mkdir(exist_ok=True)

    print("Setup done")


if __name__ == "__main__":
    main()
