import pathlib


def main():
    project_path = pathlib.Path(__file__).resolve().parent
    data_dir_path = project_path / "data"
    data_dir_path.mkdir(exist_ok=True)
    print("Setup done")


if __name__ == "__main__":
    main()
