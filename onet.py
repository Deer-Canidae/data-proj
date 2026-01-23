import pandas as pd

OnetTagReference = dict[str, str]


def load_onet_reference() -> OnetTagReference:
    df = pd.read_csv("./data/onet_classification/Onet_association_fixed.csv", sep=";")
    entry_iterator = df.itertuples(index=False, name=None)
    return dict(((str(tag), str(example)) for (tag, example) in entry_iterator))


def onet_alphabetical_tags(tag_ref: OnetTagReference) -> list[str]:
    return list(sorted(tag_ref.keys()))


__all__ = ["OnetTagReference", "load_onet_reference", "onet_alphabetical_tags"]
