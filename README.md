# script pipeline project

**IN REDACTION**

## Requirements

This project requires the [uv](https://docs.astral.sh/uv/) tool to run and CPython 3.12 (managed by uv).

Make is required for the automated pipeline.

## Setup 

Run the following command in the project's root

```sh
uv sync # sets up venv and dependencies
uv run setup.py # sets up directory structure
```


## Pipeline

To run the pipeline as described in [./doc/doc.md](./doc/doc.md) run:

```sh
make
```
```
```
