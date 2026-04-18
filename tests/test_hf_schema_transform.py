import json
import csv
from pathlib import Path

import pytest

sentence_transformers = pytest.importorskip("sentence_transformers")

from hf_schema_transform import (
    generate_schema_mapping_from_files,
    convert_csv_using_rule,
)


def test_generate_schema_mapping_and_convert_csv(tmp_path):
    temporary_directory: Path = tmp_path
    source_schema_file = temporary_directory / "schema_one.json"
    target_schema_file = temporary_directory / "schema_two.json"
    source_csv_file = temporary_directory / "source.csv"
    target_csv_file = temporary_directory / "target.csv"

    source_schema = {"firstName": "string", "age": "integer"}
    target_schema = {"name": "string", "age": "integer"}

    source_schema_file.write_text(json.dumps(source_schema))
    target_schema_file.write_text(json.dumps(target_schema))

    with source_csv_file.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["firstName", "age"])
        writer.writeheader()
        writer.writerow({"firstName": "Alice", "age": "30"})

    rule = generate_schema_mapping_from_files(
        str(source_schema_file), str(target_schema_file)
    )
    convert_csv_using_rule(str(source_csv_file), rule, str(target_csv_file))

    with target_csv_file.open("r", newline="", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        rows = list(reader)

    assert rule["firstName"] == "name"
    assert rule["age"] == "age"
    assert rows == [{"name": "Alice", "age": "30"}]
