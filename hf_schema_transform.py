"""Generate schema mapping using a HuggingFace model and apply it to CSV data."""

import csv
import json
from typing import Dict

from sentence_transformers import SentenceTransformer, util


def generate_schema_mapping(
    source_schema: Dict[str, str],
    target_schema: Dict[str, str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, str]:
    """Create a mapping between source and target schema fields using embeddings.

    The function encodes field names from both schemas using a HuggingFace model
    and matches each source field to the most similar target field.
    """

    model = SentenceTransformer(model_name)

    source_fields = list(source_schema.keys())
    target_fields = list(target_schema.keys())

    source_embeddings = model.encode(source_fields, convert_to_tensor=True)
    target_embeddings = model.encode(target_fields, convert_to_tensor=True)

    mapping_rule: Dict[str, str] = {}
    for index, source_field in enumerate(source_fields):
        similarities = util.pytorch_cos_sim(
            source_embeddings[index], target_embeddings
        )[0]
        best_index = int(similarities.argmax())
        mapping_rule[source_field] = target_fields[best_index]
    return mapping_rule


def generate_schema_mapping_from_files(
    source_schema_path: str,
    target_schema_path: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Dict[str, str]:
    """Read schema files and create a mapping between their fields."""

    with open(source_schema_path, "r", encoding="utf-8") as source_file:
        source_schema = json.load(source_file)
    with open(target_schema_path, "r", encoding="utf-8") as target_file:
        target_schema = json.load(target_file)
    return generate_schema_mapping(source_schema, target_schema, model_name)


def convert_csv_using_rule(
    source_csv_path: str,
    mapping_rule: Dict[str, str],
    target_csv_path: str,
) -> None:
    """Convert a CSV file from the source schema to the target schema."""

    with open(source_csv_path, newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        target_fieldnames = list(mapping_rule.values())
        with open(target_csv_path, "w", newline="", encoding="utf-8") as target_file:
            writer = csv.DictWriter(target_file, fieldnames=target_fieldnames)
            writer.writeheader()
            for row in reader:
                converted_row: Dict[str, str] = {}
                for source_field, target_field in mapping_rule.items():
                    converted_row[target_field] = row.get(source_field)
                writer.writerow(converted_row)


if __name__ == "__main__":
    source_schema_path = "source_schema.json"
    target_schema_path = "target_schema.json"
    source_csv_path = "source.csv"
    target_csv_path = "converted.csv"

    rule = generate_schema_mapping_from_files(source_schema_path, target_schema_path)
    convert_csv_using_rule(source_csv_path, rule, target_csv_path)
    print(json.dumps(rule, indent=2))
