"""Transform YANG data to NGSI-LD using simple rules."""

import json
from copy import deepcopy
from difflib import SequenceMatcher


class ToyLLM:
    """A very small language model for field name matching."""

    def __init__(self, ngsi_ld_template):
        self.target_fields = [
            key for key in ngsi_ld_template.keys() if key not in {"id", "type"}
        ]

    def predict_mapping(self, yang_fields):
        mapping = {}
        for field in yang_fields:
            best_target = None
            best_score = -1.0
            for candidate in self.target_fields:
                score = SequenceMatcher(None, field, candidate).ratio()
                if score > best_score:
                    best_score = score
                    best_target = candidate
            mapping[field] = f"{best_target}.value"
        return mapping


def llm_define_rules(yang_data, ngsi_ld_template):
    """Determine field mapping rules using a toy LLM model."""

    model = ToyLLM(ngsi_ld_template)
    return model.predict_mapping(yang_data.keys())


def get_value_by_path(data, path):
    """Retrieve a nested value using dot-separated path."""
    for key in path.split("."):
        data = data[key]
    return data


def set_value_by_path(data, path, value):
    """Set a nested value using dot-separated path."""
    keys = path.split(".")
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def transform_data(yang_data, ngsi_ld_template, rules):
    """Apply mapping rules to transform YANG JSON to NGSI-LD JSON."""
    result = deepcopy(ngsi_ld_template)
    for yang_field, ngsi_path in rules.items():
        value = get_value_by_path(yang_data, yang_field)
        set_value_by_path(result, ngsi_path, value)
    return result


if __name__ == "__main__":
    yang_data = {
        "name": "router1",
        "ip": "192.168.1.1",
        "location": "lab",
    }

    ngsi_ld_template = {
        "id": "urn:ngsi-ld:Device:router1",
        "type": "Device",
        "name": {"type": "Property", "value": None},
        "ip": {"type": "Property", "value": None},
        "location": {"type": "Property", "value": None},
    }

    rules = llm_define_rules(yang_data, ngsi_ld_template)
    transformed = transform_data(yang_data, ngsi_ld_template, rules)
    print(json.dumps(transformed, indent=2))
