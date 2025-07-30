"""Transform YANG data to NGSI-LD using simple rules."""

import json
from copy import deepcopy
from difflib import SequenceMatcher


class ToyLLM:
    """A slightly larger language model with simple training."""

    def __init__(self, ngsi_ld_template):
        self.target_fields = [
            key for key in ngsi_ld_template.keys() if key not in {"id", "type"}
        ]
        self.trained_mapping = {}
        # Precompute embeddings for targets
        self._target_embeddings = {
            field: self._embed(field) for field in self.target_fields
        }

    @staticmethod
    def _embed(text):
        """Embed text as a bag of character bigrams."""
        tokens = [text[i : i + 2] for i in range(len(text) - 1)]
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _similarity(self, a, b):
        """Cosine similarity between two bag-of-bigram embeddings."""
        common = set(a).intersection(b)
        numerator = sum(a[k] * b[k] for k in common)
        denom_a = sum(v * v for v in a.values()) ** 0.5
        denom_b = sum(v * v for v in b.values()) ** 0.5
        if denom_a == 0 or denom_b == 0:
            return 0.0
        return numerator / (denom_a * denom_b)

    def train(self, examples):
        """Train using (yang_field, ngsi_field) pairs."""
        for yang, ngsi in examples:
            if ngsi in self.target_fields:
                self.trained_mapping[yang] = ngsi

    def predict_mapping(self, yang_fields):
        mapping = {}
        for field in yang_fields:
            if field in self.trained_mapping:
                target = self.trained_mapping[field]
                mapping[field] = f"{target}.value"
                continue

            best_target = None
            best_score = -1.0
            field_emb = self._embed(field)
            for candidate, cand_emb in self._target_embeddings.items():
                score = self._similarity(field_emb, cand_emb)
                if score > best_score:
                    best_score = score
                    best_target = candidate
            mapping[field] = f"{best_target}.value"
        return mapping


def llm_define_rules(yang_data, ngsi_ld_template, training_examples=None):
    """Determine field mapping rules using the ToyLLM model."""

    model = ToyLLM(ngsi_ld_template)
    if training_examples:
        model.train(training_examples)
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

    training = [
        ("hostname", "name"),
        ("ipv4", "ip"),
        ("place", "location"),
    ]
    rules = llm_define_rules(yang_data, ngsi_ld_template, training)
    transformed = transform_data(yang_data, ngsi_ld_template, rules)
    print(json.dumps(transformed, indent=2))
