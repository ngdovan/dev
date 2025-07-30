import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from transform import llm_define_rules, transform_data


def test_transform():
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
    result = transform_data(yang_data, ngsi_ld_template, rules)

    expected = {
        "id": "urn:ngsi-ld:Device:router1",
        "type": "Device",
        "name": {"type": "Property", "value": "router1"},
        "ip": {"type": "Property", "value": "192.168.1.1"},
        "location": {"type": "Property", "value": "lab"},
    }

    assert result == expected


def test_transform_with_training():
    yang_data = {
        "hostname": "router1",
        "ipv4": "192.168.1.1",
        "place": "lab",
    }

    ngsi_ld_template = {
        "id": "urn:ngsi-ld:Device:router1",
        "type": "Device",
        "name": {"type": "Property", "value": None},
        "ip": {"type": "Property", "value": None},
        "location": {"type": "Property", "value": None},
    }

    training = [("hostname", "name"), ("ipv4", "ip"), ("place", "location")]
    rules = llm_define_rules(yang_data, ngsi_ld_template, training)
    result = transform_data(yang_data, ngsi_ld_template, rules)

    assert result["name"]["value"] == "router1"
    assert result["ip"]["value"] == "192.168.1.1"
    assert result["location"]["value"] == "lab"
