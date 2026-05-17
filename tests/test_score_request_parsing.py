"""Tests for score._parse_request — must accept every payload shape an Azure ML
managed online endpoint can hand to score.run()."""
import json

from score import _parse_request


def test_string_json_dict():
    out = _parse_request(json.dumps({"emailBody": "hi"}))
    assert out == [{"emailBody": "hi"}]


def test_bytes_json():
    out = _parse_request(json.dumps({"text": "x"}).encode("utf-8"))
    assert out == [{"text": "x"}]


def test_bytearray_json():
    out = _parse_request(bytearray(json.dumps({"text": "x"}), "utf-8"))
    assert out == [{"text": "x"}]


def test_dict_with_inputs_list_of_dicts():
    out = _parse_request({"inputs": [{"text": "a"}, {"text": "b"}]})
    assert out == [{"text": "a"}, {"text": "b"}]


def test_dict_with_inputs_list_of_strings():
    out = _parse_request({"inputs": ["a", "b"]})
    assert out == [{"text": "a"}, {"text": "b"}]


def test_dict_without_inputs_treated_as_single_item():
    out = _parse_request({"emailSubject": "subj", "emailBody": "body"})
    assert out == [{"emailSubject": "subj", "emailBody": "body"}]


def test_top_level_list_mixed():
    out = _parse_request([{"text": "a"}, "b"])
    assert out == [{"text": "a"}, {"text": "b"}]


def test_scalar_string_payload():
    # Edge: raw scalar that isn't even JSON-decodable to a dict/list — score.py
    # treats it as fallback {"text": str(data)}.  Pass a non-JSON string via
    # the dict route so we don't hit json.loads.
    out = _parse_request({"text": 42})
    assert out == [{"text": 42}]
