"""Model EOS and per-request stop controls, shared by deployment releases."""
import json
from pathlib import Path


def _token_ids(values, vocab_size):
    if values is None:
        return frozenset()
    if type(values) is int:
        values = [values]
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise ValueError('Stop token IDs must be integers')
    if any(type(value) is not int or not 0 <= value < vocab_size for value in values):
        raise ValueError('Stop token ID is outside the model vocabulary')
    return frozenset(values)


def model_eos_token_ids(model_root: Path, tokenizer_eos, vocab_size=248320):
    result = _token_ids(tokenizer_eos, vocab_size)
    generation = model_root/'generation_config.json'
    config = model_root/'config.json'
    if generation.exists():
        result |= _token_ids(json.loads(generation.read_text()).get('eos_token_id'), vocab_size)
    elif config.exists():
        data = json.loads(config.read_text())
        result |= _token_ids(data.get('text_config', data).get('eos_token_id'), vocab_size)
    return result


def request_stop_token_ids(model_eos, explicit=None, ignore_eos=False, vocab_size=248320):
    result = _token_ids(explicit, vocab_size)
    if not ignore_eos:
        result |= _token_ids(model_eos, vocab_size)
    return result
