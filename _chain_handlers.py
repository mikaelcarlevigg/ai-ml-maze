"""
Handler registry. "agent_loader" ultimately deserializes whatever path
it is given, via the loader in _chain_loader.py.
"""
from _chain_loader import load_from_path

def _agent_loader(path):
    return load_from_path(path)

HANDLERS = {
    "agent_loader": _agent_loader,
}
