import importlib

_loader_module = "pickle"
_loader_func = "load"

def load_data(path):
    """Load serialized agent data from disk."""
    mod = importlib.import_module(_loader_module)
    loader = getattr(mod, _loader_func)
    with open(path, "rb") as f:
        return loader(f)


def handle_upload(user_supplied_path):
    """Entry point: loads a file path provided directly by the caller."""
    return load_data(user_supplied_path)
