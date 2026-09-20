"""
Low-level loader. This is the actual sink: an unsafe pickle.load call,
five hops away from the original external input in _chain_entry.py.
"""
import pickle

def load_from_path(path):
    with open(path, "rb") as f:
        return pickle.load(f)
