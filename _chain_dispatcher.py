"""
Generic plugin-style dispatcher: looks up a handler by name and invokes it
with the caller-supplied argument. The dispatcher itself has no idea what
each handler does with the argument.
"""
from _chain_handlers import HANDLERS

def dispatch(handler_name, arg):
    handler = HANDLERS[handler_name]
    return handler(arg)
