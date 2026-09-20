"""
Simulated entry point: represents where external/user input first enters
the system (e.g. an API handler in a real service).
"""
from _chain_dispatcher import dispatch

def handle_request(request_path):
    # request_path originates from outside the process (e.g. a query
    # parameter or uploaded file path in a real API).
    return dispatch("agent_loader", request_path)
