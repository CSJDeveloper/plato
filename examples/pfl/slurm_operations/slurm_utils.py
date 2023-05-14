"""
Useful functions for slurm operations.
"""

import os
import json 
from typing import Any

import yaml

from plato.config import Loader


def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(
        os.path.join(loader.root_path, loader.construct_scalar(node))
    )
    extension = os.path.splitext(filename)[1].lstrip(".")

    with open(filename, "r", encoding="utf-8") as config_file:
        if extension in ("yaml", "yml"):
            return yaml.load(config_file, Loader)
        elif extension in ("json",):
            return json.load(config_file)
        else:
            return "".join(config_file.readlines())

def construct_join(loader: Loader, node: yaml.Node) -> Any:
    """Support os.path.join at node."""
    seq = loader.construct_sequence(node)
    return "/".join([str(i) for i in seq])


def load_yml_config(file_path: str) -> dict:
    """Load the configuration data from a yml file."""
    yaml.add_constructor("!include", construct_include, Loader)
    yaml.add_constructor("!join", construct_join, Loader)

    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as config_file:
            config = yaml.load(config_file, Loader)
    else:
        # if the configuration file does not exist, raise an error
        raise ValueError("A configuration file must be supplied.")

    return config


def get_server_scp_path(server_address: str, server_dir: str):
    """Obtain the server path."""

    return server_address + ":" + server_dir
