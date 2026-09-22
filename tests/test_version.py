"""The version pySIPNET reports about itself is the version it was packaged as."""

from importlib.metadata import version

from pysipnet.version import PYSIPNET_VERSION


def test_package_metadata_and_version_module_agree():
    """pyproject.toml and pysipnet/version.py both carry the version; RELEASING.md says set both."""
    assert version("pysipnet") == PYSIPNET_VERSION
