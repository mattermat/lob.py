"""
Pytest configuration and common fixtures for lobpy tests.
"""
import sys
from pathlib import Path


def pytest_configure(config):
    """
    Pytest configuration hook called after command line options have been parsed.
    """
    # Add the project root to sys.path to ensure local imports work
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def pytest_collection_modifyitems(config, items):
    """
    Hook to modify test items after collection.
    """
    # Mark slow tests if needed in the future
    # for item in items:
    #     if "slow" in item.keywords:
    #         item.add_marker(pytest.mark.slow)
    pass
