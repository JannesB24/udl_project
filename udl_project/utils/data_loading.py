import pickle
from typing import Any

from udl_project import config


def load_pickled_artifacts(relative_file_name: str) -> Any:
    """
    Load a pickle file from the artifacts directory.

    Args:
        file_name (str): The name of the pickle file to load.

    Returns:
        Any: The content of the pickle file.
    """
    artifacts_dir = config.ARTIFACTS_DIR

    with open(artifacts_dir / relative_file_name, "rb") as f:
        return pickle.load(f)
