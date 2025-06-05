from pathlib import Path

# relative from repository root
ARTIFACTS_DIR = Path("artifacts/")


def ensure_artifacts_dir():
    """Ensure the artifacts directory exists."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


ensure_artifacts_dir()
