import os

from src import config  # noqa: F401  (loads .env)
from src.status import STATUS_JSON


def publish_status() -> bool:
    token = os.environ.get("HF_TOKEN", "").strip()
    repo = os.environ.get("HF_STATUS_REPO", "").strip()
    if not token or not repo:
        print("[publish] HF_TOKEN or HF_STATUS_REPO missing in .env, skipping upload")
        return False
    from huggingface_hub import HfApi
    HfApi(token=token).upload_file(
        path_or_fileobj=str(STATUS_JSON),
        path_in_repo="status.json",
        repo_id=repo,
        repo_type="dataset",
        commit_message="hourly status update",
    )
    print("[publish] status.json uploaded")
    return True