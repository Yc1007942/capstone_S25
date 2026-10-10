from pathlib import Path
from uuid import uuid4

from video_ingestion.sources import VideoSource


def prepare_session(
    source: VideoSource,
    storage_root: Path,
) -> tuple[str, Path]:
    """Prepare an empty folder for one recording session;

    Args:
        source       : Video source whose source_id names the parent folder.
        storage_root : Base directory for recordings, e.g. Path("recordings").

    Returns:
        A (session_id, session_dir) tuple. The ID is a new UUID4 hex string,
        and session_dir is storage_root / source.source_id / session_id.
        For "camera_01", this creates recordings/camera_01/<session_id>/,
        including any missing parent directories.

    Raises:
        ValueError : If source.enabled is False; no directory is created.
        OSError    : If the directory cannot be created, e.g. permission denied.
    """

    if not source.enabled:
        raise ValueError("Video source is disabled")

    session_id = uuid4().hex

    session_dir = storage_root / source.source_id / session_id

    # Create folder and any missing parent folders
    session_dir.mkdir(parents=True, exist_ok=True)

    return session_id, session_dir
