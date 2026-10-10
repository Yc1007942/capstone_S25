import pytest

from video_ingestion.recording_worker import prepare_session
from video_ingestion.sources import VideoSource


def _make_source(enabled=True):
    return VideoSource(
        source_id="camera_01",
        name="Test camera",
        source_type="file",
        location="test.mp4",
        enabled=enabled,
    )


def test_prepare_session_creates_directory(tmp_path):
    source = _make_source()
    storage_root = tmp_path / "recordings"

    session_id, session_dir = prepare_session(source, storage_root)

    assert session_dir == storage_root / source.source_id / session_id
    assert session_dir.is_dir()


def test_prepare_session_creates_different_sessions(tmp_path):
    source = _make_source()

    id1, dir1 = prepare_session(source, tmp_path)
    id2, dir2 = prepare_session(source, tmp_path)

    assert id1 != id2
    assert dir1.is_dir()
    assert dir2.is_dir()


def test_prepare_session_rejects_disabled_source(tmp_path):
    source = _make_source(enabled=False)
    storage_root = tmp_path / "recordings"

    with pytest.raises(ValueError, match="Video source is disabled"):
        prepare_session(source, storage_root)

    assert not storage_root.exists()
