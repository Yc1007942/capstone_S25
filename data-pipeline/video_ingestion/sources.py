from dataclasses import dataclass
from typing import Literal

SourceType = Literal["usb", "rtsp", "file"]


@dataclass(frozen=True)
class VideoSource:
    """Describe where video source comes from.

    Attributes:
        source_id   : Identifier used in recording folder names, e.g. "camera_01".
        name        : Human-readable label, e.g. "Entrance camera".
        source_type : "usb" for a camera, "rtsp" for a stream, or "file".
        location    : Device identifier (e.g. "0"), RTSP URL, or video file path,
                      depending on source_type.
        enabled     : Whether this source may start a session; defaults to True.
    """

    source_id: str
    name: str
    source_type: SourceType
    location: str
    enabled: bool = True
