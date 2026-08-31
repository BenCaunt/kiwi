"""Compact wire format for live Kiwi occupancy maps."""

from dataclasses import dataclass
import struct
import zlib

import numpy as np


MAP_MAGIC = b"KVM1"
_HEADER = struct.Struct("<4sIIIfff")
_MAX_CELLS = 16 * 1024 * 1024


@dataclass(frozen=True)
class LiveOccupancyMap:
    data: np.ndarray
    resolution_m: float
    origin_x: float
    origin_y: float
    keyframes: int


def encode_occupancy_map(occupancy, keyframes):
    """Encode an OccupancyMap as a small, compressed Zenoh payload."""
    source = np.asarray(occupancy.data)
    resolution = float(occupancy.resolution_m)
    origin_x = float(occupancy.origin_x)
    origin_y = float(occupancy.origin_y)
    keyframes = int(keyframes)
    if (source.ndim != 2 or 0 in source.shape or
            source.size > _MAX_CELLS or
            np.any((source < -1) | (source > 100))):
        raise ValueError("occupancy map must be a bounded 2D array")
    if (not np.isfinite((resolution, origin_x, origin_y)).all() or
            resolution <= 0.0 or keyframes < 0):
        raise ValueError("invalid occupancy map metadata")
    data = source.astype(np.int8, copy=False)
    height, width = data.shape
    header = _HEADER.pack(
        MAP_MAGIC,
        width,
        height,
        keyframes,
        resolution,
        origin_x,
        origin_y,
    )
    return header + zlib.compress(data.tobytes(order="C"), level=3)


def decode_occupancy_map(payload):
    """Decode and validate one live occupancy-map payload."""
    payload = bytes(payload)
    if len(payload) < _HEADER.size:
        raise ValueError("occupancy map payload is truncated")
    magic, width, height, keyframes, resolution, origin_x, origin_y = \
        _HEADER.unpack_from(payload)
    cells = int(width) * int(height)
    if (magic != MAP_MAGIC or width == 0 or height == 0 or
            cells > _MAX_CELLS or
            not np.isfinite((resolution, origin_x, origin_y)).all() or
            resolution <= 0.0):
        raise ValueError("invalid occupancy map header")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(
            payload[_HEADER.size:], max_length=cells + 1)
    except zlib.error as exc:
        raise ValueError("invalid occupancy map data") from exc
    if (len(raw) != cells or not decompressor.eof or
            decompressor.unused_data):
        raise ValueError("occupancy map size does not match its header")
    data = np.frombuffer(raw, dtype=np.int8).reshape((height, width)).copy()
    if np.any((data < -1) | (data > 100)):
        raise ValueError("occupancy map contains invalid probabilities")
    return LiveOccupancyMap(
        data=data,
        resolution_m=float(resolution),
        origin_x=float(origin_x),
        origin_y=float(origin_y),
        keyframes=int(keyframes),
    )
