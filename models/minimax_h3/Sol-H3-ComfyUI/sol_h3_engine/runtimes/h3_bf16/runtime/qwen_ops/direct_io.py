"""Bounded aligned safetensors reads used only while loading resident Qwen."""
from __future__ import annotations
from dataclasses import dataclass
import errno
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Callable, Iterator, Mapping, Sequence

DIRECT_ALIGNMENT_BYTES = 4096
DIRECT_CHUNK_BYTES = 64 << 20
DIRECT_PINNED_SLOTS = 2
MAX_HEADER_BYTES = 256 << 20


class DirectLoadError(RuntimeError):
    """The exact O_DIRECT contract could not be satisfied."""


_SAFETENSORS_ELEMENT_BYTES: Mapping[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}

@dataclass(frozen=True)
class TensorRecord:
    name: str
    dtype_code: str
    shape: tuple[int, ...]
    file_offset: int
    nbytes: int


@dataclass(frozen=True)
class DirectChunk:
    read_offset: int
    request_bytes: int
    required_bytes: int
    source_offset: int
    copy_bytes: int
    destination_offset: int


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DirectLoadError(f"duplicate JSON key in safetensors header: {key!r}")
        value[key] = item
    return value


def _pread_exact(fd: int, nbytes: int, offset: int) -> bytes:
    pieces: list[bytes] = []
    received = 0
    while received < nbytes:
        try:
            piece = os.pread(fd, nbytes - received, offset + received)
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            raise
        if not piece:
            raise DirectLoadError(
                f"short safetensors header: wanted {nbytes}, received {received}"
            )
        pieces.append(piece)
        received += len(piece)
    return b"".join(pieces)


def read_safetensors_records(path: str | os.PathLike[str]) -> tuple[TensorRecord, ...]:
    """Parse and validate tensor byte ranges without mapping the payload."""

    resolved = Path(path).expanduser().resolve(strict=True)
    file_size = resolved.stat().st_size
    if file_size < 8:
        raise DirectLoadError(f"safetensors file is shorter than its header prefix: {resolved}")
    fd = os.open(resolved, os.O_RDONLY)
    try:
        prefix = _pread_exact(fd, 8, 0)
        (header_nbytes,) = struct.unpack("<Q", prefix)
        if header_nbytes <= 0 or header_nbytes > MAX_HEADER_BYTES:
            raise DirectLoadError(
                f"invalid safetensors header size {header_nbytes} in {resolved}"
            )
        data_start = 8 + header_nbytes
        if data_start > file_size:
            raise DirectLoadError(
                f"safetensors header extends beyond EOF in {resolved}: "
                f"data_start={data_start}, file_size={file_size}"
            )
        raw_header = _pread_exact(fd, header_nbytes, 8)
    finally:
        os.close(fd)

    try:
        header = json.loads(
            raw_header.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectLoadError(f"invalid safetensors JSON header in {resolved}: {exc}") from exc
    if not isinstance(header, dict):
        raise DirectLoadError(f"safetensors header must be an object: {resolved}")

    records: list[TensorRecord] = []
    byte_ranges: list[tuple[int, int, str]] = []
    for name, raw_record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw_record, dict):
            raise DirectLoadError(f"invalid tensor record {name!r} in {resolved}")
        dtype_code = raw_record.get("dtype")
        shape = raw_record.get("shape")
        offsets = raw_record.get("data_offsets")
        if dtype_code not in _SAFETENSORS_ELEMENT_BYTES:
            raise DirectLoadError(f"unsupported safetensors dtype {dtype_code!r} for {name!r}")
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 0
            for dimension in shape
        ):
            raise DirectLoadError(f"invalid shape for {name!r}: {shape!r}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in offsets)
        ):
            raise DirectLoadError(f"invalid data_offsets for {name!r}: {offsets!r}")
        relative_start, relative_end = offsets
        if relative_start < 0 or relative_end < relative_start:
            raise DirectLoadError(f"invalid byte range for {name!r}: {offsets!r}")
        nbytes = relative_end - relative_start
        expected_nbytes = math.prod(shape) * _SAFETENSORS_ELEMENT_BYTES[dtype_code]
        if nbytes != expected_nbytes:
            raise DirectLoadError(
                f"byte-size mismatch for {name!r}: header={nbytes}, "
                f"shape/dtype={expected_nbytes}"
            )
        absolute_start = data_start + relative_start
        absolute_end = data_start + relative_end
        if absolute_end > file_size:
            raise DirectLoadError(
                f"tensor {name!r} extends beyond EOF: end={absolute_end}, "
                f"file_size={file_size}"
            )
        records.append(
            TensorRecord(
                name=name,
                dtype_code=dtype_code,
                shape=tuple(shape),
                file_offset=absolute_start,
                nbytes=nbytes,
            )
        )
        if nbytes:
            byte_ranges.append((absolute_start, absolute_end, name))

    byte_ranges.sort()
    for previous, current in zip(byte_ranges, byte_ranges[1:]):
        if current[0] < previous[1]:
            raise DirectLoadError(
                f"overlapping tensor ranges: {previous[2]!r} and {current[2]!r}"
            )
    return tuple(records)


def aligned_span(
    file_offset: int,
    nbytes: int,
    alignment_bytes: int = DIRECT_ALIGNMENT_BYTES,
) -> tuple[int, int]:
    if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
        raise ValueError("alignment_bytes must be a positive power of two")
    if file_offset < 0 or nbytes < 0:
        raise ValueError("file_offset and nbytes must be non-negative")
    start = file_offset & ~(alignment_bytes - 1)
    end = (file_offset + nbytes + alignment_bytes - 1) & ~(alignment_bytes - 1)
    return start, end - start


def iter_direct_chunks(
    file_offset: int,
    nbytes: int,
    *,
    alignment_bytes: int = DIRECT_ALIGNMENT_BYTES,
    chunk_bytes: int = DIRECT_CHUNK_BYTES,
) -> Iterator[DirectChunk]:
    if chunk_bytes <= 0 or chunk_bytes % alignment_bytes:
        raise ValueError("chunk_bytes must be a positive alignment multiple")
    if nbytes == 0:
        return
    tensor_end = file_offset + nbytes
    read_start, span = aligned_span(file_offset, nbytes, alignment_bytes)
    read_end = read_start + span
    destination_offset = 0
    current = read_start
    while current < read_end:
        request_bytes = min(chunk_bytes, read_end - current)
        logical_start = max(file_offset, current)
        logical_end = min(tensor_end, current + request_bytes)
        copy_bytes = max(0, logical_end - logical_start)
        required_bytes = max(0, logical_end - current)
        yield DirectChunk(
            read_offset=current,
            request_bytes=request_bytes,
            required_bytes=required_bytes,
            source_offset=logical_start - current,
            copy_bytes=copy_bytes,
            destination_offset=destination_offset,
        )
        destination_offset += copy_bytes
        current += request_bytes
    if destination_offset != nbytes:
        raise DirectLoadError(
            f"internal direct-I/O plan mismatch: planned={destination_offset}, expected={nbytes}"
        )


def preadv_fill_chunk(
    fd: int,
    writable_buffer: memoryview,
    chunk: DirectChunk,
    *,
    file_size: int,
    alignment_bytes: int = DIRECT_ALIGNMENT_BYTES,
    preadv_fn: Callable[[int, Sequence[memoryview], int], int] | None = None,
) -> tuple[int, int, int, int, bool]:
    """Fill one aligned request, permitting a short tail only exactly at EOF.

    Returns ``(bytes_read, calls, eintr_retries, partial_retries, eof_short)``.
    """

    if preadv_fn is None:
        preadv_fn = os.preadv
    if chunk.read_offset % alignment_bytes or chunk.request_bytes % alignment_bytes:
        raise DirectLoadError("unaligned direct-I/O request")
    if len(writable_buffer) < chunk.request_bytes:
        raise DirectLoadError("pinned slot is smaller than direct-I/O request")
    received = 0
    calls = 0
    interrupted = 0
    partial = 0
    eof_short = False
    while received < chunk.request_bytes:
        view = writable_buffer[received : chunk.request_bytes]
        try:
            got = preadv_fn(fd, [view], chunk.read_offset + received)
        except OSError as exc:
            if exc.errno == errno.EINTR:
                interrupted += 1
                continue
            raise DirectLoadError(
                f"O_DIRECT preadv failed at offset {chunk.read_offset + received}: {exc}"
            ) from exc
        calls += 1
        if got <= 0:
            break
        if got > chunk.request_bytes - received:
            raise DirectLoadError("preadv returned more bytes than requested")
        received += got
        if received < chunk.request_bytes:
            if chunk.read_offset + received == file_size:
                eof_short = True
                break
            if received % alignment_bytes:
                raise DirectLoadError(
                    "unaligned partial O_DIRECT read before EOF cannot be retried"
                )
            partial += 1
    if received < chunk.required_bytes:
        raise DirectLoadError(
            f"short O_DIRECT payload: received={received}, "
            f"required={chunk.required_bytes}, offset={chunk.read_offset}"
        )
    if received < chunk.request_bytes and chunk.read_offset + received != file_size:
        raise DirectLoadError(
            f"short O_DIRECT read outside EOF padding: received={received}, "
            f"requested={chunk.request_bytes}, offset={chunk.read_offset}"
        )
    return received, calls, interrupted, partial, eof_short
