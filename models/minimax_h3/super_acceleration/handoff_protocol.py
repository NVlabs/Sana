#!/usr/bin/env python3
"""Small authenticated JSON RPC used by the two local pipeline pairs.

TCP on loopback is the production default because the H3 process runs in an
Enroot container while the LTX process uses the host sol-engine environment.
For debugging, ``unix:///tmp/...`` is also supported; never place the socket on
Lustre.
"""

from __future__ import annotations

import json
import hmac
import socket
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_MESSAGE_BYTES = 32 * 1024 * 1024
MAX_TENSOR_HEADER_BYTES = 64 * 1024
MAX_TENSOR_TOKEN_BYTES = 512
TENSOR_PROTOCOL = "h3-ltx-tensor-handoff-v1"
TENSOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TensorSpec:
    """Wire description for one contiguous CPU tensor.

    Production callers use the two fixed specs below.  Supplying alternate
    specs to :class:`TensorServer` is intentionally supported only so the wire
    protocol can be unit-tested without allocating the 180 MiB video buffer.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "nbytes": self.nbytes,
        }


VIDEO_SPEC = TensorSpec(
    name="video",
    dtype="bfloat16",
    shape=(1, 3, 121, 384, 672),
    nbytes=187_342_848,
)
AUDIO_SPEC = TensorSpec(
    name="audio",
    dtype="float32",
    shape=(1, 2, 161_333),
    nbytes=1_290_664,
)
MAX_TENSOR_PAYLOAD_BYTES = VIDEO_SPEC.nbytes + AUDIO_SPEC.nbytes


def _bytes_view(value: Any, *, writable: bool, spec: TensorSpec) -> memoryview:
    """Return a one-dimensional byte view over a CPU tensor or buffer.

    PyTorch does not expose BF16 tensors through Python's buffer protocol, so
    the uint8 NumPy view is used for CPU tensors.  No copy is made.  Plain
    memoryviews/bytearrays are useful for tests and non-Torch producers.
    """

    if hasattr(value, "device") and hasattr(value, "view") and hasattr(value, "numpy"):
        device = getattr(value, "device")
        if getattr(device, "type", str(device)) != "cpu":
            raise ValueError(f"{spec.name} must be a CPU tensor, got {device}")
        if hasattr(value, "is_contiguous") and not value.is_contiguous():
            raise ValueError(f"{spec.name} CPU tensor must be contiguous")
        shape = tuple(int(item) for item in value.shape)
        if shape != spec.shape:
            raise ValueError(f"{spec.name} shape must be {spec.shape}, got {shape}")
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype != spec.dtype:
            raise ValueError(f"{spec.name} dtype must be {spec.dtype}, got {dtype}")
        # Import lazily so the JSON-only protocol keeps no torch dependency.
        import torch

        view = memoryview(value.view(torch.uint8).numpy())
    else:
        try:
            view = memoryview(value)
        except TypeError as exc:
            raise TypeError(f"{spec.name} does not expose a CPU buffer") from exc
    if not view.c_contiguous:
        raise ValueError(f"{spec.name} buffer must be C-contiguous")
    if writable and view.readonly:
        raise ValueError(f"{spec.name} receive buffer must be writable")
    try:
        byte_view = view.cast("B")
    except TypeError as exc:
        raise ValueError(f"{spec.name} buffer cannot be represented as bytes") from exc
    if byte_view.nbytes != spec.nbytes:
        raise ValueError(
            f"{spec.name} buffer must contain exactly {spec.nbytes} bytes, "
            f"got {byte_view.nbytes}"
        )
    return byte_view


def _tensor_header(
    header: dict[str, Any], video_spec: TensorSpec, audio_spec: TensorSpec
) -> dict[str, Any]:
    allowed = {"token", "pair_id", "seq", "op", "metadata"}
    unknown = set(header) - allowed
    missing = {"token", "pair_id", "seq", "op"} - set(header)
    if unknown or missing:
        raise ValueError(f"invalid tensor header keys: missing={missing}, unknown={unknown}")
    token = header["token"]
    if not isinstance(token, str) or not token or len(token.encode()) > MAX_TENSOR_TOKEN_BYTES:
        raise ValueError("tensor header token must be a non-empty bounded string")
    if type(header["pair_id"]) is not int or int(header["pair_id"]) < 0:
        raise ValueError("tensor header pair_id must be a non-negative integer")
    if type(header["seq"]) is not int or int(header["seq"]) < 0:
        raise ValueError("tensor header seq must be a non-negative integer")
    if not isinstance(header["op"], str) or not header["op"] or len(header["op"]) > 64:
        raise ValueError("tensor header op must be a short non-empty string")
    metadata = header.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("tensor header metadata must be a JSON object")
    value = {
        "protocol": TENSOR_PROTOCOL,
        "schema_version": TENSOR_SCHEMA_VERSION,
        "token": token,
        "pair_id": header["pair_id"],
        "seq": header["seq"],
        "op": header["op"],
        "video": video_spec.descriptor(),
        "audio": audio_spec.descriptor(),
        "metadata": metadata,
    }
    # Validate serializability and the limit before opening a connection.
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    if len(raw) > MAX_TENSOR_HEADER_BYTES:
        raise ValueError("tensor header exceeds its wire limit")
    return value


def _validate_tensor_header(
    value: dict[str, Any],
    *,
    expected_token: str,
    expected_pair_id: int,
    expected_seq: int,
    video_spec: TensorSpec,
    audio_spec: TensorSpec,
) -> None:
    required = {
        "protocol",
        "schema_version",
        "token",
        "pair_id",
        "seq",
        "op",
        "video",
        "audio",
        "metadata",
    }
    if set(value) != required:
        raise RuntimeError(f"tensor wire header keys mismatch: {set(value) ^ required}")
    if value["protocol"] != TENSOR_PROTOCOL or value["schema_version"] != TENSOR_SCHEMA_VERSION:
        raise RuntimeError("tensor wire protocol/schema mismatch")
    token = value["token"]
    if not isinstance(token, str) or not hmac.compare_digest(token, expected_token):
        raise PermissionError("tensor handoff token mismatch")
    if type(value["pair_id"]) is not int or value["pair_id"] != expected_pair_id:
        raise RuntimeError("tensor handoff pair_id mismatch")
    if type(value["seq"]) is not int or value["seq"] != expected_seq:
        raise RuntimeError("tensor handoff sequence mismatch")
    if not isinstance(value["op"], str) or not value["op"]:
        raise RuntimeError("invalid tensor handoff operation")
    if not isinstance(value["metadata"], dict):
        raise RuntimeError("invalid tensor handoff metadata")
    if value["video"] != video_spec.descriptor():
        raise RuntimeError("video tensor descriptor mismatch")
    if value["audio"] != audio_spec.descriptor():
        raise RuntimeError("audio tensor descriptor mismatch")


def _recv_exact_into(connection: socket.socket, target: memoryview) -> None:
    offset = 0
    while offset < target.nbytes:
        received = connection.recv_into(target[offset:], min(8 * 1024 * 1024, target.nbytes - offset))
        if received <= 0:
            raise EOFError(f"tensor payload ended at {offset}/{target.nbytes} bytes")
        offset += received


def _address(endpoint: str) -> tuple[int, str | tuple[str, int]]:
    if endpoint.startswith("tcp://"):
        host_port = endpoint.removeprefix("tcp://")
        host, separator, port = host_port.rpartition(":")
        if not separator or not host or not port.isdigit():
            raise ValueError(f"invalid TCP endpoint: {endpoint!r}")
        return socket.AF_INET, (host, int(port))
    if endpoint.startswith("unix://"):
        path = endpoint.removeprefix("unix://")
        if not path.startswith("/tmp/"):
            raise ValueError("Unix handoff sockets must be node-local under /tmp")
        return socket.AF_UNIX, path
    raise ValueError(f"unsupported endpoint: {endpoint!r}")


def _read_line(stream: Any, *, max_bytes: int = MAX_MESSAGE_BYTES) -> dict[str, Any]:
    raw = stream.readline(max_bytes + 1)
    if not raw or len(raw) > max_bytes or not raw.endswith(b"\n"):
        raise RuntimeError("invalid or oversized handoff message")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError("handoff message must be a JSON object")
    return value


def _write_line(stream: Any, value: dict[str, Any]) -> None:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
    if len(raw) > MAX_MESSAGE_BYTES:
        raise RuntimeError("handoff response is too large")
    stream.write(raw)
    stream.flush()


class JsonServer:
    def __init__(self, endpoint: str, *, timeout_s: float) -> None:
        family, address = _address(endpoint)
        self.endpoint = endpoint
        self.family = family
        self.address = address
        self.sock = socket.socket(family, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        if family == socket.AF_INET:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        else:
            path = Path(str(address))
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                mode = path.lstat().st_mode
                if not stat.S_ISSOCK(mode):
                    raise FileExistsError(f"refusing non-socket endpoint {path}")
                path.unlink()
        self.sock.bind(address)
        self.sock.listen(1)

    def receive(self) -> tuple[dict[str, Any], Any]:
        connection, _ = self.sock.accept()
        connection.settimeout(self.sock.gettimeout())
        stream = connection.makefile("rwb", buffering=0)
        try:
            request = _read_line(stream)
        except Exception:
            stream.close()
            connection.close()
            raise
        return request, (stream, connection)

    @staticmethod
    def respond(handle: Any, response: dict[str, Any]) -> None:
        stream, connection = handle
        try:
            _write_line(stream, response)
        finally:
            stream.close()
            connection.close()

    def close(self) -> None:
        self.sock.close()
        if self.family == socket.AF_UNIX:
            path = Path(str(self.address))
            if path.exists() and stat.S_ISSOCK(path.lstat().st_mode):
                path.unlink()


class TensorServer(JsonServer):
    """One listener supporting both tensor staging and the existing JSON RPC.

    Stage 2 should construct one ``TensorServer`` for an endpoint, call
    :meth:`receive_into` for the binary staging connection, then use inherited
    ``receive``/``respond`` for ordinary control requests.  This avoids trying
    to bind separate binary and JSON servers to the same endpoint.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_s: float,
        video_spec: TensorSpec = VIDEO_SPEC,
        audio_spec: TensorSpec = AUDIO_SPEC,
    ) -> None:
        super().__init__(endpoint, timeout_s=timeout_s)
        if video_spec.nbytes < 0 or audio_spec.nbytes < 0:
            raise ValueError("tensor wire sizes must be non-negative")
        if video_spec.nbytes + audio_spec.nbytes > MAX_TENSOR_PAYLOAD_BYTES:
            raise ValueError("tensor wire payload exceeds the production safety limit")
        self.video_spec = video_spec
        self.audio_spec = audio_spec
        self.last_tensor_receive_timing: dict[str, float] | None = None

    def receive_into(
        self,
        video_cpu: Any,
        audio_cpu: Any,
        *,
        expected_token: str,
        expected_pair_id: int,
        expected_seq: int,
    ) -> tuple[dict[str, Any], Any]:
        """Accept one staging connection and fill caller-owned CPU buffers.

        The destination buffers may be pinned CPU torch tensors or writable
        contiguous buffer objects.  Returning before sending an ACK lets the
        Stage-2 caller enqueue/copy these buffers to CUDA first.
        """

        video_view = _bytes_view(video_cpu, writable=True, spec=self.video_spec)
        audio_view = _bytes_view(audio_cpu, writable=True, spec=self.audio_spec)
        self.last_tensor_receive_timing = None
        accept_started_ns = time.perf_counter_ns()
        connection, _ = self.sock.accept()
        accepted_ns = time.perf_counter_ns()
        connection.settimeout(self.sock.gettimeout())
        stream = connection.makefile("rwb", buffering=0)
        payload_started_ns = time.perf_counter_ns()
        try:
            header = _read_line(stream, max_bytes=MAX_TENSOR_HEADER_BYTES)
            _validate_tensor_header(
                header,
                expected_token=expected_token,
                expected_pair_id=expected_pair_id,
                expected_seq=expected_seq,
                video_spec=self.video_spec,
                audio_spec=self.audio_spec,
            )
            _recv_exact_into(connection, video_view)
            _recv_exact_into(connection, audio_view)
            # The matching sender half-closes its write side after the fixed
            # payload.  This one-byte probe rejects trailing/ambiguous data.
            if connection.recv(1):
                raise RuntimeError("tensor payload contains trailing bytes")
        except Exception:
            stream.close()
            connection.close()
            raise
        payload_completed_ns = time.perf_counter_ns()
        self.last_tensor_receive_timing = {
            "accept_wait_s": (accepted_ns - accept_started_ns) / 1_000_000_000.0,
            "payload_receive_s": (
                payload_completed_ns - payload_started_ns
            ) / 1_000_000_000.0,
        }
        return header, (stream, connection, header)

    @staticmethod
    def ack_staged(
        handle: Any,
        *,
        tensor_token: str,
        copied_to_cuda: bool,
        timing: dict[str, Any] | None = None,
        **timing_fields: Any,
    ) -> None:
        """Acknowledge staging after Stage 2 has completed/enqueued its H2D copy."""

        stream, connection, header = handle
        try:
            if not isinstance(tensor_token, str) or not tensor_token.startswith("h3tensor://"):
                raise ValueError("tensor_token must be a non-empty h3tensor:// URI")
            if timing is not None and timing_fields:
                raise ValueError("pass timing or timing keyword fields, not both")
            timing_value = timing if timing is not None else timing_fields
            if not isinstance(timing_value, dict):
                raise ValueError("ACK timing must be a JSON object")
            response = {
                "protocol": TENSOR_PROTOCOL,
                "schema_version": TENSOR_SCHEMA_VERSION,
                "status": "staged",
                "pair_id": header["pair_id"],
                "seq": header["seq"],
                "tensor_token": tensor_token,
                "copied_to_cuda": copied_to_cuda,
                "video_nbytes": int(header["video"]["nbytes"]),
                "audio_nbytes": int(header["audio"]["nbytes"]),
                "timing": timing_value,
            }
            _write_line(stream, response)
        finally:
            stream.close()
            connection.close()


def request(
    endpoint: str,
    payload: dict[str, Any],
    *,
    connect_timeout_s: float,
    response_timeout_s: float,
) -> dict[str, Any]:
    family, address = _address(endpoint)
    deadline = time.monotonic() + connect_timeout_s
    last_error: OSError | None = None
    connection: socket.socket | None = None
    while time.monotonic() < deadline:
        candidate = socket.socket(family, socket.SOCK_STREAM)
        candidate.settimeout(min(5.0, max(0.1, deadline - time.monotonic())))
        try:
            candidate.connect(address)
        except OSError as exc:
            last_error = exc
            candidate.close()
            time.sleep(0.25)
            continue
        connection = candidate
        break
    if connection is None:
        raise TimeoutError(f"could not connect to {endpoint}: {last_error}")
    # Once a request is delivered it is never retried: a response timeout may
    # mean the GPU is still executing, and replaying would duplicate inference.
    connection.settimeout(response_timeout_s)
    stream = connection.makefile("rwb", buffering=0)
    try:
        _write_line(stream, payload)
        return _read_line(stream)
    finally:
        stream.close()
        connection.close()


def _stage_tensor(
    endpoint: str,
    header: dict[str, Any],
    video_cpu: Any,
    audio_cpu: Any,
    *,
    connect_timeout_s: float,
    response_timeout_s: float,
    video_spec: TensorSpec,
    audio_spec: TensorSpec,
) -> dict[str, Any]:
    """Wire implementation shared by the fixed production API and tiny tests."""

    if video_spec.nbytes + audio_spec.nbytes > MAX_TENSOR_PAYLOAD_BYTES:
        raise ValueError("tensor wire payload exceeds the production safety limit")
    video_view = _bytes_view(video_cpu, writable=False, spec=video_spec)
    audio_view = _bytes_view(audio_cpu, writable=False, spec=audio_spec)
    wire_header = _tensor_header(header, video_spec, audio_spec)
    family, address = _address(endpoint)
    deadline = time.monotonic() + connect_timeout_s
    last_error: OSError | None = None
    connection: socket.socket | None = None
    while time.monotonic() < deadline:
        candidate = socket.socket(family, socket.SOCK_STREAM)
        candidate.settimeout(min(5.0, max(0.1, deadline - time.monotonic())))
        try:
            candidate.connect(address)
        except OSError as exc:
            last_error = exc
            candidate.close()
            time.sleep(0.25)
            continue
        connection = candidate
        break
    if connection is None:
        raise TimeoutError(f"could not connect to {endpoint}: {last_error}")
    connection.settimeout(response_timeout_s)
    stream = connection.makefile("rwb", buffering=0)
    try:
        _write_line(stream, wire_header)
        connection.sendall(video_view)
        connection.sendall(audio_view)
        connection.shutdown(socket.SHUT_WR)
        response = _read_line(stream, max_bytes=MAX_TENSOR_HEADER_BYTES)
    finally:
        stream.close()
        connection.close()

    required = {
        "protocol",
        "schema_version",
        "status",
        "pair_id",
        "seq",
        "tensor_token",
        "copied_to_cuda",
        "video_nbytes",
        "audio_nbytes",
        "timing",
    }
    if set(response) != required:
        raise RuntimeError(f"tensor staging ACK keys mismatch: {set(response) ^ required}")
    if response["protocol"] != TENSOR_PROTOCOL or response["schema_version"] != TENSOR_SCHEMA_VERSION:
        raise RuntimeError("tensor staging ACK protocol/schema mismatch")
    if response["status"] != "staged":
        raise RuntimeError(f"tensor staging failed: {response}")
    if type(response["pair_id"]) is not int or response["pair_id"] != wire_header["pair_id"]:
        raise RuntimeError("tensor staging ACK pair_id mismatch")
    if type(response["seq"]) is not int or response["seq"] != wire_header["seq"]:
        raise RuntimeError("tensor staging ACK sequence mismatch")
    token = response["tensor_token"]
    if not isinstance(token, str) or not token.startswith("h3tensor://"):
        raise RuntimeError("tensor staging ACK has an invalid tensor_token")
    if response["copied_to_cuda"] is not True:
        raise RuntimeError("Stage 2 did not acknowledge a completed CUDA copy")
    if response["video_nbytes"] != video_spec.nbytes:
        raise RuntimeError("tensor staging ACK video size mismatch")
    if response["audio_nbytes"] != audio_spec.nbytes:
        raise RuntimeError("tensor staging ACK audio size mismatch")
    if not isinstance(response["timing"], dict):
        raise RuntimeError("tensor staging ACK timing must be a JSON object")
    return response


def stage_tensor(
    endpoint: str,
    header: dict[str, Any],
    video_cpu: Any,
    audio_cpu: Any,
    *,
    connect_timeout_s: float,
    response_timeout_s: float,
) -> dict[str, Any]:
    """Stage the fixed H3 video/audio tensors and await the Stage-2 H2D ACK.

    ``header`` must contain exactly ``token``, ``pair_id``, ``seq``, ``op`` and
    optional ``metadata``.  The payload must be contiguous CPU storage for:

    * BF16 video ``[1, 3, 121, 384, 672]`` (187,342,848 bytes)
    * FP32 audio ``[1, 2, 161333]`` (1,290,664 bytes)

    Pinned torch tensors are accepted without a host-side copy.  A successful
    return always contains ``status=staged``, an ``h3tensor://`` token and
    ``copied_to_cuda=true``.
    """

    return _stage_tensor(
        endpoint,
        header,
        video_cpu,
        audio_cpu,
        connect_timeout_s=connect_timeout_s,
        response_timeout_s=response_timeout_s,
        video_spec=VIDEO_SPEC,
        audio_spec=AUDIO_SPEC,
    )
