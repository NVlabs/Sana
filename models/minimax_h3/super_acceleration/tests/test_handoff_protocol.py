#!/usr/bin/env python3
"""Lightweight CPU tests for the local JSON + tensor handoff protocol."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import handoff_protocol as protocol


class TensorProtocolTest(unittest.TestCase):
    def test_tensor_stage_then_json_control_on_one_listener(self) -> None:
        video_spec = protocol.TensorSpec("video", "uint8", (7,), 7)
        audio_spec = protocol.TensorSpec("audio", "uint8", (5,), 5)
        socket_path = Path(tempfile.gettempdir()) / f"h3-handoff-test-{os.getpid()}.sock"
        endpoint = f"unix://{socket_path}"
        server = protocol.TensorServer(
            endpoint,
            timeout_s=5.0,
            video_spec=video_spec,
            audio_spec=audio_spec,
        )
        errors: list[BaseException] = []

        def serve() -> None:
            try:
                video = bytearray(7)
                audio = bytearray(5)
                header, handle = server.receive_into(
                    video,
                    audio,
                    expected_token="secret",
                    expected_pair_id=2,
                    expected_seq=9,
                )
                self.assertEqual(bytes(video), b"VIDEO!!")
                self.assertEqual(bytes(audio), b"AUDIO")
                self.assertEqual(header["metadata"], {"phase": "hot"})
                self.assertIsNotNone(server.last_tensor_receive_timing)
                assert server.last_tensor_receive_timing is not None
                self.assertGreaterEqual(
                    server.last_tensor_receive_timing["accept_wait_s"], 0.0
                )
                self.assertGreaterEqual(
                    server.last_tensor_receive_timing["payload_receive_s"], 0.0
                )
                server.ack_staged(
                    handle,
                    tensor_token="h3tensor://pair-2/seq-9",
                    copied_to_cuda=True,
                    timing={"h2d_s": 0.001},
                )
                request_value, json_handle = server.receive()
                self.assertEqual(request_value, {"op": "consume", "seq": 9})
                server.respond(json_handle, {"status": "succeeded", "seq": 9})
            except BaseException as exc:  # make thread failures visible to unittest
                errors.append(exc)

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        try:
            ack = protocol._stage_tensor(
                endpoint,
                {
                    "token": "secret",
                    "pair_id": 2,
                    "seq": 9,
                    "op": "stage_tensor",
                    "metadata": {"phase": "hot"},
                },
                memoryview(b"VIDEO!!"),
                memoryview(b"AUDIO"),
                connect_timeout_s=2.0,
                response_timeout_s=5.0,
                video_spec=video_spec,
                audio_spec=audio_spec,
            )
            self.assertEqual(ack["status"], "staged")
            self.assertTrue(ack["copied_to_cuda"])
            self.assertEqual(ack["tensor_token"], "h3tensor://pair-2/seq-9")
            response = protocol.request(
                endpoint,
                {"op": "consume", "seq": 9},
                connect_timeout_s=2.0,
                response_timeout_s=5.0,
            )
            self.assertEqual(response, {"status": "succeeded", "seq": 9})
            worker.join(timeout=5.0)
            self.assertFalse(worker.is_alive())
            if errors:
                raise errors[0]
        finally:
            server.close()

    def test_header_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            protocol._tensor_header(
                {
                    "token": "secret",
                    "pair_id": 0,
                    "seq": 0,
                    "op": "stage_tensor",
                    "unexpected": True,
                },
                protocol.VIDEO_SPEC,
                protocol.AUDIO_SPEC,
            )

    def test_exact_buffer_size_is_required(self) -> None:
        spec = protocol.TensorSpec("video", "uint8", (4,), 4)
        with self.assertRaisesRegex(ValueError, "exactly 4 bytes"):
            protocol._bytes_view(bytearray(3), writable=True, spec=spec)


if __name__ == "__main__":
    unittest.main()
