import queue
import json
import threading
import unittest
from runtime.server import Bridge, Session
from runtime.config import load_config, ROOT


class ProtocolTests(unittest.TestCase):
    def test_order_and_backpressure(self):
        session = Session("test", {}, 1, 1)
        session.ready = True
        session.accept(0, b"frame", {"frame": 0})
        with self.assertRaises(queue.Full):
            session.accept(1, b"frame", {"frame": 1})
        self.assertEqual(session.next_frame, 1)
        with self.assertRaises(ValueError):
            session.accept(2, b"frame", {"frame": 2})
        self.assertEqual(session.collect(1, 1)[0][0], 0)
        session.accept(1, b"frame", {"frame": 1})

    def test_close_unblocks_partial_chunk(self):
        session = Session("test", {}, 1, 1)
        session.closed = True
        self.assertIsNone(session.collect(8, 1))

    def test_single_active_session(self):
        bridge = Bridge({"max_pending_frames": 48, "output_queue_chunks": 2})
        bridge.worker_ready = True
        config = {"fps": 24, "width": 1344, "height": 768}
        a = bridge.create(config)
        with self.assertRaises(ValueError):
            bridge.create(config)
        a.closed = True
        b = bridge.create(config)
        self.assertNotEqual(a.id, b.id)
        self.assertEqual(len(bridge.sessions), 1)

    def test_release_profile(self):
        config, _ = load_config(ROOT / "configs/runtime.json", ROOT / "configs/assets.example.json")
        inference = config["inference"]
        self.assertEqual(inference["models"]["backbone"]["args"]["num_layers"], 50)
        self.assertEqual(inference["diffusion"]["sampling_timesteps"]["args"]["num_sampling_steps"], 2)
        for node in inference["diffusion"].values():
            self.assertIsInstance(node["args"].get("T", 1.0), float)
        self.assertFalse(inference["meta_model"]["qwen_reference_video"])
        decoder = inference["models"]["video_decoder"]["args"]["decoder_config"]
        self.assertEqual(decoder['decoder_kwargs']['num_layers'], 24)
        self.assertNotIn('asset://native_vae', json.dumps(config))


if __name__ == "__main__":
    unittest.main()
