import io
import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
from runtime.server import Bridge, serve


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.bridge = Bridge({"backend": "passthrough", "host": "127.0.0.1", "port": 0,
                              "max_pending_frames": 1, "output_queue_chunks": 1,
                              "max_body_bytes": 12582912})
        self.bridge.worker_ready = True
        self.httpd = serve(self.bridge)
        self.url = "http://127.0.0.1:" + str(self.httpd.server_port)

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, path, body, headers=None):
        return urlopen(Request(self.url + path, data=body, headers=headers or {}, method="POST"), timeout=3)

    def test_accept_retry_close(self):
        response = self.post("/session", json.dumps({"fps": 24, "width": 1344, "height": 768}).encode())
        session_id = json.load(response)["id"]
        session = self.bridge.sessions[session_id]
        session.ready = True
        image = io.BytesIO()
        Image.new("RGB", (1344, 768)).save(image, "PNG")
        for index in (0, 1):
            try:
                response = self.post(f"/frame/{session_id}/{index}", image.getvalue(),
                                     {"X-Controls": json.dumps({"frame": index})})
                self.assertEqual(index, json.load(response)["frame"])
            except HTTPError as error:
                self.assertEqual(error.code, 429)
        self.assertEqual(session.next_frame, 1)
        self.post("/close/" + session_id, b"").close()
        self.assertTrue(session.closed)


if __name__ == "__main__":
    unittest.main()
