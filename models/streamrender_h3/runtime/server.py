"""Loopback HTTP bridge with bounded, ordered reference and RGB queues."""
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


@dataclass
class Session:
    id: str
    config: dict
    capacity: int
    output_capacity: int
    next_frame: int = 0
    ready: bool = False
    closed: bool = False
    error: str | None = None
    metrics: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.frames = queue.Queue(self.capacity)
        self.outputs = queue.Queue(self.output_capacity)
        self.viewer = threading.Lock()

    def accept(self, index, png, controls):
        with self.lock:
            if self.closed:
                raise ValueError("Session closed")
            if not self.ready:
                raise queue.Full
            if index != self.next_frame or controls.get("frame") != index:
                raise ValueError("Reference/control sequence mismatch")
            self.frames.put_nowait((index, png, controls, time.monotonic()))
            self.next_frame += 1

    def collect(self, count, timeout):
        values = []
        deadline = time.monotonic() + timeout
        while len(values) < count and not self.closed:
            if time.monotonic() > deadline:
                self.error = "Reference producer idle timeout"
                self.closed = True
                break
            try:
                values.append(self.frames.get(timeout=min(.5, max(.001, deadline - time.monotonic()))))
            except queue.Empty:
                continue
        return None if self.closed else values


class Bridge:
    def __init__(self, settings):
        self.settings = settings
        self.starts = queue.Queue(1)
        self.sessions = {}
        self.lock = threading.Lock()
        self.worker_ready = False
        self.worker_error = None

    def create(self, config):
        with self.lock:
            if not self.worker_ready:
                raise RuntimeError("GPU worker is loading/warming up")
            if any(not session.closed for session in self.sessions.values()):
                raise ValueError("One active GPU session is supported; close it before restarting")
            if config.get("fps") != 24 or config.get("width") != 1344 or config.get("height") != 768:
                raise ValueError("Engine input must be 1344x768 at 24 fps")
            # Retain at most one finished session; no unbounded history storage.
            while not self.starts.empty():
                self.starts.get_nowait()
            self.sessions.clear()
            session = Session(uuid.uuid4().hex, config,
                              self.settings["max_pending_frames"],
                              self.settings["output_queue_chunks"])
            self.sessions[session.id] = session
            self.starts.put_nowait(session)
            return session

    def fail(self, error):
        self.worker_ready = False
        self.worker_error = str(error)
        for session in self.sessions.values():
            session.error = str(error)
            session.closed = True


def serve(bridge):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def reply(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/health":
                return self.reply(200, {"ready": bridge.worker_ready, "error": bridge.worker_error,
                                        "backend": bridge.settings["backend"]})
            pieces = path.strip("/").split("/")
            if len(pieces) != 2 or pieces[1] not in bridge.sessions:
                return self.reply(404, {"error": "Unknown session"})
            session = bridge.sessions[pieces[1]]
            if pieces[0] == "status":
                return self.reply(200, {"id": session.id, "ready": session.ready,
                                        "closed": session.closed, "error": session.error,
                                        "accepted_frames": session.next_frame,
                                        "queued_frames": session.frames.qsize(),
                                        "metrics": session.metrics})
            if pieces[0] != "stream":
                return self.reply(404, {"error": "Unknown endpoint"})
            if not session.viewer.acquire(blocking=False):
                return self.reply(409, {"error": "A viewer is already connected"})
            try:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=render")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.connection.settimeout(10)
                deadline = time.monotonic()
                while not session.closed:
                    try:
                        images, metadata = session.outputs.get(timeout=.5)
                    except queue.Empty:
                        continue
                    for index, image in enumerate(images):
                        if session.closed:
                            break
                        delay = deadline - time.monotonic()
                        if delay > 0:
                            time.sleep(min(delay, 1 / 24))
                        header = (f"--render\r\nContent-Type: image/jpeg\r\n"
                                  f"Content-Length: {len(image)}\r\n"
                                  f"X-Session: {session.id}\r\n"
                                  f"X-Chunk: {metadata['round']}\r\n\r\n").encode()
                        self.wfile.write(header + image + b"\r\n")
                        self.wfile.flush()
                        deadline = max(deadline + 1 / 24, time.monotonic())
            except (BrokenPipeError, ConnectionError, TimeoutError):
                session.closed = True
            finally:
                session.viewer.release()
                self.close_connection = True

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 0 or length > bridge.settings["max_body_bytes"]:
                    self.close_connection = True
                    return self.reply(413, {"error": "Request too large"})
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Truncated request")
                pieces = urlsplit(self.path).path.strip("/").split("/")
                if pieces == ["session"]:
                    session = bridge.create(json.loads(body))
                    return self.reply(200, {"id": session.id})
                if len(pieces) < 2 or pieces[1] not in bridge.sessions:
                    return self.reply(404, {"error": "Unknown session"})
                session = bridge.sessions[pieces[1]]
                if pieces[0] == "frame" and len(pieces) == 3:
                    if len(body) < 24 or body[:8] != b"\x89PNG\r\n\x1a\n":
                        raise ValueError("Expected PNG reference frame")
                    width = int.from_bytes(body[16:20], "big")
                    height = int.from_bytes(body[20:24], "big")
                    if (width, height) != (1344, 768):
                        raise ValueError("Reference frame dimensions mismatch")
                    index = int(pieces[2])
                    controls = json.loads(self.headers.get("X-Controls", "{}"))
                    session.accept(index, body, controls)
                    return self.reply(200, {"frame": index})
                if pieces[0] == "close":
                    session.closed = True
                    return self.reply(200, {"closed": True})
                return self.reply(404, {"error": "Unknown endpoint"})
            except queue.Full:
                self.send_response(429)
                self.send_header("Content-Length", "0")
                self.send_header("Retry-After", "1")
                self.end_headers()
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except RuntimeError as error:
                self.reply(503, {"error": str(error)})

    httpd = ThreadingHTTPServer((bridge.settings["host"], bridge.settings["port"]), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="render-http")
    thread.start()
    return httpd
