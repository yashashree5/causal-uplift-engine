"""A tiny local HTTP server used instead of the real network in download tests.

It can serve static bodies with optional Range support, fail the first N requests with a
chosen status, truncate transfers mid-body, redirect, and mimic the Hugging Face Hub
metadata endpoint.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class Route:
    body: bytes = b""
    supports_range: bool = True
    status_sequence: list[int] = field(default_factory=list)  # consumed one per request before serving
    retry_after: str | None = None
    truncate_first: int = 0  # number of initial successful responses cut off halfway
    redirect_to: str | None = None
    content_type: str = "application/octet-stream"
    extra_headers: dict[str, str] = field(default_factory=dict)


class FixtureServer:
    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.requests: list[tuple[str, str, str | None]] = []  # (method, path, Range header)
        self._lock = threading.Lock()
        self._truncations: dict[str, int] = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):  # silence test output
                pass

            def do_HEAD(self):  # noqa: N802
                self._serve(head=True)

            def do_GET(self):  # noqa: N802
                self._serve(head=False)

            def _serve(self, head: bool) -> None:
                path = self.path.split("?", 1)[0]
                with server._lock:
                    server.requests.append((self.command, path, self.headers.get("Range")))
                    route = server.routes.get(path)
                    if route is None:
                        self._reply(404, b"not found")
                        return
                    if route.status_sequence:
                        status = route.status_sequence.pop(0)
                        headers = (
                            {"Retry-After": route.retry_after} if route.retry_after and status in (429, 503) else {}
                        )
                        self._reply(status, b"error", headers)
                        return
                    if route.redirect_to:
                        self.send_response(307)
                        self.send_header("Location", route.redirect_to)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    body = route.body
                    start = 0
                    status = 200
                    rng = self.headers.get("Range")
                    headers = dict(route.extra_headers)
                    headers["Content-Type"] = route.content_type
                    headers["Accept-Ranges"] = "bytes" if route.supports_range else "none"
                    if rng and route.supports_range and rng.startswith("bytes="):
                        start = int(rng[6:].split("-", 1)[0])
                        if start >= len(body):
                            self._reply(416, b"", {"Content-Range": f"bytes */{len(body)}"})
                            return
                        status = 206
                        headers["Content-Range"] = f"bytes {start}-{len(body) - 1}/{len(body)}"
                    payload = body[start:]
                    truncate = False
                    if route.truncate_first and server._truncations.get(path, 0) < route.truncate_first and not head:
                        server._truncations[path] = server._truncations.get(path, 0) + 1
                        truncate = True
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    if head:
                        return
                    if truncate:
                        self.wfile.write(payload[: max(1, len(payload) // 2)])
                        self.wfile.flush()
                        self.close_connection = True
                        return
                    self.wfile.write(payload)

            def _reply(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=lambda: self._httpd.serve_forever(poll_interval=0.01), daemon=True)

    # ---- lifecycle
    def start(self) -> "FixtureServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "FixtureServer":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    # ---- helpers
    def add(self, path: str, body: bytes = b"", **kwargs) -> Route:
        route = Route(body=body, **kwargs)
        self.routes[path] = route
        return route

    def add_hub_repo(
        self,
        repo_id: str,
        revision: str,
        files: dict[str, bytes],
        *,
        resolved_id: str | None = None,
        commit_sha: str = "c0ffee" * 6 + "abcd",
        lfs_sha256: dict[str, str] | None = None,
    ) -> None:
        """Serve Hub-style metadata and ``resolve`` URLs for ``files`` (path-in-repo -> bytes)."""
        import hashlib

        siblings = []
        for name, body in files.items():
            sha = (lfs_sha256 or {}).get(name, hashlib.sha256(body).hexdigest())
            siblings.append({"rfilename": name, "size": len(body), "lfs": {"sha256": sha, "size": len(body)}})
        payload = {"id": resolved_id or repo_id, "sha": commit_sha, "siblings": siblings}
        self.add(
            f"/api/datasets/{repo_id}/revision/{revision}",
            json.dumps(payload).encode(),
            content_type="application/json",
        )
        for name, body in files.items():
            sha = (lfs_sha256 or {}).get(name, hashlib.sha256(body).hexdigest())
            self.add(
                f"/datasets/{repo_id}/resolve/{revision}/{name}",
                body,
                extra_headers={
                    "X-Repo-Commit": commit_sha,
                    "X-Linked-Etag": f'"{sha}"',
                    "X-Linked-Size": str(len(body)),
                },
            )

    def count(self, path: str, method: str = "GET") -> int:
        return sum(1 for m, p, _ in self.requests if p == path and m == method)
