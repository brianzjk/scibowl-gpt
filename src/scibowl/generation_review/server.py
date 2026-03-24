from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from scibowl.generation_review.store import GeneratedQuestionReviewStore, dump_json


def run_generated_question_review_server(
    *,
    runs_path: Path,
    output_path: Path,
    reviewer_id: str,
    host: str = "127.0.0.1",
    port: int = 8775,
    title: str = "Generated Question Review",
) -> None:
    store = GeneratedQuestionReviewStore(runs_path, output_path, reviewer_id)
    html = (Path(__file__).with_name("review_app.html")).read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                page = html.replace("__APP_TITLE__", title).replace("__REVIEWER_ID__", reviewer_id)
                self._respond_html(page)
                return
            if parsed.path == "/api/summary":
                self._respond_json(store.summary())
                return
            if parsed.path == "/api/session":
                query = parse_qs(parsed.query)
                filter_name = query.get("filter", ["unreviewed"])[0]
                self._respond_json(
                    {
                        "items": store.session_items(filter_name=filter_name),
                        "summary": store.summary(),
                    }
                )
                return
            if parsed.path.startswith("/api/question/"):
                draft_id = parsed.path.removeprefix("/api/question/")
                self._respond_json(store.question_payload(draft_id))
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            payload = json.loads(body.decode("utf-8"))
            updated = store.save_review(
                draft_id=payload["draft_id"],
                difficulty=payload.get("difficulty"),
                quality=payload.get("quality"),
                comment=payload.get("comment"),
                verifier_scores=payload.get("verifier_scores"),
                verifier_comment=payload.get("verifier_comment"),
            )
            self._respond_json({"question": updated, "summary": store.summary()})

        def log_message(self, format: str, *args: object) -> None:
            return

        def _respond_html(self, payload: str) -> None:
            data = payload.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _respond_json(self, payload: dict[str, object]) -> None:
            data = dump_json(payload)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving generated-question review app at http://{host}:{port}")
    print(f"Runs source: {runs_path}")
    print(f"Review output: {output_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
