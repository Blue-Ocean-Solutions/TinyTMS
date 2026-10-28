"""One-page web check-in (stdlib only: wsgiref, email for multipart, html, json).

    python -m tinytms.web --port 8765 --receipts-dir receipts

GET  /                      form: load photo, document photo, voice note, mode, explain, language
POST /checkin               multipart -> the same pipeline as the CLI -> HTML with witnesses, verdict, receipt link
GET  /receipts/<name>.json  the signed receipt (safe names only: [A-Za-z0-9._-]+, no "..")

Binds 127.0.0.1 by default. Uploads live in a temporary directory for the length of the request; the
receipt keeps only each file's name and sha256.
"""

from __future__ import annotations

import argparse
import email.policy
import html
import os
import re
import sys
import tempfile
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Callable, Iterable

from .pipeline import EXPLAIN_MODES, MODES, CheckinResult, ClientFactory, InputError, run_checkin
from .receipt import redact_machine_paths
from .verify import DEFAULT_MAX_WEIGHT_KG
from .witness import DocWitness, VisionWitness, VoiceWitness

MAX_UPLOAD_BYTES = 40 * 1024 * 1024
SAFE_NAME = re.compile(r"[A-Za-z0-9._-]+")
FILE_FIELDS = ("load", "doc", "voice")
LANGS = ("es", "en")
# The page is a dock-side demo on one machine. Requests must name this machine (Host allowlist stops DNS
# rebinding), a browser POST must come from this origin (Origin/Referer check stops cross-site forms), and
# spending Meta API credits from the page needs an explicit --allow-api at start-up.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
PAID_MODES = ("api", "auto")


def host_name(value: str) -> str:
    """'localhost:8765' -> 'localhost', '[::1]:8765' -> '::1'; lower-case, no port."""
    v = (value or "").strip().lower()
    if v.startswith("["):
        return v[1:v.find("]")] if "]" in v else v[1:]
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


def origin_host(header: str) -> str | None:
    """Host of an Origin/Referer header; None when it cannot be parsed (e.g. 'null')."""
    from urllib.parse import urlsplit

    try:
        return (urlsplit(header.strip()).hostname or "").lower() or None
    except ValueError:
        return None

StartResponse = Callable[..., Any]

CSS = """
body{font:16px/1.45 system-ui,-apple-system,sans-serif;max-width:760px;margin:2rem auto;padding:0 1rem;color:#1b1f24}
h1{font-size:1.4rem;margin-bottom:.2rem} .sub{color:#57606a;margin-top:0}
fieldset{border:1px solid #d0d7de;border-radius:8px;margin:1rem 0;padding:.8rem 1rem}
label{display:block;margin:.5rem 0 .2rem;font-weight:600} select,input[type=file]{font-size:1rem}
button{font-size:1rem;padding:.55rem 1.2rem;border:0;border-radius:6px;background:#1f6feb;color:#fff;cursor:pointer}
.v{font-size:2rem;font-weight:800;padding:.4rem .9rem;border-radius:8px;display:inline-block}
.ACCEPT{background:#dafbe1;color:#116329} .HOLD{background:#ffebe9;color:#a40e26}
table{border-collapse:collapse;width:100%;margin:.6rem 0} td,th{border-bottom:1px solid #d0d7de;padding:.35rem;text-align:left;vertical-align:top}
.OK{color:#116329;font-weight:700} .ERROR,.NO_MODEL{color:#a40e26;font-weight:700}
.ex{background:#f6f8fa;border-left:4px solid #1f6feb;padding:.6rem .9rem;border-radius:4px}
small,.muted{color:#57606a} code{font-size:.9em}
"""


def _page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html lang=\"es\"><head><meta charset=\"utf-8\">"
            f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>").encode()


def _e(x: object) -> str:
    return html.escape("" if x is None else str(x))


def _options(values: Iterable[str], selected: str) -> str:
    return "".join(f"<option value=\"{_e(v)}\"{' selected' if v == selected else ''}>{_e(v)}</option>" for v in values)


def fixture_cases(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and SAFE_NAME.fullmatch(p.name) and (p / "vision.FIXTURE.json").is_file())


def form_html(cases: list[str]) -> bytes:
    body = f"""
<h1>tinyTMS · check-in</h1>
<p class="sub">Tres testigos y un juez. Los modelos informan; el juez (Python determinista) decide; el recibo va firmado.</p>
<form method="post" action="/checkin" enctype="multipart/form-data">
<fieldset><legend>Entradas</legend>
<label for="load">Foto de la carga</label><input type="file" id="load" name="load" accept="image/jpeg,image/png">
<label for="doc">Foto de la carta de porte / BOL</label><input type="file" id="doc" name="doc" accept="image/jpeg,image/png">
<label for="voice">Nota de voz del chofer (WAV)</label><input type="file" id="voice" name="voice" accept="audio/wav,.wav">
</fieldset>
<fieldset><legend>Cómo</legend>
<label for="mode">Modo</label><select id="mode" name="mode">{_options(MODES, "fixture")}</select>
<label for="fixture">Caso FIXTURE (sólo modo fixture; sin ficheros usa su media de ejemplo)</label>
<select id="fixture" name="fixture">{_options(cases, "accept" if "accept" in cases else (cases[0] if cases else ""))}</select>
<label for="explain">Explicar un HOLD al chofer</label><select id="explain" name="explain">{_options(EXPLAIN_MODES, "none")}</select>
<label for="lang">Idioma de la explicación</label><select id="lang" name="lang">{_options(LANGS, "es")}</select>
</fieldset>
<button type="submit">Registrar llegada</button>
</form>
<p class="muted"><small>fixture: testigos escritos a mano, ningún modelo · api: Meta Model API · local: Muse Glimmer en
esta máquina (sin voz local: siempre retiene) · auto: API y, si un testigo cae por red/HTTP o falta la llave, local.</small></p>
"""
    return _page("tinyTMS check-in", body)


def _witness_row(name: str, w: Any) -> str:
    seen = ""
    if isinstance(w, DocWitness):
        seen = _e(f"pieces={w.pieces} · weight_kg={w.weight_kg} · seal={w.seal} · hazmat={w.hazmat}")
    elif isinstance(w, VoiceWitness):
        seen = _e(f"count={w.count} · seal={w.seal}") + (f"<br><small>«{_e(w.transcript)}»</small>" if w.transcript else "")
    elif isinstance(w, VisionWitness):
        seen = _e(f"{w.concept} count={w.count}")
    if w.status != "OK":
        seen = ""
    return (f"<tr><th>{_e(name)}</th><td class=\"{_e(w.status)}\">{_e(w.status)}</td>"
            f"<td>{_e(w.provider)}<br><small>{_e(w.model or '-')}</small></td><td>{seen}"
            f"{('<br><small>' + _e(w.detail) + '</small>') if w.detail else ''}</td></tr>")


def result_html(res: CheckinResult) -> bytes:
    v = res.verdict
    reasons = "".join(f"<li><code>{_e(r)}</code></li>" for r in v.reasons) or "<li>ninguna</li>"
    rows = "".join(_witness_row(k, w) for k, w in res.witnesses.items())
    fb = "".join(f"<li>{_e(k)} → local-mlx: {_e(why)}</li>" for k, why in res.fallback.items())
    ex = ""
    if res.explanation is not None:
        e = res.explanation
        ex = (f"<h2>Explicación al chofer</h2><div class=\"ex\">{_e(e.get('text') or '')}"
              f"<br><small>{_e(e.get('status'))} · {_e(e.get('provider'))} {_e(e.get('model'))} · {_e(e.get('language'))}"
              f"{(' · ' + _e(e.get('detail'))) if e.get('detail') else ''}</small></div>"
              "<p class=\"muted\"><small>La explicación la escribe un modelo a partir de las razones del juez; "
              "no cambia el veredicto.</small></p>")
    prov = f"<p class=\"muted\">{_e(res.body['provenance'])}</p>" if "provenance" in res.body else ""
    body = f"""
<h1>tinyTMS · resultado</h1>
<p><span class="v {_e(v.decision)}">{_e(v.decision)}</span> <span class="muted">modo {_e(res.body['mode'])}</span></p>
{prov}
<h2>Razones del juez</h2><ul>{reasons}</ul>
<h2>Testigos</h2>
<table><tr><th>testigo</th><th>estado</th><th>proveedor / modelo</th><th>qué vio</th></tr>{rows}</table>
{('<h2>Respaldo local</h2><ul>' + fb + '</ul>') if fb else ''}
{ex}
<p>Recibo firmado: <a href="/receipts/{_e(res.path.name)}">{_e(res.path.name)}</a></p>
<p><a href="/">← otro check-in</a></p>
"""
    return _page(f"tinyTMS {v.decision}", body)


def parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    """(fields, files) from a multipart/form-data body. files: name -> (client filename, bytes)."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise InputError("expected multipart/form-data")
    head = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("latin-1")
    msg = BytesParser(policy=email.policy.HTTP).parsebytes(head + body)
    if not msg.is_multipart():
        raise InputError("malformed multipart body")
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not isinstance(name, str):
            continue
        data = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if filename is not None:
            if data:  # an empty file input means "not given"
                files[name] = (filename, data)
        else:
            fields[name] = data.decode("utf-8", "replace").strip()
    return fields, files


def _safe_filename(client_name: str, field: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "_", client_name.replace("\\", "/").rsplit("/", 1)[-1]).strip("._")
    return (base or field)[:80]


def make_app(
    receipts_dir: str | Path = "receipts",
    keys_dir: str | Path = "keys",
    fixtures_dir: str | Path = "fixtures",
    calls_log: str | Path | None = None,
    max_weight_kg: float = DEFAULT_MAX_WEIGHT_KG,
    client_factory: ClientFactory | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    allow_api: bool = False,
    allowed_hosts: Iterable[str] | None = None,
) -> Callable[[dict[str, Any], StartResponse], list[bytes]]:
    rdir, fdir = Path(receipts_dir), Path(fixtures_dir)
    hosts = {h.lower() for h in (allowed_hosts if allowed_hosts is not None else LOOPBACK_HOSTS)}

    def send(start: StartResponse, status: str, body: bytes, ctype: str = "text/html; charset=utf-8") -> list[bytes]:
        start(status, [("Content-Type", ctype), ("Content-Length", str(len(body))),
                       ("X-Content-Type-Options", "nosniff")])
        return [body]

    def error(start: StartResponse, status: str, msg: str) -> list[bytes]:
        body = _page("tinyTMS error", f"<h1>{_e(status)}</h1><p>{_e(redact_machine_paths(msg))}</p>"
                                     "<p><a href=\"/\">← volver</a></p>")
        return send(start, status, body)

    def checkin(environ: dict[str, Any], start: StartResponse) -> list[bytes]:
        # A browser POST carries Origin (or at least Referer); either must name this machine. A request
        # with neither is not a browser form and cannot be cross-site.
        for header in ("HTTP_ORIGIN", "HTTP_REFERER"):
            if environ.get(header):
                if origin_host(environ[header]) not in hosts:
                    return error(start, "403 Forbidden", "cross-site request refused: the form must be posted from this server")
                break
        try:
            n = int(environ.get("CONTENT_LENGTH") or 0)
        except ValueError:
            return error(start, "400 Bad Request", "bad Content-Length")
        if n < 0:
            return error(start, "400 Bad Request", "bad Content-Length")
        if n > max_upload_bytes:
            return error(start, "413 Payload Too Large", f"upload larger than {max_upload_bytes} bytes")
        try:
            fields, files = parse_multipart(environ["wsgi.input"].read(n), environ.get("CONTENT_TYPE", ""))
            mode = fields.get("mode", "fixture")
            if mode not in MODES:
                raise InputError(f"mode must be one of {MODES}")
            explain = fields.get("explain", "none")
            lang = fields.get("lang", "es")
            if not allow_api and (mode in PAID_MODES or explain == "api"):
                return error(start, "403 Forbidden",
                             "this page may not spend Meta API credits: start the server with --allow-api")
            fixture_dir = None
            if mode == "fixture":
                case = fields.get("fixture", "")
                if case not in fixture_cases(fdir):
                    raise InputError(f"unknown fixture case {case!r}")
                fixture_dir = fdir / case
            with tempfile.TemporaryDirectory(prefix="tinytms-web-") as tmp:
                paths: dict[str, Path | None] = {}
                for f in FILE_FIELDS:
                    if f in files:
                        p = Path(tmp) / f / _safe_filename(files[f][0], f)
                        p.parent.mkdir()
                        p.write_bytes(files[f][1])
                        paths[f] = p
                    else:
                        paths[f] = None
                res = run_checkin(mode, load=paths["load"], doc=paths["doc"], voice=paths["voice"],
                                  fixture_dir=fixture_dir, max_weight_kg=max_weight_kg, receipts_dir=rdir,
                                  keys_dir=keys_dir, calls_log=calls_log, explain=explain, lang=lang,
                                  client_factory=client_factory, origin="web")
        except InputError as e:
            return error(start, "400 Bad Request", str(e))
        except Exception as e:  # noqa: BLE001 - operational failure, shown literally (paths redacted)
            return error(start, "500 Internal Server Error", f"{type(e).__name__}: {e}")
        return send(start, "200 OK", result_html(res))

    def receipt(name: str, start: StartResponse) -> list[bytes]:
        if not SAFE_NAME.fullmatch(name) or ".." in name or not name.endswith(".json"):
            return error(start, "404 Not Found", "no such receipt")
        p = rdir / name
        if not p.is_file() or p.resolve().parent != rdir.resolve():
            return error(start, "404 Not Found", "no such receipt")
        return send(start, "200 OK", p.read_bytes(), "application/json; charset=utf-8")

    def app(environ: dict[str, Any], start: StartResponse) -> list[bytes]:
        method, path = environ.get("REQUEST_METHOD", "GET"), environ.get("PATH_INFO", "/") or "/"
        if host_name(environ.get("HTTP_HOST") or environ.get("SERVER_NAME", "")) not in hosts:
            return error(start, "403 Forbidden", "host not allowed: this page answers only to this machine")
        if path == "/" and method in ("GET", "HEAD"):
            return send(start, "200 OK", form_html(fixture_cases(fdir)))
        if path == "/checkin":
            if method != "POST":
                return error(start, "405 Method Not Allowed", "POST a multipart form to /checkin")
            return checkin(environ, start)
        if path.startswith("/receipts/") and method in ("GET", "HEAD"):
            return receipt(path[len("/receipts/"):], start)
        return error(start, "404 Not Found", "not found")

    return app


def main(argv: list[str] | None = None) -> int:
    from wsgiref.simple_server import make_server

    ap = argparse.ArgumentParser(prog="tinytms.web")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--receipts-dir", default="receipts")
    ap.add_argument("--keys-dir", default=os.environ.get("TINYTMS_KEYS_DIR", "keys"))
    ap.add_argument("--fixtures-dir", default="fixtures")
    ap.add_argument("--calls-log", default=None)
    ap.add_argument("--max-weight-kg", type=float, default=DEFAULT_MAX_WEIGHT_KG)
    ap.add_argument("--allow-api", action="store_true",
                    help="let the page use the Meta Model API (modes api/auto, explain=api); off by default")
    a = ap.parse_args(argv)
    app = make_app(a.receipts_dir, a.keys_dir, a.fixtures_dir, a.calls_log, a.max_weight_kg,
                   allow_api=a.allow_api, allowed_hosts=set(LOOPBACK_HOSTS) | {a.host.lower()})
    with make_server(a.host, a.port, app) as srv:
        print(f"tinyTMS web on http://{a.host}:{a.port}/  (Ctrl-C to stop)", file=sys.stderr)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
