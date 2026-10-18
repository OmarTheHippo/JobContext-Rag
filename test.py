"""Live diagnostic tests for the Job-Context Client, run against your real API.

Run from the app folder (next to jobcontext.txt), after Settings are saved in the app:

    py test.py embeddings      # embedding model: which request shapes work or hang
    py test.py chat            # chat model: plain vs streamed replies, short vs long
    py test.py all

Options:
    --timeout SECONDS    longest to wait for one request (default 120)

Each variation is sent once, with no retries, and timed: how long until the reply
started, how long in total, and how big it was. Results are printed and saved to
test-results-YYYYMMDD-HHMM.txt next to this file. Your API key is never printed or
saved. Uses the Base URL, models and key from the app's Settings.

Add new live tests as functions named test_<name>(app, out) and list them in TESTS.
"""
import importlib.machinery
import importlib.util
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT = 120.0

# Neutral sample text (no real project content), cut to the lengths each test needs.
SAMPLE = (
    "The contractor shall inspect the existing equipment, document its condition, and "
    "recommend repairs or replacement. Inspections cover piping, valves, pumps, tanks, "
    "controls, electrical connections and safety devices. Findings are summarised in a "
    "report with photographs, a prioritised list of deficiencies, and cost estimates. "
)


def sample(chars: int, n: int) -> list[str]:
    text = SAMPLE * (chars // len(SAMPLE) + 1)
    return [f"Passage {i + 1}. " + text[: chars - 12] for i in range(n)]


def load_app():
    """Load jobcontext.txt as a module to reuse its saved settings and key."""
    sys.path.insert(0, str(HERE))
    loader = importlib.machinery.SourceFileLoader("jobcontext_app", str(HERE / "jobcontext.txt"))
    spec = importlib.util.spec_from_loader("jobcontext_app", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Out:
    """Prints and keeps every line for the results file."""
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(text)


def probe(url: str, body: dict, headers: dict, stream: bool = False) -> dict:
    """Send one request (no retries) and time it. Never raises."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    started, reply_at, received, first_piece_at = time.time(), None, 0, None
    result = {"request_kb": len(data) / 1024}
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            reply_at = time.time() - started
            result["status"] = r.status
            result["content_type"] = r.headers.get("Content-Type", "")
            result["encoding"] = r.headers.get("Content-Encoding", "")
            chunks = []
            while True:
                chunk = r.read(4096 if stream else 65536)
                if not chunk:
                    break
                if first_piece_at is None:
                    first_piece_at = time.time() - started
                chunks.append(chunk)
                received += len(chunk)
            raw = b"".join(chunks)
        result.update(ok=True, raw=raw)
    except urllib.error.HTTPError as e:
        result.update(ok=False, status=e.code,
                      error=f"HTTP {e.code}: {e.read().decode(errors='replace')[:200]}")
    except Exception as e:                                  # noqa: BLE001 - report anything
        reason = getattr(e, "reason", None) or e
        where = ("before any reply arrived" if reply_at is None
                 else f"while downloading the reply ({received // 1024:,} KB received)")
        result.update(ok=False, error=f"{type(reason).__name__}: {reason} - {where}")
    result.update(total=time.time() - started, reply_at=reply_at,
                  first_piece_at=first_piece_at, reply_kb=received / 1024)
    return result


def line(label: str, r: dict, extra: str = "") -> str:
    if r["ok"]:
        timing = f"reply after {r['reply_at']:.1f}s, done {r['total']:.1f}s"
        return f"  OK    {label:<46} {timing}, {r['reply_kb']:,.0f} KB {extra}".rstrip()
    return f"  FAIL  {label:<46} after {r['total']:.1f}s - {r['error']}"


# ---- embeddings ------------------------------------------------------------------------

def test_embeddings(app, out):
    s = app.load_settings()
    url, key = app._embeddings_endpoint(s), app.get_api_key()
    model = (s.get("embedding_model") or "").strip()
    out("EMBEDDINGS")
    out(f"  endpoint: {url}")
    out(f"  model:    {model or '(not set!)'}")
    if not (url and key and model):
        out("  Set the Base URL, API key and embedding model in the app's Settings first.")
        return

    plain = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    compact = dict(plain, **{"Accept-Encoding": "gzip", "Accept": "application/json"})

    def body(texts, b64):
        b = {"model": model, "input": texts}
        if b64:
            b["encoding_format"] = "base64"
        return b

    def dims(r):
        try:
            raw = r["raw"]
            if r.get("encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            first = json.loads(raw)["data"][0]["embedding"]
            return f"({len(first) if isinstance(first, list) else len(first) * 3 // 16} dims)"
        except Exception:                                   # noqa: BLE001
            return "(reply not readable as embeddings)"

    cases = [
        # label, texts, base64?, headers
        ("1 short text, plain numbers", sample(200, 1), False, plain),
        ("1 short text, base64 + gzip", sample(200, 1), True, compact),
        ("5 x 1,000 chars, plain numbers", sample(1000, 5), False, plain),
        ("5 x 1,000 chars, base64 + gzip", sample(1000, 5), True, compact),
        ("20 x 1,000 chars, plain numbers", sample(1000, 20), False, plain),
        ("20 x 1,000 chars, base64 + gzip", sample(1000, 20), True, compact),
        ("20 x 3,000 chars, base64 + gzip", sample(3000, 20), True, compact),
        ("20 x 1,000 chars, plain, other User-Agent", sample(1000, 20), False,
         dict(plain, **{"User-Agent": "JobContext-test/1.0"})),
    ]
    results = {}
    out(f"  Each request waits up to {TIMEOUT:.0f}s. Requests that hang will take that long.")
    for label, texts, b64, headers in cases:
        r = probe(url, body(texts, b64), headers)
        results[label] = r
        out(line(label, r, dims(r) if r["ok"] else ""))

    # Several at once, as the app does while indexing.
    out("  4 requests at once (5 x 1,000 chars each, base64 + gzip):")
    parallel = []
    threads = [threading.Thread(target=lambda: parallel.append(
        probe(url, body(sample(1000, 5), True), compact))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, r in enumerate(parallel, 1):
        out(line(f"   at once #{i}", r))

    # The app's own code path (one attempt, no retries).
    out("  The app's own embedding call (20 x 1,000 chars, one attempt):")
    started = time.time()
    try:
        vectors = app._genai_embed(sample(1000, 20), s, attempts=1)
        out(f"  OK    app._genai_embed                             {time.time() - started:.1f}s, "
            f"{len(vectors)} vectors of {len(vectors[0])} dims, format used: {app._EMBED_FORMAT[0]}")
    except Exception as e:                                   # noqa: BLE001
        out(f"  FAIL  app._genai_embed                             after {time.time() - started:.1f}s - {e}")

    # What the pattern suggests.
    ok = {k: v["ok"] for k, v in results.items()}
    out("")
    out("  What this suggests:")
    if not any(ok.values()):
        out("  - Nothing worked, even one short text: check the endpoint, model name and key,")
        out("    and whether the gateway expects a different request format or path.")
    elif ok["20 x 1,000 chars, base64 + gzip"] and not ok["20 x 1,000 chars, plain numbers"]:
        out("  - Large plain-number replies fail but compact ones work: reply size is the")
        out("    problem. The app now asks for base64 + gzip, which should fix indexing.")
    elif not ok["20 x 1,000 chars, plain numbers"] and ok["20 x 1,000 chars, plain, other User-Agent"]:
        out("  - The same request works with a different User-Agent: the gateway or proxy is")
        out("    treating Python's default User-Agent differently. Tell Claude.")
    elif all(ok.values()) and all(r["ok"] for r in parallel):
        out("  - Everything worked here. If indexing still fails, the difference is in how many")
        out("    requests run back to back; send this file and the app's console log.")
    elif all(ok.values()):
        out("  - Single requests work but several at once fail: set Settings > Simultaneous")
        out("    API requests to 1.")
    else:
        out("  - Mixed results: send this file to Claude.")


# ---- chat ------------------------------------------------------------------------------

def test_chat(app, out):
    s = app.load_settings()
    url, key, model = app._chat_endpoint(s), app.get_api_key(), app._model(s)
    out("CHAT")
    out(f"  endpoint: {url}")
    out(f"  model:    {model or '(not set!)'}")
    if not (url and key and model):
        out("  Set the Base URL, API key and model in the app's Settings first.")
        return
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    short = [{"role": "user", "content": "Reply with exactly: OK"}]
    long_ = [{"role": "user", "content": "Write about 600 words describing how to plan a "
                                          "routine equipment inspection. Plain prose."}]
    for label, messages, stream in (
        ("short reply, plain", short, False),
        ("short reply, streamed", short, True),
        ("long reply (~600 words), plain", long_, False),
        ("long reply (~600 words), streamed", long_, True),
    ):
        body = {"model": model, "messages": messages, "max_tokens": 2000}
        if stream:
            body["stream"] = True
        r = probe(url, body, headers, stream=stream)
        extra = ""
        if r["ok"] and stream:
            first = r["first_piece_at"]
            extra = (f"(first piece after {first:.1f}s, type {r.get('content_type', '')})"
                     if first is not None else "")
        out(line(label, r, extra))


TESTS = {"embeddings": test_embeddings, "chat": test_chat}


def main():
    global TIMEOUT
    args = sys.argv[1:]
    if "--timeout" in args:
        TIMEOUT = float(args[args.index("--timeout") + 1])
    names = [a for a in args if not a.startswith("--") and a in TESTS or a == "all"]
    if not names:
        print(__doc__)
        return
    if "all" in names:
        names = list(TESTS)

    out = Out()
    app = load_app()
    out(f"Job-Context Client {app.APP_VERSION} live tests - {time.strftime('%Y-%m-%d %H:%M')}")
    out(f"Python {sys.version.split()[0]}")
    out()
    for name in names:
        TESTS[name](app, out)
        out()

    path = HERE / time.strftime("test-results-%Y%m%d-%H%M.txt")
    path.write_text("\n".join(out.lines) + "\n", encoding="utf-8")
    print(f"Saved to {path.name} (no API key in it).")


if __name__ == "__main__":
    main()
