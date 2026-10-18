"""Live diagnostic tests for the Job-Context Client, run against your real API.

Run from the app folder (next to jobcontext.txt), after Settings are saved in the app:

    py test.py replica         # your own system-prompt test, then step by step to the app's
    py test.py system          # are system messages and conversation turns honoured?
    py test.py openai          # the app's requests sent through the openai library
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
    """Prints each line and appends it to the results file straight away, so a run
    stopped with Ctrl+C still leaves its results."""
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, text=""):
        print(text, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


# Python's urllib (used by the app) routes requests through the proxy set in Windows'
# internet settings; many other clients (e.g. the openai library) don't.
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def probe(url: str, body: dict, headers: dict, stream: bool = False,
          direct: bool = False) -> dict:
    """Send one request (no retries) and time it. Never raises.
    direct=True bypasses any system proxy."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    started, reply_at, received, first_piece_at = time.time(), None, 0, None
    result = {"request_kb": len(data) / 1024}
    try:
        opener = DIRECT.open if direct else urllib.request.urlopen
        with opener(req, timeout=TIMEOUT) as r:
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

    # 1. How does the request travel? These run first and are quick to compare.
    proxies = urllib.request.getproxies()
    shown = {k: v.split("@")[-1] for k, v in proxies.items()}      # hide any credentials
    out(f"  system proxy Python sees: {shown or 'none'}")
    out("  How the request travels (one short text each):")
    first = {}
    for label, direct in (("via system proxy (as the app does)", False),
                          ("direct, bypassing system proxy", True)):
        r = probe(url, {"model": model, "input": ["test sentence"]}, plain, direct=direct)
        first[label] = r
        out(line(label, r))
    try:
        import openai                                             # what Cline's test used
        started = time.time()
        try:
            client = openai.OpenAI(base_url=url.rsplit("/embeddings", 1)[0], api_key=key,
                                   timeout=TIMEOUT, max_retries=0)
            resp = client.embeddings.create(model=model, input=["test sentence"])
            out(f"  OK    {'openai library (like the Cline test)':<46} "
                f"{time.time() - started:.1f}s, {len(resp.data[0].embedding)} dims")
            first["openai"] = {"ok": True}
        except Exception as e:                                   # noqa: BLE001
            out(f"  FAIL  {'openai library (like the Cline test)':<46} after "
                f"{time.time() - started:.1f}s - {type(e).__name__}: {str(e)[:150]}")
            first["openai"] = {"ok": False}
    except ImportError:
        out("  (openai library not installed - skipped)")
    via_proxy = first["via system proxy (as the app does)"]["ok"]
    direct_ok = first["direct, bypassing system proxy"]["ok"]
    if direct_ok and not via_proxy:
        out("  -> Direct works but the system proxy route fails: the proxy is the problem.")
        out("     Tell Claude; the app can be set to bypass it.")
    out("")

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

    # 2. Request shapes, all DIRECT, changing one thing at a time. The timings of the
    #    ones that work matter as much as which fail: if successes slow down as
    #    requests grow and failures all stop near the same time, the gateway has a
    #    time limit for non-streamed requests.
    cases = [
        # label, texts, base64?, headers
        ("1 x 200 chars, plain numbers", sample(200, 1), False, plain),
        ("1 x 200 chars, base64", sample(200, 1), True, plain),
        ("1 x 200 chars, plain + gzip/Accept headers", sample(200, 1), False, compact),
        ("1 x 1,000 chars", sample(1000, 1), False, plain),
        ("1 x 3,000 chars", sample(3000, 1), False, plain),
        ("1 x 12,000 chars", sample(12000, 1), False, plain),
        ("5 x 200 chars", sample(200, 5), False, plain),
        ("20 x 200 chars", sample(200, 20), False, plain),
        ("5 x 1,000 chars", sample(1000, 5), False, plain),
        ("20 x 1,000 chars", sample(1000, 20), False, plain),
    ]
    results = {}
    out("  Request shapes (direct; one thing changed at a time):")
    out(f"  Each request waits up to {TIMEOUT:.0f}s. Requests that hang will take that long.")
    for label, texts, b64, headers in cases:
        r = probe(url, body(texts, b64), headers, direct=True)
        results[label] = r
        out(line(label, r, dims(r) if r["ok"] else ""))

    # 3. Several at once, as the app does while indexing.
    out("  4 requests at once (direct, 1 x 200 chars each):")
    parallel = []
    threads = [threading.Thread(target=lambda: parallel.append(
        probe(url, body(sample(200, 1), False), plain, direct=True))) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i, r in enumerate(parallel, 1):
        out(line(f"   at once #{i}", r))

    # 4. The app's own code path (its Settings route, one attempt, no retries).
    for n in (1, 20):
        label = f"app._genai_embed, {n} x 1,000 chars"
        started = time.time()
        try:
            vectors = app._genai_embed(sample(1000, n), s, attempts=1)
            out(f"  OK    {label:<46} {time.time() - started:.1f}s, {len(vectors)} vectors of "
                f"{len(vectors[0])} dims, format used: {app._EMBED_FORMAT[0]}")
        except Exception as e:                               # noqa: BLE001
            out(f"  FAIL  {label:<46} after {time.time() - started:.1f}s - {str(e)[:160]}")

    # What the pattern suggests.
    oks = [(k, v["total"]) for k, v in results.items() if v["ok"]]
    fails = [v["total"] for v in results.values() if not v["ok"]]
    out("")
    out("  What this suggests:")
    if oks:
        out(f"  - Successful requests took {min(t for _, t in oks):.1f}s to "
            f"{max(t for _, t in oks):.1f}s (slowest: {max(oks, key=lambda x: x[1])[0]}).")
    if fails:
        out(f"  - Failed requests stopped after {min(fails):.0f}s to {max(fails):.0f}s.")
    if fails and max(fails) - min(fails) < 5 and min(fails) > 30:
        out("  - Failures all stop at about the same time: likely a time limit on")
        out("    non-streamed requests. The largest shape that worked is a safe batch size.")
    if not oks:
        out("  - Nothing worked directly either: send this file to Claude.")
    out("  Send this file to Claude.")


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
    out(f"  app's network route (Settings): {s.get('proxy_mode', 'env')}")
    out("  How the request travels (short reply, not streamed):")
    for label, direct in (("via system proxy", False), ("direct, bypassing system proxy", True)):
        out(line(label, probe(url, {"model": model, "messages": short, "max_tokens": 1024},
                              headers, direct=direct)))
    # Same short question, different reply limits: if larger limits fail although the
    # reply is still "OK", the model is probably spending time before answering (e.g.
    # hidden reasoning), and non-streamed requests run into a time limit.
    out("  Reply limit (max_tokens), same short question, direct:")
    # Not below ~1,000: reasoning models spend part of the limit on hidden thinking, so a
    # tiny limit can leave an empty reply even for "OK".
    for limit in (1024, 2000, 8192):
        for stream in (False, True):
            body = {"model": model, "messages": short, "max_tokens": limit}
            if stream:
                body["stream"] = True
            r = probe(url, body, headers, stream=stream, direct=True)
            label = f"max_tokens {limit:,}, {'streamed' if stream else 'plain'}"
            extra = ""
            if r["ok"] and not stream:
                try:
                    usage = json.loads(r["raw"]).get("usage") or {}
                    details = usage.get("completion_tokens_details") or {}
                    extra = (f"(reply tokens {usage.get('completion_tokens', '?')}, reasoning "
                             f"tokens {details.get('reasoning_tokens', 'not reported')})")
                except Exception:                            # noqa: BLE001
                    pass
            elif r["ok"] and r["first_piece_at"] is not None:
                extra = f"(first piece after {r['first_piece_at']:.1f}s)"
            out(line(label, r, extra))
    long_ = [{"role": "user", "content": "Write about 600 words describing how to plan a "
                                          "routine equipment inspection. Plain prose."}]
    out("  Long reply (~600 words), max_tokens 2,000, direct:")
    for stream in (False, True):
        body = {"model": model, "messages": long_, "max_tokens": 2000}
        if stream:
            body["stream"] = True
        r = probe(url, body, headers, stream=stream, direct=True)
        out(line("streamed" if stream else "plain", r))


# ---- openai library ---------------------------------------------------------------------

def test_openai(app, out):
    """The requests the app needs, sent through the openai library (what worked in
    the Cline test), to see whether switching the app to it would fix things."""
    out("OPENAI LIBRARY")
    try:
        import openai
    except ImportError:
        out("  The openai library is not installed: py -m pip install openai")
        return
    s = app.load_settings()
    key = app.get_api_key()
    base = app._chat_endpoint(s)
    base = base[: -len(app.CHAT_PATH)] if base.endswith(app.CHAT_PATH) else base
    chat_model, embed_model = app._model(s), (s.get("embedding_model") or "").strip()
    out(f"  openai library {getattr(openai, '__version__', '?')}, base URL {base}")
    out(f"  chat model {chat_model or '(not set!)'}, embedding model {embed_model or '(not set!)'}")
    client = openai.OpenAI(base_url=base, api_key=key, timeout=TIMEOUT, max_retries=0)

    def timed(label, fn):
        started = time.time()
        try:
            detail = fn()
            out(f"  OK    {label:<46} {time.time() - started:.1f}s {detail}")
        except Exception as e:                               # noqa: BLE001
            out(f"  FAIL  {label:<46} after {time.time() - started:.1f}s - "
                f"{type(e).__name__}: {str(e)[:150]}")

    if embed_model:
        for n, chars in ((1, 200), (5, 1000), (20, 1000), (20, 3000)):
            timed(f"embeddings {n} x {chars:,} chars",
                  lambda n=n, chars=chars: "({} vectors of {} dims)".format(
                      *(lambda r: (len(r.data), len(r.data[0].embedding)))(
                          client.embeddings.create(model=embed_model, input=sample(chars, n)))))

    if chat_model:
        short = [{"role": "user", "content": "Reply with exactly: OK"}]
        long_ = [{"role": "user", "content": "Write about 600 words describing how to plan a "
                                              "routine equipment inspection. Plain prose."}]

        def plain(messages):
            r = client.chat.completions.create(model=chat_model, messages=messages, max_tokens=2000)
            return f"({len(r.choices[0].message.content or '')} chars)"

        def streamed(messages):
            text = ""
            for chunk in client.chat.completions.create(model=chat_model, messages=messages,
                                                        max_tokens=2000, stream=True):
                if chunk.choices and chunk.choices[0].delta.content:
                    text += chunk.choices[0].delta.content
            return f"({len(text)} chars)"

        timed("chat short, plain, max_tokens 2,000", lambda: plain(short))
        timed("chat short, streamed", lambda: streamed(short))
        timed("chat long (~600 words), plain", lambda: plain(long_))
        timed("chat long (~600 words), streamed", lambda: streamed(long_))


# ---- system messages and conversation turns --------------------------------------------

def test_system(app, out):
    """Which parts of a request does the gateway pass to the model? Uses cooperative
    instructions (a marker at the start of the reply, a code word to recall) that don't
    fight the user's question, sent through the app's own request code."""
    out("SYSTEM MESSAGES AND CONVERSATION TURNS")
    s = app.load_settings()
    if not app._model(s):
        out("  Set the Base URL, API key and model in the app's Settings first.")
        return
    out(f"  app setting: message format {s.get('message_format', 'combined')}, "
        f"API client {s.get('api_client', 'auto')}")
    provider = app.PROVIDERS[s.get("provider", "genai")]
    marker = "[JC-OK]"
    rule = f"Formatting rule: begin every reply with the exact text {marker} followed by a space."
    sky = [{"role": "user", "content": "In one short sentence: what colour is the sky on a clear day?"}]
    turns = [
        {"role": "user", "content": "Please remember this code word for later: HYDRANT."},
        {"role": "assistant", "content": "Noted - I'll remember the code word."},
        {"role": "user", "content": "What was the code word I gave you? Reply with only the word."},
    ]
    transcript = ("# Prior Conversation Transcript\n[USER]\nPlease remember this code word for "
                  "later: HYDRANT.\n\n[ASSISTANT]\nNoted - I'll remember the code word.\n\n"
                  "# Current User Request\nWhat was the code word I gave you? Reply with only "
                  "the word.")

    def run(label, system, messages, fmt, expect, stream):
        started = time.time()
        label = f"{label}, {'streamed' if stream else 'not streamed'}"
        try:
            # Reset the app's "gateway doesn't stream" memory so each probe really
            # uses the mode it asks for.
            app._STREAM_REJECTED[0] = False
            reply = provider(system, messages, dict(s, message_format=fmt, stream=stream),
                             1).strip()
            ok = expect.lower() in reply.lower()
            out(f"  {'OK  ' if ok else 'MISS'}  {label:<58} {time.time() - started:.1f}s - "
                f"replied: {reply[:60]!r}")
            return ok
        except Exception as e:                               # noqa: BLE001
            out(f"  FAIL  {label:<58} after {time.time() - started:.1f}s - {str(e)[:140]}")
            return None              # the request errored: says nothing about the model

    base = run("1. plain request", "Answer briefly.",
               [{"role": "user", "content": "Reply with exactly this word and nothing else: "
                                            "PINEAPPLE"}], "native", "PINEAPPLE", False)
    native_rule = {st: run("2. rule as a system message", rule, sky, "native", marker, st)
                   for st in (False, True)}
    combined_rule = {st: run("3. rule inside the user message (Combined)", rule, sky,
                             "combined", marker, st) for st in (False, True)}
    native_turns = {st: run("4. earlier turns as real messages (Native)", "Answer briefly.",
                            turns, "native", "HYDRANT", st) for st in (False, True)}
    combined_turns = {st: run("5. earlier turns as a transcript (Combined)", "Answer briefly.",
                              [{"role": "user", "content": transcript}], "combined", "HYDRANT",
                              st) for st in (False, True)}

    out("")
    out("  What this suggests:")
    if not base:
        out("  - The model didn't follow even a plain request: the other results can't be read.")
        out("    Run again; if it repeats, send this file to Claude.")
        return
    # True = followed, False = MISS (model didn't follow), None = FAIL (request error).
    def worked(*values):
        return all(v is not False for v in values) and any(v is True for v in values)

    errors = sum(v is None for d in (native_rule, combined_rule, native_turns, combined_turns)
                 for v in d.values())
    if errors:
        out(f"  - {errors} request(s) FAILED with an error (see the FAIL lines). The model never")
        out("    saw those, so they say nothing about instructions; run again to check them.")
    plain_native = worked(native_rule[False], native_turns[False])
    streamed_native = worked(native_rule[True], native_turns[True])
    if plain_native and not streamed_native and False in (native_rule[True], native_turns[True]):
        out("  - System messages and earlier turns work when NOT streamed but are lost when")
        out("    streamed: the gateway's streaming path drops them. Tell Claude - the app can")
        out("    send Native-format requests without streaming.")
    elif plain_native and streamed_native:
        out("  - The gateway passes system messages and earlier turns either way: Native")
        out("    format works.")
    else:
        if False in native_rule.values():
            out("  - System messages are not reaching the model (or are ignored).")
        if False in native_turns.values():
            out("  - Earlier turns sent as separate messages are not reaching the model.")
        out("  - Keep Settings > Message format on Combined.")
    if False in combined_rule.values():
        out("  - Standing instructions inside the user message were not followed (MISS):")
        out("    layer prompts in Combined format may be ignored. Send this file to Claude.")
    if False in combined_turns.values():
        out("  - The model didn't recall the code word from a transcript: send this file to Claude.")


# ---- replica of the user's own system-prompt test ---------------------------------------

def test_replica(app, out):
    """Reproduce a system-prompt test that is known to work on this gateway, then change one
    thing at a time towards what the app sends, to find the difference that breaks it."""
    out("REPLICA OF YOUR SYSTEM-PROMPT TEST")
    try:
        import openai
    except ImportError:
        out("  The openai library is not installed: py -m pip install openai")
        return
    s = app.load_settings()
    key, model = app.get_api_key(), app._model(s)
    base = app._chat_endpoint(s)
    base = base[: -len(app.CHAT_PATH)] if base.endswith(app.CHAT_PATH) else base
    if not (base and key and model):
        out("  Set the Base URL, API key and model in the app's Settings first.")
        return
    max_tokens = int(s.get("max_tokens") or 4000)
    out(f"  model from Settings: {model}")
    out(f"  base URL: {base}")
    out(f"  reply limit in Settings (max_tokens): {max_tokens:,}")
    out("  Compare the model above with the one in your own test script.")
    client = openai.OpenAI(base_url=base, api_key=key, timeout=TIMEOUT, max_retries=0)

    codeword_system = ("You are a helpful assistant. Regardless of user query, you must always "
                       "conclude your final response with the exact token: CODEWORD 42")
    photosynthesis = [{"role": "user", "content": "Explain photosynthesis"}]
    jc_rule = "Formatting rule: begin every reply with the exact text [JC-OK] followed by a space."
    sky = [{"role": "user", "content": "In one short sentence: what colour is the sky on a clear day?"}]
    turns = [
        {"role": "user", "content": "Please remember this code word for later: HYDRANT."},
        {"role": "assistant", "content": "Noted - I'll remember the code word."},
        {"role": "user", "content": "What was the code word I gave you? Reply with only the word."},
    ]

    def direct(messages, **extra):
        r = client.chat.completions.create(model=model, messages=messages, **extra)
        return r.choices[0].message.content or ""

    def check(label, fn, expect):
        started = time.time()
        try:
            reply = fn().strip()
            ok = expect.lower() in reply.lower()
            tail = reply[-60:] if len(reply) > 60 else reply
            out(f"  {'OK  ' if ok else 'MISS'}  {label:<52} {time.time() - started:.1f}s - "
                f"reply ends: {tail!r}")
            return ok
        except Exception as e:                               # noqa: BLE001
            out(f"  FAIL  {label:<52} after {time.time() - started:.1f}s - {str(e)[:400]}")
            return False

    sys_msg = lambda text: [{"role": "system", "content": text}]
    a = check("A. your codeword test, direct, no reply limit",
              lambda: direct(sys_msg(codeword_system) + photosynthesis), "CODEWORD 42")
    b = check("B. same, with the app's reply limit",
              lambda: direct(sys_msg(codeword_system) + photosynthesis, max_tokens=max_tokens),
              "CODEWORD 42")
    check("B2. same, limit sent as max_completion_tokens",
          lambda: direct(sys_msg(codeword_system) + photosynthesis,
                         max_completion_tokens=max_tokens), "CODEWORD 42")
    check("B3. same, max_tokens=8,192",
          lambda: direct(sys_msg(codeword_system) + photosynthesis, max_tokens=8192),
          "CODEWORD 42")
    native = dict(s, message_format="native", stream=False)
    app._LIMIT_LEARNED.clear()
    c = check("C. same, through the app's own code (Native)",
              lambda: app.PROVIDERS["genai"](codeword_system, photosynthesis, native, 1),
              "CODEWORD 42")
    d = check("D. the [JC-OK] rule, direct, no reply limit",
              lambda: direct(sys_msg(jc_rule) + sky), "[JC-OK]")
    e = check("E. earlier turns, direct, no reply limit",
              lambda: direct(sys_msg("Answer briefly.") + turns), "HYDRANT")

    learned = app._LIMIT_LEARNED
    if learned:
        out(f"  (the app found the reply limit works as: "
            f"{'no limit' if learned['param'] == 'none' else learned['param']}"
            f"{', capped at ' + format(learned['cap'], ',') if learned.get('cap') else ''})")
    out("")
    out("  The exact request body the app sends for C (no key):")
    body = {"model": model, "messages": app._wire_messages(codeword_system, photosynthesis, native)}
    body.update(app._limit_kwargs(learned.get("param", "max_tokens"),
                                  min(max_tokens, learned.get("cap") or max_tokens)))
    for text_line in json.dumps(body, indent=2).splitlines():
        out("    " + text_line)

    out("")
    out("  What this suggests:")
    if a and not b and c:
        out("  - The model rejects max_tokens as the app first sent it; the app now detects")
        out("    what the model accepts (see the line above the request body) and C works.")
    elif a and not b:
        out("  - The reply limit (max_tokens) breaks it, and the app's detection didn't fix C:")
        out("    send this file to Claude.")
    elif a and b and not c:
        out("  - Direct works but the app's code doesn't: compare the body above with your")
        out("    script's request and send this file to Claude.")
    elif a and b and c and not d:
        out("  - System messages work; the model just didn't follow the [JC-OK] rule. Native")
        out("    format should be fine - send this file to Claude.")
    elif not a:
        out("  - Even your own test fails with this model: check whether your script used a")
        out("    different model than the one in Settings.")
    if a and not e:
        out("  - Earlier turns are not passed to the model even when sent directly.")
    if a and b and c and d and e:
        out("  - Everything works direct and through the app: send this file to Claude.")


TESTS = {"replica": test_replica, "system": test_system, "openai": test_openai, "embeddings": test_embeddings,
         "chat": test_chat}


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

    path = HERE / time.strftime("test-results-%Y%m%d-%H%M.txt")
    out = Out(path)
    print(f"Results are being saved to {path.name} as they come (no API key in it).")
    app = load_app()
    out(f"Job-Context Client {app.APP_VERSION} live tests - {time.strftime('%Y-%m-%d %H:%M')}")
    out(f"Python {sys.version.split()[0]}")
    out()
    try:
        for name in names:
            TESTS[name](app, out)
            out()
    except KeyboardInterrupt:
        out("(stopped with Ctrl+C)")
    print(f"Saved to {path.name} (no API key in it).")


if __name__ == "__main__":
    main()
