# JobContext-Rag

A local chat client for a GenAI API that carries **standing context** into every call
and **searches your own documents** to ground its answers. It is the document-search
version of [JobContext](https://github.com/OmarTheHippo/JobContext).

- **Layers.** Each chat uses a **Role** layer (who you are), any number of **Project**
  layers, and any number of **Agent** layers (specialist roles such as legal or design
  review). The text you write for a layer is its **layer prompt**.
- **Repositories.** Each layer can have a folder of documents (.txt, .md, .csv, .pdf,
  .docx, .xlsx) indexed for search; pick it with **Browse…** or paste its path. Indexing and search run on your PC; only
  the passages selected for a question are sent to the API.
- **Retrieval planner.** After a first search, an optional planning call reads the
  question, the recent conversation and the layer prompts, then runs targeted searches
  in any selected layer: project facts needed to apply a rule, or the requirements
  that apply to a project's scope.
- **Project briefs.** After a project's folder is indexed, the app writes one summary
  card per file, then combines them into a brief (overview, cost, scope, document
  checklist, conflicts, gaps, inventory). The brief guides the planner and is sent with
  chats about that project. What it looks for is set in Settings > Brief guidance.

## Install

Requires Python 3.10+. Put all files in one folder and double-click `install.bat`. It
finds Python (`py`, `python` or `python3`), installs the packages in
`requirements-rag.txt` if they're missing, creates the `data` folder, and writes
`Start Job-Context.bat`. No admin rights needed.

Manual install, from the folder:

```
py -m pip install -r requirements-rag.txt
py jobcontext.txt --install
py jobcontext.txt
```

With the local embedding model, Chroma downloads it (all-MiniLM-L6-v2, about 80 MB) the
first time a folder is indexed. With an API embedding model nothing is downloaded.

## Settings

- **Base URL, model and API key** for your GenAI API. Requests go to
  `<base URL>/chat/completions` in the common "chat completions" format; adapt the
  three marked lines in `_genai_complete()` if your API differs. The key is saved in
  `data\settings.json`, encrypted with Windows DPAPI for your Windows account.
- **Token limits.** Set the model's **context window** (the most one request can hold)
  and the reply limit; document passages and chat history are calculated from these
  unless you override them. Settings shows the breakdown.
- **Document search.** Choose the built-in local embedding model (no network; reads
  ~800 characters per passage) or an API embedding model on the same base URL and key
  (requests go to `<base URL>/embeddings`; name the model explicitly). With an API
  model, prose passages default to 3,000 characters; spreadsheet and CSV passages stay
  around 1,000 so single rows aren't buried. Texts are sent 20 per request. Changing
  the model or passage size requires re-indexing; the editor flags affected folders.
- **Reply limit.** Models differ in how they accept a reply limit (`max_tokens`,
  `max_completion_tokens`, a lower maximum, or none at all). On Automatic, the app
  finds what the model accepts on the first request and reuses it; Settings can fix it.
- **Message format.** Combined (default): layer prompts and the earlier conversation go
  into one user message, which works with gateways that forward only the last user
  message. Native: a real system message and real conversation turns, for gateways
  that forward the whole conversation. `py test.py system` shows which parts of a
  request reach the model and which format to use.
- **API client.** Requests go through the `openai` Python library when it is installed
  (it is in `requirements-rag.txt`). On some agency networks the gateway stalls requests
  from Python's built-in client until they are dropped (WinError 10054 after ~90 s)
  while the same requests through the library work. The built-in client remains as a
  fallback and can be chosen in Settings.
- **Network route.** By default the app connects like most API client libraries:
  it ignores the proxy in Windows' internet settings and only honours `HTTPS_PROXY` /
  `HTTP_PROXY` environment variables. Some agency proxies hold non-streamed requests
  until they are dropped (WinError 10054 after ~90 s); `py test.py embeddings` shows
  whether that is happening. Settings can switch to the Windows proxy or to no proxy.
- **Streaming and retries.** Replies are streamed (keeps slow requests such as briefs
  from being cut off by proxies); dropped connections and busy responses are retried.
- **Brief guidance.** What project briefs should look for, such as the documents a
  complete project file must contain. The repository ships a generic example; enter your
  organisation's list here (it is stored in `data\settings.json`, not in the code).

## Files

| File | Contents |
|---|---|
| `jobcontext.txt` | The app: Python plus the web page's HTML, CSS and JavaScript (runs with `py jobcontext.txt`) |
| `jobcontext_rag.py` | Text extraction, chunking, local indexing and search (Chroma) |
| `requirements-rag.txt` | Python packages: document search and the `openai` API client |
| `install.bat` | Installer for Windows |
| `test.py` | Live diagnostic tests against your real API (see below) |

Everything you create lives in `data\` next to the app (layers, chats, briefs, the local
search index, settings) and is excluded from git.

## Live diagnostic tests

`test.py` sends a set of small, timed requests to your real API using the app's saved
Settings, to pin down connection problems:

```
py test.py replica
py test.py system
py test.py openai
py test.py embeddings
py test.py chat
py test.py all
```

Each variation is sent once (no retries) and reported as OK or FAIL with timings and
sizes, plus a short diagnosis. Results are saved to `test-results-<date>.txt` (no API
key in it; excluded from git). New live tests can be added as functions in `test.py`.

## Known limits

- Scanned PDFs have no extractable text (no OCR); such files are flagged as poorly read.
- The local embedding model reads only the first ~256 tokens of a passage, so local
  passages are kept to about 800 characters; an API embedding model allows larger ones.
- Briefs and answers come from a language model: treat them as a review aid and check
  them against the source documents.
