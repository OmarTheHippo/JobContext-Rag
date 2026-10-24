# JobContext-Rag

A local chat client for a GenAI API that carries **standing context** into every call
and **searches your own documents** to ground its answers. It is the document-search
version of [JobContext](https://github.com/OmarTheHippo/JobContext).

- **Layers.** Each chat uses a **Role** layer (who you are), any number of **Project**
  layers, and any number of **Agent** layers (specialist roles such as legal or design
  review). The text you write for a layer is its **layer prompt**.
- **Repositories.** Each layer can have a folder of documents indexed for search; pick
  it with **Browse…** or paste its path. Indexing and search run on your PC; only the
  passages selected for a question are sent to the API. File types: .txt, .md, .csv,
  .pdf, .docx, .xlsx/.xlsm and saved emails (.eml) always; with the optional packages
  also .pptx, old Excel .xls, Outlook .msg (with attachments), AES-encrypted PDFs, and
  OCR of scanned PDF pages and images.
- **Index report.** After indexing, the layer shows what was not read and why: files
  that failed (e.g. password-protected), files read only in part (e.g. scanned pages
  without OCR), and file types that aren't read, with what to do about each.
- **Search by meaning and by keyword.** Each search combines a meaning-based (vector)
  search with exact keyword matching (BM25), so a project number, form or drawing
  number or acronym mentioned once in a long document is still found. Only words that
  are rare in a layer's documents trigger keyword matches, so long questions full of
  common words don't; each layer gets a few reserved places for keyword matches so
  neither kind of search crowds out the other. The source list under each answer tags
  passages found by keyword. No re-index needed; switch it off in Settings.
- **Retrieval planner.** After a first search, an optional planning call reads the
  question, the recent conversation and the layer prompts, then runs targeted searches
  in any selected layer: project facts needed to apply a rule, or the requirements
  that apply to a project's scope. It works with any selected layer (a Project is not
  required) and can name exact keywords (IDs, acronyms) to match word for word.
- **Stated vs inferred.** Answers and briefs present something as fact only when a
  document states it; a connection the model draws between documents (same year,
  place or system) is presented as a possible link, with what it rests on and what
  would confirm it.
- **Project briefs.** After a project's folder is indexed, the app writes one summary
  card per file, then combines them into a brief (overview, cost, scope, document
  checklist, conflicts, gaps, inventory). The brief guides the planner and is sent with
  chats about that project. What it looks for is set in Settings > Brief guidance.

## Install

Requires Python 3.10+. Put all files in one folder and double-click `install.bat`. It
finds Python (`py`, `python` or `python3`), installs the core packages in
`requirements.txt` if they're missing, then installs each optional group in
`requirements-optional.txt` on its own, creates the `data` folder, writes
`Start Job-Context.bat`, and shows a summary of what did and didn't install. No admin
rights needed. Run it again to retry an optional group (e.g. on another network).

| Packages | Required? | What they do |
|---|---|---|
| chromadb, pypdf, python-docx, openpyxl, openai | **Core** (`requirements.txt`) | Document search; PDF, Word and Excel text; the API client |
| python-pptx | Optional | PowerPoint (.pptx): slide text, tables, speaker notes |
| xlrd | Optional | Old Excel (.xls) |
| olefile, compressed-rtf | Optional | Outlook email (.msg): header, body, attachments |
| cryptography | Optional | Opening AES-encrypted ("restricted") PDFs (pypdfium2 from the OCR group also reads them) |
| pypdfium2, Pillow, winrt-* | Optional | OCR of scanned PDF pages and images with Windows' own OCR engine (no models to download; uses the OCR languages installed in Windows) |

Without an optional group the app still works; those files are listed in the index
report as "needs ...". Settings > Document search shows which file types are on.

Manual install, from the folder:

```
py -m pip install -r requirements.txt
py jobcontext.txt --install
py jobcontext.txt
```

`--install` also installs the optional groups; add `--no-optional` to skip them.

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
  **Keyword matching** (on by default) and **OCR** (on by default, used when indexing)
  can be switched off here.
- **Reply limit.** Models differ in how they accept a reply limit (`max_tokens`,
  `max_completion_tokens`, a lower maximum, or none at all). On Automatic, the app
  finds what the model accepts on the first request and reuses it; Settings can fix it.
- **Message format.** Combined (default): layer prompts and the earlier conversation go
  into one user message, which works with gateways that forward only the last user
  message. Native: a real system message and real conversation turns, for gateways
  that forward the whole conversation. `py test.py system` shows which parts of a
  request reach the model and which format to use.
- **API client.** Requests go through the `openai` Python library when it is installed
  (it is in `requirements.txt`). On some agency networks the gateway stalls requests
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
| `requirements.txt` | Core Python packages: document search, PDF/Word/Excel readers, the `openai` API client |
| `requirements-optional.txt` | Optional packages, one group per file type (see Install) |
| `requirements-rag.txt` | Kept for older instructions; installs `requirements.txt` |
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

- OCR quality depends on the scan; OCR'd passages are labelled "(OCR)" in the sources.
  OCR results are cached in `data\rag\ocr` by file content, so re-indexing doesn't
  repeat them. Old Word (.doc), old PowerPoint (.ppt) and drawing files aren't read.
- The keyword index is built in memory on a layer's first search after the app starts
  or the layer is re-indexed. Its speed on very large document sets is not yet measured.
- Keyword matching treats a word as distinctive when it appears in at most about 2% of
  a layer's passages; in a very small folder some ordinary words also qualify, which
  adds at most a few passages.
- The local embedding model reads only the first ~256 tokens of a passage, so local
  passages are kept to about 800 characters; an API embedding model allows larger ones.
- Briefs and answers come from a language model: treat them as a review aid and check
  them against the source documents.
