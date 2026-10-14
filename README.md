# JobContext-Rag

A local chat client for a GenAI API that carries **standing context** into every call
and **searches your own documents** to ground its answers. It is the document-search
version of [JobContext](https://github.com/OmarTheHippo/JobContext).

- **Layers.** Each chat uses a **Role** layer (who you are), any number of **Project**
  layers, and any number of **Agent** layers (specialist roles such as legal or design
  review). The text you write for a layer is its **layer prompt**.
- **Repositories.** Each layer can have a folder of documents (.txt, .md, .csv, .pdf,
  .docx, .xlsx) indexed locally for search. Indexing and search run on your PC; only
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

Chroma downloads its small embedding model (all-MiniLM-L6-v2, about 80 MB) the first
time a folder is indexed.

## Settings

- **Base URL, model and API key** for your GenAI API. Requests go to
  `<base URL>/chat/completions` in the common "chat completions" format; adapt the
  three marked lines in `_genai_complete()` if your API differs. The key is saved in
  `data\settings.json`, encrypted with Windows DPAPI for your Windows account.
- **Token limits.** Set the model's **context window** (the most one request can hold)
  and the reply limit; document passages and chat history are calculated from these
  unless you override them. Settings shows the breakdown.
- **Brief guidance.** What project briefs should look for, such as the documents a
  complete project file must contain. The repository ships a generic example; enter your
  organisation's list here (it is stored in `data\settings.json`, not in the code).

## Files

| File | Contents |
|---|---|
| `jobcontext.txt` | The app: Python plus the web page's HTML, CSS and JavaScript (runs with `py jobcontext.txt`) |
| `jobcontext_rag.py` | Text extraction, chunking, local indexing and search (Chroma) |
| `requirements-rag.txt` | Python packages for document search |
| `install.bat` | Installer for Windows |

Everything you create lives in `data\` next to the app (layers, chats, briefs, the local
search index, settings) and is excluded from git.

## Known limits

- Scanned PDFs have no extractable text (no OCR); such files are flagged as poorly read.
- The embedding model reads only the first ~256 tokens of a passage, so passages are
  kept to about 800 characters. Re-index existing folders after changing chunk settings;
  the app flags indexes built with different settings.
- Briefs and answers come from a language model: treat them as a review aid and check
  them against the source documents.
