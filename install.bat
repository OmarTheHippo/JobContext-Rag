@echo off
rem Job-Context Client (RAG) installer. Put this file next to jobcontext.txt,
rem jobcontext_rag.py and requirements-rag.txt, then double-click it. No admin rights.
cd /d "%~dp0"

for %%F in (jobcontext.txt jobcontext_rag.py requirements-rag.txt) do (
  if not exist %%F (
    echo %%F must be in the same folder as install.bat.
    pause
    exit /b 1
  )
)

rem Find a working Python by actually running it. (Checking with "where" isn't reliable,
rem and Windows has a fake "python" that only prints "Python was not found".)
set "PY="
py -c "import sys" >nul 2>nul && set "PY=py"
if not defined PY python -c "import sys" >nul 2>nul && set "PY=python"
if not defined PY python3 -c "import sys" >nul 2>nul && set "PY=python3"
if not defined PY (
  echo Couldn't start Python with "py", "python" or "python3" from this window.
  echo If one of them works in your own terminal, open that terminal in this folder and run:
  echo     py -m pip install -r requirements-rag.txt
  echo     py jobcontext.txt --install
  pause
  exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo Python 3.10 or newer is required.
  %PY% --version
  pause
  exit /b 1
)
echo Using: %PY%

rem Packages: document search (Chroma, PDF/Word/Excel readers) and the openai
rem library used to call the API. Installed once.
%PY% -c "import chromadb, pypdf, docx, openpyxl, openai" >nul 2>nul
if errorlevel 1 (
  echo Installing document-search packages from requirements-rag.txt...
  %PY% -m pip install -r requirements-rag.txt
  if errorlevel 1 (
    echo Package installation failed. Check your network or package mirror, then run:
    echo     %PY% -m pip install -r requirements-rag.txt
    pause
    exit /b 1
  )
) else (
  echo Document-search packages already installed.
)

%PY% jobcontext.txt --install
echo.
pause
