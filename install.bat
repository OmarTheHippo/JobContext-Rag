@echo off
rem Job-Context Client (RAG) installer. Put this file next to jobcontext.txt,
rem jobcontext_rag.py, requirements.txt and requirements-optional.txt, then
rem double-click it. No admin rights.
cd /d "%~dp0"

for %%F in (jobcontext.txt jobcontext_rag.py requirements.txt) do (
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
  echo     py -m pip install -r requirements.txt
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

rem Core packages (the app needs these): document search (Chroma, PDF/Word/Excel
rem readers) and the openai library used to call the API. Installed once.
%PY% -c "import chromadb, pypdf, docx, openpyxl, openai" >nul 2>nul
if errorlevel 1 (
  echo Installing core packages from requirements.txt...
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Package installation failed. Check your network or package mirror, then run:
    echo     %PY% -m pip install -r requirements.txt
    pause
    exit /b 1
  )
) else (
  echo Core packages already installed.
)

rem Then the app's own setup. It also installs each optional group from
rem requirements-optional.txt on its own (PowerPoint, old Excel, Outlook email,
rem encrypted PDFs, OCR) and shows a summary of what did and didn't install.
%PY% jobcontext.txt --install
echo.
pause
