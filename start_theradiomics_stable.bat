@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if "%PYTHON_EXE%"=="" (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_EXE=%CD%\.venv\Scripts\python.exe"
    ) else (
        set "PYTHON_EXE=python"
    )
)

if "%PORT%"=="" set "PORT=8765"

echo ============================================================
echo THERADIOMICS - stable launcher
echo ============================================================
echo Python: %PYTHON_EXE%
echo Dashboard: http://127.0.0.1:%PORT%
echo.

start "" "http://127.0.0.1:%PORT%"
bun run server.ts
pause
