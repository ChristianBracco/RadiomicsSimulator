@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo THERADIOMICS - Cross-platform launcher
echo Sistema rilevato: Windows
echo ============================================================
echo.

where bun >nul 2>nul
if errorlevel 1 (
    echo [ERRORE] Bun non trovato nel PATH.
    echo Installa Bun oppure aggiungi bun.exe al PATH.
    echo.
    pause
    exit /b 1
)

REM Su Windows normalmente Python e' python.
REM Se l'utente ha impostato PYTHON_EXE, la rispettiamo.
if "%PYTHON_EXE%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    ) else (
        where py >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_EXE=py -3"
        ) else (
            echo [ERRORE] Python non trovato.
            echo Installa Python 3 oppure imposta PYTHON_EXE.
            echo Esempio:
            echo   set PYTHON_EXE=python
            echo.
            pause
            exit /b 1
        )
    )
)

if "%PORT%"=="" set "PORT=8765"

echo Cartella progetto:
echo %CD%
echo.
echo Python usato:
echo %PYTHON_EXE%
echo.
echo Porta:
echo %PORT%
echo.
echo Avvio dashboard:
echo http://127.0.0.1:%PORT%
echo.

start "" "http://127.0.0.1:%PORT%"

bun run server.ts

echo.
echo Backend terminato.
pause
