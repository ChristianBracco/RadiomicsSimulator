@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo THERADIOMICS Dashboard Backend
echo ============================================================
echo.
echo Cartella:
echo %CD%
echo.
echo Installo/controllo dipendenze minime...
python -m pip install fastapi uvicorn python-multipart
echo.
echo Avvio backend su:
echo http://127.0.0.1:8765
echo.
echo Per chiudere: CTRL+C
echo ============================================================
echo.

python -m uvicorn dashboard_backend:app --host 127.0.0.1 --port 8765 --reload

pause
