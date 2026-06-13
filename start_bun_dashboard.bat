@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo THERADIOMICS Bun Python Bridge
echo ============================================================
echo.
echo Cartella:
echo %CD%
echo.
echo Avvio dashboard su:
echo http://127.0.0.1:8765
echo.
echo Per chiudere: CTRL+C
echo ============================================================
echo.

bun run server.ts

pause
