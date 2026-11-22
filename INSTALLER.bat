@echo off
cd /d "%~dp0"
echo === Installation de Zakira_bilal_car (une seule fois, 5 a 10 minutes, Internet necessaire) ===
set "PY="
py -3.13 -c "print()" >nul 2>&1 && set "PY=py -3.13"
if not defined PY py -3.12 -c "print()" >nul 2>&1 && set "PY=py -3.12"
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3,12),(3,13)) else 1)" >nul 2>&1 && set "PY=python"
if not defined PY (
  echo Python 3.12 ou 3.13 est necessaire. Installe Python 3.13 depuis https://www.python.org/downloads/
  echo en cochant "Add python.exe to PATH", puis relance INSTALLER.bat.
  pause
  exit /b 1
)
echo Python trouve : %PY%
if not exist .venv\Scripts\python.exe %PY% -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install numpy==2.2.6 Pillow==11.2.1 matplotlib==3.10.3 opencv-python==4.13.0.90
.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -c "import torch, racing.drive, racing.inference; print('=== Installation OK : lance maintenant LANCER_VOITURE.bat ===')"
if errorlevel 1 echo === Probleme pendant l'installation : lis les messages ci-dessus ===
pause
