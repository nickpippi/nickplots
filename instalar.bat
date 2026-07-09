@echo off
title Plot Lab - Instalacao

echo.
echo  Plot Lab - Instalacao
echo  =====================
echo.
echo  Instalando dependencias... pode demorar alguns minutos.
echo.
pause

python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERRO: Python nao encontrado no PATH.
    echo  Instale o Python em https://python.org
    echo  e marque "Add Python to PATH" durante a instalacao.
    echo.
    pause
    exit /b 1
)

python -m pip install --upgrade pip --quiet

python -m pip install customtkinter matplotlib seaborn pandas scipy scikit-learn openpyxl shapely pywebview --quiet

if errorlevel 1 (
    echo.
    echo  Erro na instalacao. Tente clicar com botao direito
    echo  em instalar.bat e escolher "Executar como administrador".
    echo.
    pause
    exit /b 1
)

echo.
echo  Instalacao concluida! Abrindo o Plot Lab...
echo.

cd /d "%~dp0"
start "" python main_web.py
