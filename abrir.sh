#!/bin/bash

# Garante que o script está rodando no diretório onde o arquivo está localizado
cd "$(dirname "$0")" || exit

# Se o ambiente virtual existir, ativa-o antes de rodar o programa
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Executa o programa
python3 main_web.py