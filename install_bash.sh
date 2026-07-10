#!/bin/bash

echo ""
echo "  Plot Lab - Instalação"
echo "  ====================="
echo ""
echo "  Instalando dependencias... pode demorar alguns minutos." [cite: 2]
echo ""
read -p "Pressione [Enter] para continuar..."

# Verifica se o Python 3 está instalado (macOS utiliza python3 por padrão)
if ! command -v python3 &> /dev/null; then
    echo ""
    echo "  ERRO: Python 3 não encontrado no PATH."
    echo "  Instale o Python em https://python.org ou via Homebrew (brew install python)."
    echo ""
    exit 1
fi

# Cria o diretório base como referência segura
cd "$(dirname "$0")" || exit

# Arquitetura Robusta: Criação de um Virtual Environment (venv)
# Isso previne o erro "externally-managed-environment" comum em macOS recentes
echo "  Configurando ambiente virtual isolado (venv)..."
python3 -m venv venv
source venv/bin/activate

# Atualiza o pip e instala as dependências no ambiente virtual
python3 -m pip install --upgrade pip --quiet
python3 -m pip install customtkinter matplotlib seaborn pandas scipy scikit-learn openpyxl shapely pywebview --quiet

if [ $? -ne 0 ]; then
    echo ""
    echo "  Erro na instalacao. Verifique sua conexao com a internet ou permissoes do diretorio." [cite: 3]
    echo ""
    exit 1
fi

echo ""
echo "  Instalacao concluida! Abrindo o Plot Lab..." [cite: 4]
echo ""

# Executa o programa principal
python3 main_web.py