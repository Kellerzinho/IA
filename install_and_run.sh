#!/bin/bash

echo "===== ETHARA MONITOR - INSTALADOR E EXECUTOR ====="
echo

# Verifica se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "Python3 não encontrado. Por favor, instale Python 3.8 ou superior."
    echo "Em sistemas baseados em Debian/Ubuntu: sudo apt install python3 python3-pip"
    echo "Em sistemas baseados em RHEL/Fedora: sudo dnf install python3 python3-pip"
    exit 1
fi

display_menu() {
    clear
    echo "===== ETHARA MONITOR - MENU PRINCIPAL ====="
    echo
    echo "1. Instalar dependências"
    echo "2. Configurar sistema"
    echo "3. Construir executável"
    echo "4. Executar programa (via Python)"
    echo "5. Executar programa (via Executável, se disponível)"
    echo "6. Sair"
    echo
    read -p "Escolha uma opção (1-6): " opcao
    
    case $opcao in
        1) instalar_dependencias ;;
        2) configurar_sistema ;;
        3) construir_executavel ;;
        4) executar_python ;;
        5) executar_exe ;;
        6) echo "Saindo..."; exit 0 ;;
        *) echo "Opção inválida"; sleep 2; display_menu ;;
    esac
}

instalar_dependencias() {
    clear
    echo "Instalando dependências..."
    python3 install_dependencies.py
    echo
    read -p "Pressione Enter para continuar..."
    display_menu
}

configurar_sistema() {
    clear
    echo "Iniciando configuração do sistema..."
    python3 config_manager.py
    echo
    read -p "Pressione Enter para continuar..."
    display_menu
}

construir_executavel() {
    clear
    echo "Construindo executável..."
    python3 build_executable.py
    echo
    read -p "Pressione Enter para continuar..."
    display_menu
}

executar_python() {
    clear
    echo "Executando via Python..."
    python3 people.py
    display_menu
}

executar_exe() {
    clear
    if [ -f "dist/ethara_monitor" ]; then
        echo "Executando o programa..."
        ./dist/ethara_monitor
    else
        echo "Executável não encontrado. Por favor, construa o executável primeiro (opção 3)."
        read -p "Pressione Enter para continuar..."
    fi
    display_menu
}

# Tornar o script executável
chmod +x "$0"

# Exibir menu
display_menu 