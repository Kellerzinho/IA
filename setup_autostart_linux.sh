#!/bin/bash

echo "Configurando inicialização automática do Ethara Monitor no Linux..."

# Determina o diretório onde este script está localizado
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Verifica se o usuário tem permissões de sudo
if ! command -v sudo &> /dev/null; then
    echo "Comando sudo não encontrado. Você precisa de permissões de administrador."
    exit 1
fi

# Cria o script de inicialização
cat > "$SCRIPT_DIR/start_ethara.sh" << 'EOL'
#!/bin/bash

echo "Iniciando Ethara Monitor automaticamente..."

# Determina o diretório onde este script está localizado
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Verifica se o executável existe
if [ -f "$SCRIPT_DIR/dist/ethara_monitor" ]; then
    echo "Executando via executável..."
    nohup "$SCRIPT_DIR/dist/ethara_monitor" > "$SCRIPT_DIR/ethara.log" 2>&1 &
else
    echo "Executável não encontrado. Tentando executar via Python..."
    
    # Verifica se Python está instalado
    if ! command -v python3 &> /dev/null; then
        echo "Python3 não encontrado. Não foi possível iniciar o programa."
        exit 1
    fi
    
    echo "Iniciando via Python..."
    nohup python3 "$SCRIPT_DIR/people.py" > "$SCRIPT_DIR/ethara.log" 2>&1 &
fi

# Armazena o PID do processo para possível referência futura
echo $! > "$SCRIPT_DIR/ethara.pid"

echo "Ethara Monitor iniciado com PID $(cat $SCRIPT_DIR/ethara.pid)!"
echo "Output redirecionado para ethara.log"
EOL

# Dá permissão de execução para o script de início
chmod +x "$SCRIPT_DIR/start_ethara.sh"

# Cria um arquivo de serviço systemd
SERVICE_FILE="/tmp/ethara-monitor.service"

cat > $SERVICE_FILE << EOL
[Unit]
Description=Ethara Monitor - Sistema de Monitoramento Inteligente
After=network.target

[Service]
ExecStart=$SCRIPT_DIR/start_ethara.sh
WorkingDirectory=$SCRIPT_DIR
Restart=on-failure
User=$USER

[Install]
WantedBy=multi-user.target
EOL

# Copia o arquivo para o diretório de serviços do systemd
echo "Instalando serviço systemd (requer senha de administrador)..."
sudo cp $SERVICE_FILE /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ethara-monitor.service

echo "Serviço criado. Para iniciar o serviço agora, execute:"
echo "sudo systemctl start ethara-monitor.service"
echo ""
echo "Para verificar o status:"
echo "sudo systemctl status ethara-monitor.service"
echo ""
echo "O Ethara Monitor agora será iniciado automaticamente na próxima inicialização do sistema."
echo "Para testar agora, execute:"
echo "$SCRIPT_DIR/start_ethara.sh" 