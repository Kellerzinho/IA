# MONITOR DE MESAS - SISTEMA DE DETECCAO

Este sistema monitora a ocupacao de mesas atraves de cameras RTSP,
utilizando inteligencia artificial com YOLO e CUDA.

## REQUISITOS DO SISTEMA

1. Windows 10 ou superior
2. Python 3.8 ou superior
3. Placa de video NVIDIA com suporte a CUDA
4. Drivers NVIDIA atualizados
5. Acesso a uma camera via RTSP (ex: DVR Intelbras)

## ESTRUTURA DO PROJETO

- **iniciar_monitor.bat** - Script principal para iniciar o sistema
- **people.py** - Codigo principal do sistema de deteccao
- **roboflow.pt** - Modelo de IA treinado para deteccao
- **config.json** - Arquivo de configuracao do sistema
- **install_dependencies.py** - Script para instalar dependencias
- **build_executable.py** - Script para criar versao executavel
- **config_manager.py** - Ferramenta para configurar o sistema
- **setup_autostart_windows.bat** - Configura inicializacao automatica no Windows
- **install_and_run.bat** - Menu de instalacao e execucao
- **INSTRUCOES.html** - Documentacao detalhada com interface visual

## COMO EXECUTAR

### Metodo 1 (Mais simples):
- De um duplo clique no arquivo **"iniciar_monitor.bat"**

### Metodo 2 (Criar atalho):
1. Clique com o botao direito na area de trabalho
2. Selecione "Novo" > "Atalho"
3. No campo "Digite o local do item", insira:
   ```
   %windir%\system32\cmd.exe /c "cd /d "%~dp0" && iniciar_monitor.bat"
   ```
4. Clique em "Avancar"
5. De o nome "Monitor de Mesas" ao atalho
6. Clique em "Concluir"

## CONFIGURACAO RTSP

Para DVRs Intelbras, use o formato:
```
rtsp://USUARIO:SENHA@IP_DO_DVR:554/cam/realmonitor?channel=NUMERO_CAMERA&subtype=0
```

onde:
- USUARIO: nome de usuario do DVR (geralmente "admin")
- SENHA: senha de acesso ao DVR
- IP_DO_DVR: endereco IP do DVR (ex: 192.168.1.100)
- NUMERO_CAMERA: numero do canal da camera (1, 2, 3, etc.)
- subtype=0: stream principal (alta qualidade)
- subtype=1: stream secundario (menor qualidade)

## TECLAS DE CONTROLE

Quando o programa estiver rodando, utilize:
- **Q**: Encerra o programa
- **D**: Ativa/desativa modo debug
- **P**: Mostra/oculta caixas de pessoas
- **L**: Mostra/oculta linhas de associacao

## SOLUCAO DE PROBLEMAS

1. **Se ocorrer erro CUDA:**
   - Verifique se os drivers NVIDIA estao atualizados
   - Execute o comando "nvidia-smi" no prompt para verificar a GPU

2. **Se a camera RTSP nao conectar:**
   - Verifique se a URL esta correta
   - Teste a URL em outro player (como VLC)
   - Verifique se ha firewall bloqueando a conexao

3. **Logs e diagnostico:**
   - Verifique o arquivo "table_tracker.log" para mais detalhes de erros

## SUPORTE

Em caso de problemas, consulte a documentacao detalhada no arquivo INSTRUCOES.html. 