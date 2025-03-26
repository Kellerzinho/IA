# Monitor Ethara- Sistema de Monitoramento Inteligente

O Monitor Ethara é um sistema de visão computacional que utiliza IA para monitorar mesas e pessoas em ambientes do Cortez, identificando status de ocupação e necessidades de atendimento.

## Visão Geral

Este sistema utiliza detecção de objetos baseada em YOLO (You Only Look Once) para identificar:
- Diferentes tipos de mesas (capacidade de 2 a 14 lugares)
- Pessoas sentadas
- Gestos (mãos levantadas) para solicitação de atendimento

## Requisitos de Hardware

⚠️ **IMPORTANTE**: Este sistema requer obrigatoriamente uma **GPU com suporte a CUDA** para funcionar.

- **GPU NVIDIA** com pelo menos 4GB de VRAM
- **Drivers NVIDIA** atualizados
- **CUDA Toolkit** versão 11.0 ou superior instalado

O sistema verifica automaticamente a disponibilidade de GPU com CUDA e terminará a execução caso não seja detectada.

## Componentes do Sistema

### Arquivo `roboflow.pt`

O arquivo `roboflow.pt` é um modelo de deep learning pré-treinado no formato PyTorch (extensão `.pt`). Este modelo foi treinado para reconhecer:

1. **Mesas de diferentes tamanhos** (Classes 0-6)
2. **Mãos levantadas** (Classe 7)
3. **Pessoas sentadas** (Classe 8)

#### Sobre o Treinamento do Dataset

O modelo foi treinado utilizando a plataforma Roboflow com uma arquitetura YOLOv8, seguindo estas etapas:

1. **Coleta de dados**: Foram capturadas imagens representativas do ambiente-alvo, como mesas, clientes sentados e gestos de mãos.

2. **Anotação**: As imagens foram manualmente anotadas para identificar e rotular os objetos de interesse (mesas, pessoas, mãos).

3. **Aumento de dados**: Técnicas de data augmentation foram aplicadas para expandir o conjunto de treinamento, incluindo:
   - Rotações
   - Alterações de brilho e contraste
   - Ruído
   - Espelhamento horizontal

4. **Treinamento**: O modelo foi treinado através de aprendizado supervisionado, usando a arquitetura YOLOv8 para detecção de objetos em tempo real.

5. **Validação**: O modelo foi validado em um conjunto separado de dados para garantir sua precisão em condições reais.

6. **Exportação**: Após o treinamento e validação, o modelo foi exportado no formato PyTorch (`.pt`).

### Arquivo `people.py`

Este é o arquivo principal do sistema que contém toda a lógica de processamento de vídeo e detecção. Suas principais funcionalidades incluem:

1. **Detecção de objetos**:
   - Carrega o modelo YOLOv8 (`roboflow.pt`) utilizando GPU (CUDA)
   - Processa frames de vídeo de uma câmera RTSP (via protocolo TCP)
   - Detecta mesas, pessoas e mãos levantadas

2. **Gerenciamento de mesas**:
   - Rastreia mesas ao longo do tempo
   - Calcula estado de ocupação (vazia, ocupada, cheia)
   - Detecta quando uma mesa precisa de atendimento

3. **Associação pessoa-mesa**:
   - Atribui pessoas detectadas às mesas mais próximas
   - Monitora ocupação de cada mesa baseada na quantidade de pessoas
   - Calcula taxas de ocupação globais do estabelecimento

4. **Detecção de solicitações de atendimento**:
   - Identifica mãos levantadas
   - Associa os gestos às mesas mais próximas
   - Sinaliza mesas que precisam de atendimento

5. **Interface visual**:
   - Apresenta uma visualização do estado atual do sistema
   - Mostra cores diferentes para cada estado de mesa
   - Exibe estatísticas de ocupação

6. **API de notificação**:
   - Envia atualizações de estado para um dashboard externo via HTTP
   - Fornece dados em formato JSON para integração com outros sistemas

7. **Otimizações de desempenho**:
   - Utiliza CUDA para aceleração em GPU
   - Configura streaming RTSP via TCP para maior estabilidade
   - Implementa mecanismos de recuperação e tolerância a falhas de conexão

### Arquivo `config.json`

O arquivo de configuração `config.json` permite personalizar o sistema sem modificar o código. Ele contém:

1. **Configurações da câmera**:
   - URL da câmera RTSP ou fonte de vídeo

2. **Configurações do dashboard**:
   - URL para envio de notificações

3. **Parâmetros do modelo**:
   - Caminho para o modelo
   - Níveis de confiança para detecção geral, mãos e pessoas
   - Configurações de IOU (Intersection over Union)

4. **Configurações de visualização**:
   - Opção para mostrar ou ocultar caixas de detecção de pessoas
   - Opção para mostrar ou ocultar linhas de associação pessoa-mesa
   - Modo de debug para depuração

5. **Parâmetros de tracking**:
   - Velocidade de atualização (alpha)
   - Tempos de confirmação e timeout
   - Parâmetros de distância para associação

6. **Capacidade das mesas**:
   - Definição da quantidade de lugares para cada tipo de mesa

### Utilidades de Configuração

- **`config_manager.py`**: Ferramenta interativa para configurar o sistema sem editar manualmente o arquivo JSON
- **Menu de instalação**: Opção específica para configurar o sistema interativamente

## Estrutura do Projeto

```
|- people.py                     # Arquivo principal com a lógica do sistema
|- roboflow.pt                   # Modelo de IA pré-treinado
|- config.json                   # Arquivo de configuração
|- config_manager.py             # Utilitário de configuração interativa
|
|- install_dependencies.py       # Script para instalar dependências
|- build_executable.py           # Script para criar o executável
|
|- install_and_run.bat           # Menu de instalação e execução (Windows)
|- install_and_run.sh            # Menu de instalação e execução (Linux)
|
|- setup_autostart_windows.bat   # Configuração de inicialização automática (Windows)
|- setup_autostart_linux.sh      # Configuração de inicialização automática (Linux)
```

## Requisitos do Sistema

- Python 3.8 ou superior
- GPU NVIDIA com suporte a CUDA
- CUDA Toolkit 11.0 ou superior
- PyTorch (versão com suporte a CUDA)
- OpenCV
- Ultralytics (biblioteca YOLOv8)
- Numpy
- Requests

## Configuração de Rede

O sistema foi configurado para utilizar o protocolo RTSP sobre TCP para maior estabilidade e confiabilidade na conexão com as câmeras. Isto é especialmente útil em redes com possibilidade de packet loss, onde o protocolo UDP (padrão do RTSP) pode gerar falhas na transmissão.

A configuração `OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp` é aplicada automaticamente pelo sistema.

## Instalação e Execução

### Windows

1. Verifique se a GPU com suporte CUDA está instalada e funcionando
2. Execute o arquivo `install_and_run.bat`
3. Selecione a opção 1 para instalar dependências
4. Selecione a opção 2 para configurar o sistema (URLs da câmera e dashboard, etc.)
5. Selecione a opção 3 para construir o executável (opcional)
6. Execute o programa via Python (opção 4) ou via executável (opção 5)

### Linux

1. Verifique se a GPU com suporte CUDA está instalada e funcionando
2. Dê permissão de execução ao script: `chmod +x install_and_run.sh`
3. Execute o arquivo `./install_and_run.sh`
4. Selecione a opção 1 para instalar dependências
5. Selecione a opção 2 para configurar o sistema (URLs da câmera e dashboard, etc.)
6. Selecione a opção 3 para construir o executável (opcional)
7. Execute o programa via Python (opção 4) ou via executável (opção 5)

## Configuração Manual

Para configurar manualmente o sistema, você pode:

1. Editar diretamente o arquivo `config.json` com um editor de texto
2. Usar o utilitário de configuração: `python config_manager.py`
3. Definir valores específicos via linha de comando:
   - `python config_manager.py camera rtsp://usuario:senha@ip/stream`
   - `python config_manager.py dashboard http://seu-servidor/api/endpoint`
   - `python config_manager.py model caminho/para/modelo.pt`

## Implantação em Servidor com Anydesk

Para implantar este sistema em um servidor acessível via Anydesk:

1. Certifique-se que o servidor possui GPU com suporte a CUDA
2. Instale o Anydesk no servidor destino
3. Transfira os arquivos do projeto para o servidor
4. Execute o script de instalação apropriado (conforme sistema operacional do servidor)
5. Configure o sistema usando a opção 2 do menu ou `config_manager.py`
6. Execute o programa e verifique o funcionamento
7. Configure o programa para iniciar automaticamente com o sistema usando os scripts:
   - Windows: `setup_autostart_windows.bat`
   - Linux: `setup_autostart_linux.sh`

## Notas Adicionais

- O sistema **requer obrigatoriamente** uma GPU com suporte a CUDA para processamento
- O streaming RTSP é configurado para usar TCP, o que melhora a estabilidade em redes com possibilidade de packet loss
- O sistema requer acesso a uma câmera IP com protocolo RTSP
- Todas as configurações são armazenadas em `config.json` e podem ser modificadas sem recompilar o executável 