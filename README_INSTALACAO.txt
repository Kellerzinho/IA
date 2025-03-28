[1;33m============================================================
[1;36m      MONITOR DE MESAS - SISTEMA DE DETECÇÃO
[1;33m============================================================[0m

[1;32mEste sistema monitora a ocupação de mesas através de câmeras RTSP,
utilizando inteligência artificial com YOLO e CUDA.[0m

[1;36;4m## REQUISITOS DO SISTEMA ##[0m

[1;37m1. Windows 10 ou superior
2. Python 3.8 ou superior
3. Placa de vídeo NVIDIA com suporte a CUDA
4. Drivers NVIDIA atualizados
5. Acesso a uma câmera via RTSP (ex: DVR Intelbras)[0m

[1;36;4m## COMO EXECUTAR ##[0m

[1;32mMétodo 1 (Mais simples):[0m
[1;37m- Dê um duplo clique no arquivo [1;33m"iniciar_monitor.bat"[1;37m[0m

[1;32mMétodo 2 (Criar atalho):[0m
[1;37m1. Siga as instruções no arquivo [1;33m"Executar Projeto.lnk"[1;37m
2. Use o atalho criado na área de trabalho[0m 

[1;36;4m## PRIMEIROS PASSOS ##[0m

[1;37mAo executar pela primeira vez:

1. O sistema verificará se todas as dependências estão instaladas
2. Se não existir um arquivo config.json, será criado automaticamente
3. Edite o arquivo config.json para configurar a URL da câmera RTSP
   Exemplo: [1;33m"url": "rtsp://admin:senha@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"[1;37m
4. Certifique-se de que o arquivo do modelo (roboflow.pt) está no diretório[0m

[1;36;4m## CONFIGURAÇÃO RTSP ##[0m

[1;32mPara DVRs Intelbras, use o formato:[0m
[1;33mrtsp://USUARIO:SENHA@IP_DO_DVR:554/cam/realmonitor?channel=NUMERO_CAMERA&subtype=0[0m

[1;37monde:
- USUARIO: nome de usuário do DVR (geralmente "admin")
- SENHA: senha de acesso ao DVR
- IP_DO_DVR: endereço IP do DVR (ex: 192.168.1.100)
- NUMERO_CAMERA: número do canal da câmera (1, 2, 3, etc.)
- subtype=0: stream principal (alta qualidade)
- subtype=1: stream secundário (menor qualidade)[0m

[1;36;4m## TECLAS DE CONTROLE ##[0m

[1;37mQuando o programa estiver rodando, utilize:
- [1;33mQ[1;37m: Encerra o programa
- [1;33mD[1;37m: Ativa/desativa modo debug
- [1;33mP[1;37m: Mostra/oculta caixas de pessoas
- [1;33mL[1;37m: Mostra/oculta linhas de associação[0m

[1;36;4m## SOLUÇÃO DE PROBLEMAS ##[0m

[1;31m1. Se ocorrer erro CUDA:[0m
   [1;37m- Verifique se os drivers NVIDIA estão atualizados
   - Execute o comando "nvidia-smi" no prompt para verificar a GPU[0m

[1;31m2. Se a câmera RTSP não conectar:[0m
   [1;37m- Verifique se a URL está correta
   - Teste a URL em outro player (como VLC)
   - Verifique se há firewall bloqueando a conexão[0m

[1;31m3. Logs e diagnóstico:[0m
   [1;37m- Verifique o arquivo "table_tracker.log" para mais detalhes de erros[0m

[1;36;4m## SUPORTE ##[0m

[1;32mEm caso de problemas, entre em contato com a equipe de suporte técnico.[0m

[1;33m============================================================[0m

[1;37mNota: Para visualizar corretamente as cores, visualize este arquivo em
um terminal ou prompt de comando que suporte códigos de cores ANSI.[0m 