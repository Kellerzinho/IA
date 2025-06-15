import cv2
import numpy as np
from ultralytics import YOLO
import time
import logging
import sys
import requests
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from collections import defaultdict
import json
import torch
import threading
from collections import OrderedDict
import os
import re
import psutil  # Para monitoramento de memória

# Forçar o uso de CUDA (GPU) para processamento
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Verificação do dispositivo disponível
if not torch.cuda.is_available():
    print("ERRO: GPU com suporte a CUDA não disponível. Este sistema requer GPU para funcionar.")
    print("Por favor, verifique a instalação do CUDA e os drivers da GPU.")
    sys.exit(1)

# Configura dispositivo para GPU (CUDA)
device = torch.device('cuda')
print(f"Usando dispositivo: {device} - {torch.cuda.get_device_name(0)}")
print(f"Memória GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Constante para o arquivo que guarda a última porta
LAST_PORT_FILE = 'last_used_port.txt'

# Adicionar após as importações e variáveis globais iniciais, antes da classe AppConfig
# Dicionário global para controle de IDs únicos por câmera
global_table_ids = {}
global_ids_lock = threading.Lock()
global_next_id = 1

def get_unique_table_id(camera_id):
    """
    Gera um ID único para uma mesa baseado na câmera.
    Mantém IDs diferentes entre câmeras usando um dicionário global.
    """
    global global_next_id
    with global_ids_lock:
        # Garantir que camera_id é uma string
        camera_id = str(camera_id)
        
        # Inicializar o contador para esta câmera se não existir
        if camera_id not in global_table_ids:
            global_table_ids[camera_id] = []
        
        # Encontrar próximo ID único
        new_id = global_next_id
        global_next_id += 1
        
        # Registrar o ID para esta câmera
        global_table_ids[camera_id].append(new_id)
        
        return new_id

def get_camera_table_ids(camera_id):
    """
    Retorna a lista de IDs de mesas atribuídos a uma câmera específica.
    """
    with global_ids_lock:
        camera_id = str(camera_id)
        return global_table_ids.get(camera_id, []).copy()

def print_table_id_distribution():
    """
    Imprime a distribuição atual de IDs de mesas por câmera.
    Útil para depuração.
    """
    with global_ids_lock:
        print("\n=== Distribuição de IDs de Mesas por Câmera ===")
        for camera_id, ids in sorted(global_table_ids.items()):
            print(f"Câmera {camera_id}: {len(ids)} mesas - IDs: {sorted(ids)}")
        print(f"Próximo ID global: {global_next_id}")
        print("==================================================\n")

def remove_global_table_id(camera_id, table_id):
    """
    Remove um ID de mesa específico do sistema global quando a mesa é excluída.
    """
    with global_ids_lock:
        camera_id = str(camera_id)
        if camera_id in global_table_ids:
            if table_id in global_table_ids[camera_id]:
                global_table_ids[camera_id].remove(table_id)
                # Se a lista ficou vazia, remove a entrada da câmera
                if not global_table_ids[camera_id]:
                    del global_table_ids[camera_id]

def sync_global_table_ids(camera_id, active_table_ids):
    """
    Sincroniza os IDs globais com os IDs atualmente ativos na câmera.
    Remove IDs que não existem mais na câmera.
    """
    with global_ids_lock:
        camera_id = str(camera_id)
        if camera_id in global_table_ids:
            # Manter apenas IDs que ainda existem na câmera
            global_table_ids[camera_id] = [tid for tid in global_table_ids[camera_id] if tid in active_table_ids]
            # Se a lista ficou vazia, remove a entrada da câmera
            if not global_table_ids[camera_id]:
                del global_table_ids[camera_id]

def cleanup_global_ids():
    """
    Limpeza periódica do sistema de IDs globais.
    Redefine o próximo ID global baseado no maior ID atualmente em uso.
    """
    global global_next_id
    with global_ids_lock:
        if global_table_ids:
            # Encontra o maior ID em uso
            all_ids = []
            for camera_ids in global_table_ids.values():
                all_ids.extend(camera_ids)
            
            if all_ids:
                max_id = max(all_ids)
                # Define o próximo ID como max_id + 1
                global_next_id = max_id + 1
            else:
                # Se não há IDs em uso, reinicia do 1
                global_next_id = 1
        else:
            # Se não há câmeras, reinicia do 1
            global_next_id = 1

@dataclass
class AppConfig:
    cls_to_lugares: Dict[int, int] = field(default_factory=dict)
    colors: Dict[str, Tuple[int]] = field(default_factory=lambda: {
        'CHEIA': (0, 0, 255),
        'VAZIA': (0, 255, 0),
        'STANDBY': (128, 128, 128),
        'ATENDIMENTO': (0, 255, 255),
        'MAO': (0, 165, 255),
        'OCUPADA': (255, 165, 0),
        'PESSOA': (128, 0, 128)
    })
    tracking_params: Dict[str, float] = field(default_factory=dict)
    
    # Configurações do modelo
    model_conf: float = 0.1
    model_mao_conf: float = 0.8
    model_pessoa_conf: float = 0.3
    model_iou: float = 0.4
    model_agnostic_nms: bool = True
    model_path: str = 'roboflow.pt'
    
    # Configurações de câmeras
    cameras: List[Dict[str, str]] = field(default_factory=list)
    
    # Classe para pessoas
    people_cls_id: int = 8
    
    # Opções de visualização
    show_people_boxes: bool = True
    show_association_lines: bool = True
    debug_mode: bool = False
    
    # Nomes das classes
    cls_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Mesa 10L", 1: "Mesa 12L", 2: "Mesa 14L", 3: "Mesa 2L",
        4: "Mesa 4L", 5: "Mesa 6L", 6: "Mesa 8L", 7: "Mão", 8: "Pessoa"
    })
    
    # Configuração do método HTTP para dashboard
    dashboard_http_method: str = 'PUT'
    
    @classmethod
    def load_from_json(cls, json_file='config.json'):
        """Carrega configurações a partir de um arquivo JSON."""
        config = cls()
        
        try:
            if not os.path.exists(json_file):
                print(f"Arquivo de configuração {json_file} não encontrado. Usando valores padrão.")
                return config
                
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Câmeras (múltiplas)
            if 'cameras' in data:
                config.cameras = data['cameras']
                print(f"Configuradas {len(config.cameras)} câmeras")
            
            # Dashboard
            if 'dashboard' in data:
                if 'url' in data['dashboard']:
                    config.dashboard_url = data['dashboard']['url']
                if 'http_method' in data['dashboard']:
                    config.dashboard_http_method = data['dashboard']['http_method'].upper()
            
            # Model params
            if 'model' in data:
                model_data = data['model']
                if 'path' in model_data:
                    config.model_path = model_data['path']
                if 'conf' in model_data:
                    config.model_conf = float(model_data['conf'])
                if 'mao_conf' in model_data:
                    config.model_mao_conf = float(model_data['mao_conf'])
                if 'pessoa_conf' in model_data:
                    config.model_pessoa_conf = float(model_data['pessoa_conf'])
                if 'iou' in model_data:
                    config.model_iou = float(model_data['iou'])
                if 'agnostic_nms' in model_data:
                    config.model_agnostic_nms = bool(model_data['agnostic_nms'])
            
            # Detection options
            if 'detection' in data:
                detection_data = data['detection']
                if 'show_people_boxes' in detection_data:
                    config.show_people_boxes = bool(detection_data['show_people_boxes'])
                if 'show_association_lines' in detection_data:
                    config.show_association_lines = bool(detection_data['show_association_lines'])
                if 'debug_mode' in detection_data:
                    config.debug_mode = bool(detection_data['debug_mode'])
            
            # Tracking params
            if 'tracking_params' in data:
                config.tracking_params = data['tracking_params']
            
            # Table capacities
            if 'table_capacities' in data:
                config.cls_to_lugares = {int(k): int(v) for k, v in data['table_capacities'].items()}
            
            print("Configurações carregadas com sucesso de", json_file)
        except Exception as e:
            print(f"Erro ao carregar configurações: {str(e)}. Usando valores padrão.")
        
        return config


def is_near(box1, box2, threshold=100):
    """Verifica se o centro do box1 está dentro da distância 'threshold' do centro do box2."""
    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2) < threshold


def update_box(old_box, new_box, alpha=0.3):
    """Faz média ponderada das coordenadas antigas e novas."""
    return tuple(
        int(alpha*new + (1-alpha)*old)
        for old, new in zip(old_box, new_box)
    )


class TableManager:
    def __init__(self, config: AppConfig, camera_id=None, camera_name=None):
        self.config = config
        
        self.tables: Dict[int, Dict] = {}  # ID da mesa -> dados da mesa
        self.pending_new_tables: Dict[Tuple[int, Tuple], float] = {}
        self.maos_detectadas: Dict[Tuple[int, Tuple], float] = {}
        self.state_history: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
        self.mesas_notificadas = set()
        self.next_creation_index = 1
        
        # Identificação da câmera
        self.camera_id = camera_id
        self.camera_name = camera_name
        
        self.setup_logger()

    def setup_logger(self):
        self.logger = logging.getLogger('Monitor')
        # Muda o nível para DEBUG para capturar mensagens de depuração
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler sem filtro
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))
        # Remover o uso do filtro personalizado
        ch.setLevel(logging.INFO)  # Define que apenas mensagens INFO ou mais críticas vão para o console

        # Arquivo único para todos os logs de dashboard
        dh = logging.FileHandler('dashboard.log', encoding='utf-8')
        dh.setFormatter(formatter)
        dh.setLevel(logging.DEBUG)

        # Arquivo único para todas as câmeras
        fh = logging.FileHandler('monitor.log', encoding='utf-8')
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.addHandler(dh)

    def notificar_dashboard(self, mensagem: str):
        """Envia notificação para o dashboard via thread separada."""
        def enviar_notificacao(mensagem):
            try:
                self.logger.debug(f"[DASHBOARD] Tentando enviar: {mensagem[:100]}...")
                
                headers = {'Content-Type': 'application/json'}
                self.logger.debug(f"[DASHBOARD] URL destino: {self.config.dashboard_url}")
                self.logger.debug(f"[DASHBOARD] Método HTTP: {self.config.dashboard_http_method}")
                
                # Usa o método PUT para enviar as notificações
                resp = requests.put(
                    self.config.dashboard_url,
                    data=mensagem,
                    headers=headers,
                    timeout=3
                )
                
                if resp.status_code == 200:
                    self.logger.debug(f"[DASHBOARD] Notificação enviada com sucesso")
                else:
                    self.logger.error(f"[DASHBOARD] Falha ao enviar notificação: {resp.status_code}")
                    self.logger.error(f"[DASHBOARD] Resposta: {resp.text[:100]}...")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"[DASHBOARD] Falha na conexão: {str(e)}")
                self.logger.error(f"[DASHBOARD] URL tentada: {self.config.dashboard_url}")
            except Exception as e:
                self.logger.error(f"[DASHBOARD] Erro inesperado: {str(e)}")
                import traceback
                self.logger.error(f"[DASHBOARD] Traceback: {traceback.format_exc()}")
    
        # Verificação básica de mensagem
        try:
            # Valida se a mensagem é JSON válido
            json_obj = json.loads(mensagem)
            
            # Se ainda não processamos o camera_id em outro lugar, fazemos aqui
            if "camera_id" in json_obj and isinstance(json_obj["camera_id"], str) and json_obj["camera_id"].startswith("camera"):
                # Extrair apenas o número da câmera
                json_obj["camera_id"] = self._extract_camera_number(json_obj["camera_id"])
                # Recodifica para JSON
                mensagem = json.dumps(json_obj)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"[DASHBOARD] Mensagem inválida (não é JSON): {str(e)}")
            return
            
        self.logger.debug(f"[DASHBOARD] Iniciando thread para envio de notificação")
        thread = threading.Thread(target=enviar_notificacao, args=(mensagem,))
        thread.daemon = True
        thread.start()

    def initialize_tables(self, detected_boxes: List[Tuple[int, Tuple]], current_time: float):
        """Inicializa as mesas com base nas detecções recebidas na primeira leitura."""
        filtered_boxes = []
        # Ordenação inicial: (y, x) => processar de cima p/ baixo, esquerda p/ direita
        for cls, coords in sorted(detected_boxes, key=lambda x: (x[1][1], x[1][0])):
            if not any(is_near(ex_coords, coords, 80) for _, ex_coords in filtered_boxes):
                filtered_boxes.append((cls, coords))
    
        self.tables = {}
        for i, (cls, coords) in enumerate(filtered_boxes, start=1):
            # Obter ID único para esta mesa baseado na câmera
            unique_id = get_unique_table_id(self.camera_id)
            
            self.tables[unique_id] = {
                'cls': cls,
                'coords': coords,
                'last_seen': current_time,
                'state': 'VAZIA',
                'pending_state': 'VAZIA',
                'state_change_time': current_time,
                'precisa_atendimento': False,
                'ultimo_atendimento': None,
                'notificada_atendimento': False,
                'creation_index': self.next_creation_index,
                'occupant_count': 0,
                'last_occupancy_change': current_time,
                'pending_occupancy': None
            }
            self.next_creation_index += 1
            
            if self.config.debug_mode:
                self.logger.info(f"Mesa {unique_id} criada: tipo={cls} ({self.config.cls_names.get(cls, 'Desconhecida')}), lugares={self.config.cls_to_lugares.get(cls, 0)}")
        
        self.mesas_notificadas.clear()

    def update_states(self, detected_boxes: List[Tuple[int, Tuple]], current_time: float):
        """Faz a "two-pass merge" (tracking manual) para as mesas."""
        old_tables = dict(self.tables)
        updated_tables = {}
        used_ids = set()

        # ---------- PASSO 1: Identifica possíveis merges ----------
        merges = {}
        for det_cls, det_coords in detected_boxes:
            # Primeiro, procura por mesas ativas próximas
            active_matching_ids = [
                tid for tid, tdata in old_tables.items()
                if tdata['state'] != 'STANDBY' and is_near(tdata['coords'], det_coords, self.config.tracking_params['distance_threshold'])
            ]
            
            # Se não encontrou mesas ativas, procura por mesas STANDBY próximas para reativar
            standby_matching_ids = [
                tid for tid, tdata in old_tables.items()
                if tdata['state'] == 'STANDBY' and is_near(tdata['coords'], det_coords, self.config.tracking_params['distance_threshold'])
            ]
            
            # Prioriza mesas ativas, mas se não há ativas, considera STANDBY
            matching_ids = active_matching_ids if active_matching_ids else standby_matching_ids

            if not matching_ids:
                # Se não achamos mesa ativa nem STANDBY próxima, vira "pending"
                self.pending_new_tables[(det_cls, det_coords)] = current_time
            else:
                # Se tem match, escolhe ID MAIOR
                chosen_id = max(matching_ids)
                if chosen_id not in merges:
                    merges[chosen_id] = set()
                for mid in matching_ids:
                    merges[chosen_id].add(mid)
                used_ids.update(matching_ids)
                
                # Se reativando uma mesa STANDBY, marca para log
                if old_tables[chosen_id]['state'] == 'STANDBY':
                    if self.config.debug_mode:
                        self.logger.info(f"Mesa STANDBY {chosen_id} será reativada por detecção próxima")

        # ---------- PASSO 2: Aplica merges ----------
        det_in_merge = {}
        for det_cls, det_coords in detected_boxes:
            for chosen_id, group_ids in merges.items():
                if chosen_id in group_ids:
                    if is_near(self.tables[chosen_id]['coords'], det_coords, self.config.tracking_params['distance_threshold']):
                        det_in_merge[chosen_id] = (det_cls, det_coords)
                    else:
                        for mid in group_ids:
                            if is_near(self.tables[mid]['coords'], det_coords, self.config.tracking_params['distance_threshold']):
                                det_in_merge[chosen_id] = (det_cls, det_coords)
                                break

        for chosen_id, group_ids in merges.items():
            if chosen_id not in det_in_merge:
                # Se não houve detecção para o chosen_id
                final_det_cls = self.tables[chosen_id]['cls']
                final_det_coords = self.tables[chosen_id]['coords']
            else:
                final_det_cls, final_det_coords = det_in_merge[chosen_id]

            # IDs que não são chosen_id => ficam STANDBY
            for mid in group_ids:
                if mid == chosen_id:
                    continue
                if mid in old_tables:
                    mid_data = old_tables[mid]
                    del old_tables[mid]
                else:
                    mid_data = updated_tables.get(mid, None)
                    if mid in updated_tables:
                        del updated_tables[mid]

                if mid_data is not None:
                    mid_data['state'] = "STANDBY"
                    updated_tables[mid] = mid_data

            # Atualiza chosen_id
            self._update_existing_table(chosen_id, final_det_cls, final_det_coords, current_time, updated_tables)

        # ---------- PASSO 3: Processa pendentes e ausentes ----------
        self._process_pending_tables(current_time, updated_tables)
        self._update_missing_tables(current_time, updated_tables, used_ids, old_tables)
        self._update_history_and_alerts(current_time)

        # ---------- PASSO 4: Salva e reordena ----------
        self.tables = self._sort_tables_by_creation(updated_tables)

        # ---------- PASSO 5: Limpa STANDBY obsoletas ----------
        self._cleanup_standby_tables()

    def _process_pending_tables(self, current_time, updated_tables):
        """Tenta confirmar novas mesas pendentes."""
        to_remove = []
        for (det_cls, det_coords), detected_time in list(self.pending_new_tables.items()):
            
            # CORREÇÃO 1: Verificar proximidade apenas com mesas ATIVAS (não STANDBY)
            active_tables = [
                mesa for mesa in self.tables.values() 
                if mesa['state'] != 'STANDBY'
            ]
            
            too_close_to_active = any(
                is_near(mesa['coords'], det_coords, self.config.tracking_params['distance_threshold'])
                for mesa in active_tables
            )
            
            # CORREÇÃO 2: Verificar se há mesa STANDBY próxima para reativar
            standby_tables = [
                (tid, mesa) for tid, mesa in self.tables.items() 
                if mesa['state'] == 'STANDBY'
            ]
            
            nearby_standby = None
            for tid, mesa in standby_tables:
                if is_near(mesa['coords'], det_coords, self.config.tracking_params['distance_threshold']):
                    nearby_standby = (tid, mesa)
                    break
            
            # Se não está perto de mesa ativa e já passou confirm_time
            if not too_close_to_active and (current_time - detected_time >= self.config.tracking_params['confirm_time']):
                
                # CORREÇÃO 3: Se há mesa STANDBY próxima, reativa ela ao invés de criar nova
                if nearby_standby:
                    standby_id, standby_data = nearby_standby
                    
                    # Reativa a mesa STANDBY
                    standby_data['state'] = 'VAZIA'
                    standby_data['cls'] = det_cls  # Atualiza classe se necessário
                    standby_data['coords'] = update_box(standby_data['coords'], det_coords, 0.5)
                    standby_data['last_seen'] = current_time
                    standby_data['state_change_time'] = current_time
                    standby_data['occupant_count'] = 0
                    standby_data['last_occupancy_change'] = current_time
                    standby_data['pending_occupancy'] = None
                    
                    updated_tables[standby_id] = standby_data
                    to_remove.append((det_cls, det_coords))
                    
                    if self.config.debug_mode:
                        self.logger.info(f"Mesa STANDBY {standby_id} reativada: tipo={det_cls} ({self.config.cls_names.get(det_cls, 'Desconhecida')})")
                
                else:
                    # CORREÇÃO 4: Só cria nova mesa se não há STANDBY próxima
                    new_id = get_unique_table_id(self.camera_id)
                    
                    updated_tables[new_id] = {
                        'cls': det_cls,
                        'coords': det_coords,
                        'last_seen': current_time,
                        'state': 'VAZIA',
                        'pending_state': 'VAZIA',
                        'state_change_time': current_time,
                        'precisa_atendimento': False,
                        'ultimo_atendimento': None,
                        'creation_index': self.next_creation_index,
                        'occupant_count': 0,
                        'last_occupancy_change': current_time,
                        'pending_occupancy': None
                    }
                    self.next_creation_index += 1
                    to_remove.append((det_cls, det_coords))
                    
                    if self.config.debug_mode:
                        self.logger.info(f"Nova mesa {new_id} criada: tipo={det_cls} ({self.config.cls_names.get(det_cls, 'Desconhecida')}), lugares={self.config.cls_to_lugares.get(det_cls, 0)}")

        for key in to_remove:
            del self.pending_new_tables[key]

    def _update_missing_tables(self, current_time, updated_tables, used_ids, old_tables):
        """Se mesa não detectada neste frame e passou max_missed_time, vira STANDBY. Caso contrário, mantém."""
        for tid, tdata in old_tables.items():
            if (tid not in used_ids) and (tid not in updated_tables):
                time_missing = current_time - tdata['last_seen']
                if time_missing > self.config.tracking_params['max_missed_time']:
                    tdata['state'] = "STANDBY"
                    updated_tables[tid] = tdata

    def _update_existing_table(self, table_id, det_cls, det_coords, current_time, updated_tables):
        """Faz a atualização de uma mesa existente com a nova detecção (fusão)."""
        table_data = self.tables[table_id]
        new_coords = update_box(table_data['coords'], det_coords, self.config.tracking_params['alpha'])

        # Atualiza coords / last_seen
        table_data['coords'] = new_coords
        table_data['last_seen'] = current_time
        
        # Se a mesa estava em STANDBY, reativa ela
        if table_data['state'] == 'STANDBY':
            old_state = table_data['state']
            table_data['state'] = 'VAZIA'
            table_data['cls'] = det_cls  # Atualiza classe se necessário
            table_data['state_change_time'] = current_time
            table_data['occupant_count'] = 0
            table_data['last_occupancy_change'] = current_time
            table_data['pending_occupancy'] = None
            table_data['precisa_atendimento'] = False
            table_data['ultimo_atendimento'] = None
            
            if self.config.debug_mode:
                self.logger.info(f"Mesa {table_id} reativada de {old_state} para VAZIA")

        updated_tables[table_id] = table_data

    def _log_state_change(self, table_id, old_state, new_state):
        """Faz log e envia notificação quando a mesa muda de estado."""
        camera_info = f"[{self.camera_name or self.camera_id or 'Câmera desconhecida'}] " if self.camera_id or self.camera_name else ""
        self.logger.info(f"{camera_info}Mesa {table_id} mudou de {old_state} para {new_state}")
        
        # Se ficou VAZIA e antes era CHEIA ou OCUPADA, notificar "mesa liberada"
        if new_state == 'VAZIA' and old_state in ['CHEIA', 'OCUPADA']:
            lugares = self.config.cls_to_lugares.get(self.tables[table_id]['cls'], 0)
            msg_terminal = f"{camera_info}LIBEROU {lugares} LUGARES - Mesa {table_id}"
            self.logger.info(msg_terminal)
            
            # Extrair apenas o número da câmera
            camera_num = self._extract_camera_number(self.camera_id)
            
            # Estrutura da câmera
            camera_data = {
                "tipo": "mesa_liberada",
                "mesa_id": table_id,
                "lugares": lugares,
                "timestamp": time.time(),
                "camera_id": camera_num
            }
            
            # Enviar para o dashboard
            msg_dashboard = {
                "restaurante": "CSVL",
                "cameras": [camera_data]
            }
            self.notificar_dashboard(json.dumps(msg_dashboard))

    def _update_history_and_alerts(self, current_time):
        """Salva histórico de estados para cada mesa."""
        for table_id, table_data in self.tables.items():
            self.state_history[table_id].append((current_time, table_data['state']))

    def _sort_tables_by_creation(self, tables_dict: Dict[int, Dict]) -> OrderedDict:
        """Ordena as mesas pelo campo 'creation_index' e retorna um OrderedDict."""
        sorted_items = sorted(
            tables_dict.items(), 
            key=lambda item: item[1]['creation_index']
        )
        return OrderedDict(sorted_items)

    def _cleanup_standby_tables(self):
        """
        Remove mesas STANDBY nas seguintes condições:
        1. Que não foram vistas há muito tempo (mas protege as que estão entre mesas ativas)
        2. Limita o número total de mesas em STANDBY para evitar acúmulo (mas protege as que estão entre mesas ativas)
        """
        current_time = time.time()
        standby_timeout = self.config.tracking_params.get('standby_timeout', 300)  # 5 minutos
        
        active_tables = [tid for tid, tdata in self.tables.items() 
                         if tdata['state'] != 'STANDBY']
        
        if not active_tables:
            return
        
        # Obter creation_index das mesas ativas para determinar "gaps"
        active_creation_indices = [self.tables[tid]['creation_index'] for tid in active_tables]
        min_active_index = min(active_creation_indices)
        max_active_index = max(active_creation_indices)
        
        # Função para verificar se uma mesa STANDBY está protegida (entre mesas ativas)
        def is_protected_standby(table_data):
            creation_index = table_data['creation_index']
            return min_active_index < creation_index < max_active_index
        
        # Lista para rastrear IDs removidos
        removed_ids = []
        
        # 1. Remover STANDBY muito antigas (mas proteger as que estão entre mesas ativas)
        to_remove_by_time = []
        for tid, tdata in self.tables.items():
            if tdata['state'] == 'STANDBY':
                time_in_standby = current_time - tdata['last_seen']
                
                # Se está entre mesas ativas (por creation_index), não remove por timeout
                if time_in_standby > standby_timeout and not is_protected_standby(tdata):
                    to_remove_by_time.append(tid)
        
        for tid in to_remove_by_time:
            del self.tables[tid]
            removed_ids.append(tid)
            if self.config.debug_mode:
                self.logger.info(f"Removida mesa STANDBY {tid} (timeout: {standby_timeout}s, não está entre mesas ativas)")
        
        # 2. Limitar o número máximo de mesas em STANDBY (mas proteger as que estão entre mesas ativas)
        standby_tables = [tid for tid, tdata in self.tables.items() 
                          if tdata['state'] == 'STANDBY']
        
        max_standby_tables = 3
        
        if len(standby_tables) > max_standby_tables:
            # Separar mesas STANDBY protegidas das não protegidas
            protected_standby = []
            unprotected_standby = []
            
            for tid in standby_tables:
                if is_protected_standby(self.tables[tid]):
                    protected_standby.append(tid)
                else:
                    unprotected_standby.append(tid)
            
            # Se temos mesas protegidas + não protegidas > limite, remove apenas das não protegidas
            total_standby = len(protected_standby) + len(unprotected_standby)
            
            if total_standby > max_standby_tables:
                # Calcular quantas precisamos remover
                to_remove_count = total_standby - max_standby_tables
                
                # Primeiro, tenta remover apenas das não protegidas
                if len(unprotected_standby) >= to_remove_count:
                    # Ordenar não protegidas por last_seen (as mais antigas primeiro)
                    sorted_unprotected = sorted(
                        [(tid, self.tables[tid]['last_seen']) for tid in unprotected_standby],
                        key=lambda x: x[1]
                    )
                    
                    to_remove_by_limit = [tid for tid, _ in sorted_unprotected[:to_remove_count]]
                    
                    for tid in to_remove_by_limit:
                        del self.tables[tid]
                        removed_ids.append(tid)
                        if self.config.debug_mode:
                            self.logger.info(f"Removida mesa STANDBY {tid} (limite excedido, não protegida)")
                else:
                    # Se não há suficientes não protegidas, remove todas as não protegidas
                    for tid in unprotected_standby:
                        del self.tables[tid]
                        removed_ids.append(tid)
                        if self.config.debug_mode:
                            self.logger.info(f"Removida mesa STANDBY {tid} (limite excedido, não protegida)")
                    
                    # E avisa que há muitas mesas protegidas
                    if self.config.debug_mode:
                        self.logger.info(f"AVISO: {len(protected_standby)} mesas STANDBY protegidas (entre ativas) excedem limite, mas não foram removidas")
        
        # 3. NOVO: Sincronizar IDs globais após remoções
        if removed_ids:
            # Remove IDs específicos do sistema global
            for tid in removed_ids:
                remove_global_table_id(self.camera_id, tid)
            
            # Sincroniza todos os IDs desta câmera
            current_table_ids = list(self.tables.keys())
            sync_global_table_ids(self.camera_id, current_table_ids)
            
            if self.config.debug_mode:
                self.logger.info(f"Sincronizados IDs globais da câmera {self.camera_id}: removidos {removed_ids}")
        
        # Reordenar após a remoção
        self.tables = self._sort_tables_by_creation(self.tables)

    # -------------------------------------------------------
    #   MÃO LEVANTADA & ATENDIMENTO
    # -------------------------------------------------------
    def process_maos_levantadas(self, maos: List[Tuple[int, Tuple]], current_time: float):
        """Verifica se 'mão' confirmada está próxima de uma mesa CHEIA => pedido de atendimento."""
        # Adiciona detecções novas
        for mao_cls, mao_coords in maos:
            # Verificar se as coordenadas são válidas
            if len(mao_coords) != 4:
                continue
            self.maos_detectadas[(mao_cls, mao_coords)] = current_time

        # Verifica se persistiu tempo suficiente => mão confirmada
        maos_confirmadas = []
        for (mao_cls, mao_coords), detection_time in list(self.maos_detectadas.items()):
            if (current_time - detection_time) >= self.config.tracking_params['mao_confirm_time']:
                maos_confirmadas.append(mao_coords)
                del self.maos_detectadas[(mao_cls, mao_coords)]

        # Distância ponderada
        def dist_mao_mesa(mao, mesa):
            # Verificar validade das coordenadas
            if len(mao) != 4 or len(mesa) != 4:
                return float('inf')
            
            mx = (mao[0] + mao[2]) / 2
            my = mao[3]  # base da mão
            tx = (mesa[0] + mesa[2]) / 2
            ty = (mesa[1] + mesa[3]) / 2
            dx = abs(mx - tx) * self.config.tracking_params['peso_horizontal']
            dy = abs(my - ty) * self.config.tracking_params['peso_vertical']
            return np.sqrt(dx**2 + dy**2)

        # Se a mão confirmada estiver próxima de uma mesa CHEIA, vira "ATENDIMENTO"
        for mao_coords in maos_confirmadas:
            melhor_id = None
            menor_dist = float('inf')
            for tid, tdata in self.tables.items():
                # Só processa mesas CHEIas ou OCUPADAs
                if tdata['state'] not in ['CHEIA', 'OCUPADA']:
                    continue
                
                d = dist_mao_mesa(mao_coords, tdata['coords'])

                # Check outliers
                mao_cx = (mao_coords[0] + mao_coords[2]) / 2
                mesa_cx = (tdata['coords'][0] + tdata['coords'][2]) / 2
                dif_h = abs(mao_cx - mesa_cx)

                mao_by = mao_coords[3]
                mesa_ty = tdata['coords'][1]
                dif_v = mao_by - mesa_ty

                if (dif_h > self.config.tracking_params['mao_horizontal_threshold']
                    or dif_v > self.config.tracking_params['mao_vertical_threshold']):
                    continue
                
                if d < menor_dist:
                    menor_dist = d
                    melhor_id = tid

            if melhor_id is not None:
                if not self.tables[melhor_id]['precisa_atendimento']:
                    self.tables[melhor_id]['precisa_atendimento'] = True
                    self.tables[melhor_id]['state'] = "ATENDIMENTO"
                    self.tables[melhor_id]['ultimo_atendimento'] = current_time
                    
                    if melhor_id not in self.mesas_notificadas:
                        camera_info = f"[{self.camera_name or self.camera_id or 'Câmera desconhecida'}] " if self.camera_id or self.camera_name else ""
                        msg_terminal = f"{camera_info}ATENDIMENTO: Mesa {melhor_id}"
                        self.logger.info(msg_terminal)
                        
                        # Extrair apenas o número da câmera
                        camera_num = self._extract_camera_number(self.camera_id)
                        
                        # Estrutura da câmera
                        camera_data = {
                            "tipo": "atendimento_requisitado",
                            "mesa_id": melhor_id,
                            "timestamp": current_time,
                            "camera_id": camera_num
                        }
                        
                        # Estrutura de notificação
                        msg_dashboard = {
                            "restaurante": "CSVL",
                            "cameras": [camera_data]
                        }
                        self.notificar_dashboard(json.dumps(msg_dashboard))
                        
                        self.mesas_notificadas.add(melhor_id)

        # Timeout de atendimento
        for tid, tdata in list(self.tables.items()):
            if tdata['precisa_atendimento']:
                ultimo = tdata['ultimo_atendimento']
                if ultimo is not None:
                    if (current_time - ultimo) > self.config.tracking_params['atendimento_timeout']:
                        # se ainda tem pessoas, volta a CHEIA; senão, VAZIA
                        lugares = self.config.cls_to_lugares.get(tdata['cls'], 0)
                        old_state = tdata['state']
                        
                        # Determina o novo estado com base na contagem de ocupantes
                        if tdata['occupant_count'] >= lugares:
                            new_state = 'CHEIA'
                        elif tdata['occupant_count'] > 0:
                            new_state = 'OCUPADA'
                        else:
                            new_state = 'VAZIA'
                        
                        # Atualiza o estado explicitamente
                        if self.config.debug_mode:
                            self.logger.info(f"Timeout atendimento Mesa {tid}: mudando de {old_state} para {new_state} (occupant_count={tdata['occupant_count']}, lugares={lugares})")
                        
                        # Atualiza estado e notifica a mudança
                        tdata['state'] = new_state
                        
                        # Só chama _log_state_change se realmente houve mudança
                        if old_state != new_state:
                            self._log_state_change(tid, old_state, new_state)

                        # Limpa flags de atendimento
                        tdata['precisa_atendimento'] = False
                        tdata['ultimo_atendimento'] = None
                        if tid in self.mesas_notificadas:
                            self.mesas_notificadas.remove(tid)
                        
                        self.logger.info(f"Atendimento da Mesa {tid} expirado")

    # -------------------------------------------------------
    #   ATRIBUIR PESSOAS ÀS MESAS
    # -------------------------------------------------------
    def assign_people_to_tables(self, people: List[Tuple[int, Tuple]]):
        """
        Conta quantas pessoas há em cada mesa (com base em proximidade),
        e atualiza occupant_count. Atualiza estado (VAZIA, OCUPADA, CHEIA)
        de acordo com occupant_count em comparação com lugares.
        """
        # Inicializa estrutura para rastrear mudanças significativas
        previous_counts = {tid: tdata['occupant_count'] for tid, tdata in self.tables.items()}
        
        # Zera occupant_count antes de recalcular
        for tdata in self.tables.values():
            tdata['occupant_count'] = 0

        # Função de distância (centro da pessoa vs centro da mesa)
        def dist_person_table(person_box, table_box):
            # Verificar se os boxes têm formato válido
            if len(person_box) != 4 or len(table_box) != 4:
                return float('inf')
            
            # Usamos o centro da pessoa e da mesa para calcular distância
            px = (person_box[0] + person_box[2]) / 2
            py = (person_box[1] + person_box[3]) / 2
            tx = (table_box[0] + table_box[2]) / 2
            ty = (table_box[1] + table_box[3]) / 2
            
            # Distância euclidiana simples
            return np.sqrt((px - tx)**2 + (py - ty)**2)

        # Lista para armazenar associações pessoa-mesa para visualização
        person_table_associations = []
        
        # PARTE 1: Associar pessoas às mesas
        for p_cls, p_coords in people:
            # Para cada pessoa, acha a mesa mais próxima
            melhor_id = None
            menor_dist = float('inf')
            
            # Removendo logs de debug de processamento de pessoa
            # Esses logs são muito frequentes e poluem o console
            
            for tid, tdata in self.tables.items():
                # Se mesa está em STANDBY, ignore
                if tdata['state'] == 'STANDBY':
                    continue
                    
                d = dist_person_table(p_coords, tdata['coords'])
                
                # Removendo logs de distância para cada mesa
                # Esses logs são muito frequentes e poluem o console
                    
                if d < menor_dist:
                    menor_dist = d
                    melhor_id = tid
            
            # Usa o parâmetro de configuração para a distância máxima
            max_dist = self.config.tracking_params.get('pessoa_to_table_max_dist', 200)
            
            if melhor_id is not None and menor_dist < max_dist:
                self.tables[melhor_id]['occupant_count'] += 1
                person_table_associations.append((p_coords, self.tables[melhor_id]['coords'], melhor_id))
                
                # Removendo logs de associação de pessoa à mesa
                # Esses logs são muito frequentes e poluem o console
                
                # Garante que a mesa tenha as propriedades necessárias
                # Inicializa o campo last_occupancy_change se ainda não existir
                if 'last_occupancy_change' not in self.tables[melhor_id]:
                    self.tables[melhor_id]['last_occupancy_change'] = time.time()
                if 'pending_occupancy' not in self.tables[melhor_id]:
                    self.tables[melhor_id]['pending_occupancy'] = None

        # Armazena as associações para visualização
        self.person_table_associations = person_table_associations
        
        # PARTE 2: Atualizar estados com base na contagem
        current_time = time.time()
        state_change_delay = self.config.tracking_params.get('state_change_delay', 1)
        
        for tid, tdata in self.tables.items():
            if tdata['state'] == 'STANDBY':
                continue  # não mexe em mesas STANDBY
                
            lugares = self.config.cls_to_lugares.get(tdata['cls'], 0)
            old_state = tdata['state']
            current_count = tdata['occupant_count']
            previous_count = previous_counts.get(tid, 0)
            
            # Força inicialização de campos necessários
            if 'last_occupancy_change' not in tdata:
                tdata['last_occupancy_change'] = current_time
            if 'pending_occupancy' not in tdata:
                tdata['pending_occupancy'] = None
                
            # Determina diretamente o novo estado com base na contagem
            if current_count == 0:
                target_state = 'VAZIA'
            elif current_count >= lugares:
                target_state = 'CHEIA'
            else:
                target_state = 'OCUPADA'
            
            # Verifica mudanças na contagem ou no estado alvo
            if (current_count != previous_count) or (tdata['pending_occupancy'] != target_state):
                # Registra o momento da mudança e atualiza estado pendente
                tdata['last_occupancy_change'] = current_time
                tdata['pending_occupancy'] = target_state
                    
                # Removendo logs frequentes de mudança de ocupação
                # São muito frequentes e poluem o console
            
            # Se está em atendimento, não muda o estado
            if tdata['precisa_atendimento']:
                continue
                
            # Aplica o estado pendente após o delay
            if tdata['pending_occupancy'] is not None:
                time_since_change = current_time - tdata['last_occupancy_change']
                
                # Quando passar o tempo de delay OU
                # Quando a mesa já tenha registrado ocupantes por frame consecutivos
                if time_since_change >= state_change_delay:
                    new_state = tdata['pending_occupancy']
                    
                    # Atualiza o estado se for diferente do atual
                    if new_state != old_state:
                        # Mantemos este log pois é importante registrar mudanças de estado
                        if self.config.debug_mode:
                            self.logger.info(f"MUDANÇA ESTADO Mesa {tid}: {old_state} -> {new_state} (ocupantes={current_count}/{lugares})")
                        
                        tdata['state'] = new_state
                        self._log_state_change(tid, old_state, new_state)
                    # Removendo logs de estado mantido (muito frequentes)
                    
                    # Não reinicia o estado pendente para permitir que atualizações futuras ocorram mais rapidamente
                    # quando a situação atual já tiver sido confirmada
                # Removendo logs de aguardando delay (muito frequentes)

    # -------------------------------------------------------
    #   STATÍSTICAS
    # -------------------------------------------------------
    def get_occupancy_stats(self):
        """
        Retorna estatísticas gerais de ocupação.
        - total_mesas: total de mesas (ativas ou standby)
        - ocupadas: quantas mesas têm occupant_count > 0
        - atendimento: quantas mesas estão em estado de atendimento
        - standby: quantas mesas estão em standby
        - occupant_sum: soma total de pessoas
        - capacity_sum: soma total de lugares
        - taxa_ocupacao: occupant_sum / capacity_sum
        """
        total_mesas = len(self.tables)
        atendimento = sum(1 for t in self.tables.values() if t['precisa_atendimento'])
        standby_count = sum(1 for t in self.tables.values() if t['state'] == 'STANDBY')

        # Mesas ativas (não standby)
        active_tables = [t for t in self.tables.values() if t['state'] != 'STANDBY']
        occupant_sum = sum(t['occupant_count'] for t in active_tables)
        capacity_sum = sum(self.config.cls_to_lugares.get(t['cls'], 0) for t in active_tables)

        ocupadas = sum(1 for t in active_tables if t['occupant_count'] > 0)
        taxa = (occupant_sum / capacity_sum) if capacity_sum > 0 else 0.0
        
        return {
            'total_mesas': total_mesas,
            'ocupadas': ocupadas,
            'atendimento': atendimento,
            'standby': standby_count,
            'occupant_sum': occupant_sum,
            'capacity_sum': capacity_sum,
            'taxa_ocupacao': taxa
        }

    def _extract_camera_number(self, camera_id):
        """Extrai apenas o número da câmera do ID (ex: 'camera11' -> '11')"""
        if isinstance(camera_id, str) and camera_id.startswith("camera"):
            return camera_id.replace("camera", "")
        return camera_id

    def build_current_states_snapshot(self, current_time: float) -> Dict:
        """Monta o snapshot das mesas para enviar ao dashboard."""
        # Obter estatísticas para incluir taxa de ocupação
        stats = self.get_occupancy_stats()
        
        # Extrair apenas o número da câmera
        camera_num = self._extract_camera_number(self.camera_id)
        
        # Cria a estrutura da câmera individual
        camera_data = {
            "tipo": "estado_atual",
            "timestamp": current_time,
            "camera_id": camera_num,
            "taxa_ocupacao": stats['taxa_ocupacao'],
            "total_lugares": stats['capacity_sum'],
            "total_pessoas": stats['occupant_sum'],
            "mesas": []
        }
    
        for table_id, tdata in sorted(self.tables.items(), key=lambda x: x[1]['creation_index']):
            estado = tdata['state']
            
            # Pula mesas em estado STANDBY
            if estado == 'STANDBY':
                continue
                
            precisa_atendimento = tdata.get('precisa_atendimento', False)
        
            tempo_restante = None
            if precisa_atendimento and tdata['ultimo_atendimento'] is not None:
                tempo_restante = self.config.tracking_params['atendimento_timeout'] - (current_time - tdata['ultimo_atendimento'])
                tempo_restante = max(0, tempo_restante)
        
            camera_data["mesas"].append({
                "mesa_id": table_id,
                "estado": estado,
                "lugares": self.config.cls_to_lugares.get(tdata['cls'], 0),
                "occupant_count": tdata['occupant_count'],
                "precisa_atendimento": precisa_atendimento,
                "tempo_restante_atendimento": tempo_restante,
            })
        
        return camera_data

    # -------------------------------------------------------
    #   DESENHO NA TELA
    # -------------------------------------------------------
    def draw_interface(self, frame, current_time, detected_maos=[], detected_people=[]):
        """
        Desenha as mesas na ordem do dict self.tables (creation_index ascendente).
        Agora também desenha as pessoas detectadas.
        """
        # Desenhar mesas
        for table_id, tdata in sorted(self.tables.items(), key=lambda x: x[1]['creation_index']):
            x1, y1, x2, y2 = tdata['coords']
            estado = tdata['state']
            color = self.config.colors.get(estado, (255, 255, 255))
            
            # Se está em atendimento, cor especial
            if tdata['precisa_atendimento']:
                color = self.config.colors['ATENDIMENTO']
                est_txt = "ATENDIMENTO"
            else:
                est_txt = estado

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{table_id}: {est_txt}", (x1+5, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Se está em atendimento, mostrar tempo restante
            if tdata['precisa_atendimento'] and tdata['ultimo_atendimento'] is not None:
                tempo_espera = current_time - tdata['ultimo_atendimento']
                tempo_restante = self.config.tracking_params['atendimento_timeout'] - tempo_espera
                texto_espera = f"Espera: {max(0, tempo_restante):.0f}s"
                cv2.putText(frame, texto_espera, (x1+5, y2-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            else:
                # Mostra occupant_count e lugares
                lugares = self.config.cls_to_lugares.get(tdata['cls'], 0)
                occ_txt = f"{tdata['occupant_count']}/{lugares} lugares"
                cv2.putText(frame, occ_txt, (x1+5, y2-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            # Mostrar classe da mesa em modo debug
            if self.config.debug_mode:
                mesa_cls = tdata['cls']
                mesa_cls_name = self.config.cls_names.get(mesa_cls, f"Cls {mesa_cls}")
                cv2.putText(frame, mesa_cls_name, (x1+5, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Variável global para armazenar snapshots das câmeras
global_camera_snapshots = {}
global_snapshot_lock = threading.Lock()

def add_camera_snapshot(camera_id, snapshot_data):
    """Adiciona o snapshot de uma câmera ao armazenamento global."""
    with global_snapshot_lock:
        global_camera_snapshots[camera_id] = snapshot_data

def get_all_camera_snapshots():
    """Retorna uma cópia de todos os snapshots de câmeras."""
    with global_snapshot_lock:
        return global_camera_snapshots.copy()

def send_combined_snapshots():
    """Envia um único snapshot combinado com todas as câmeras."""
    with global_snapshot_lock:
        if not global_camera_snapshots:
            return  # Nada para enviar
            
        # Cria a estrutura consolidada
        combined_snapshot = {
            "restaurante": "CSVL",
            "cameras": list(global_camera_snapshots.values())
        }
        
        # Envia a notificação combinada
        combined_json = json.dumps(combined_snapshot)
        
        try:
            # Obter configuração do dashboard
            config = AppConfig.load_from_json()
            headers = {'Content-Type': 'application/json'}
            
            # Log para debug
            print(f"[DASHBOARD] Enviando snapshot combinado com {len(global_camera_snapshots)} câmeras")
            
            # Usa o método PUT para enviar as notificações
            resp = requests.put(
                config.dashboard_url,
                data=combined_json,
                headers=headers,
                timeout=3
            )
            
            if resp.status_code == 200:
                print(f"[DASHBOARD] Notificação combinada enviada com sucesso")
            else:
                print(f"[DASHBOARD] Falha ao enviar notificação combinada: {resp.status_code}")
        except Exception as e:
            print(f"[DASHBOARD] Erro ao enviar notificação combinada: {str(e)}")

def get_last_used_port():
    """Lê a última porta usada do arquivo."""
    try:
        if os.path.exists(LAST_PORT_FILE):
            with open(LAST_PORT_FILE, 'r') as f:
                return f.read().strip()
    except Exception as e:
        print(f"Erro ao ler última porta usada: {e}")
    return None

def save_last_used_port(port):
    """Salva a porta usada no arquivo."""
    try:
        with open(LAST_PORT_FILE, 'w') as f:
            f.write(str(port))
    except Exception as e:
        print(f"Erro ao salvar última porta usada: {e}")

def get_dynamic_camera_url(base_url):
    """Solicita a porta ao usuário e atualiza a URL."""
    last_port = get_last_used_port()
    prompt = f"Digite a porta para a câmera RTSP"
    if last_port:
        prompt += f" (última: {last_port}, deixe em branco para usar): "
    else:
        prompt += ": "

    while True:
        user_input = input(prompt).strip()
        if not user_input:
            if last_port:
                selected_port = last_port
                print(f"Usando a última porta salva: {selected_port}")
                break
            else:
                print("Nenhuma porta anterior encontrada. Por favor, insira uma porta.")
        else:
            # Validação simples (verifica se é número)
            if user_input.isdigit():
                selected_port = user_input
                break
            else:
                print("Entrada inválida. Por favor, insira um número de porta.")

    # Tenta substituir a porta na URL base
    # Regex para encontrar @host:porta/
    match = re.search(r"(@[^:]+):(\d+)/", base_url)
    if match:
        new_url = base_url[:match.start(2)] + selected_port + base_url[match.end(2):]
        print(f"URL da câmera configurada para: {new_url}")
        save_last_used_port(selected_port) # Salva a porta que será usada
        return new_url
    else:
        print("Formato da URL inválido no config.json. Não foi possível encontrar '@host:porta/'. Usando URL original.")
        return base_url


def calculate_iou(box1, box2):
    """Calcula Intersection over Union (IoU) para dois bounding boxes."""
    # Verificar se os boxes têm formato válido
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
        
    # Verificar se as coordenadas fazem sentido
    if box1[2] <= box1[0] or box1[3] <= box1[1] or box2[2] <= box2[0] or box2[3] <= box2[1]:
        return 0.0
    
    # Coordenadas da interseção
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Área da interseção
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    intersection = width * height
    
    # Áreas dos boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Área da união
    union = box1_area + box2_area - intersection
    
    # IoU - evitar divisão por zero
    if union <= 0:
        return 0.0
        
    return intersection / union


def aplicar_mascara_regiao(frame, camera_id):
    """
    Aplica mascaramento (retângulos pretos) em regiões específicas da imagem
    para as câmeras 8, 9 e 11.
    
    Args:
        frame: Imagem a ser processada (formato OpenCV)
        camera_id: ID da câmera (para escolher máscara apropriada)
    
    Returns:
        frame: Imagem com regiões mascaradas
    """
    # Região para câmera 8 (mesmas máscaras da câmera 9)
    if camera_id == '8':
        # Região direita superior da imagem
        cv2.rectangle(frame, (int(frame.shape[1]*0.7), 0), 
                     (frame.shape[1], frame.shape[0]), 
                     (0, 0, 0), -1)
        
    # Região para câmera 9
    elif camera_id == '9':
        # Região direita superior da imagem
        cv2.rectangle(frame, (int(frame.shape[1]*0.5), 0), 
                     (frame.shape[1], frame.shape[0]), 
                     (0, 0, 0), -1)
        
        # Região superior esquerda
        cv2.rectangle(frame, (int(frame.shape[1]*0.25), 0), 
                     (int(frame.shape[1]), int(frame.shape[0]*0.5)), 
                     (0, 0, 0), -1)
    
    # Região para câmera 11
    elif camera_id == '11':
        # Lado esquerdo da imagem
        cv2.rectangle(frame, (0, 0), 
                     (int(frame.shape[1]*0.2), frame.shape[0]), 
                     (0, 0, 0), -1)
        
        cv2.rectangle(frame, (0, int(frame.shape[1]*0.8)), 
                     (int(frame.shape[1]), int(frame.shape[0])), 
                     (0, 0, 0), -1)
    
    return frame


def process_people_detections(detected_people_raw, previous_people):
    """
    Processa detecções de pessoas para:
    1. Filtrar overlaps excessivos (NMS adicional)
    2. Estabilizar detecções usando frames anteriores
    3. Melhorar persistência de pessoas detectadas
    
    Args:
        detected_people_raw: Lista de tuplas (cls_id, coords, conf)
        previous_people: Lista de tuplas (cls_id, coords) do frame anterior
        
    Returns:
        Lista filtrada de tuplas (cls_id, coords)
    """
    # Se não há detecções, verifica o histórico
    if not detected_people_raw and previous_people:
        # Mantém detecções anteriores mas com confiança reduzida
        # Isso ajuda a manter pessoas que temporariamente não foram detectadas
        print(f"Sem detecções novas, usando {len(previous_people)} detecções anteriores com menor confiança")
        return previous_people
    
    if not detected_people_raw:
        return []
    
    # Ordena por confiança (maior para menor)
    detected_people_raw.sort(key=lambda x: x[2], reverse=True)
    
    # Implementa um NMS suavizado para remover overlaps, mas com threshold mais baixo
    filtered_detections = []
    for i, (cls_id, coords, conf) in enumerate(detected_people_raw):
        should_keep = True
        
        # Verifica overlap com detecções já aceitas
        for _, accepted_coords, _ in filtered_detections:
            iou = calculate_iou(coords, accepted_coords)
            if iou > 0.4:  # Reduzido de 0.5 para 0.4 para ser mais conservador com overlaps
                should_keep = False
                break
        
        if should_keep:
            filtered_detections.append((cls_id, coords, conf))
    
    # Adiciona o mecanismo de "memória": associa com detecções anteriores e suaviza
    final_detections = []
    
    # Primeiro, adiciona as novas detecções filtradas
    for cls_id, coords, conf in filtered_detections:
        best_match = None
        best_dist = float('inf')
        
        # Procura correspondência com detecções anteriores
        for prev_cls, prev_coords in previous_people:
            if len(coords) != 4 or len(prev_coords) != 4:
                continue
            
            current_center = ((coords[0] + coords[2]) / 2, (coords[1] + coords[3]) / 2)
            prev_center = ((prev_coords[0] + prev_coords[2]) / 2, (prev_coords[1] + prev_coords[3]) / 2)
            
            dist = np.sqrt((current_center[0] - prev_center[0])**2 + 
                           (current_center[1] - prev_center[1])**2)
            
            if dist < best_dist:
                best_dist = dist
                best_match = prev_coords
        
        final_coords = coords  # por padrão, usa a detecção atual
        
        # Suaviza posição através de média ponderada com detecção anterior se próxima
        if best_match is not None and best_dist < 80:  # Aumentado de 50 para 80 pixels
            # Quanto mais próximo, mais confiança damos ao frame anterior
            alpha = min(0.8, best_dist / 80.0)  # Alpha entre 0 e 0.8 com base na distância
            final_coords = update_box(best_match, coords, alpha)
            
        final_detections.append((cls_id, final_coords))
    
    # Segundo, verifica se há detecções anteriores que não foram associadas a novas
    # e as mantém por um tempo para evitar oscilações
    prev_centers = set()
    for cls_id, coords in final_detections:
        center_x = (coords[0] + coords[2]) / 2
        center_y = (coords[1] + coords[3]) / 2
        prev_centers.add((center_x, center_y))
    
    # Adiciona pessoas do frame anterior que não foram associadas (persistência)
    for prev_cls, prev_coords in previous_people:
        prev_center_x = (prev_coords[0] + prev_coords[2]) / 2
        prev_center_y = (prev_coords[1] + prev_coords[3]) / 2
        
        # Verifica se essa pessoa já está próxima de alguma das novas detecções
        found_close = False
        for cx, cy in prev_centers:
            dist = np.sqrt((cx - prev_center_x)**2 + (cy - prev_center_y)**2)
            if dist < 100:  # Se estiver dentro de 100 pixels, considere a mesma pessoa
                found_close = True
                break
        
        # Se não encontramos ninguém próximo, mantemos a detecção anterior
        if not found_close:
            final_detections.append((prev_cls, prev_coords))
    
    return final_detections


def process_camera(camera_config, config):
    """Processa uma câmera específica em uma thread separada."""
    camera_id = camera_config.get('id', 'camera_desconhecida')
    camera_name = camera_config.get('name', f'Câmera {camera_id}')
    camera_url = camera_config.get('url', '')
    
    print(f"Iniciando processamento da {camera_name} ({camera_id})")
    
    # Validar camera_id antes de usar
    if camera_id is None or camera_id == '':
        camera_id = f"camera_{id(camera_config)}"  # Gera ID baseado no hash do objeto se não existir
        print(f"AVISO: ID da câmera não definido, usando ID gerado: {camera_id}")
    
    # Informações de log sobre IDs únicos
    print(f"Camera {camera_id}: sistema utilizando IDs de mesa únicos globais")
    
    # Criar gerenciador de mesas específico para esta câmera
    table_manager = TableManager(config, camera_id, camera_name)
    
    # Testar conectividade com o dashboard
    try:
        table_manager.logger.info(f"Testando conexão com o dashboard: {config.dashboard_url}")
        test_response = requests.get(
            config.dashboard_url, 
            timeout=5
        )
        table_manager.logger.info(f"Conexão com dashboard OK: {test_response.status_code}")
    except Exception as e:
        table_manager.logger.warning(f"Aviso: Não foi possível conectar ao dashboard: {str(e)}")
        table_manager.logger.warning(f"As notificações podem não ser entregues. Verifique config.dashboard_url")
    
    # Carrega modelo YOLO e força o uso de CUDA
    model = YOLO(config.model_path)
    model.to('cuda')
    
    # Variáveis para reconexão e monitoramento
    reconnection_attempts = 0
    max_reconnection_attempts = 5  # Número máximo de tentativas de reconexão
    capture_restart_count = 0  # Contador para estatísticas
    last_reconnection = time.time()
    last_mem_check = time.time()  # Para monitoramento de memória
    
    # Função para criar e configurar uma nova captura de câmera
    def create_camera_capture():
        cap = cv2.VideoCapture(camera_url)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduz o buffer para minimizar atrasos

        cap.set(cv2.CAP_PROP_FPS, 15)        # Limitar FPS da captura
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduzir resolução
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)   # se necessário

        return cap
    
    # Abre stream da câmera com configurações iniciais
    cap = create_camera_capture()
    
    if not cap.isOpened():
        table_manager.logger.error(f"Não foi possível abrir a câmera {camera_name} ({camera_id}). URL: {camera_url}")
        return

    # Cria uma janela específica para esta câmera
    window_name = f'Monitor - Camera {camera_id}'  # Nome simplificado sem caracteres especiais
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    last_stats_time = time.time()
    last_second_notification = time.time()
    fail_count = 0
    max_fail = 10  # Reduzido para 10 tentativas antes de reconectar
    
    # Estrutura para armazenar pessoas detectadas anteriormente
    previous_people = []
    
    # Contagem de classes para debug
    class_counter = defaultdict(int)
    last_counter_reset = time.time()

    try:
        while True:
            current_time = time.time()
            
            # Monitoramento de memória a cada minuto
            if current_time - last_mem_check > 60:
                proc = psutil.Process()
                mem_info = proc.memory_info()
                table_manager.logger.info(f"Uso de memória ({camera_name}): {mem_info.rss / 1024 / 1024:.1f} MB")
                last_mem_check = current_time
            
            # Removido: Limpeza periódica dos IDs globais (a cada 10 minutos)
            
            ret, frame = cap.read()
            if ret:
                # Extrai apenas o número da câmera para aplicar a máscara correta
                camera_numero = camera_id
                if isinstance(camera_id, str) and camera_id.startswith("camera"):
                    camera_numero = camera_id.replace("camera", "")
                
                # Aplica máscara em regiões específicas para câmeras 8, 9 e 11
                if camera_numero in ['8', '9', '11']:
                    frame = aplicar_mascara_regiao(frame, camera_numero)
                    
            if not ret:
                fail_count += 1
                # Reduzimos a frequência deste log para não poluir o console
                if fail_count == 1 or fail_count % 3 == 0:  # Mostrar apenas a primeira falha e depois a cada 3
                    table_manager.logger.warning(f"Falha ao ler frame da {camera_name} ({fail_count}/{max_fail})")
                time.sleep(1)
                
                if fail_count >= max_fail:
                    table_manager.logger.error(f"Falhas repetidas na {camera_name}. Tentando reconexão robusta...")
                    capture_restart_count += 1
                    
                    # Libera completamente os recursos
                    cap.release()
                    cv2.destroyWindow(window_name)  # Fecha a janela para liberar recursos
                    time.sleep(3)  # Espera maior para liberar recursos de rede
                    
                    # Recria a janela
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, 1280, 720)
                    
                    # Tenta diferentes configurações de reconexão
                    reconnect_success = False
                    for attempt in range(2):  # Reduzido para 2 tentativas
                        try:
                            table_manager.logger.info(f"Tentativa de reconexão {attempt+1}/2 para {camera_name}")
                            
                            # Cria um novo objeto VideoCapture com diferentes configurações
                            if attempt == 0:
                                # Primeira tentativa: configuração padrão
                                cap = cv2.VideoCapture(camera_url)
                                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            else:
                                # Segunda tentativa: timeout maior
                                cap = cv2.VideoCapture(camera_url)
                                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 segundos
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            
                            # Testa a conexão
                            test_ret = cap.grab()
                            if test_ret:
                                table_manager.logger.info(f"Reconexão bem-sucedida para {camera_name} (tentativa {attempt+1})")
                                reconnect_success = True
                                break
                            else:
                                cap.release()
                                time.sleep(2)
                        except Exception as e:
                            table_manager.logger.error(f"Erro na tentativa {attempt+1} de reconexão: {str(e)}")
                            time.sleep(2)
                    
                    if not reconnect_success:
                        table_manager.logger.error(f"Todas as tentativas de reconexão falharam para {camera_name}. Tentando novamente em 30 segundos...")
                        time.sleep(30)
                        cap = create_camera_capture()
                    
                    table_manager.logger.info(f"Reiniciando processamento da {camera_name} (reinício #{capture_restart_count})")
                    fail_count = 0
                    last_reconnection = time.time()
                    continue
                continue
            fail_count = 0

            # Zera contador de classes a cada 5 segundos
            if current_time - last_counter_reset > 5:
                class_counter = defaultdict(int)
                last_counter_reset = current_time

            # Inferência com CUDA
            results = model.predict(
                frame,
                conf=config.model_conf,
                iou=config.model_iou,
                agnostic_nms=config.model_agnostic_nms,
                verbose=False,
                device='cuda'
            )
            dets = results[0]

            detected_boxes = []  # mesas
            detected_maos = []
            detected_people_raw = []  # detecções brutas

            for box in dets.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = tuple(map(int, box.xyxy[0].tolist()))
                
                # Incrementa contador de classes (para debug)
                class_counter[cls_id] += 1

                if cls_id == 7:  # mão
                    if conf < config.model_mao_conf:
                        continue
                    detected_maos.append((cls_id, coords))
                elif cls_id == config.people_cls_id:  # pessoa
                    if conf < config.model_pessoa_conf:
                        continue
                    detected_people_raw.append((cls_id, coords, conf))
                else:
                    # Classes de mesa (0..6)
                    detected_boxes.append((cls_id, coords))
            
            # Processa e filtra detecções de pessoas
            detected_people = process_people_detections(detected_people_raw, previous_people)
            previous_people = [(cls, coords) for cls, coords, _ in detected_people_raw]

            # Se não há mesas ainda, inicializa
            if not table_manager.tables:
                table_manager.initialize_tables(detected_boxes, current_time)
            else:
                table_manager.update_states(detected_boxes, current_time)

            # Atribuir pessoas às mesas
            table_manager.assign_people_to_tables(detected_people)

            # Processa mãos levantadas
            table_manager.process_maos_levantadas(detected_maos, current_time)
            
            # Desenha interface
            table_manager.draw_interface(frame, current_time, detected_maos, detected_people)
            
            # Adiciona identificação da câmera na imagem
            cv2.putText(
                frame, 
                f"{camera_name}", 
                (10, frame.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
            
            # Mostrar contagem de classes em modo debug
            if config.debug_mode:
                debug_y = 60
                for cls_id, count in sorted(class_counter.items()):
                    cls_name = config.cls_names.get(cls_id, f"Classe {cls_id}")
                    debug_text = f"{cls_name}: {count}"
                    cv2.putText(frame, debug_text, (10, debug_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    debug_y += 20
            
            cv2.imshow(window_name, frame)

            # Notificação de estado a cada 1s
            if (current_time - last_second_notification) >= 1:
                camera_data = table_manager.build_current_states_snapshot(current_time)
                
                # Removendo logs de debug de snapshot que são muito frequentes
                # Log das mesas no snapshot
                # if config.debug_mode:
                #     table_manager.logger.debug(f"Adicionando snapshot da {camera_name} com {len(camera_data['mesas'])} mesas:")
                #     for mesa in camera_data['mesas']:
                #         table_manager.logger.debug(f"  Mesa {mesa['mesa_id']}: estado={mesa['estado']}, ocupantes={mesa['occupant_count']}/{mesa['lugares']}")
                
                # Adiciona o snapshot desta câmera ao armazenamento global
                add_camera_snapshot(camera_data['camera_id'], camera_data)
                
                # Registra timestamp da última notificação enviada
                last_second_notification = current_time

            # Log estatísticas a cada 300s
            if (current_time - last_stats_time) > 300:
                stats = table_manager.get_occupancy_stats()
                table_manager.logger.info(
                    f"Estatísticas {camera_name} | Mesas: {stats['total_mesas']} | "
                    f"Ocupadas (mesas): {stats['ocupadas']} | "
                    f"Atendimento: {stats['atendimento']} | "
                    f"Standby: {stats['standby']} | "
                    f"Pessoas: {stats['occupant_sum']}/{stats['capacity_sum']} | "
                    f"Taxa: {stats['taxa_ocupacao']:.0%}"
                )
                last_stats_time = current_time

            # Processar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif chr(key) in toggle_keys:
                # Alternar configuração booleana correspondente
                attr_name = toggle_keys[chr(key)]
                setattr(config, attr_name, not getattr(config, attr_name))
                print(f"Alternando {attr_name} para {getattr(config, attr_name)}")

    except Exception as e:
        table_manager.logger.error(f"Erro no loop principal da {camera_name}: {str(e)}")
        import traceback
        table_manager.logger.error(traceback.format_exc())
    finally:
        cap.release()
        cv2.destroyWindow(window_name)
        table_manager.logger.info(f"Processamento da {camera_name} encerrado")


def update_camera_ports():
    """Solicita a porta ao usuário e atualiza todas as URLs das câmeras no config.json."""
    last_port = get_last_used_port()
    prompt = f"Digite a porta para todas as câmeras RTSP"
    if last_port:
        prompt += f" (última: {last_port}, aperte Enter para usar): "
    else:
        prompt += ": "

    while True:
        user_input = input(prompt).strip()
        if not user_input:
            if last_port:
                selected_port = last_port
                print(f"Usando a última porta salva: {selected_port}")
                break
            else:
                print("Nenhuma porta anterior encontrada. Por favor, insira uma porta.")
        else:
            # Validação simples (verifica se é número)
            if user_input.isdigit():
                selected_port = user_input
                break
            else:
                print("Entrada inválida. Por favor, insira um número de porta.")
    
    # Salva a porta selecionada para uso futuro
    save_last_used_port(selected_port)
    
    # Atualiza o arquivo config.json
    try:
        # Carrega o arquivo atual
        with open('config.json', 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Atualiza as URLs das câmeras
        for camera in config_data['cameras']:
            url = camera['url']
            # Regex para encontrar @host:porta/
            match = re.search(r"(@[^:]+):(\d+)/", url)
            if match:
                new_url = url[:match.start(2)] + selected_port + url[match.end(2):]
                camera['url'] = new_url
                print(f"URL da {camera.get('name', camera.get('id', 'Câmera'))} atualizada para: {new_url}")
            else:
                print(f"Formato da URL inválido para {camera.get('name', camera.get('id', 'Câmera'))}. Não foi possível atualizar.")
        
        # Salva as alterações
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        
        print("Arquivo config.json atualizado com sucesso!")
        return selected_port
    except Exception as e:
        print(f"Erro ao atualizar config.json: {str(e)}")
        return selected_port


def main():
    # Atualiza as portas das câmeras
    update_camera_ports()
    
    # Carregar configurações do arquivo JSON (após atualização das portas)
    config = AppConfig.load_from_json()
    
    # Verificar se temos câmeras configuradas
    if not config.cameras:
        print("ERRO: Nenhuma câmera configurada no arquivo config.json")
        return
    
    # Definir teclas para controle
    global toggle_keys
    toggle_keys = {
        'p': 'show_people_boxes',   # 'p' para ligar/desligar visualização de pessoas
        'l': 'show_association_lines',  # 'l' para ligar/desligar linhas de associação
        'd': 'debug_mode'  # 'd' para ligar/desligar modo debug
    }
    
    # Iniciar thread para envio combinado de snapshots
    def snapshot_sender():
        while True:
            send_combined_snapshots()
            time.sleep(1)  # Envia atualizações combinadas a cada 1 segundo
    
    snapshot_thread = threading.Thread(target=snapshot_sender, daemon=True)
    snapshot_thread.start()
    
    # Thread para monitorar a distribuição de IDs
    def id_distribution_monitor():
        while True:
            print_table_id_distribution()
            time.sleep(60)  # Mostra distribuição a cada 60 segundos
    
    id_monitor_thread = threading.Thread(target=id_distribution_monitor, daemon=True)
    id_monitor_thread.start()
    
    # Criar thread para cada câmera
    threads = []
    for camera_config in config.cameras:
        thread = threading.Thread(
            target=process_camera,
            args=(camera_config, config),
            daemon=True
        )
        threads.append(thread)
        thread.start()
        # Pequeno delay para não iniciar todas as câmeras simultaneamente
        time.sleep(1)
    
    # Aguardar todas as threads
    for thread in threads:
        thread.join()

    print("Sistema encerrado")


if __name__ == "__main__":
    main()
