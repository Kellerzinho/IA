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
            
            # Camera
            if 'camera' in data and 'url' in data['camera']:
                config.camera_url = data['camera']['url']
            
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
    def __init__(self, config: AppConfig):
        self.config = config
        
        self.tables: Dict[int, Dict] = {}  # ID da mesa -> dados da mesa
        self.pending_new_tables: Dict[Tuple[int, Tuple], float] = {}
        self.maos_detectadas: Dict[Tuple[int, Tuple], float] = {}
        self.state_history: Dict[int, List[Tuple[float, str]]] = defaultdict(list)
        self.mesas_notificadas = set()
        self.next_creation_index = 1
        
        self.setup_logger()

    def setup_logger(self):
        self.logger = logging.getLogger('TableTracker')
        # Muda o nível para DEBUG para capturar mensagens de depuração
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setStream(open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1))

        # Arquivo específico para logs de debug com dashboard
        dh = logging.FileHandler('dashboard_debug.log', encoding='utf-8')
        dh.setFormatter(formatter)
        dh.setLevel(logging.DEBUG)

        fh = logging.FileHandler('table_tracker.log', encoding='utf-8')
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)  # O arquivo principal continua com INFO

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
            json.loads(mensagem)
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
            self.tables[i] = {
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
                self.logger.info(f"Mesa {i} criada: tipo={cls} ({self.config.cls_names.get(cls, 'Desconhecida')}), lugares={self.config.cls_to_lugares.get(cls, 0)}")
        
        self.mesas_notificadas.clear()

    def update_states(self, detected_boxes: List[Tuple[int, Tuple]], current_time: float):
        """Faz a "two-pass merge" (tracking manual) para as mesas."""
        old_tables = dict(self.tables)
        updated_tables = {}
        used_ids = set()

        # ---------- PASSO 1: Identifica possíveis merges ----------
        merges = {}
        for det_cls, det_coords in detected_boxes:
            matching_ids = [
                tid for tid, tdata in old_tables.items()
                if is_near(tdata['coords'], det_coords, self.config.tracking_params['distance_threshold'])
            ]

            if not matching_ids:
                # Se não achamos mesa ativa, vira "pending"
                self.pending_new_tables[(det_cls, det_coords)] = current_time
            else:
                # Se tem match, escolhe ID MAIOR
                chosen_id = max(matching_ids)
                if chosen_id not in merges:
                    merges[chosen_id] = set()
                for mid in matching_ids:
                    merges[chosen_id].add(mid)
                used_ids.update(matching_ids)

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
            too_close = any(
                is_near(mesa['coords'], det_coords, self.config.tracking_params['distance_threshold'])
                for mesa in self.tables.values()
            )
            # Se não está perto de nenhuma mesa e já passou confirm_time
            if not too_close and (current_time - detected_time >= self.config.tracking_params['confirm_time']):
                new_id = max([*self.tables.keys(), *updated_tables.keys()], default=0) + 1
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
                    # Garantir que os campos de controle de estado estejam presentes
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

        updated_tables[table_id] = table_data

    def _log_state_change(self, table_id, old_state, new_state):
        """Faz log e envia notificação quando a mesa muda de estado."""
        self.logger.info(f"Mesa {table_id} mudou de {old_state} para {new_state}")
        
        # Se ficou VAZIA e antes era CHEIA ou OCUPADA, notificar "mesa liberada"
        if new_state == 'VAZIA' and old_state in ['CHEIA', 'OCUPADA']:
            lugares = self.config.cls_to_lugares.get(self.tables[table_id]['cls'], 0)
            msg_terminal = f"LIBEROU {lugares} LUGARES - Mesa {table_id}"
            self.logger.info(msg_terminal)
            
            # Enviar para o dashboard
            msg_dashboard = {
                "tipo": "mesa_liberada",
                "mesa_id": table_id,
                "lugares": lugares,
                "timestamp": time.time()
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
        """Remove mesas STANDBY cujo ID seja maior que o maior ID ativo, e reordena."""
        active_tables = [tid for tid, tdata in self.tables.items() 
                         if tdata['state'] != 'STANDBY']
        
        if not active_tables:
            return
        
        max_active_id = max(active_tables)
        to_remove = [
            tid for tid, tdata in self.tables.items()
            if tdata['state'] == 'STANDBY' and tid > max_active_id
        ]
        
        for tid in to_remove:
            del self.tables[tid]
            self.logger.info(f"Removida mesa STANDBY {tid} (ID > {max_active_id})")

        # Reordenar após a remoção:
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
                # Só processa mesas CHEIAs ou OCUPADAs
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
                        msg_terminal = f"ATENDIMENTO: Mesa {melhor_id}"
                        self.logger.info(msg_terminal)
                        
                        msg_dashboard = {
                            "tipo": "atendimento_requisitado",
                            "mesa_id": melhor_id,
                            "timestamp": current_time,
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
            
            if self.config.debug_mode:
                self.logger.info(f"Processando pessoa: centro=({(p_coords[0] + p_coords[2])/2:.1f}, {(p_coords[1] + p_coords[3])/2:.1f})")
                
            for tid, tdata in self.tables.items():
                # Se mesa está em STANDBY, ignore
                if tdata['state'] == 'STANDBY':
                    continue
                    
                d = dist_person_table(p_coords, tdata['coords'])
                
                if self.config.debug_mode:
                    self.logger.info(f"  Distância para Mesa {tid}: {d:.1f} pixels")
                    
                if d < menor_dist:
                    menor_dist = d
                    melhor_id = tid
            
            # Usa o parâmetro de configuração para a distância máxima
            max_dist = self.config.tracking_params.get('pessoa_to_table_max_dist', 200)
            
            if melhor_id is not None and menor_dist < max_dist:
                self.tables[melhor_id]['occupant_count'] += 1
                person_table_associations.append((p_coords, self.tables[melhor_id]['coords'], melhor_id))
                
                if self.config.debug_mode:
                    self.logger.info(f"  ASSOCIADA à Mesa {melhor_id} (distância: {menor_dist:.1f}px)")
                
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
                
                # Registra no log a mudança de ocupação
                if self.config.debug_mode:
                    self.logger.info(f"Mesa {tid}: Mudança de ocupação {previous_count} -> {current_count} (alvo: {target_state})")
            
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
                        if self.config.debug_mode:
                            self.logger.info(f"MUDANÇA ESTADO Mesa {tid}: {old_state} -> {new_state} (delay={time_since_change:.1f}s, ocupantes={current_count}/{lugares})")
                        
                        tdata['state'] = new_state
                        self._log_state_change(tid, old_state, new_state)
                    else:
                        if self.config.debug_mode:
                            self.logger.info(f"Estado mantido Mesa {tid}: {old_state} (ocupantes={current_count}/{lugares})")
                    
                    # Não reinicia o estado pendente para permitir que atualizações futuras ocorram mais rapidamente
                    # quando a situação atual já tiver sido confirmada
                elif self.config.debug_mode and time_since_change > 0:
                    self.logger.info(f"Aguardando delay Mesa {tid}: pendente={tdata['pending_occupancy']}, atual={old_state}, tempo={time_since_change:.1f}/{state_change_delay}s")

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

    def build_current_states_snapshot(self, current_time: float) -> Dict:
        """Monta o snapshot das mesas para enviar ao dashboard."""
        snapshot = {
            "tipo": "estado_atual",
            "timestamp": current_time,
            "mesas": []
        }
    
        for table_id, tdata in sorted(self.tables.items(), key=lambda x: x[1]['creation_index']):
            estado = tdata['state']
            precisa_atendimento = tdata.get('precisa_atendimento', False)
        
            tempo_restante = None
            if precisa_atendimento and tdata['ultimo_atendimento'] is not None:
                tempo_restante = self.config.tracking_params['atendimento_timeout'] - (current_time - tdata['ultimo_atendimento'])
                tempo_restante = max(0, tempo_restante)
        
            snapshot["mesas"].append({
                "mesa_id": table_id,
                "estado": estado,
                "lugares": self.config.cls_to_lugares.get(tdata['cls'], 0),
                "occupant_count": tdata['occupant_count'],
                "precisa_atendimento": precisa_atendimento,
                "tempo_restante_atendimento": tempo_restante,
            })
    
        return snapshot

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

        # Desenhar pessoas detectadas (se habilitado)
        if self.config.show_people_boxes:
            for p_cls, p_coords in detected_people:
                x1, y1, x2, y2 = p_coords
                color = self.config.colors['PESSOA']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Texto básico ou detalhado dependendo do modo
                if self.config.debug_mode:
                    pessoa_cls_name = self.config.cls_names.get(p_cls, f"Cls {p_cls}")
                    cv2.putText(frame, f"{pessoa_cls_name} ({p_cls})", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    cv2.putText(frame, "PESSOA", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Desenhar linhas de associação entre pessoas e mesas (se habilitado)
        if self.config.show_association_lines and hasattr(self, 'person_table_associations'):
            for person_box, table_box, table_id in self.person_table_associations:
                # Usa a parte inferior central da pessoa (pés)
                px = (person_box[0] + person_box[2]) // 2  # x central
                py = person_box[3]  # y mais baixo (pés da pessoa)
                
                # Centro da mesa permanece o mesmo
                tx = (table_box[0] + table_box[2]) // 2
                ty = (table_box[1] + table_box[3]) // 2
                
                # Desenha a linha de associação
                cv2.line(frame, (px, py), (tx, ty), (0, 255, 255), 1)

        # Desenhar mãos detectadas
        for mao_cls, mao_coords in detected_maos:
            x1, y1, x2, y2 = mao_coords
            color = self.config.colors['MAO']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if self.config.debug_mode:
                mao_cls_name = self.config.cls_names.get(mao_cls, f"Cls {mao_cls}")
                cv2.putText(frame, f"{mao_cls_name} ({mao_cls})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                cv2.putText(frame, "MAO", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mostra estatísticas
        stats = self.get_occupancy_stats()
        stxt = (
            f"Mesas: {stats['total_mesas']} | "
            f"Ocupadas: {stats['ocupadas']} | "
            f"Atendimento: {stats['atendimento']} | "
            f"Standby: {stats['standby']} | "
            f"Pessoas: {stats['occupant_sum']}/{stats['capacity_sum']} | "
            f"Taxa: {stats['taxa_ocupacao']:.0%}"
        )
        cv2.putText(frame, stxt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    
        # Mostrar informações adicionais em modo debug
        if self.config.debug_mode:
            debug_info = (
                f"Modo: BOUNDING BOX | "
                f"Pessoas detectadas: {len(detected_people)} | "
                f"Maos detectadas: {len(detected_maos)} | "
                f"Modelo: {self.config.model_path}"
            )
            cv2.putText(frame, debug_info, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


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


def main():
    # Carregar configurações do arquivo JSON
    config = AppConfig.load_from_json()
    table_manager = TableManager(config)
    
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
    
    # Abre stream da câmera com configurações RTSP/TCP
    cap = cv2.VideoCapture(config.camera_url)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    
    if not cap.isOpened():
        print("Não foi possível abrir a câmera RTSP. Verifique config.camera_url.")
        return

    cv2.namedWindow('Monitor de Mesas', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Monitor de Mesas', 1920, 1080)

    last_stats_time = time.time()
    last_second_notification = time.time()
    fail_count = 0
    max_fail = 20  # Tolerância a falhas consecutivas
    
    # Estrutura para armazenar pessoas detectadas anteriormente
    previous_people = []
    
    # Definir teclas para controle
    toggle_keys = {
        'p': 'show_people_boxes',   # 'p' para ligar/desligar visualização de pessoas
        'l': 'show_association_lines',  # 'l' para ligar/desligar linhas de associação
        'd': 'debug_mode'  # 'd' para ligar/desligar modo debug
    }
    
    # Contagem de classes para debug
    class_counter = defaultdict(int)
    last_counter_reset = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                print("Falha ao ler frame, tentando novamente...")
                time.sleep(1)
                if fail_count >= max_fail:
                    print("Falha repetida em ler frames. Encerrando...")
                    break
                continue
            fail_count = 0

            current_time = time.time()

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
            
            # Mostrar contagem de classes em modo debug
            if config.debug_mode:
                debug_y = 60
                for cls_id, count in sorted(class_counter.items()):
                    cls_name = config.cls_names.get(cls_id, f"Classe {cls_id}")
                    debug_text = f"{cls_name}: {count}"
                    cv2.putText(frame, debug_text, (10, debug_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    debug_y += 20
            
            cv2.imshow('Monitor de Mesas', frame)

            # Notificação de estado a cada 1s
            if (current_time - last_second_notification) >= 1:
                snapshot = table_manager.build_current_states_snapshot(current_time)
                
                # Log das mesas no snapshot
                if config.debug_mode:
                    table_manager.logger.debug(f"Enviando snapshot com {len(snapshot['mesas'])} mesas:")
                    for mesa in snapshot['mesas']:
                        table_manager.logger.debug(f"  Mesa {mesa['mesa_id']}: estado={mesa['estado']}, ocupantes={mesa['occupant_count']}/{mesa['lugares']}")
                
                # Enviar notificação do snapshot atual
                snapshot_json = json.dumps(snapshot)
                table_manager.notificar_dashboard(snapshot_json)
                
                # Registra timestamp da última notificação enviada
                last_second_notification = current_time

            # Log estatísticas a cada 300s
            if (current_time - last_stats_time) > 300:
                stats = table_manager.get_occupancy_stats()
                table_manager.logger.info(
                    f"Estatísticas | Mesas: {stats['total_mesas']} | "
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
        table_manager.logger.error(f"Erro no loop principal: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        table_manager.logger.info("Sistema encerrado")


if __name__ == "__main__":
    main()
