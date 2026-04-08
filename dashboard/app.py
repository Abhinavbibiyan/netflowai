'''import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify
import pandas as pd
import json

app = Flask(__name__)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summary')
def summary():
    df = pd.read_csv(os.path.join(DATA_DIR, 'flows_scored.csv'))
    total = len(df)
    detected = int(df['predicted_anomaly'].sum())
    actual = int(df['is_anomaly'].sum())
    tp = int(((df['predicted_anomaly'] == 1) & (df['is_anomaly'] == 1)).sum())
    precision = round(tp / detected, 3) if detected > 0 else 0
    recall = round(tp / actual, 3) if actual > 0 else 0
    return jsonify({
        'total_flows': total,
        'detected_anomalies': detected,
        'actual_anomalies': actual,
        'precision': precision,
        'recall': recall
    })

@app.route('/api/alerts')
def alerts():
    with open(os.path.join(DATA_DIR, 'alerts.json')) as f:
        data = json.load(f)
    return jsonify(data)

@app.route('/api/timeline')
def timeline():
    df = pd.read_csv(os.path.join(DATA_DIR, 'flows_scored.csv'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['minute'] = df['timestamp'].dt.floor('min').astype(str)
    timeline = df.groupby('minute').agg(
        total=('anomaly_score', 'count'),
        anomalies=('predicted_anomaly', 'sum'),
        avg_score=('anomaly_score', 'mean')
    ).reset_index()
    return jsonify(timeline.to_dict(orient='records'))

@app.route('/api/protocol_stats')
def protocol_stats():
    df = pd.read_csv(os.path.join(DATA_DIR, 'flows_scored.csv'))
    stats = df.groupby('protocol').agg(
        total=('anomaly_score', 'count'),
        anomalies=('predicted_anomaly', 'sum')
    ).reset_index()
    return jsonify(stats.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)'''
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
import pickle
import json
import shap
import numpy as np
import torch
import torch.nn as nn
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from scapy.all import sniff, IP, TCP, UDP, conf

conf.verb = 0

app = Flask(__name__)
app.config['SECRET_KEY'] = 'netflowai-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
FEATURES = ['tcp_flags', 'packets', 'bytes', 'duration',
            'ttl', 'udp_tcp_ratio', 'bytes_per_packet']

# ── in-memory state ──────────────────────────────────────────────
alerts        = []          # list of alert dicts pushed to browser
all_flows     = []          # every scored flow
flow_buffer   = {}          # src_ip+dst_ip+proto → accumulated packets
ALERT_LOCK    = threading.Lock()
WINDOW_SECS   = 5           # aggregate packets into 5-second flows

# ── load model ───────────────────────────────────────────────────
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),  nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),  nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

model     = Autoencoder(len(FEATURES))
model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'model.pt'), map_location='cpu'))
model.eval()

with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(DATA_DIR, 'threshold.txt')) as f:
    THRESHOLD = float(f.read().strip())

# background for SHAP (50 zero-vectors is fast and good enough for live use)
background_np = np.zeros((50, len(FEATURES)), dtype=np.float32)

def model_error_fn(x_np):
    t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        recon = model(t)
        return ((t - recon) ** 2).mean(dim=1).numpy()

explainer = shap.KernelExplainer(model_error_fn, background_np)

# ── scoring ──────────────────────────────────────────────────────
def score_flow(flow_dict):
    try:
        row = np.array([[
            flow_dict['tcp_flags'],
            flow_dict['packets'],
            flow_dict['bytes'],
            flow_dict['duration'],
            flow_dict['ttl'],
            flow_dict['udp_tcp_ratio'],
            flow_dict['bytes_per_packet'],
        ]], dtype=np.float32)

        row_scaled = scaler.transform(row)
        t = torch.tensor(row_scaled, dtype=torch.float32)
        with torch.no_grad():
            recon = model(t)
            score = float(((t - recon) ** 2).mean().item())

        is_anomaly = score > THRESHOLD

        shap_vals = explainer.shap_values(row_scaled, nsamples=30, silent=True)
        shap_dict = {FEATURES[i]: round(float(shap_vals[0][i]), 4)
                     for i in range(len(FEATURES))}

        result = {
            'timestamp'    : time.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip'       : flow_dict['src_ip'],
            'dst_ip'       : flow_dict['dst_ip'],
            'protocol'     : flow_dict['protocol'],
            'packets'      : flow_dict['packets'],
            'bytes'        : flow_dict['bytes'],
            'anomaly_score': round(score, 6),
            'is_anomaly'   : is_anomaly,
            'shap'         : shap_dict,
        }

        with ALERT_LOCK:
            all_flows.append(result)
            if is_anomaly:
                alerts.append(result)
                if len(alerts) > 200:
                    alerts.pop(0)

        socketio.emit('new_flow', result)
        if is_anomaly:
            socketio.emit('new_alert', result)

    except Exception as e:
        print(f'[score_flow error] {e}')

# ── flow aggregator (flushes every WINDOW_SECS seconds) ──────────
def flush_flows():
    while True:
        time.sleep(WINDOW_SECS)
        with ALERT_LOCK:
            snapshot = dict(flow_buffer)
            flow_buffer.clear()

        for key, fd in snapshot.items():
            if fd['packets'] > 0:
                fd['duration']        = WINDOW_SECS
                fd['bytes_per_packet'] = fd['bytes'] / fd['packets']
                threading.Thread(target=score_flow, args=(fd,), daemon=True).start()

# ── scapy packet handler ──────────────────────────────────────────
def handle_packet(pkt):
    if not pkt.haslayer(IP):
        return

    src = pkt[IP].src
    dst = pkt[IP].dst
    ttl = pkt[IP].ttl

    if pkt.haslayer(TCP):
        proto     = 'TCP'
        tcp_flags = int(pkt[TCP].flags)
        is_udp    = 0
    elif pkt.haslayer(UDP):
        proto     = 'UDP'
        tcp_flags = 0
        is_udp    = 1
    else:
        proto     = 'ICMP'
        tcp_flags = 0
        is_udp    = 0

    pkt_len = len(pkt)
    key     = f'{src}|{dst}|{proto}'

    with ALERT_LOCK:
        if key not in flow_buffer:
            flow_buffer[key] = {
                'src_ip'       : src,
                'dst_ip'       : dst,
                'protocol'     : proto,
                'tcp_flags'    : tcp_flags,
                'packets'      : 0,
                'bytes'        : 0,
                'duration'     : 0,
                'ttl'          : ttl,
                'udp_tcp_ratio': is_udp,
                'bytes_per_packet': 0,
            }
        flow_buffer[key]['packets'] += 1
        flow_buffer[key]['bytes']   += pkt_len

# ── start sniffer in background ───────────────────────────────────
def start_sniffer():
    print('[sniffer] starting — capturing all interfaces...')
    sniff(prn=handle_packet, store=False, filter='ip')

# ── Flask routes ──────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/alerts')
def get_alerts():
    with ALERT_LOCK:
        return jsonify(list(reversed(alerts[-50:])))

@app.route('/api/summary')
def get_summary():
    with ALERT_LOCK:
        total    = len(all_flows)
        detected = len(alerts)
    return jsonify({
        'total_flows'       : total,
        'detected_anomalies': detected,
        'threshold'         : round(THRESHOLD, 6),
    })

@app.route('/api/threshold')
def get_threshold():
    return jsonify({'threshold': round(THRESHOLD, 6)})

# ── main ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    threading.Thread(target=start_sniffer, daemon=True).start()
    threading.Thread(target=flush_flows,   daemon=True).start()
    print(f'[app] threshold = {THRESHOLD:.6f}')
    print('[app] dashboard → http://localhost:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
