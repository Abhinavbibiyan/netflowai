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
import socket as sock

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from scapy.all import sniff, IP, TCP, UDP, conf

conf.verb = 0

app = Flask(__name__)
app.config['SECRET_KEY'] = 'netflowai-secret'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
FEATURES = ['tcp_flags', 'packets', 'bytes', 'duration',
            'ttl', 'udp_tcp_ratio', 'bytes_per_packet']

alerts      = []
all_flows   = []
flow_buffer = {}
dns_cache   = {}
ALERT_LOCK  = threading.Lock()
WINDOW_SECS = 5

# ── model ────────────────────────────────────────────────────────
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

model = Autoencoder(len(FEATURES))
model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'model.pt'), map_location='cpu'))
model.eval()

with open(os.path.join(DATA_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(DATA_DIR, 'threshold.txt')) as f:
    THRESHOLD = float(f.read().strip())

background_np = np.zeros((50, len(FEATURES)), dtype=np.float32)

def model_error_fn(x_np):
    t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        recon = model(t)
        return ((t - recon) ** 2).mean(dim=1).numpy()

explainer = shap.KernelExplainer(model_error_fn, background_np)

# ── helpers ───────────────────────────────────────────────────────
def resolve_hostname(ip):
    if ip in dns_cache:
        return dns_cache[ip]
    try:
        host = sock.gethostbyaddr(ip)[0]
    except Exception:
        host = ip
    dns_cache[ip] = host
    return host

def classify_threat(flow, shap_dict):
    proto      = flow['protocol']
    pkts       = flow['packets']
    bpp        = flow['bytes_per_packet']
    tcp_flags  = flow['tcp_flags']
    dst_ip     = flow['dst_ip']
    udp_ratio  = flow['udp_tcp_ratio']
    score      = flow['anomaly_score']

    top_feature = max(shap_dict, key=lambda k: abs(shap_dict[k]))

    # multicast / broadcast
    if dst_ip.startswith('239.') or dst_ip.startswith('224.') or dst_ip == '255.255.255.255':
        return {
            'threat_type': 'Multicast / Broadcast',
            'severity'   : 'Low',
            'reason'     : f'Traffic sent to multicast/broadcast address {dst_ip}. '
                           f'Common for mDNS, SSDP, or network discovery. Usually benign '
                           f'but high volume may indicate a network scanner.',
            'action'     : 'Monitor volume. Block if unexpected.'
        }

    # UDP flood
    if proto == 'UDP' and pkts > 500:
        return {
            'threat_type': 'UDP Flood (DoS)',
            'severity'   : 'Critical',
            'reason'     : f'Extremely high UDP packet count ({pkts} pkts) in a {WINDOW_SECS}s window. '
                           f'Bytes per packet: {bpp:.0f}. '
                           f'This pattern matches a volumetric denial-of-service attack.',
            'action'     : 'Block source IP immediately.'
        }

    # port scan (SYN only, tiny packets)
    if proto == 'TCP' and tcp_flags == 2 and bpp < 100:
        return {
            'threat_type': 'Port Scan (SYN)',
            'severity'   : 'High',
            'reason'     : f'TCP SYN packets with no ACK/data (flags=2, {bpp:.0f} bytes/pkt). '
                           f'Classic SYN scan — attacker probing for open ports.',
            'action'     : 'Block source IP. Check what ports were targeted.'
        }

    # data exfiltration (large outbound TCP flows)
    if proto == 'TCP' and bpp > 1000 and flow['bytes'] > 100000:
        return {
            'threat_type': 'Possible Data Exfiltration',
            'severity'   : 'High',
            'reason'     : f'Large TCP flow: {flow["bytes"]:,} bytes at {bpp:.0f} bytes/pkt '
                           f'over {WINDOW_SECS}s to {dst_ip}. '
                           f'High byte volume to external host may indicate data theft.',
            'action'     : 'Inspect destination. Check if host is known/trusted.'
        }

    # suspicious UDP (not multicast, not flood)
    if proto == 'UDP' and udp_ratio == 1.0:
        return {
            'threat_type': 'Suspicious UDP Traffic',
            'severity'   : 'Medium',
            'reason'     : f'Anomalous UDP flow to {dst_ip}. '
                           f'Top SHAP driver: {top_feature} (score contribution: {shap_dict[top_feature]:+.3f}). '
                           f'Could be DNS tunneling, C2 beaconing, or unusual application traffic.',
            'action'     : 'Verify destination. Check if UDP is expected on this host.'
        }

    # generic high score
    severity = 'Critical' if score > THRESHOLD * 3 else 'High' if score > THRESHOLD * 2 else 'Medium'
    return {
        'threat_type': 'Anomalous Flow',
        'severity'   : severity,
        'reason'     : f'Reconstruction error {score:.4f} exceeds threshold {THRESHOLD:.4f} '
                       f'(×{score/THRESHOLD:.1f}). '
                       f'Top anomaly driver: {top_feature} '
                       f'(SHAP={shap_dict[top_feature]:+.3f}). '
                       f'Protocol: {proto}, packets: {pkts}, bytes: {flow["bytes"]:,}.',
        'action'     : 'Investigate traffic pattern for this src/dst pair.'
    }

def shap_explanation(shap_dict):
    lines = []
    for feat, val in sorted(shap_dict.items(), key=lambda x: -abs(x[1])):
        direction = 'pushed score UP (more anomalous)' if val > 0 else 'pulled score DOWN (more normal)'
        lines.append(f'{feat}: {val:+.4f} — {direction}')
    return lines

# ── scoring ───────────────────────────────────────────────────────
def score_flow(flow_dict):
    try:
        row = np.array([[
            flow_dict['tcp_flags'], flow_dict['packets'],
            flow_dict['bytes'],     flow_dict['duration'],
            flow_dict['ttl'],       flow_dict['udp_tcp_ratio'],
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

        src_host = resolve_hostname(flow_dict['src_ip'])
        dst_host = resolve_hostname(flow_dict['dst_ip'])
        threat   = classify_threat({**flow_dict, 'anomaly_score': score}, shap_dict)
        shap_exp = shap_explanation(shap_dict)

        result = {
            'timestamp'      : time.strftime('%Y-%m-%d %H:%M:%S'),
            'src_ip'         : flow_dict['src_ip'],
            'src_host'       : src_host,
            'dst_ip'         : flow_dict['dst_ip'],
            'dst_host'       : dst_host,
            'protocol'       : flow_dict['protocol'],
            'packets'        : flow_dict['packets'],
            'bytes'          : flow_dict['bytes'],
            'bytes_per_packet': round(flow_dict['bytes_per_packet'], 1),
            'ttl'            : flow_dict['ttl'],
            'anomaly_score'  : round(score, 6),
            'threshold'      : round(THRESHOLD, 6),
            'score_ratio'    : round(score / THRESHOLD, 2),
            'is_anomaly'     : is_anomaly,
            'shap'           : shap_dict,
            'shap_explanation': shap_exp,
            'threat_type'    : threat['threat_type'],
            'severity'       : threat['severity'],
            'reason'         : threat['reason'],
            'action'         : threat['action'],
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

def flush_flows():
    while True:
        time.sleep(WINDOW_SECS)
        with ALERT_LOCK:
            snapshot = dict(flow_buffer)
            flow_buffer.clear()
        for key, fd in snapshot.items():
            if fd['packets'] > 0:
                fd['duration']         = WINDOW_SECS
                fd['bytes_per_packet'] = fd['bytes'] / fd['packets']
                threading.Thread(target=score_flow, args=(fd,), daemon=True).start()

def handle_packet(pkt):
    if not pkt.haslayer(IP):
        return
    src = pkt[IP].src
    dst = pkt[IP].dst
    ttl = pkt[IP].ttl
    if pkt.haslayer(TCP):
        proto = 'TCP'; tcp_flags = int(pkt[TCP].flags); is_udp = 0
    elif pkt.haslayer(UDP):
        proto = 'UDP'; tcp_flags = 0; is_udp = 1
    else:
        proto = 'ICMP'; tcp_flags = 0; is_udp = 0
    key = f'{src}|{dst}|{proto}'
    with ALERT_LOCK:
        if key not in flow_buffer:
            flow_buffer[key] = {
                'src_ip': src, 'dst_ip': dst, 'protocol': proto,
                'tcp_flags': tcp_flags, 'packets': 0, 'bytes': 0,
                'duration': 0, 'ttl': ttl, 'udp_tcp_ratio': is_udp,
                'bytes_per_packet': 0,
            }
        flow_buffer[key]['packets'] += 1
        flow_buffer[key]['bytes']   += len(pkt)

def start_sniffer():
    print('[sniffer] starting...')
    sniff(prn=handle_packet, store=False, filter='ip')

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
        total = len(all_flows); detected = len(alerts)
    return jsonify({'total_flows': total, 'detected_anomalies': detected,
                    'threshold': round(THRESHOLD, 6)})

@app.route('/api/threshold')
def get_threshold():
    return jsonify({'threshold': round(THRESHOLD, 6)})

if __name__ == '__main__':
    threading.Thread(target=start_sniffer, daemon=True).start()
    threading.Thread(target=flush_flows,   daemon=True).start()
    print(f'[app] threshold = {THRESHOLD:.6f}')
    print('[app] dashboard → http://localhost:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
