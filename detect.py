import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import pickle
import json
import os

FEATURES = ['tcp_flags', 'packets', 'bytes', 'duration',
            'ttl', 'udp_tcp_ratio', 'bytes_per_packet']

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def reconstruction_error(model, X_tensor):
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        errors = ((X_tensor - recon) ** 2).mean(dim=1).numpy()
    return errors

def run_detection():
    df = pd.read_csv('data/flows.csv')
    X_raw = df[FEATURES].fillna(0).values

    with open('data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X = scaler.transform(X_raw)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    model = Autoencoder(len(FEATURES))
    model.load_state_dict(torch.load('data/model.pt', map_location='cpu'))

    with open('data/threshold.txt') as f:
        threshold = float(f.read().strip())

    errors = reconstruction_error(model, X_tensor)
    df['anomaly_score'] = np.round(errors, 6)
    df['predicted_anomaly'] = (errors > threshold).astype(int)

    print(f"Total flows: {len(df)}")
    print(f"Detected anomalies: {df['predicted_anomaly'].sum()}")
    print(f"Threshold: {threshold:.6f}")

    background = torch.tensor(X[:200], dtype=torch.float32)

    def model_recon_error(x_np):
        t = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            recon = model(t)
            return ((t - recon) ** 2).mean(dim=1).numpy()

    explainer = shap.KernelExplainer(model_recon_error, background.numpy()[:50])

    anomaly_idx = df[df['predicted_anomaly'] == 1].index[:30].tolist()
    X_anomalies = X[anomaly_idx]

    print(f"Computing SHAP values for {len(anomaly_idx)} anomalies...")
    shap_values = explainer.shap_values(X_anomalies, nsamples=50, silent=True)

    results = []
    for i, idx in enumerate(anomaly_idx):
        row = df.iloc[idx]
        sv = shap_values[i].tolist()
        results.append({
            'index': int(idx),
            'timestamp': str(row['timestamp']),
            'src_ip': str(row['src_ip']),
            'dst_ip': str(row['dst_ip']),
            'protocol': str(row['protocol']),
            'anomaly_score': round(float(row['anomaly_score']), 6),
            'is_actual_anomaly': int(row['is_anomaly']),
            'shap': {FEATURES[j]: round(sv[j], 4) for j in range(len(FEATURES))}
        })

    os.makedirs('data', exist_ok=True)
    with open('data/alerts.json', 'w') as f:
        json.dump(results, f, indent=2)

    df.to_csv('data/flows_scored.csv', index=False)
    print("Saved: data/alerts.json, data/flows_scored.csv")

if __name__ == '__main__':
    run_detection()
