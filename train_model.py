import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

FEATURES = ['tcp_flags', 'packets', 'bytes', 'duration',
            'ttl', 'udp_tcp_ratio', 'bytes_per_packet']

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train():
    df = pd.read_csv('data/flows.csv')
    normal = df[df['is_anomaly'] == 0][FEATURES].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(normal.values)
    X_train, X_val = train_test_split(X, test_size=0.1, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

    model = Autoencoder(len(FEATURES))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Training autoencoder...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train_t), X_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), X_val_t).item()
            print(f"  Epoch {epoch+1}/100 — train loss: {loss.item():.6f}  val loss: {val_loss:.6f}")

    model.eval()
    with torch.no_grad():
        recon = model(X_train_t)
        errors = ((X_train_t - recon) ** 2).mean(dim=1).numpy()

    threshold = float(np.percentile(errors, 95))
    print(f"\nAnomaly threshold (95th percentile): {threshold:.6f}")

    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), 'data/model.pt')
    with open('data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('data/threshold.txt', 'w') as f:
        f.write(str(threshold))

    print("Saved: data/model.pt, data/scaler.pkl, data/threshold.txt")

if __name__ == '__main__':
    train()
