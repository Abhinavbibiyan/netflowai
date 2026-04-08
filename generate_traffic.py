import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

PROTOCOLS = ['TCP', 'UDP', 'ICMP']
PRIVATE_SUBNETS = [
    '192.168.1.', '192.168.2.', '10.0.0.', '10.0.1.', '172.16.0.'
]

def random_ip(subnet=None):
    if subnet:
        return subnet + str(random.randint(1, 254))
    return random.choice(PRIVATE_SUBNETS) + str(random.randint(1, 254))

def random_port(privileged=False):
    if privileged:
        return random.choice([22, 80, 443, 3306, 5432, 8080, 8443])
    return random.randint(1024, 65535)

def generate_normal_flow(ts):
    proto = random.choices(PROTOCOLS, weights=[0.6, 0.3, 0.1])[0]
    duration = round(random.uniform(0.01, 30.0), 4)
    packets = random.randint(2, 200)
    bytes_count = packets * random.randint(64, 1500)
    tcp_flags = random.randint(0, 63) if proto == 'TCP' else 0
    ttl = random.choice([64, 128, 255])
    return {
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'src_ip': random_ip(),
        'dst_ip': random_ip(),
        'src_port': random_port(),
        'dst_port': random_port(privileged=True),
        'protocol': proto,
        'tcp_flags': tcp_flags,
        'packets': packets,
        'bytes': bytes_count,
        'duration': duration,
        'ttl': ttl,
        'udp_tcp_ratio': 1.0 if proto == 'UDP' else 0.0,
        'bytes_per_packet': round(bytes_count / packets, 2),
        'is_anomaly': 0
    }

def generate_port_scan(ts):
    src = random_ip()
    return {
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'src_ip': src,
        'dst_ip': random_ip(),
        'src_port': random_port(),
        'dst_port': random.randint(1, 1024),
        'protocol': 'TCP',
        'tcp_flags': 2,
        'packets': 1,
        'bytes': 60,
        'duration': round(random.uniform(0.0001, 0.01), 6),
        'ttl': 64,
        'udp_tcp_ratio': 0.0,
        'bytes_per_packet': 60.0,
        'is_anomaly': 1
    }

def generate_udp_flood(ts):
    src = random_ip()
    pkts = random.randint(5000, 20000)
    b = pkts * random.randint(500, 1472)
    return {
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'src_ip': src,
        'dst_ip': random_ip(),
        'src_port': random.randint(1024, 65535),
        'dst_port': random.randint(1, 1024),
        'protocol': 'UDP',
        'tcp_flags': 0,
        'packets': pkts,
        'bytes': b,
        'duration': round(random.uniform(0.1, 2.0), 4),
        'ttl': 128,
        'udp_tcp_ratio': 1.0,
        'bytes_per_packet': round(b / pkts, 2),
        'is_anomaly': 1
    }

def generate_data_exfil(ts):
    pkts = random.randint(100, 500)
    b = pkts * random.randint(1200, 1500)
    return {
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'src_ip': random_ip(),
        'dst_ip': '203.0.113.' + str(random.randint(1, 254)),
        'src_port': random_port(),
        'dst_port': random.choice([443, 8443, 4444]),
        'protocol': 'TCP',
        'tcp_flags': 24,
        'packets': pkts,
        'bytes': b,
        'duration': round(random.uniform(60.0, 300.0), 2),
        'ttl': 64,
        'udp_tcp_ratio': 0.0,
        'bytes_per_packet': round(b / pkts, 2),
        'is_anomaly': 1
    }

def generate_dataset(n_normal=3000, n_anomaly=300):
    rows = []
    base_time = datetime(2024, 1, 1, 8, 0, 0)

    for i in range(n_normal):
        ts = base_time + timedelta(seconds=i * random.uniform(0.5, 3.0))
        rows.append(generate_normal_flow(ts))

    anomaly_generators = [generate_port_scan, generate_udp_flood, generate_data_exfil]
    for i in range(n_anomaly):
        ts = base_time + timedelta(seconds=i * random.uniform(1.0, 5.0))
        gen = random.choice(anomaly_generators)
        rows.append(gen(ts))

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs('data', exist_ok=True)
    df.to_csv('data/flows.csv', index=False)
    print(f"Generated {len(df)} flows ({n_normal} normal, {n_anomaly} anomalies)")
    print(f"Saved to data/flows.csv")
    return df

if __name__ == '__main__':
    generate_dataset()
