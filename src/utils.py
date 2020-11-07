import os
import random
import itertools
import numpy as np
import pandas as pd


data_path = os.path.join('..', 'data')
results_path = os.path.join('..', 'results')

# iBKS 105 beacons
ibks105_D0 = [
    'F0:C1:64:9B:71:8A', 'DF:47:ED:03:9C:24', 'D3:30:0D:56:C7:70', 'C7:F8:84:6B:03:A4', 'DA:4A:C9:76:10:39',
    'C7:9F:30:FE:E6:ED', 'C5:53:84:B9:11:46', 'EC:8A:32:11:DB:F3', 'F8:5F:B8:BE:8C:76', 'E5:B6:B3:71:CF:B1'
]

# iBKS plus beacons  (3B:43:A1:9E:5E:9A not working)
ibksplus_D0 = [
    'E1:A3:C1:C7:3F:1A', 'EE:67:B9:E8:A8:87', '3B:43:A1:9E:5E:9A', 'FD:E8:09:40:4E:6A', 'EE:C8:47:1D:2F:9B',
    'F5:65:FB:0C:D8:10', 'E3:88:8A:F5:83:2C', 'C8:30:3E:55:3E:27', 'E1:48:EA:70:98:5E', 'F8:EC:F7:70:78:B9'
]


UJI_D0 = []
for j in range(len(ibks105_D0)):
    UJI_D0.append(ibks105_D0[j])
    UJI_D0.append(ibksplus_D0[j])


def find_max(dff, rmean):
    t_max = None
    timestamps = dff['timestamp'].values
    if dff.shape[0] > rmean:
        if rmean <= 1:
            rssi_values = dff['rssi'].values
        else:
            rssi_values = dff['rssi'].rolling(rmean, min_periods=1).mean().values
        t_max = timestamps[np.argmax(rssi_values)]
    return t_max, rmean


def get_beacon_combinations(max_len=250):
    m0 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    m1 = [1, 3, 7, 9, 11, 13, 15, 17, 19]
    m2 = m0 + m1
    combinations = []
    min_n = 2
    for i in range(min_n, len(m0) + 1):
        combs = list(itertools.combinations(m0, i))
        if len(combs) > max_len:
            combs = random.sample(combs, max_len)
        combinations.extend(combs)
    for i in range(min_n, len(m1) + 1):
        combs = list(itertools.combinations(m1, i))
        if len(combs) > max_len:
            combs = random.sample(combs, max_len)
        combinations.extend(combs)
    for i in range(2, len(m2) + 1):
        combs = list(itertools.combinations(m2, i))
        if len(combs) > max_len:
            combs = random.sample(combs, max_len)
        combinations.extend(combs)

    return combinations


def process_df(df, walk_id, device, rmean, combination):
    beacons = set([UJI_D0[i] for i in combination])
    df = df.loc[(df['mac'].isin(beacons))
                & (df['walk_id'] == walk_id) & (df['device'] == device) & (df['rmean'] == rmean)]
    if not df.empty and set(pd.unique(df['mac'])) == beacons:
        return get_track_stats(df, walk_id, device, rmean, combination)

    return None


def get_track_stats(df, walk_id, device, rmean, combination):
    v_hat = []
    pairs = list(itertools.combinations(combination, 2))
    for pair in pairs:
        beacons = [UJI_D0[i] for i in pair]
        d = abs(0.3 * (pair[1] - pair[0]))
        tmax1 = df.loc[df['mac'] == beacons[0], 'ts_max'].iloc[0]
        tmax2 = df.loc[df['mac'] == beacons[1], 'ts_max'].iloc[0]
        with np.errstate(divide='raise'):
            try:
                v_pair = 1000 * abs(d / (tmax2 - tmax1))
            except FloatingPointError:
                v_pair = 0
        if 0.2 < v_pair < 2.0:
            v_hat.append(v_pair)

    if len(v_hat) > 0:
        pred_mean = np.mean(v_hat)
        pred_median = np.median(v_hat)
        p40 = np.percentile(v_hat, 40)
        p45 = np.percentile(v_hat, 45)
        p50 = np.percentile(v_hat, 50)
        p55 = np.percentile(v_hat, 55)
        p60 = np.percentile(v_hat, 60)
        p65 = np.percentile(v_hat, 65)
        p70 = np.percentile(v_hat, 70)
        p75 = np.percentile(v_hat, 75)
        p80 = np.percentile(v_hat, 80)
        p85 = np.percentile(v_hat, 85)
        p90 = np.percentile(v_hat, 90)
    else:
        return None

    speed = df['speed'].values[0]
    direction = df['direction'].values[0]
    error_mean = abs(speed - pred_mean)
    beacons = ' '.join([str(item) for item in combination])
    length = len(combination)
    even = all(c % 2 == 0 for c in combination)
    if even:
        m = 0
    else:
        odd = all(c % 2 != 0 for c in combination)
        if odd:
            m = 1
        else:
            m = 2

    row = {
        'walk_id': walk_id, 'direction': direction, 'device': device, 'speed': speed, 'rmean': rmean,
        'beacons': beacons, 'nbeacons': length, 'beacon_models': m, 'pred_mean': pred_mean, 'pred_median': pred_median,
        'error_mean': error_mean, 'p40': p40, 'p45': p45, 'p50': p50, 'p55': p55, 'p60': p60, 'p65': p65, 'p70': p70,
        'p75': p75, 'p80': p80, 'p85': p85, 'p90': p90
    }

    return row
