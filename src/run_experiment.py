import itertools
import os
import random
import sys
import multiprocessing
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from utils import data_path, get_beacon_combinations, process_df, results_path, find_max


random.seed(42)
nc = multiprocessing.cpu_count()
print(f'Number of cores: {nc}. Using {nc - 1}')

# PREPROCESSING
# mac,rssi,device,timestamp,user,direction,walk_id,speed
df = pd.read_csv(os.path.join(data_path, 'ble-gspeed.csv'))
all_tracks = pd.unique(df['walk_id']).tolist()
all_macs = pd.unique(df['mac'])
all_devices = pd.unique(df['device'])
all_rmeans = list(range(1, 15, 2))
main_loop = [item for item in itertools.product(all_tracks, all_macs, all_devices)]
main_loop = tqdm(main_loop, file=sys.stdout, ncols=80, leave=False)
size = len(main_loop) * len(all_rmeans)
columns = ['walk_id', 'mac', 'direction', 'user', 'device', 'speed', 'n', 'ts_max', 'rmean']
results = pd.DataFrame(columns=columns, index=range(size))
count = 0

for walk_id, mac, device in main_loop:
    dfi = df.loc[(df['walk_id'] == walk_id) & (df['mac'] == mac) & (df['device'] == device)]
    if not dfi.empty:
        dfi = dfi.sort_values(by=['timestamp'])
        direction = dfi['direction'].iloc[0]
        user = dfi['user'].iloc[0]
        speed = dfi['speed'].iloc[0]
        n = dfi.shape[0]
        ts_maxss = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(find_max)(dfi, rmean) for rmean in
                                                                    all_rmeans)
        for ts_max, rmean in ts_maxss:
            if ts_max is not None:
                row = {'walk_id': walk_id, 'mac': mac, 'direction': direction, 'user': user, 'device': device,
                       'speed': speed, 'n': n, 'ts_max': ts_max, 'rmean': rmean}
                results.iloc[count] = row
                count += 1

results = results.iloc[:count]
filename = os.path.join(results_path, 'preprocessed.csv')
results.to_csv(filename, index=False)
print(f'results saved to file {filename}')

# EXPERIMENT
# walk_id,mac,direction,user,device,speed,n,ts_max,rmean
df = pd.read_csv(os.path.join(results_path, 'preprocessed.csv'), index_col=None)
if len(sys.argv) > 1:
    all_devices = sys.argv[1].split(',')
else:
    # a650,14df,d884,38b8
    all_devices = pd.unique(df['device'])

all_tracks = pd.unique(df['walk_id']).tolist()
all_rmeans = pd.unique(df['rmean'])
max_comb_length = 250
ncols = 80

print('processing devices :', all_devices)

all_combinations = get_beacon_combinations(max_comb_length)

device_loop = tqdm(all_devices, leave=False, file=sys.stdout, ncols=ncols)
for device in device_loop:
    device_loop.set_description(f'device: {device:13s}')
    df_d = df.loc[df['device'] == device]
    df_len = len(all_combinations) * len(all_tracks) * len(all_rmeans)
    df_stats = pd.DataFrame(columns=['walk_id', 'direction', 'device', 'speed', 'rmean', 'beacons', 'nbeacons',
                                     'beacon_models', 'method', 'pred_mean', 'pred_median', 'error_mean',
                                     'p40', 'p45', 'p50', 'p55', 'p60', 'p65', 'p70', 'p75',
                                     'p80', 'p85', 'p90'],
                            index=range(df_len))
    count = 0
    track_loop = tqdm(all_tracks, leave=False, file=sys.stdout, ncols=ncols)
    for walk_id in track_loop:
        track_loop.set_description(f'walk id: {walk_id:3d}{"":9s}')
        df_w = df_d.loc[df['walk_id'] == walk_id]
        inner_loop = tqdm(itertools.product(all_rmeans, all_combinations), leave=False, file=sys.stdout,
                          total=len(all_rmeans) * len(all_combinations), ncols=ncols)
        rows = Parallel(n_jobs=nc - 1)(delayed(process_df)(df_w, walk_id, device, *params)
                                       for params in inner_loop)
        for row in rows:
            if row is not None:
                df_stats.iloc[count] = row
                count += 1
    df_stats = df_stats.iloc[:count]
    filename = os.path.join(results_path, f'results_{device}.csv')
    df_stats.to_csv(filename, index=False)
    print(f'results saved to file {filename}')
