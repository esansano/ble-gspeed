import warnings
import os
import sys
import itertools
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from utils import results_path, UJI_D0, data_path
from joblib import Parallel, delayed
from tqdm import tqdm


def user_speed_distribution():
    cmap1 = plt.get_cmap('tab20c')
    color1 = 0
    colors1 = [cmap1(i) for i in range(color1, color1 + 4)]
    df = pd.read_csv(os.path.join(data_path, 'ble-gspeed.csv'))
    users = sorted(pd.unique(df['user']).tolist())
    walk_ids = sorted(pd.unique(df['walk_id']).tolist())
    plt.figure(figsize=(10, 10), dpi=200)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    speeds = []
    speeds_hist = []
    for i, user in enumerate(users):
        user_speeds = pd.unique(df.loc[(df['user'] == user), 'speed']).tolist()
        speeds.append(user_speeds)
        speeds_hist.extend(user_speeds)
    vp = ax0.violinplot(speeds, showmedians=True, widths=0.75)

    for i, user in enumerate(users):
        count = len(pd.unique(df.loc[df['user'] == user]['walk_id']))
        ax1.bar(x=i + 1, height=100 * count/len(walk_ids), width=0.6, color=colors1[1], alpha=0.75)

    ax2.hist(speeds_hist, bins=100, color=colors1[1])

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        pc = vp[partname]
        pc.set_edgecolor('k')
        pc.set_linewidth(1)

    for pc in vp['bodies']:
        pc.set_facecolor(colors1[1])
        pc.set_edgecolor(colors1[0])
        pc.set_linewidth(1)
        pc.set_alpha(0.75)

    ax0.set_xlim(0.5, 13.5)
    # ax0.get_xaxis().set_visible(False)
    ax0.set_xticks(users)
    ax0.set_xticklabels(users)
    ax0.set_xlabel('user id', horizontalalignment='right', x=0.95)
    ax0.set_ylabel('speed (m/s)')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.set_title('Distribution of speeds recorded by user')
    # ax0.spines['bottom'].set_visible(False)
    ax0.text(x=-0.4, y=1.9, s='(a)', fontweight='bold', fontsize=16)

    ax1.set_xlim(0.5, 13.5)
    ax1.set_ylim(0, 22)
    ax1.set_xticks(users)
    ax1.set_xticklabels(users)
    ax1.set_xlabel('user id', horizontalalignment='right', x=0.95)
    ax1.set_ylabel('% of walks')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_title('Number of walks recorded by user')
    ax1.text(x=-0.4, y=21, s='(b)', fontweight='bold', fontsize=16)

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlabel('speed (m/s)', horizontalalignment='right', x=0.95)
    ax2.set_xlim(0.2, 1.8)
    ax2.set_ylim(0, 15)
    ax2.set_ylabel('# of walks')
    ax2.set_title('Distribution of speeds')
    ax2.text(x=0.088, y=14, s='(c)', fontweight='bold', fontsize=16)
    # ax1.set_xticks(ticks=[1, 2, 3, 4], minor=False)
    # ax1.set_yticks(ticks=[0, 10, 20, 30, 40, 50], minor=False)

    plt.tight_layout()
    plt.show()


def device_beacon_rssi_distribution():
    cmap2 = plt.get_cmap('tab20')
    scolors = [cmap2(14), cmap2(15)]
    bcolors = [cmap2(0), cmap2(1), cmap2(4), cmap2(5)]
    dcolors = [cmap2(6), cmap2(7), cmap2(8), cmap2(9),
               cmap2(10), cmap2(11), cmap2(18), cmap2(19)]

    df = pd.read_csv(os.path.join(data_path, 'ble-gspeed.csv'))
    df = df.loc[df['mac'].isin(UJI_D0)]
    devices = pd.unique(df['device']).tolist()
    macs = pd.unique(df['mac']).tolist()
    device_rates = rssi_device_scan_rates()
    mac_rates = rssi_beacon_advertising_rates()
    model_rates = rssi_model_advertising_rates()
    macs_105 = [mac for i, mac in enumerate(UJI_D0) if i % 2 == 0 and mac in macs]
    macs_plus = [mac for i, mac in enumerate(UJI_D0) if i % 2 != 0 and mac in macs]

    model_rssi = [
        df.loc[df['mac'].isin(macs_105)]['rssi'].values.tolist(),
        df.loc[df['mac'].isin(macs_plus)]['rssi'].values.tolist()
    ]

    fig = plt.figure(figsize=(12, 10), dpi=300)

    gs = fig.add_gridspec(5, 2, left=0.065, bottom=0.05, right=0.975, top=0.975, wspace=0.2, hspace=0.05,
                          height_ratios=[2.3, 1, 0.5, 2.3, 1], width_ratios=[1.4, 1])
    ax11 = fig.add_subplot(gs[0, 0])
    ax12 = fig.add_subplot(gs[1, 0])
    ax21 = fig.add_subplot(gs[0, 1])
    ax22 = fig.add_subplot(gs[1, 1])
    ax31 = fig.add_subplot(gs[3, :])
    ax32 = fig.add_subplot(gs[4, :])

    rssi_lst = []

    for i, device in enumerate(devices):
        rssi_lst.append(df.loc[df['device'] == device].rssi.values)
        if device in device_rates:
            ax12.bar(x=i + 1, height=device_rates[device], width=0.6, color=dcolors[1 + i * 2], alpha=0.75)
    vp = ax11.violinplot(rssi_lst, showmedians=True, widths=0.75)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        pc = vp[partname]
        pc.set_edgecolor(scolors[0])
        pc.set_linewidth(1)

    for i, pc in enumerate(vp['bodies']):
        pc.set_facecolor(dcolors[i * 2 + 1])
        pc.set_edgecolor(dcolors[i * 2])
        pc.set_linewidth(1)
        pc.set_alpha(0.75)

    vp = ax21.violinplot(model_rssi, showmedians=True, widths=0.5)
    for i, model_rate in enumerate(model_rates):
        c = bcolors[1] if i % 2 == 0 else bcolors[3]
        ax22.bar(x=i + 1, height=model_rate, width=0.5, color=c, alpha=0.75)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        pc = vp[partname]
        pc.set_edgecolor(scolors[0])
        pc.set_linewidth(1)

    for i, pc in enumerate(vp['bodies']):
        c1, c2 = (bcolors[1], bcolors[0]) if i % 2 == 0 else (bcolors[3], bcolors[2])
        pc.set_facecolor(c1)
        pc.set_edgecolor(c2)
        pc.set_linewidth(1)
        pc.set_alpha(0.75)

    rssi_lst = []
    for i, mac in enumerate(UJI_D0):
        rssi_data = df.loc[df['mac'] == mac].rssi.values
        if len(rssi_data) > 0:
            rssi_lst.append(rssi_data)
        else:
            rssi_lst.append([np.nan, np.nan])

        ax32.axhline(y=2.5, xmin=0.005, xmax=0.995, color=scolors[1], alpha=0.3, linestyle='--')
        if mac in mac_rates:
            # c = colors1[1] if i % 2 == 0 else colors2[1]
            for k, device in enumerate(['a650', '14df', 'd884', '38b8']):
                height = mac_rates[mac][device][0] / mac_rates[mac][device][1]
                c = dcolors[k * 2 + 1]
                ax32.bar(x=i + 1 - 0.225 + k * 0.15, height=height, width=0.13, color=c, alpha=0.75)

    vp = ax31.violinplot(rssi_lst, showmedians=True, widths=0.75)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        pc = vp[partname]
        pc.set_edgecolor(scolors[0])
        pc.set_linewidth(1)

    for i, pc in enumerate(vp['bodies']):
        c1, c2 = (bcolors[1], bcolors[0]) if i % 2 == 0 else (bcolors[3], bcolors[2])
        pc.set_facecolor(c1)
        pc.set_edgecolor(c2)
        pc.set_linewidth(1)
        pc.set_alpha(0.75)

    dpatches = [
        mpatches.Patch(color=dcolors[1]),
        mpatches.Patch(color=dcolors[3]),
        mpatches.Patch(color=dcolors[5]),
        mpatches.Patch(color=dcolors[7])]
    ax11.legend(dpatches, ['a650', '14df', 'd884', '38b8'], fontsize=11, frameon=False)
    ax11.set_xlim(0.5, 5.1)
    ax12.set_xlim(0.5, 5.1)
    ax11.set_ylabel('RSSI')
    ax12.set_ylabel('scanning rate (n/sec)', labelpad=18)
    ax11.get_xaxis().set_visible(False)
    ax11.spines['bottom'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    ax12.spines['top'].set_visible(False)
    ax11.spines['right'].set_visible(False)
    ax12.spines['right'].set_visible(False)
    ax12.set_xlabel('smartwatch')
    ax12.set_xticks(ticks=[1, 2, 3, 4], minor=False)
    ax12.set_yticks(ticks=[0, 10, 20, 30, 40, 50], minor=False)
    ax12.set_xticklabels(devices)
    ax11.set_title('RSSI distribution and scanning rate')
    ax11.text(x=0.0, y=-30, s='(a)', fontweight='bold', fontsize=14)

    patch_105 = mpatches.Patch(color=bcolors[1])
    patch_plus = mpatches.Patch(color=bcolors[3])
    ax21.legend((patch_105, patch_plus), ['iBKS 105', 'iBKS plus'], fontsize=11, frameon=False)
    ax21.set_xlim(0.3, 3.0)
    ax22.set_xlim(0.3, 3.0)
    ax21.set_ylabel('RSSI')
    ax22.set_ylabel('advertising rate (n/sec)', labelpad=10)
    ax21.get_xaxis().set_visible(False)
    ax21.spines['bottom'].set_visible(False)
    ax21.spines['top'].set_visible(False)
    ax22.spines['top'].set_visible(False)
    ax21.spines['right'].set_visible(False)
    ax22.spines['right'].set_visible(False)
    ax22.set_xlabel('beacon model')
    ax21.set_xticks(ticks=[1, 2], minor=False)
    ax22.set_xticks(ticks=[1, 2], minor=False)
    ax22.set_yticks(ticks=[0, 20, 40, 60, 80, 100], minor=False)
    ax22.set_xticklabels(['iBKS 105', 'iBKS plus'])
    ax21.set_title('RSSI distribution and advertising rate')
    ax21.text(x=-0.17, y=-30, s='(b)', fontweight='bold', fontsize=14)

    patch_105 = mpatches.Patch(color=bcolors[1])
    patch_plus = mpatches.Patch(color=bcolors[3])
    ax31.legend((patch_105, patch_plus), ['iBKS 105', 'iBKS plus'], fontsize=11, frameon=False)
    ax31.set_xlim(0.5, 20.5)
    ax32.set_xlim(0.5, 20.5)
    ax31.set_ylabel('RSSI')
    ax32.set_ylabel('advertising rate (n/sec)', labelpad=16)
    ax31.get_xaxis().set_visible(False)
    ax31.spines['bottom'].set_visible(False)
    ax31.spines['top'].set_visible(False)
    ax32.spines['top'].set_visible(False)
    ax31.spines['right'].set_visible(False)
    ax32.spines['right'].set_visible(False)
    ax32.set_xlabel('beacon')
    ax32.set_xticks(ticks=np.arange(1, len(UJI_D0) + 1), minor=False)
    ax32.set_yticks(ticks=np.arange(0, 4.1, .5), minor=False)
    ax32.set_xticklabels([m[:5] for m in UJI_D0])
    ax31.set_title('RSSI distribution and advertising rate for each beacon')
    ax31.text(x=-0.6, y=-30, s='(c)', fontweight='bold', fontsize=14)

    plt.show()


def rssi_device_scan_rates():

    def process(d):
        dfi = df.loc[(df['device'] == d) & (df['mac'].isin(UJI_D0))]
        t, n = 0, 0
        for w in walks:
            dfiw = dfi.loc[(df['walk_id'] == w)]
            if not dfiw.empty:
                t += max(dfiw['timestamp']) - min(dfiw['timestamp'])
                n += dfiw.shape[0]

        if t > 0:
            return d, n, t / 1000
        else:
            return None

    df = pd.read_csv(os.path.join(data_path, 'ble-gspeed.csv'))
    walks = pd.unique(df['walk_id']).tolist()
    devices = pd.unique(df['device']).tolist()
    total = len(devices)
    pb = tqdm(devices, leave=False, file=sys.stdout, ncols=60, total=total)
    rows = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(process)(device) for device in pb)
    results = {device: 0 for device in devices}
    for row in rows:
        if row is not None:
            results[row[0]] = row[1] / row[2]
    return results


def rssi_model_advertising_rates():

    df = pd.read_csv(os.path.join(data_path, 'ble-gspeed.csv'))
    walks = pd.unique(df['walk_id']).tolist()
    macs = pd.unique(df['mac']).tolist()
    macs_105 = [mac for i, mac in enumerate(UJI_D0) if i % 2 == 0 and mac in macs]
    macs_plus = [mac for i, mac in enumerate(UJI_D0) if i % 2 != 0 and mac in macs]
    model_rates = []

    for model_macs in [macs_105, macs_plus]:
        dfi = df.loc[(df['mac'].isin(model_macs))]
        t, n = 0, 0
        for walk in walks:
            dfiw = dfi.loc[dfi['walk_id'] == walk]
            t += max(dfiw['timestamp']) - min(dfiw['timestamp'])
            n += dfiw.shape[0]
        model_rates.append(1000 * n / t)

    return model_rates


def rssi_beacon_advertising_rates():

    def process(m, d):
        dfi = df.loc[(df['mac'] == m) & (df['device'] == d)]
        t, n = 0, 0
        for w in walks:
            dfiw = dfi.loc[dfi['walk_id'] == w]
            if not dfiw.empty:
                t += max(dfiw['timestamp']) - min(dfiw['timestamp'])
                n += dfiw.shape[0]

        if t > 0:
            return m, d, n, t / 1000
        else:
            return None

    df = pd.read_csv(os.path.join(data_path, 'ble-gspeed.csv'))
    walks = pd.unique(df['walk_id']).tolist()
    devices = pd.unique(df['device']).tolist()
    macs = pd.unique(df['mac']).tolist()
    total = len(UJI_D0) * len(devices)
    pb = tqdm(itertools.product(UJI_D0, devices), leave=False, file=sys.stdout, ncols=60, total=total)
    rows = Parallel(n_jobs=multiprocessing.cpu_count() - 1)(delayed(process)(*params) for params in pb)
    results = {mac: {device: 0 for device in devices} for mac in macs}
    for row in rows:
        if row is not None:
            results[row[0]][row[1]] = row[2], row[3]
    return results


def get_statistics(df, methods, combined=False, percentiles=None):
    all_rmeans = pd.unique(df['rmean']).tolist()
    all_methods = pd.unique(df['method']).tolist()
    all_bmodels = pd.unique(df['beacon_models']).tolist()
    all_devices = pd.unique(df['device']).tolist()
    all_devices.append('all')
    best_rmeans = []
    for i, m in enumerate(all_methods):
        min_error = 1
        best_rmeans.append(0)
        for rmean in all_rmeans:
            err_mean = np.mean(df.loc[(df['rmean'] == rmean) & (df['method'] == m), 'error_mean'].tolist())
            if err_mean < min_error:
                min_error = err_mean
                best_rmeans[i] = rmean

    stats_df = pd.DataFrame(columns=['device', 'nbeacons', 'error', 'cmin', 'cmax', 'beacon_models', 'rmeans'],
                            index=range(len(all_bmodels) * 19 * 5))
    rmeans = ' '.join([str(rmean) for rmean in best_rmeans])
    all_tracks = pd.unique(df['walk_id'])

    for didx, device in enumerate(all_devices):
        for b, bmodels in enumerate(all_bmodels):
            dfm = df.loc[df['beacon_models'] == bmodels]
            if device != 'all':
                dfm = dfm.loc[dfm['device'] == device]
            for j in range(1, 20):
                dfmb = dfm.loc[dfm['nbeacons'] == j]
                speeds = np.zeros(len(all_tracks))
                speeds.fill(np.nan)
                pred_means = np.zeros((len(all_tracks), len(methods)))
                pred_means.fill(np.nan)
                pred_medians = np.zeros((len(all_tracks), len(methods)))
                pred_medians.fill(np.nan)

                for k, m in enumerate(methods):
                    if combined:
                        dfs = []
                        for mtd in all_methods:
                            dfs.append(dfmb.loc[(dfmb['rmean'] == best_rmeans[k]) & (dfmb['method'] == mtd)])
                        dfmbrm = pd.concat(dfs)
                    else:
                        dfmbrm = dfmb.loc[(dfmb['rmean'] == best_rmeans[k]) & (dfmb['method'] == m)]

                    if not dfmbrm.empty:
                        for i, walk_id in enumerate(all_tracks):
                            dfmbrmw = dfmbrm.loc[(dfmbrm['walk_id'] == walk_id)]
                            if not dfmbrmw.empty:
                                speeds[i] = dfmbrmw.iloc[0]['speed']
                                if percentiles is None:
                                    predictions = dfmbrmw['pred_mean'].values
                                    pred = np.mean(predictions)
                                else:
                                    predictions = dfmbrmw[f'p{percentiles[k]}'].values
                                    pred = np.mean(predictions)
                                pred_means[i, k] = pred

                                pred_medians[i, k] = np.mean(dfmbrmw['pred_median'].values)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    pred_means = np.nanmean(pred_means, axis=1)
                    error_means = np.abs(speeds - pred_means)

                    meanc = np.nanmean(error_means)
                    stdc = np.nanstd(error_means)
                    idx = didx * 19 * 3 + b * 19 + j - 1
                    stats_df.iloc[idx]['device'] = device
                    stats_df.iloc[idx]['nbeacons'] = j
                    stats_df.iloc[idx]['error'] = meanc
                    stats_df.iloc[idx]['cmin'] = meanc - stdc
                    stats_df.iloc[idx]['cmax'] = meanc + stdc
                    stats_df.iloc[idx]['beacon_models'] = bmodels

    stats_df['rmeans'] = rmeans
    return stats_df


def plot_single_results(devices, rmean=None):
    dfs = []
    for d in devices:
        filename = os.path.join(results_path, f'results_{d}.csv')
        dfs.append(pd.read_csv(filename))
    # walk_id,direction,device,speed,rmean,beacons,nbeacons,beacon_models,method,pred_mean,pred_median,error_mean,error_median
    df = pd.concat(dfs)
    df1 = df.loc[df['rmean'] == 1]
    df2 = df.loc[df['rmean'] == rmean]
    # all_bmodels = [2]
    all_bmodels = [0, 1]
    models_str = ['iBKS 105', 'iBKS plus']

    fig, ax = plt.subplots(len(all_bmodels), len(devices), figsize=(4 * len(devices), 8), dpi=200, sharey='row')
    cmap = plt.get_cmap('tab10')
    colors = [cmap(0), cmap(2), cmap(3), cmap(4), cmap(8)]
    linestyles = ['-', '--']

    # device,nbeacons,error,cmin,cmax,beacon_models,rmeans
    for k, (df, method) in enumerate(zip([df1, df2], ['raw data', 'smoothed data'])):
        dfm = get_statistics(df, ['ts_max'])
        dfm = dfm.drop(columns=['rmeans'])
        for j, m in enumerate(all_bmodels):
            for i, d in enumerate(devices):
                dfc = dfm.loc[(dfm['beacon_models'] == m) & (dfm['device'] == d)]
                # device,nbeacons,error,cmin,cmax,beacon_models
                if not dfc.empty:
                    arr = dfc.drop(columns=['device', 'beacon_models']).values.astype(np.float)[:10, :]
                    ax[j, i].plot(arr[:, 0], arr[:, 1], linewidth=3, linestyle=linestyles[k],
                                  c=colors[k], alpha=0.75, label=method)
                    ax[j, i].yaxis.grid(which='major', color='k', linestyle='-', linewidth=1, alpha=0.15)
                    ax[j, i].spines['right'].set_visible(False)
                    ax[j, i].spines['top'].set_visible(False)
                    ax[j, i].set_xlim(0.5, 10.5)
                    ax[j, i].set_ylim(0.0, 0.4)
                    ax[j, i].set_xticks(range(1, 11))
                    if j == 0:
                        ax[j, i].set_title(d, fontsize=14, fontweight='bold')
                    elif j == len(all_bmodels) - 1:
                        ax[j, i].set_xlabel('number of beacons')
                    if i == 0:
                        x = -1.5
                        ax[j, i].text(x, 0.1, models_str[j], rotation=90, verticalalignment='center',
                                      fontsize=14, fontweight='bold')
                        ax[j, i].set_ylabel('error (m/s)')
                        ax[j, i].yaxis.set_label_coords(-0.13, 0.8)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    # labels = ['gait speed']
    fig.suptitle('Average error in gait speed estimation', fontsize=16, fontweight='bold')
    fig.legend(handles, labels, loc='upper right', fontsize='large', bbox_to_anchor=(0.995, 0.915))
    fig.tight_layout()
    plt.show()


def plot_both_models(devices, rmean=None):
    dfs = []
    for d in devices:
        filename = os.path.join(results_path, f'results_{d}.csv')
        dfs.append(pd.read_csv(filename))
    # walk_id,direction,device,speed,rmean,beacons,nbeacons,beacon_models,method,pred_mean,pred_median,error_mean,error_median
    df = pd.concat(dfs)
    df1 = df.loc[df['rmean'] == 1]
    df2 = df.loc[df['rmean'] == rmean]
    all_bmodels = [2]
    models_str = ['iBKS 105 + iBKS plus']

    fig, ax = plt.subplots(len(all_bmodels), len(devices), figsize=(4 * len(devices), 4), dpi=200, sharey='row')
    cmap = plt.get_cmap('tab10')
    colors = [cmap(0), cmap(2), cmap(3), cmap(4), cmap(8)]
    linestyles = ['-', '--']

    # device,nbeacons,error,cmin,cmax,beacon_models,rmeans
    for k, (df, method) in enumerate(zip([df1, df2], ['raw data', 'smoothed data'])):
        dfm = get_statistics(df, ['ts_max'])
        dfm = dfm.drop(columns=['rmeans'])
        for i, d in enumerate(devices):
            dfc = dfm.loc[(dfm['beacon_models'] == 2) & (dfm['device'] == d)]
            # device,nbeacons,error,cmin,cmax,beacon_models
            if not dfc.empty:
                arr = dfc.drop(columns=['device', 'beacon_models']).values.astype(np.float)[:10, :]
                ax[i].plot(arr[:, 0], arr[:, 1], linewidth=3, linestyle=linestyles[k],
                           c=colors[k], alpha=0.75, label=method)
                ax[i].yaxis.grid(which='major', color='k', linestyle='-', linewidth=1, alpha=0.15)
                ax[i].spines['right'].set_visible(False)
                ax[i].spines['top'].set_visible(False)
                ax[i].set_xlim(0.5, 10.5)
                ax[i].set_ylim(0.0, 0.4)
                ax[i].set_xticks(range(1, 11))
                ax[i].set_title(d, fontsize=14, fontweight='bold')
                ax[i].set_xlabel('number of beacons')
                if i == 0:
                    x = -1.5
                    ax[i].text(x, 0.1, models_str[0], rotation=90, verticalalignment='center',
                               fontsize=12, fontweight='bold')
                    ax[i].set_ylabel('error (m/s)')
                    ax[i].yaxis.set_label_coords(-0.13, 0.8)
    handles, labels = ax[0].get_legend_handles_labels()
    # labels = ['gait speed']
    fig.suptitle('Average error in gait speed estimation', fontsize=16, fontweight='bold')
    fig.legend(handles, labels, loc='upper right', fontsize='large', bbox_to_anchor=(0.995, 0.8515))
    fig.tight_layout()
    plt.show()


def plot_all_devices(devices, rmean=None):
    dfs = []
    for d in devices:
        filename = os.path.join(results_path, f'results_{d}.csv')
        dfs.append(pd.read_csv(filename))
    # walk_id,direction,device,speed,rmean,beacons,nbeacons,beacon_models,method,pred_mean,pred_median,error_mean,error_median
    df = pd.concat(dfs)
    df1 = df.loc[df['rmean'] == 1]
    df2 = df.loc[df['rmean'] == rmean]
    all_bmodels = [0, 1, 2]
    models_str = ['iBKS 105', 'iBKS plus', 'iBKS 105 + iBKS plus']

    fig, ax = plt.subplots(1, len(all_bmodels), figsize=(4 * len(devices), 4), dpi=200)
    cmap = plt.get_cmap('tab10')
    colors = [cmap(0), cmap(2), cmap(3), cmap(4), cmap(8)]
    linestyles = ['-', '--']

    # device,nbeacons,error,cmin,cmax,beacon_models,rmeans
    for k, (df, method) in enumerate(zip([df1, df2], ['raw data', 'smoothed data'])):
        dfm = get_statistics(df, ['ts_max'])
        dfm = dfm.drop(columns=['rmeans'])
        d = 'all'
        for j, m in enumerate(all_bmodels):
            dfc = dfm.loc[(dfm['beacon_models'] == m) & (dfm['device'] == d)]
            # device,nbeacons,error,cmin,cmax,beacon_models
            if not dfc.empty:
                arr = dfc.drop(columns=['device', 'beacon_models']).values.astype(np.float)[:10, :]
                ax[j].plot(arr[:, 0], arr[:, 1], linewidth=3, linestyle=linestyles[k],
                           c=colors[k], alpha=0.75, label=method)
                ax[j].yaxis.grid(which='major', color='k', linestyle='-', linewidth=1, alpha=0.15)
                ax[j].spines['right'].set_visible(False)
                ax[j].spines['top'].set_visible(False)
                ax[j].set_xlim(0.5, 10.5)
                ax[j].set_ylim(0.0, 0.4)
                ax[j].set_xticks(range(1, 11))
                # ax[j].set_title('all smartwatches', fontsize=14, fontweight='bold')
                ax[j].set_xlabel('number of beacons')
                if j == 0:
                    ax[j].text(-1.0, 0.1, models_str[j], rotation=90, verticalalignment='center',
                               fontsize=13, fontweight='bold')
                elif j == 1:
                    ax[j].text(-1, 0.1, models_str[j], rotation=90, verticalalignment='center',
                               fontsize=13, fontweight='bold')
                else:
                    ax[j].text(-1, 0.1, models_str[j], rotation=90, verticalalignment='center',
                               fontsize=12, fontweight='bold')

                ax[j].set_ylabel('error (m/s)')
                ax[j].yaxis.set_label_coords(-0.10, 0.85)
    handles, labels = ax[0].get_legend_handles_labels()
    # labels = ['gait speed']
    fig.suptitle('Average error in gait speed estimation (all smartwatches)', fontsize=16, fontweight='bold')
    fig.legend(handles, labels, loc='upper right', fontsize='large', bbox_to_anchor=(0.995, 0.915))
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    smartwatches = ['a650', '14df', 'd884', '38b8']

    # plot figure 3
    user_speed_distribution()
    # plot figure 4
    device_beacon_rssi_distribution()
    # plot figure 6
    plot_single_results(devices=smartwatches, rmean=13)
    # plot figure 7
    plot_both_models(devices=smartwatches, rmean=13)
    # plot figure 8
    plot_all_devices(devices=smartwatches, rmean=13)
