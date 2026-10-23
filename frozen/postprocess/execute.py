import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import logging
from detectors.median_smoother import label_smoother
from src.detectors import stat_detector, median_smoothing, kmeans_detector,  mean_smoothing
from src.detectors.metrics import windowdiff_pk_score as wd_pk
import time
import json
import pandas as pd

NAMES = [
    'aci_iot_2023',
    'active_wiretap',
    'cic2017',
    'cic2018',
    'fuzzing',
    'os_scan',
    'ssdp_flood',
    'ssl_renegotiation',
    'video_injection'
]
ROOT_PATH = './data/'
TARGET_PATH = './data/'
RES_PATH = './experiments/'
logging.basicConfig(level=logging.INFO)


def get_numpy_files(path):
    return [os.path.abspath(os.path.join(path, f)) for f in os.listdir(path) if f.endswith('.npy')]

def get_array_and_info(paths):
    info_ar = dict()
    info_or = dict()
    for string in paths:
        name = string.split("\\")[-1].replace('.npy', '')
        split_name = name.split('_ar_')
        if split_name[-1] == 'True':
            info_ar[split_name[0].upper()] = np.load(string, allow_pickle=True)
        else:
            info_or[split_name[0].upper()] = np.load(string, allow_pickle=True)
    return info_ar, info_or

def compute_scores(detection_score: np.ndarray, target: np.ndarray):
    """
    This function computes the scores of a given detection.
    :param detection_score: Detection score.
    :param target: Target.
    :return: FPR, TPR, best FPR, best TPR, best threshold.
    """
    fprs = list()
    tprs = list()
    best_ratio = -1
    best_fpr, best_tpr, best_th = None, None, None

    for th in np.linspace(0, 0.3, 10_000):
        binary_preds = (detection_score >= th).astype(int)
        fpr = np.sum((binary_preds == 1) & (target == 0)) / np.sum(target == 0)
        tpr = np.sum((binary_preds == 1) & (target == 1)) / np.sum(target == 1)
        fprs.append(fpr)
        tprs.append(tpr)
        ratio = (1 - fpr) * tpr / ((1 - fpr) + tpr + 1e-9)
        if ratio > best_ratio:
            best_ratio = ratio
            best_fpr = fpr
            best_tpr = tpr
            best_th = th
    return fprs, tprs, best_fpr, best_tpr, best_th

def plot_frame(frame, target, path, name, boost=1):
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.gca()
        order = np.linspace(0, len(frame), len(frame))
        xticks = np.linspace(0, len(frame) * boost, 60)
        yticks = np.linspace(0, max(frame), 30)
        plt.xticks(xticks, rotation=90)
        plt.yticks(yticks)
        plt.xlim((0, len(frame)))
        plt.ylim((0, max(frame) * 1.01))
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        plt.scatter(order, frame, label='Frame', color='black', alpha=0.7, zorder=2, s=0.25)
        plt.fill_between(order, max(frame) * 1.1 * target, color='RED', alpha=0.3, zorder=1)
        ax.axvspan(0, 200_000 // boost, alpha=0.3, hatch='xx', edgecolor='black', zorder=1, linewidth=0.5)
        ax.axvspan(0, 50_000 // boost, alpha=0.3, hatch='xx', edgecolor='black', zorder=1, linewidth=0.9)
        plt.xlabel('Packet number', fontsize=16)
        plt.ylabel('$\log_{10}({RMSE})$', fontsize=16)
        plt.title(f'{name}', fontsize=21)
        fig.tight_layout()
        plt.savefig(f'{path}/{name}_frame.png', dpi=220)
        plt.savefig(f'{path}/{name}_frame.eps', dpi=220)
        plt.savefig(f'{path}/{name}_frame.pdf', dpi=220)
    except Exception as e:
        plt.close()
        logging.error(f'Error plotting frame: {e}: {f'{path}/{name}_frame'}')
        plot_frame(frame[::2], target[::2], path, name, boost*2)
    finally:
        plt.show()


def plot_detection_score(detection_score, target, th, path, name, boost=1):
    try:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.gca()
        order = np.linspace(0, len(detection_score), len(detection_score))
        xticks = np.linspace(0, len(detection_score) * boost, 60)
        yticks = np.linspace(0, max(detection_score), 30)
        plt.xticks(xticks, rotation=90)
        plt.yticks(yticks)
        plt.xlim((0, len(detection_score)))
        plt.ylim((0, max(detection_score) * 1.01))
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)
        plt.plot(detection_score, label='Detection Score', color='black', alpha=1, zorder=2, linewidth=1.8)
        plt.fill_between(order, max(detection_score) * 1.1 * target, color='RED', alpha=0.1, zorder=1)
        plt.axhline(y=th, color='green', linestyle='--', linewidth=1.5, label='Threshold')
        ax.axvspan(0, 200_000 // boost, alpha=0.1, hatch='xx', edgecolor='black', zorder=1, linewidth=0.5)
        ax.axvspan(0, 50_000 // boost, alpha=0.1, hatch='xx', edgecolor='black', zorder=1, linewidth=0.9)
        plt.xlabel('Packet number', fontsize=16)
        plt.ylabel('Detection Score', fontsize=16)
        plt.title(f'{name}', fontsize=21)
        fig.tight_layout()
        plt.savefig(f'{path}/{name}_score.png', dpi=220)
        plt.savefig(f'{path}/{name}_score.eps', dpi=220)
        plt.savefig(f'{path}/{name}_score.pdf', dpi=220)
    except Exception as e:
        plt.close()
        logging.error(f'Error plotting detection score: {e}: {f'{path}/{name}_score'}')
        plot_detection_score(detection_score[::2], target[::2], th, path, name, boost*2)
    finally:
        plt.show()


def plot_auc(best_fpr, best_tpr, best_fpr_smooth, best_tpr_smooth,
             fprs, tprs, fprs_smooth, tprs_smooth,
             path, name):

    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)

    X, Y = np.meshgrid(x, y)

    Z = 2 * (1 - X) * Y / (Y - X + 1 + 1e-9)

    plt.pcolormesh(X, Y, Z, shading='auto', cmap='plasma', alpha=0.3)

    num_contours = 30
    contour = plt.contour(X, Y, Z, levels=num_contours, colors='black', linewidths=0.8, alpha=0.5, linestyles='dashed',
                          label='F1-Scores')
    diag_x = np.linspace(0, 1, 15)
    diag_y = 1 - diag_x
    manual_positions = list(zip(diag_x, diag_y))

    plt.clabel(contour, inline=True, fontsize=6, manual=manual_positions, fmt='%.2f')
    plt.scatter(fprs, tprs, marker='.', color='gray', s=10, label='ROC Curve (base)')
    plt.scatter([best_fpr], [best_tpr], marker='o', color='red', s=15, label='Best F1-Score (base)')
    plt.scatter(fprs_smooth, tprs_smooth, marker='.', color='black', s=10, label='ROC Curve (smoothed)')
    plt.scatter([best_fpr_smooth], [best_tpr_smooth], marker='o', color='green', s=15, label='Best F1-Score (smoothed)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name.upper()}')

    plt.xticks(np.arange(0, 1.025, 0.025), rotation=90)
    plt.yticks(np.arange(0, 1.025, 0.025))

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.xlim((-0., 1))
    plt.ylim((-0., 1))

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='dashed', lw=0.8, label='F1-Scores'),
        Line2D([0], [0], marker='.', color='gray', markersize=5, linestyle='None', label='ROC Curve (base)'),
        Line2D([0], [0], marker='.', color='black', markersize=5, linestyle='None', label='ROC Curve (smoothed)'),
        Line2D([0], [0], marker='o', color='red', markersize=5, linestyle='None', label='Best F1-Score (base)'),
        Line2D([0], [0], marker='o', color='green', markersize=5, linestyle='None', label='Best F1-Score (smoothed)')
    ]

    ax.legend(
        loc='lower right',
        frameon=True,
        facecolor='white',
        edgecolor='black',
        fontsize=8,
        framealpha=1.0,
        fancybox=False,
        bbox_to_anchor=(1, 0.),
        handles=legend_elements
    )
    fig.tight_layout()

    plt.savefig(f'{path}/{name}_auc.png', dpi=220)
    plt.savefig(f'{path}/{name}_auc.eps', dpi=220)
    plt.savefig(f'{path}/{name}_auc.pdf', dpi=220)

    plt.show()


def compute_metrics(detection_raw, detection_smoothing, target_raw, target_smoothing, threshold_raw, threshold_smooth,
                    path, name):
    """
    This function computes the metrics of a given detection.
    :param detection_raw:
    :param detection_smoothing:
    :param target_raw:
    :param target_smoothing:
    :param threshold:
    :return:
    """
    decision_raw = (detection_raw >= threshold_raw).astype(int)
    decision_smooth = (detection_smoothing >= threshold_smooth).astype(int)
    tp_raw = np.sum((decision_raw == 1) & (target_raw == 1))
    fp_raw = np.sum((decision_raw == 1) & (target_raw == 0))
    tn_raw = np.sum((decision_raw == 0) & (target_raw == 0))
    fn_raw = np.sum((decision_raw == 0) & (target_raw == 1))
    precision_raw = tp_raw / (tp_raw + fp_raw + 1e-9)
    recall_raw = tp_raw / (tp_raw + fn_raw + 1e-9)
    f1_raw = 2 * precision_raw * recall_raw / (precision_raw + recall_raw + 1e-9)
    wd_raw, pk_raw = wd_pk(target_raw, decision_raw)
    tp_smooth = np.sum((decision_smooth == 1) & (target_smoothing == 1))
    fp_smooth = np.sum((decision_smooth == 1) & (target_smoothing == 0))
    tn_smooth = np.sum((decision_smooth == 0) & (target_smoothing == 0))
    fn_smooth = np.sum((decision_smooth == 0) & (target_smoothing == 1))
    precision_smooth = tp_smooth / (tp_smooth + fp_smooth + 1e-9)
    recall_smooth = tp_smooth / (tp_smooth + fn_smooth + 1e-9)
    f1_smooth = 2 * precision_smooth * recall_smooth / (precision_smooth + recall_smooth + 1e-9)
    wd_smooth, pk_smooth = wd_pk(target_smoothing, decision_smooth)
    metrics = {
            'Precision Raw': float(precision_raw),
            'Recall Raw': float(recall_raw),
            'F1-Score Raw': float(f1_raw),
            'Window Diff Raw': float(wd_raw),
            'Pk Raw': float(pk_raw),
            'TP Raw': float(tp_raw),
            'FP Raw': float(fp_raw),
            'TN Raw': float(tn_raw),
            'FN Raw': float(fn_raw),
            'Threshold Raw': float(threshold_raw),
            'Precision Smooth': float(precision_smooth),
            'Recall Smooth': float(recall_smooth),
            'F1-Score Smooth': float(f1_smooth),
            'Window Diff Smooth': wd_smooth,
            'Pk Smooth': float(pk_smooth),
            'TP Smooth': float(tp_smooth),
            'FP Smooth': float(fp_smooth),
            'TN Smooth': float(tn_smooth),
            'FN Smooth': float(fn_smooth),
            'Threshold Smooth': float(threshold_smooth)
    }

    np.save(f'{path}/__{name}_values.npy',
            np.array([detection_raw, detection_smoothing, target_raw, target_smoothing]))

    with open(f'{path}/__{name}_metrics.json', 'w') as file:
        json.dump(metrics, file, indent=4)
    return metrics


if __name__ == '__main__':
    for name in NAMES:
        res_path = f'{RES_PATH}{name}/'
        root_path = f'{ROOT_PATH}{name}/'
        target_path = f'{TARGET_PATH}{name}/target.npy'
        numpy_files = get_numpy_files(root_path)
        data_ar, data_or = get_array_and_info(numpy_files)
        target = np.load(target_path, allow_pickle=True).astype(int)
        metrics = list()
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        logging.info(f'Processing {name}...')

        for data, is_ar_name in zip([data_ar, data_or], ['Autoregressive', 'Full Sequence']):
            for key, frame in data.items():
                target = target[:len(frame)]
                if len(frame) - len(target) > 0:
                    logging.info(f"Cutting target len (-300k)...")
                    frame = frame[len(frame) - len(target):]
                if not os.path.exists(f'{res_path}/__{is_ar_name} - {key}_values.npy') and key != 'TARGET':
                    logging.info(f'Framing {is_ar_name} - {key}: {res_path}')
                    tik = time.perf_counter()
                    plot_frame(frame, target, res_path, f'{is_ar_name} - {key}')
                    logging.info(f'Detecting {is_ar_name} - {key}...')
                    detection_score = stat_detector(frame, window_length=10_000)
                    logging.info(f'Computing scores {is_ar_name} - {key}...')
                    fprs, tprs, best_fpr, best_tpr, best_th = compute_scores(detection_score, target)
                    logging.info(f'Plotting scores {is_ar_name} - {key}...')
                    plot_detection_score(detection_score, target, best_th, res_path, f'{is_ar_name} - {key}')
                    logging.info(f'Computing smoothed scores {is_ar_name} - {key}...')
                    detection_smooth, target_smooth = median_smoothing(detection_score, window_length=50_000), label_smoother(target, 10_000)
                    fprs_smooth, tprs_smooth, best_fpr_smooth, best_tpr_smooth, best_th_smooth = compute_scores(detection_smooth, target_smooth)
                    logging.info(f'Plotting scores (smooth) {is_ar_name} - {key}...')
                    plot_detection_score(detection_smooth, target_smooth, best_th_smooth, res_path, f'{is_ar_name} - {key} - SMOOTH')
                    logging.info(f'Plotting AUC {is_ar_name} - {key}...')
                    plot_auc(best_fpr, best_tpr, best_fpr_smooth, best_tpr_smooth,
                             fprs, tprs, fprs_smooth, tprs_smooth,
                             res_path, f'{is_ar_name} - {key}')
                    logging.info(f'Computing metrics for {is_ar_name} - {key}...')
                    metrics.append(compute_metrics(detection_score, detection_smooth, target, target_smooth,
                                                   best_th, best_th_smooth, res_path, f'{is_ar_name} - {key}'))
                    logging.info(f'Metrics: {metrics[-1]}')
                    logging.info(f'Finished {is_ar_name} - {key} in {time.perf_counter() - tik:.2f} seconds.')
                else:
                    logging.info(f'File {is_ar_name} - {key} already exists. Skipping...')

        df = pd.DataFrame(metrics)
        df.to_csv(f'{RES_PATH}/metrics.csv', index=False)
