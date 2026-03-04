"""MaNo Attack Comparison Pipeline.

Runs Clean MaNo, Norm Maximization Attack, and Norm Minimization Attack
side-by-side for each corruption/severity and produces comparison results.

Usage:
    python main_attack.py --arch resnet18 --dataname cifar10 --severity -1 \
        --gpu 0 --cifar_data_path ./datasets/cifar10 \
        --cifar_corruption_path ./datasets/CIFAR-10-C \
        --attack_eps 0.031 --attack_steps 20
"""

import argparse
import os
import numpy as np
import time
import torch

from models.utils import get_model, get_imagenet_model
from data.utils import build_dataloader
from algs.mano import MaNo
from algs.mano_attack import (
    MaNoMaxAttack, MaNoMinAttack,
    MaNoMaxAttackFast, MaNoMinAttackFast,
)
from utils.logging_utils import (
    init_wandb, log_iteration, log_summary,
    log_scatter_to_wandb, save_results_json,
    log_artifacts_to_wandb, finish_wandb,
)

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='MaNo Attack Comparison')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--gpu', type=str, default=None)
parser.add_argument('--cifar_data_path', default='../datasets/Cifar10', type=str)
parser.add_argument('--cifar_corruption_path', default='../datasets/Cifar10/CIFAR-10-C', type=str)
parser.add_argument('--corruption', default='all', type=str)
parser.add_argument('--severity', default=0, type=int)
parser.add_argument('--dataname', default='cifar10', type=str)
parser.add_argument('--num_classes', default=10, type=int)
parser.add_argument('--num_samples', default=50000, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--threshold', default=0.5, type=float)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--norm_type', default=4, type=int)
parser.add_argument('--source', default='None', type=str)

# Attack parameters
parser.add_argument('--attack_mode', default='analytical', type=str,
                    choices=['pgd', 'fgsm', 'analytical'],
                    help='Attack method: pgd (iterative), fgsm (1-step), analytical (zero-cost)')
parser.add_argument('--attack_eps', default=None, type=float,
                    help='L-inf budget in pixel space (default: 8/255)')
parser.add_argument('--attack_alpha', default=None, type=float,
                    help='PGD step size in pixel space (default: 2/255)')
parser.add_argument('--attack_steps', default=20, type=int,
                    help='Number of PGD steps')
parser.add_argument('--attack_gamma', default=0.05, type=float,
                    help='Margin for norm minimization soft targets')
parser.add_argument('--attack_tau', default=0.01, type=float,
                    help='Temperature for analytical attack (default: 0.01)')

# wandb & output
parser.add_argument('--wandb_project', default=None, type=str)
parser.add_argument('--wandb_group', default=None, type=str)
parser.add_argument('--wandb_tags', default=None, type=str,
                    help='Comma-separated tags for wandb')
parser.add_argument('--results_dir', default='results_attack', type=str)

args = vars(parser.parse_args())

# Defaults for eps/alpha; FGSM overrides
if args['attack_eps'] is None:
    args['attack_eps'] = 8 / 255
if args['attack_alpha'] is None:
    args['attack_alpha'] = 2 / 255
if args['attack_mode'] == 'fgsm':
    args['attack_steps'] = 1
    args['attack_alpha'] = args['attack_eps']  # single full-step

# Select attack classes based on mode
if args['attack_mode'] == 'analytical':
    MaxAttackCls = MaNoMaxAttackFast
    MinAttackCls = MaNoMinAttackFast
else:  # pgd or fgsm (fgsm = pgd with steps=1)
    MaxAttackCls = MaNoMaxAttack
    MinAttackCls = MaNoMinAttack

if args["gpu"] is not None:
    device = torch.device(f"cuda:{args['gpu']}")
else:
    device = torch.device('cpu')

# ── Dataset config ───────────────────────────────────────────────────
num_class_dict = {
    "cifar10": 10, "cifar100": 100, "tinyimagenet": 200, "pacs": 7,
    "imagenet": 1000, "office_home": 65, "wilds_rr1": 1139,
    "entity30": 30, "entity13": 13, "living17": 17, "nonliving26": 26,
    "domainnet": 345,
}
args["num_classes"] = num_class_dict[args["dataname"]]


def correlation2(v1, v2):
    return float(np.corrcoef(v1, v2)[0, 1] ** 2)


def spearman(v1, v2):
    from scipy import stats
    return float(stats.spearmanr(v1, v2).correlation)


def make_comparison_scatter(clean_scores, max_scores, min_scores, clean_accs,
                            color_labels, title, save_path):
    """Three-method comparison scatter plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(clean_scores, clean_accs, c='tab:blue', label='Clean MaNo',
               alpha=0.6, edgecolors='k', linewidths=0.3, s=40, marker='o')
    ax.scatter(max_scores, clean_accs, c='tab:red', label='Max Attack',
               alpha=0.6, edgecolors='k', linewidths=0.3, s=40, marker='^')
    ax.scatter(min_scores, clean_accs, c='tab:green', label='Min Attack',
               alpha=0.6, edgecolors='k', linewidths=0.3, s=40, marker='s')

    # Regression lines
    for xs, color, ls in [(clean_scores, 'tab:blue', '-'),
                          (max_scores, 'tab:red', '--'),
                          (min_scores, 'tab:green', ':')]:
        xs_arr = np.array(xs)
        ys_arr = np.array(clean_accs)
        if len(xs_arr) > 1 and np.std(xs_arr) > 1e-8:
            slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
            x_line = np.linspace(xs_arr.min(), xs_arr.max(), 100)
            ax.plot(x_line, slope * x_line + intercept, ls, color=color, linewidth=1.2, alpha=0.7)

    # Annotation box
    def _r2(x, y):
        return float(np.corrcoef(x, y)[0, 1] ** 2) if len(x) > 1 and np.std(x) > 1e-8 else 0.0
    def _sp(x, y):
        return float(sp_stats.spearmanr(x, y).correlation) if len(x) > 1 else 0.0

    # Combined data points
    cm_s = list(clean_scores) + list(max_scores)
    cm_a = list(clean_accs) + list(clean_accs)
    cn_s = list(clean_scores) + list(min_scores)
    cn_a = list(clean_accs) + list(clean_accs)
    cmn_s = list(clean_scores) + list(max_scores) + list(min_scores)
    cmn_a = list(clean_accs) + list(clean_accs) + list(clean_accs)

    text = (
        "Clean:         R\u00b2={:.4f}  \u03c1={:.4f}\n"
        "Clean+Max:     R\u00b2={:.4f}  \u03c1={:.4f}\n"
        "Clean+Min:     R\u00b2={:.4f}  \u03c1={:.4f}\n"
        "Clean+Max+Min: R\u00b2={:.4f}  \u03c1={:.4f}"
    ).format(
        _r2(clean_scores, clean_accs), _sp(clean_scores, clean_accs),
        _r2(cm_s, cm_a), _sp(cm_s, cm_a),
        _r2(cn_s, cn_a), _sp(cn_s, cn_a),
        _r2(cmn_s, cmn_a), _sp(cmn_s, cmn_a),
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props, family='monospace')

    ax.set_xlabel('MaNo Score')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(title)
    ax.legend(loc='upper right')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    # Also save as PNG for wandb image logging
    png_path = os.path.splitext(save_path)[0] + '.png'
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('[scatter] Saved to {} and {}'.format(save_path, png_path))


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataname = args['dataname']
    arch = args['arch']
    results_dir = args['results_dir']

    # wandb init
    wandb_tags = args['wandb_tags'].split(',') if args['wandb_tags'] else None
    args['alg'] = 'attack_{}'.format(args['attack_mode'])  # for wandb run name
    wandb_run = init_wandb(args, project=args['wandb_project'],
                           group=args['wandb_group'], tags=wandb_tags)

    # Corruption list
    if dataname in ('cifar10', 'cifar100'):
        corruption_list = [
            "brightness", "contrast", "defocus_blur", "elastic_transform",
            "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur",
            "impulse_noise", "jpeg_compression", "motion_blur", "pixelate",
            "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur",
        ]
        max_severity = 5
    elif dataname == 'tinyimagenet':
        corruption_list = [
            "brightness", "contrast", "defocus_blur", "elastic_transform",
            "fog", "frost", "gaussian_noise", "glass_blur", "impulse_noise",
            "jpeg_compression", "motion_blur", "pixelate", "shot_noise",
            "snow", "zoom_blur",
        ]
        max_severity = 5
    elif dataname == 'pacs':
        corruption_list = ['art_painting', 'cartoon', 'photo', 'sketch_pacs']
        max_severity = 1
    elif dataname == 'office_home':
        corruption_list = ['Art', 'Clipart', 'Product', 'Real_World']
        max_severity = 1
    elif dataname == 'domainnet':
        corruption_list = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        max_severity = 1
    else:
        raise ValueError('Unsupported dataset: {}'.format(dataname))

    is_domain = dataname in ('pacs', 'office_home', 'domainnet')

    # Load base model once
    if ('imagenet' in dataname) and (args['num_classes'] == 1000):
        base_model = get_imagenet_model(arch, args['num_classes'], args['seed']).to(device)
    elif not is_domain:
        save_dir = './checkpoints/{}_{}'.format(dataname, arch)
        base_model = get_model(arch, args['num_classes'], args['seed']).to(device)
        base_model.load_state_dict(torch.load('{}/base_model.pt'.format(save_dir), map_location=device, weights_only=False))
    else:
        base_model = None  # domain branch loads per source

    # Attack info string (for logging) and file suffix (for filenames)
    mode = args['attack_mode']
    if mode == 'analytical':
        atk_info = 'mode=analytical, tau={}'.format(args['attack_tau'])
        atk_suffix = 'analytical_tau{}'.format(args['attack_tau'])
    elif mode == 'fgsm':
        atk_info = 'mode=fgsm, eps={:.4f}'.format(args['attack_eps'])
        atk_suffix = 'fgsm_eps{:.4f}'.format(args['attack_eps'])
    else:  # pgd
        atk_info = 'mode={}, eps={:.4f}, alpha={:.4f}, steps={}, gamma={}'.format(
            mode, args['attack_eps'], args['attack_alpha'], args['attack_steps'], args['attack_gamma'])
        atk_suffix = 'pgd_eps{:.4f}_alpha{:.4f}_steps{}_gamma{}'.format(
            args['attack_eps'], args['attack_alpha'], args['attack_steps'], args['attack_gamma'])

    # Accumulators
    clean_scores, max_scores, min_scores = [], [], []
    clean_accs, max_adv_accs, min_adv_accs = [], [], []
    clean_times, max_times, min_times = [], [], []
    color_labels = []
    iteration_data = []

    if is_domain:
        total_iters = len(corruption_list) * (len(corruption_list) - 1)
    else:
        total_iters = len(corruption_list) * max_severity

    print('=' * 90)
    print('MaNo Attack Comparison | {} | {} | {}'.format(dataname, arch, device))
    print('Attack config: {}'.format(atk_info))
    print('Corruptions: {}, Total iters: {}'.format(len(corruption_list), total_iters))
    print('=' * 90)
    header = '{:<6} {:<22} {:<4} | {:>10} {:>10} {:>10} | {:>8} {:>10} {:>10}'.format(
        'Iter', 'Corruption', 'Sev', 'Clean', 'MaxAtk', 'MinAtk', 'Acc%', 'MaxAdv%', 'MinAdv%')
    print(header)
    print('-' * 90)

    wall_start = time.time()
    iter_idx = [0]

    def run_iteration(corruption, severity, source=None):
        """Run clean + max + min for one (corruption, severity) pair."""
        iter_idx[0] += 1

        args['corruption'] = corruption
        args['severity'] = severity
        if source is not None:
            args['source'] = source

        val_loader = build_dataloader(dataname, args, skip_resize=True)

        # Determine base model for domain datasets
        bm = base_model
        if is_domain:
            save_dir = './checkpoints/{}_{}'.format(source, arch)
            bm = get_model(arch, args['num_classes'], args['seed']).to(device)
            bm.load_state_dict(torch.load('{}/base_model.pt'.format(save_dir), map_location=device, weights_only=False))

        # ── Clean MaNo ──
        t0 = time.time()
        alg_clean = MaNo(val_loader, device, args, base_model=bm)
        c_score = float(alg_clean.evaluate())
        c_time = time.time() - t0
        c_acc = float(alg_clean.test())
        phi = float(alg_clean.phi)

        # ── Max Attack ──
        t0 = time.time()
        alg_max = MaxAttackCls(val_loader, device, args, base_model=bm)
        m_score = float(alg_max.evaluate())
        m_time = time.time() - t0
        m_adv_acc = float(alg_max.adv_acc)

        # ── Min Attack ──
        t0 = time.time()
        alg_min = MinAttackCls(val_loader, device, args, base_model=bm)
        n_score = float(alg_min.evaluate())
        n_time = time.time() - t0
        n_adv_acc = float(alg_min.adv_acc)

        # Accumulate
        clean_scores.append(c_score)
        max_scores.append(m_score)
        min_scores.append(n_score)
        clean_accs.append(c_acc)
        max_adv_accs.append(m_adv_acc)
        min_adv_accs.append(n_adv_acc)
        clean_times.append(c_time)
        max_times.append(m_time)
        min_times.append(n_time)
        color_labels.append(corruption)

        elapsed = time.time() - wall_start
        eta = elapsed / iter_idx[0] * (total_iters - iter_idx[0])
        label = '{}:{}'.format(source, corruption) if source else corruption

        print('{:<6} {:<22} {:<4} | {:>10.4f} {:>10.4f} {:>10.4f} | {:>7.2f}% {:>9.2f}% {:>9.2f}%  ({:.0f}s/{:.0f}s)'.format(
            '{}/{}'.format(iter_idx[0], total_iters), label, severity,
            c_score, m_score, n_score, c_acc, m_adv_acc, n_adv_acc, elapsed, eta))

        iteration_data.append({
            'iter': iter_idx[0],
            'corruption': corruption,
            'severity': severity,
            'source': source,
            'phi': phi,
            'clean_score': c_score, 'max_score': m_score, 'min_score': n_score,
            'clean_acc': c_acc, 'max_adv_acc': m_adv_acc, 'min_adv_acc': n_adv_acc,
            'clean_time': c_time, 'max_time': m_time, 'min_time': n_time,
        })

        # wandb per-iteration logging
        if wandb_run is not None:
            wandb_run.log({
                'iter': iter_idx[0],
                'corruption': corruption,
                'severity': severity,
                'phi': phi,
                'clean_score': c_score, 'max_score': m_score, 'min_score': n_score,
                'clean_acc': c_acc, 'max_adv_acc': m_adv_acc, 'min_adv_acc': n_adv_acc,
                'clean_time': c_time, 'max_time': m_time, 'min_time': n_time,
            })

    # ── Iterate ──────────────────────────────────────────────────────
    if is_domain:
        for corruption in corruption_list:
            for source in corruption_list:
                if corruption != source:
                    run_iteration(corruption, 1, source=source)
    else:
        for corruption in corruption_list:
            for severity in range(1, max_severity + 1):
                run_iteration(corruption, severity)

    total_wall = time.time() - wall_start

    # ── Summary ──────────────────────────────────────────────────────
    # Clean only
    c_r2 = correlation2(clean_scores, clean_accs)
    c_sp = spearman(clean_scores, clean_accs)

    # Clean + Max (combined data points)
    cm_scores_all = clean_scores + max_scores
    cm_accs_all = clean_accs + clean_accs
    cm_r2 = correlation2(cm_scores_all, cm_accs_all)
    cm_sp = spearman(cm_scores_all, cm_accs_all)

    # Clean + Min (combined data points)
    cn_scores_all = clean_scores + min_scores
    cn_accs_all = clean_accs + clean_accs
    cn_r2 = correlation2(cn_scores_all, cn_accs_all)
    cn_sp = spearman(cn_scores_all, cn_accs_all)

    # Clean + Max + Min (all combined)
    cmn_scores_all = clean_scores + max_scores + min_scores
    cmn_accs_all = clean_accs + clean_accs + clean_accs
    cmn_r2 = correlation2(cmn_scores_all, cmn_accs_all)
    cmn_sp = spearman(cmn_scores_all, cmn_accs_all)

    print('=' * 90)
    print('Summary: {} | {} | {} | {}'.format(dataname, arch, atk_info, device))
    print('-' * 90)
    print('{:<18} {:>8} {:>10}'.format('Combination', 'R\u00b2', 'Spearman'))
    print('{:<18} {:>8.4f} {:>10.4f}'.format('Clean', c_r2, c_sp))
    print('{:<18} {:>8.4f} {:>10.4f}'.format('Clean+Max', cm_r2, cm_sp))
    print('{:<18} {:>8.4f} {:>10.4f}'.format('Clean+Min', cn_r2, cn_sp))
    print('{:<18} {:>8.4f} {:>10.4f}'.format('Clean+Max+Min', cmn_r2, cmn_sp))
    print('-' * 90)
    print('{:<14} {:>11} {:>10} {:>10}'.format(
        'Method', 'Mean Score', 'Mean Time', 'AdvAcc%'))
    print('{:<14} {:>11.4f} {:>9.2f}s {:>10}'.format(
        'Clean', np.mean(clean_scores), np.mean(clean_times), '-'))
    print('{:<14} {:>11.4f} {:>9.2f}s {:>9.2f}%'.format(
        'Max Attack', np.mean(max_scores), np.mean(max_times), np.mean(max_adv_accs)))
    print('{:<14} {:>11.4f} {:>9.2f}s {:>9.2f}%'.format(
        'Min Attack', np.mean(min_scores), np.mean(min_times), np.mean(min_adv_accs)))
    print('-' * 90)
    print('Score shift  | Max: {:+.4f} ({:+.1f}%)  Min: {:+.4f} ({:+.1f}%)'.format(
        np.mean(max_scores) - np.mean(clean_scores),
        100 * (np.mean(max_scores) - np.mean(clean_scores)) / (np.mean(clean_scores) + 1e-10),
        np.mean(min_scores) - np.mean(clean_scores),
        100 * (np.mean(min_scores) - np.mean(clean_scores)) / (np.mean(clean_scores) + 1e-10)))
    print('Total wall time: {:.1f}s'.format(total_wall))
    print('=' * 90)

    # ── Scatter plot ─────────────────────────────────────────────────
    scatter_path = os.path.join(results_dir, dataname,
                                'attack_{}_{}_{}.pdf'.format(dataname, arch, atk_suffix))
    make_comparison_scatter(
        clean_scores, max_scores, min_scores, clean_accs, color_labels,
        title='MaNo Attack | {} | {} | {}'.format(dataname, arch, atk_info),
        save_path=scatter_path,
    )

    # ── JSON results ─────────────────────────────────────────────────
    json_path = os.path.join(results_dir, dataname,
                             'attack_{}_{}_{}.json'.format(dataname, arch, atk_suffix))
    save_results_json({
        'dataname': dataname, 'arch': arch,
        'attack_config': {
            'mode': args['attack_mode'],
            'eps': args['attack_eps'], 'alpha': args['attack_alpha'],
            'steps': args['attack_steps'], 'gamma': args['attack_gamma'],
            'tau': args['attack_tau'],
        },
        'summary': {
            'clean':         {'mean_score': float(np.mean(clean_scores)), 'R2': c_r2, 'spearman': c_sp},
            'clean_max':     {'R2': cm_r2, 'spearman': cm_sp},
            'clean_min':     {'R2': cn_r2, 'spearman': cn_sp},
            'clean_max_min': {'R2': cmn_r2, 'spearman': cmn_sp},
            'max_attack':    {'mean_score': float(np.mean(max_scores)),
                              'mean_adv_acc': float(np.mean(max_adv_accs))},
            'min_attack':    {'mean_score': float(np.mean(min_scores)),
                              'mean_adv_acc': float(np.mean(min_adv_accs))},
        },
        'raw': {
            'clean_scores': clean_scores, 'max_scores': max_scores, 'min_scores': min_scores,
            'clean_accs': clean_accs, 'max_adv_accs': max_adv_accs, 'min_adv_accs': min_adv_accs,
            'color_labels': color_labels,
        },
        'total_wall_time': total_wall,
        'iterations': iteration_data,
    }, json_path)

    # ── wandb summary + artifacts + finish ────────────────────────────
    if wandb_run is not None:
        wandb_run.summary['clean_R2'] = c_r2
        wandb_run.summary['clean_spearman'] = c_sp
        wandb_run.summary['clean_max_R2'] = cm_r2
        wandb_run.summary['clean_max_spearman'] = cm_sp
        wandb_run.summary['clean_min_R2'] = cn_r2
        wandb_run.summary['clean_min_spearman'] = cn_sp
        wandb_run.summary['clean_max_min_R2'] = cmn_r2
        wandb_run.summary['clean_max_min_spearman'] = cmn_sp
        wandb_run.summary['clean_mean_score'] = float(np.mean(clean_scores))
        wandb_run.summary['max_mean_score'] = float(np.mean(max_scores))
        wandb_run.summary['min_mean_score'] = float(np.mean(min_scores))
        wandb_run.summary['total_wall_time'] = total_wall
    log_scatter_to_wandb(wandb_run, scatter_path)
    log_artifacts_to_wandb(wandb_run, json_path, scatter_path)
    finish_wandb(wandb_run)
