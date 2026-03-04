"""Combine 3 architectures x 3 datasets attack scatter plots into a 3x3 grid figure.

Usage:
    python combine_grid.py --results_dir results_attack --attack_suffix analytical_tau0.01
    python combine_grid.py --results_dir results_attack --attack_suffix pgd_eps0.0314_alpha0.0078_steps20_gamma0.05
"""

import argparse
import json
import os
import sys

import numpy as np


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def make_grid(results_dir, attack_suffix, output_path,
              archs=('resnet18', 'resnet50', 'wrn_50_2'),
              datasets=('cifar10', 'cifar100', 'tinyimagenet')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    fig, axes = plt.subplots(len(archs), len(datasets), figsize=(18, 16))

    def _r2(x, y):
        x, y = np.array(x), np.array(y)
        return float(np.corrcoef(x, y)[0, 1] ** 2) if len(x) > 1 and np.std(x) > 1e-8 else 0.0

    def _sp(x, y):
        x, y = np.array(x), np.array(y)
        return float(sp_stats.spearmanr(x, y).correlation) if len(x) > 1 else 0.0

    missing = []
    for row, arch in enumerate(archs):
        for col, dname in enumerate(datasets):
            ax = axes[row, col]
            json_path = os.path.join(results_dir, dname,
                                     'attack_{}_{}_{}.json'.format(dname, arch, attack_suffix))
            if not os.path.exists(json_path):
                missing.append(json_path)
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12, color='gray')
                ax.set_title('{} / {}'.format(arch, dname), fontsize=11)
                continue

            data = load_json(json_path)
            raw = data.get('raw', {})
            clean_scores = raw.get('clean_scores', [])
            max_scores = raw.get('max_scores', [])
            min_scores = raw.get('min_scores', [])
            clean_accs = raw.get('clean_accs', [])

            # Scatter points
            ax.scatter(clean_scores, clean_accs, c='tab:blue', label='Clean',
                       alpha=0.6, edgecolors='k', linewidths=0.3, s=25, marker='o')
            ax.scatter(max_scores, clean_accs, c='tab:red', label='Max',
                       alpha=0.6, edgecolors='k', linewidths=0.3, s=25, marker='^')
            ax.scatter(min_scores, clean_accs, c='tab:green', label='Min',
                       alpha=0.6, edgecolors='k', linewidths=0.3, s=25, marker='s')

            # Regression lines
            for xs, color, ls in [(clean_scores, 'tab:blue', '-'),
                                  (max_scores, 'tab:red', '--'),
                                  (min_scores, 'tab:green', ':')]:
                xs_arr = np.array(xs)
                ys_arr = np.array(clean_accs)
                if len(xs_arr) > 1 and np.std(xs_arr) > 1e-8:
                    slope, intercept = np.polyfit(xs_arr, ys_arr, 1)
                    x_line = np.linspace(xs_arr.min(), xs_arr.max(), 100)
                    ax.plot(x_line, slope * x_line + intercept, ls, color=color,
                            linewidth=1.0, alpha=0.7)

            # Combined correlations annotation
            cm_s = clean_scores + max_scores
            cm_a = clean_accs + clean_accs
            cn_s = clean_scores + min_scores
            cn_a = clean_accs + clean_accs
            cmn_s = clean_scores + max_scores + min_scores
            cmn_a = clean_accs + clean_accs + clean_accs

            text = (
                "C:     R\u00b2={:.3f} \u03c1={:.3f}\n"
                "C+Mx:  R\u00b2={:.3f} \u03c1={:.3f}\n"
                "C+Mn:  R\u00b2={:.3f} \u03c1={:.3f}\n"
                "All:   R\u00b2={:.3f} \u03c1={:.3f}"
            ).format(
                _r2(clean_scores, clean_accs), _sp(clean_scores, clean_accs),
                _r2(cm_s, cm_a), _sp(cm_s, cm_a),
                _r2(cn_s, cn_a), _sp(cn_s, cn_a),
                _r2(cmn_s, cmn_a), _sp(cmn_s, cmn_a),
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.02, text, transform=ax.transAxes, fontsize=7,
                    verticalalignment='bottom', bbox=props, family='monospace')

            ax.set_title('{} / {}'.format(arch, dname), fontsize=11)

            if col == 0:
                ax.set_ylabel('Test Accuracy (%)')
            if row == len(archs) - 1:
                ax.set_xlabel('MaNo Score')
            if row == 0 and col == len(datasets) - 1:
                ax.legend(loc='upper right', fontsize=8)

    if missing:
        print('[grid] Warning: missing {} JSON files: {}'.format(len(missing), missing[:3]))

    fig.suptitle('MaNo Attack Comparison Grid ({})'.format(attack_suffix), fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('[grid] Saved 3x3 grid to {}'.format(output_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results_attack', type=str)
    parser.add_argument('--attack_suffix', required=True, type=str,
                        help='Attack suffix, e.g. analytical_tau0.01 or pgd_eps0.0314_alpha0.0078_steps20_gamma0.05')
    parser.add_argument('--output', default=None, type=str,
                        help='Output path (default: results_dir/grid_<suffix>.pdf)')
    args = parser.parse_args()

    output = args.output or os.path.join(args.results_dir, 'grid_{}.pdf'.format(args.attack_suffix))
    make_grid(args.results_dir, args.attack_suffix, output)
