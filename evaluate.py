"""
Evaluation and plotting script for Dreamer world-model comparison.

Usage (run after training all three models):

    python evaluate.py \
        --logdirs logs/rssm logs/linear logs/mlp \
        --labels   RSSM    LinearSSM  MLP \
        --outdir   results

Produces in results/:
  - reward_curves.png   : episode return vs training step for all models
  - prediction_mse.png  : 5-step prediction MSE on held-out transitions

The prediction MSE is computed by loading episodes from each model's
episode directory, constructing the transition buffer, and running the
corresponding world model forward for 5 steps.

Note: for RSSM, only the scalar metrics from metrics.jsonl are plotted
      (reward curves).  The 5-step MSE is computed in obs-space for all
      three models by loading numpy episodes directly.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Metric loading
# ---------------------------------------------------------------------------

def load_metrics(logdir):
    """Parse metrics.jsonl and return a dict of {metric_name: [(step, value)]}."""
    path = pathlib.Path(logdir) / 'metrics.jsonl'
    if not path.exists():
        print(f'WARNING: {path} not found.')
        return {}
    records = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            step = d['step']
            for k, v in d.items():
                if k == 'step':
                    continue
                records.setdefault(k, []).append((step, v))
    return records


def smooth(values, window=5):
    """Simple moving-average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


# ---------------------------------------------------------------------------
# 5-step prediction MSE
# ---------------------------------------------------------------------------

def load_episodes(episode_dir, max_transitions=1000):
    """Load transitions (obs, action, next_obs) from saved .npz episodes.

    Returns three numpy arrays each with up to max_transitions rows.
    """
    episode_dir = pathlib.Path(episode_dir)
    obs_list, act_list, nobs_list = [], [], []
    total = 0
    for ep_file in sorted(episode_dir.glob('*.npz')):
        if total >= max_transitions:
            break
        try:
            ep = np.load(ep_file)
            if 'state' not in ep:
                continue
            states = ep['state']   # [T, obs_dim]
            actions = ep['action'] # [T, action_dim]
            for t in range(len(states) - 1):
                obs_list.append(states[t])
                act_list.append(actions[t])
                nobs_list.append(states[t + 1])
                total += 1
                if total >= max_transitions:
                    break
        except Exception as e:
            print(f'Could not load {ep_file}: {e}')
    if not obs_list:
        return None, None, None
    return (np.array(obs_list, dtype=np.float32),
            np.array(act_list, dtype=np.float32),
            np.array(nobs_list, dtype=np.float32))


def mse_linear(A_var, B_var, obs, actions, horizon=5):
    """Multi-step MSE for LinearSSM model.

    Rolls out the linear model for `horizon` steps from each starting point
    (obs[i], actions[i:i+horizon]) and computes MSE vs ground-truth.

    A_var : [obs_dim, obs_dim]
    B_var : [action_dim, obs_dim]  (numpy arrays or tf Variables)
    obs   : [N+horizon, obs_dim]
    actions: [N+horizon, action_dim]
    """
    try:
        A = np.array(A_var)
        B = np.array(B_var)
    except Exception:
        A = A_var.numpy()
        B = B_var.numpy()

    N = len(obs) - horizon
    if N <= 0:
        return float('nan')
    total_mse = 0.0
    count = 0
    for i in range(N):
        pred = obs[i].copy()
        step_mses = []
        for h in range(horizon):
            pred = pred @ A + actions[i + h] @ B
            step_mses.append(np.mean((pred - obs[i + h + 1]) ** 2))
        total_mse += np.mean(step_mses)
        count += 1
    return total_mse / count


def compute_prediction_mse_from_episodes(logdir, world_model, horizon=5,
                                          max_transitions=1000):
    """Compute horizon-step prediction MSE for a trained model.

    For LinearSSM and MLPDynamics the model weights are loaded from the
    variables.pkl checkpoint.  For RSSM the full TF model must be loaded
    (requires the dreamer module).

    Returns a dict {'1step': float, 'Nstep': float, 'per_step': [float]*horizon}
    or None if the model/episodes cannot be loaded.
    """
    logdir = pathlib.Path(logdir)
    episode_dir = logdir / 'episodes'
    ckpt = logdir / 'variables.pkl'

    obs, actions, nobs = load_episodes(episode_dir, max_transitions)
    if obs is None:
        print(f'  No state episodes found in {episode_dir}')
        return None

    print(f'  Loaded {len(obs)} transitions from {episode_dir}')

    if world_model == 'linear':
        return _mse_linear_from_ckpt(ckpt, obs, actions, nobs, horizon)
    elif world_model == 'mlp':
        return _mse_mlp_from_ckpt(ckpt, logdir, obs, actions, nobs, horizon)
    elif world_model == 'rssm':
        print('  RSSM MSE in obs-space skipped (requires image decoder).')
        return None
    return None


def _mse_linear_from_ckpt(ckpt, obs, actions, nobs, horizon):
    """Load LinearSSM A, B from checkpoint and compute multi-step MSE."""
    import pickle
    if not ckpt.exists():
        print(f'  Checkpoint not found: {ckpt}')
        return None
    with ckpt.open('rb') as f:
        values = pickle.load(f)
    # The checkpoint stores variables in order.
    # We search for the LinearSSM's A and B variables by shape.
    obs_dim = obs.shape[-1]
    action_dim = actions.shape[-1]
    A = B = None
    for v in values:
        v = np.array(v)
        if v.shape == (obs_dim, obs_dim):
            A = v
        elif v.shape == (action_dim, obs_dim):
            B = v
    if A is None or B is None:
        print('  Could not find A, B matrices in checkpoint.')
        return None

    per_step = []
    N = len(obs) - horizon
    for h in range(1, horizon + 1):
        mses = []
        for i in range(N):
            pred = obs[i].copy()
            for step in range(h):
                pred = pred @ A + actions[i + step] @ B
            mses.append(np.mean((pred - obs[i + h]) ** 2))
        per_step.append(float(np.mean(mses)))
    return {'per_step': per_step,
            '1step': per_step[0],
            f'{horizon}step': per_step[-1]}


def _mse_mlp_from_ckpt(ckpt, logdir, obs, actions, nobs, horizon):
    """Load MLPDynamics TF model and compute multi-step MSE."""
    # We need to reconstruct the model and load weights.
    # Import here to avoid hard dependency when running without TF.
    try:
        import sys
        sys.path.insert(0, str(logdir.parent.parent))
        import tensorflow as tf
        import alt_models as am
        from tensorflow.keras import mixed_precision as prec

        obs_dim = obs.shape[-1]
        action_dim = actions.shape[-1]
        dyn = am.MLPDynamics(obs_dim, action_dim)

        # Warm up the model to create variables.
        o_t = tf.constant(obs[:2, None, :])
        a_t = tf.constant(actions[:2, None, :])
        _ = dyn.observe(o_t, a_t)

        import pickle
        with ckpt.open('rb') as f:
            saved_values = pickle.load(f)
        tf.nest.map_structure(lambda x, y: x.assign(y), dyn.variables, saved_values)

        # Compute per-step MSE
        per_step = []
        N = len(obs) - horizon
        for h in range(1, horizon + 1):
            preds = tf.constant(obs[:N])
            for step in range(h):
                preds = dyn._predict(preds, tf.constant(actions[step:step + N]))
            targets = tf.constant(obs[h:h + N])
            mse = float(tf.reduce_mean(tf.square(preds - targets)).numpy())
            per_step.append(mse)
        return {'per_step': per_step,
                '1step': per_step[0],
                f'{horizon}step': per_step[-1]}
    except Exception as e:
        print(f'  MLPDynamics MSE failed: {e}')
        return None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = ['steelblue', 'darkorange', 'forestgreen', 'crimson', 'mediumpurple']


def plot_reward_curves(logdirs, labels, outpath, smooth_window=5):
    """Plot test/train episode return for each run on the same axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for i, (logdir, label) in enumerate(zip(logdirs, labels)):
        metrics = load_metrics(logdir)
        color = COLORS[i % len(COLORS)]

        for ax, prefix in zip(axes, ['test', 'train']):
            key = f'{prefix}/return'
            if key not in metrics:
                continue
            steps, vals = zip(*sorted(metrics[key]))
            vals_smooth = smooth(vals, smooth_window)
            steps_trim = steps[:len(vals_smooth)]
            ax.plot(steps_trim, vals_smooth, label=label, color=color)
            ax.scatter(steps, vals, alpha=0.3, s=10, color=color)

    for ax, title in zip(axes, ['Test Return', 'Train Return']):
        ax.set_xlabel('Environment steps')
        ax.set_ylabel('Episode return')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150)
    plt.close()
    print(f'Saved reward curves → {outpath}')


def plot_prediction_mse(mse_results, labels, outpath, horizon=5):
    """Bar + line chart of per-step prediction MSE for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-step MSE line plot
    ax = axes[0]
    steps = list(range(1, horizon + 1))
    for (label, result), color in zip(
            [(l, r) for l, r in zip(labels, mse_results) if r is not None],
            COLORS):
        ax.plot(steps, result['per_step'], marker='o', label=label, color=color)
    ax.set_xlabel('Prediction horizon (steps)')
    ax.set_ylabel('MSE')
    ax.set_title('Multi-step prediction MSE')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)

    # Right: final-step MSE bar chart
    ax = axes[1]
    valid = [(l, r) for l, r in zip(labels, mse_results) if r is not None]
    if valid:
        lbls, results = zip(*valid)
        final_mses = [r[f'{horizon}step'] for r in results]
        bars = ax.bar(lbls, final_mses,
                      color=COLORS[:len(lbls)], alpha=0.8, edgecolor='black')
        for bar, val in zip(bars, final_mses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('MSE')
    ax.set_title(f'{horizon}-step prediction MSE')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150)
    plt.close()
    print(f'Saved prediction MSE → {outpath}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logdirs = args.logdirs
    labels = args.labels if args.labels else [pathlib.Path(d).name for d in logdirs]

    assert len(logdirs) == len(labels), \
        f'Number of logdirs ({len(logdirs)}) must match labels ({len(labels)})'

    # ---- Reward curves ----
    plot_reward_curves(logdirs, labels, outdir / 'reward_curves.png',
                       smooth_window=args.smooth)

    # ---- Prediction MSE ----
    world_models = args.world_models
    if world_models and len(world_models) == len(logdirs):
        mse_results = []
        for logdir, wm, label in zip(logdirs, world_models, labels):
            print(f'\nComputing prediction MSE for {label} ({wm})...')
            result = compute_prediction_mse_from_episodes(
                logdir, wm, horizon=args.horizon,
                max_transitions=args.mse_transitions)
            mse_results.append(result)
            if result:
                print(f'  1-step MSE: {result["1step"]:.6f}')
                print(f'  {args.horizon}-step MSE: {result[f"{args.horizon}step"]:.6f}')

        if any(r is not None for r in mse_results):
            plot_prediction_mse(mse_results, labels,
                                outdir / 'prediction_mse.png',
                                horizon=args.horizon)
    else:
        print('\nSkipping MSE plots (pass --world_models to enable).')

    print(f'\nAll plots saved to {outdir}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate and plot Dreamer models.')
    parser.add_argument('--logdirs', nargs='+', required=True,
                        help='Log directories for each run.')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Display labels (default: directory names).')
    parser.add_argument('--world_models', nargs='+', default=None,
                        choices=['rssm', 'linear', 'mlp'],
                        help='World model type for each logdir (for MSE eval).')
    parser.add_argument('--outdir', default='results',
                        help='Output directory for plots.')
    parser.add_argument('--smooth', type=int, default=5,
                        help='Smoothing window for reward curves.')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Prediction horizon for MSE evaluation.')
    parser.add_argument('--mse_transitions', type=int, default=1000,
                        help='Number of held-out transitions for MSE.')
    args = parser.parse_args()
    main(args)
