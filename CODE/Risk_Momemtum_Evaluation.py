import os, glob, argparse, json, math, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (roc_auc_score, average_precision_score,accuracy_score, roc_curve,precision_recall_curve, confusion_matrix)
import torch
import torch.nn as nn
import torch.nn.functional as F
warnings.filterwarnings("ignore")


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu  = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu    = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho   = nn.Parameter(torch.Tensor(out_features))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -3.0)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + F.softplus(self.weight_rho) * torch.randn_like(self.weight_mu)
            b = self.bias_mu   + F.softplus(self.bias_rho)   * torch.randn_like(self.bias_mu)
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x.mean(dim=2), self.weight)


class UStringModel(nn.Module):
    def __init__(self, feat_dim=4096, proj_dim=256, gcn_dim=512, hidden=64, T=50):
        super().__init__()
        self.phi_x = nn.Sequential(nn.Linear(feat_dim, proj_dim), nn.ReLU(inplace=True))
        self.enc_gcn1 = SimpleGCNLayer(proj_dim, gcn_dim)
        self.enc_gcn2 = SimpleGCNLayer(proj_dim, gcn_dim)
        self.self_aggregation = nn.Linear(1, T, bias=False)
        self.predictor = nn.ModuleDict({
            'l1': BayesianLinear(feat_dim + gcn_dim + proj_dim, hidden),
            'l2': BayesianLinear(hidden, 2),
        })
        self.predictor_aux = nn.ModuleDict({
            'dense1': nn.Linear(gcn_dim, hidden),
            'dense2': nn.Linear(hidden, 2),
        })

    @staticmethod
    def reparameterise(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
    def forward_one_pass(self, x):
        x_proj   = self.phi_x(x)
        gcn_mu   = self.enc_gcn1(x_proj)
        gcn_lv   = self.enc_gcn2(x_proj)
        z        = self.reparameterise(gcn_mu, gcn_lv)
        feat     = torch.cat([x.mean(dim=2), z, x_proj.mean(dim=2)], dim=-1)
        h        = F.relu(self.predictor['l1'](feat))
        logits   = self.predictor['l2'](h)
        return F.softmax(logits, dim=-1)[:, :, 0]

    def forward(self, x, mc_samples=10):
        self.train()
        curves = torch.stack([self.forward_one_pass(x) for _ in range(mc_samples)])
        mean_curve = curves.mean(0)
        return mean_curve.mean(1), mean_curve


def load_model(ckpt_path, device):
    model = UStringModel().to(device)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("[WARN] No checkpoint — using random weights")
        return model, False
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        for k in ("model_state_dict","model_state","state_dict","model","net"):
            if k in raw:
                raw = raw[k]; break
    md = model.state_dict()
    matched = {k: v for k, v in raw.items() if k in md and v.shape == md[k].shape}
    md.update(matched)
    model.load_state_dict(md, strict=False)
    pct = 100 * len(matched) / len(md)
    print(f"Checkpoint loaded: {len(matched)}/{len(md)} tensors ({pct:.0f}%)")
    return model, pct > 50

def load_test_data(data_root):
    X_all, y_all, names = [], [], []
    for label, subdir in [(1,"test_positive"), (0,"test_negative")]:
        d = os.path.join(data_root, subdir)
        files = sorted(glob.glob(os.path.join(d, "*.npz")))
        print(f"  {subdir}: {len(files)} files")
        for fp in tqdm(files, ncols=70):
            try:
                feat = np.load(fp, allow_pickle=True)["data"].astype(np.float32)
                X_all.append(feat); y_all.append(label)
                names.append(os.path.basename(fp))
            except Exception as e:
                print(f"  skip {os.path.basename(fp)}: {e}")
    X = np.stack(X_all)
    y = np.array(y_all)
    print(f"  Total: {len(y)}  (pos={y.sum()}, neg={(y==0).sum()})")
    return X, y, names

@torch.no_grad()
def run_inference(model, X, mc_samples=10, batch_size=8, seed=42, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    scores, curves = [], []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i+batch_size]).to(device)
        torch.manual_seed(seed + i)
        with torch.enable_grad():
            s, c = model(xb, mc_samples)
        scores.append(s.detach().cpu().numpy())
        curves.append(c.detach().cpu().numpy())
    return np.concatenate(scores), np.concatenate(curves)

def compute_ema(signal, alpha=0.3):
    N, T = signal.shape
    ema = np.zeros_like(signal)
    ema[:, 0] = signal[:, 0]
    for t in range(1, T):
        ema[:, t] = alpha * signal[:, t] + (1 - alpha) * ema[:, t-1]
    return ema


def apply_risk_momentum_correct(risk_curves, gamma=0.3, ema_alpha=0.3,base_threshold=0.5, delta=0.15):
    R = risk_curves.copy()                          
    R_prev   = np.concatenate([R[:, :1], R[:, :-1]], axis=1)
    raw_M    = R - R_prev                           
    M_smooth = compute_ema(raw_M, alpha=ema_alpha)  
    M_pos    = np.clip(M_smooth, 0, None)           
    R_star   = np.clip(R + gamma * M_pos, 0, 1)    
    thresholds = base_threshold - delta * M_pos     
    thresholds = np.clip(thresholds, 0.2, base_threshold)
    T = R_star.shape[1]
    weights = np.ones(T)
    weights[int(T*0.4):] = 3.0          
    weights = weights / weights.sum()
    m2_scores = (R_star * weights).sum(axis=1)# (N,)
    return R_star, M_smooth, M_pos, thresholds, m2_scores


def compute_tta_adaptive(risk_curves, labels, thresholds, fps=10.0):
    T    = risk_curves.shape[1]
    ttas = []
    for i, lbl in enumerate(labels):
        if lbl != 1:
            continue
        accident_frame = T - 1   
        for t in range(T):
            if risk_curves[i, t] > thresholds[i, t]:
                frames_early = accident_frame - t
                if frames_early > 0:
                    ttas.append(frames_early / fps)
                break
    return float(np.mean(ttas)) if ttas else 0.0


def compute_tta_fixed(risk_curves, labels, threshold=0.5, fps=10.0):
    T    = risk_curves.shape[1]
    ttas = []
    for i, lbl in enumerate(labels):
        if lbl != 1:
            continue
        accident_frame = T - 1
        hits = np.where(risk_curves[i] > threshold)[0]
        if len(hits):
            frames_early = accident_frame - hits[0]
            if frames_early > 0:
                ttas.append(frames_early / fps)
    return float(np.mean(ttas)) if ttas else 0.0


def compute_metrics(scores, risk_curves, y_true, fps=10.0,threshold=0.5, thresholds_adaptive=None, label=""):
    if thresholds_adaptive is not None:
        tta = compute_tta_adaptive(risk_curves, y_true, thresholds_adaptive, fps)
        best_acc, best_thresh = 0, 0.5
        for th in np.arange(0.1, 0.95, 0.01):
            acc = accuracy_score(y_true, (scores >= th).astype(int))
            if acc > best_acc:
                best_acc, best_thresh = acc, th
        preds = (scores >= best_thresh).astype(int)
    else:
        tta   = compute_tta_fixed(risk_curves, y_true, threshold, fps)
        preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    return {
        "acc"  : float(accuracy_score(y_true, preds)),
        "auc"  : float(roc_auc_score(y_true, scores)),
        "ap"   : float(average_precision_score(y_true, scores)),
        "tta"  : tta,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "prec" : float(tp / (tp + fp + 1e-9)),
        "rec"  : float(tp / (tp + fn + 1e-9)),
    }


def print_results(m1, m2, weights_ok):
    wt = "pretrained" if weights_ok else "random weights"
    print("\n" + "╔" + "═"*72 + "╗")
    print("║FINAL RESULTS — CCD TEST SET" + " "*41 + "║")
    print("╠" + "═"*72 + "╣")
    print(f"║{'Model':<40} {'Acc':>6} {'AUC':>7} {'AP':>7} {'TTA(s)':>8}  ║")
    print("╠" + "═"*72 + "╣")
    for name, m in [(f"M1: UString ({wt})", m1),
                    ("M2: UString + Risk Momentum [OURS]", m2)]:
        print(f"║{name:<40} {m['acc']*100:>5.2f}% "
              f"{m['auc']*100:>6.2f}% {m['ap']*100:>6.2f}% "
              f"{m['tta']:>7.2f}s  ║")
    print("╚" + "═"*72 + "╝")
    d_acc = (m2['acc'] - m1['acc']) * 100
    d_auc = (m2['auc'] - m1['auc']) * 100
    d_ap  = (m2['ap']  - m1['ap'])  * 100
    d_tta =  m2['tta'] - m1['tta']
    print("\n  Gains (M2 − M1):")
    print(f"Δ Accuracy  = {d_acc:+.2f}%")
    print(f"Δ AUC-ROC   = {d_auc:+.2f}%")
    print(f"Δ Avg Prec  = {d_ap:+.2f}%")
    tta_str = "earlier warning ✓" if d_tta < 0 else "later (check gamma)"
    print(f"Δ TTA = {d_tta:+.2f}s  ({tta_str})")
    print(f"\n  Confusion matrices:")
    for tag, m in [("M1", m1), ("M2", m2)]:
        print(f"{tag}: TP={m['tp']} TN={m['tn']} FP={m['fp']} FN={m['fn']}"
              f"Prec={m['prec']:.3f} Rec={m['rec']:.3f}")


def make_plots(m1, m2, m1_scores, m2_scores, m1_risk, R_star,M_smooth, M_pos, thresholds_adaptive,y, fps=10.0, out_dir="results_correct"):
    os.makedirs(out_dir, exist_ok=True)
    T = m1_risk.shape[1]
    t_ax = np.arange(T) / fps
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("M1 (UString)  vs  M2 (UString + Correct Risk Momentum)",fontsize=13, fontweight="bold")
    for scores, label, col, ls in [
        (m1_scores, f"M1: UString  AUC={m1['auc']:.3f}", "#2196F3", "--"),
        (m2_scores, f"M2: +Momentum  AUC={m2['auc']:.3f}", "#E91E63", "-"),
    ]:
        fpr, tpr, _ = roc_curve(y, scores)
        axes[0].plot(fpr, tpr, color=col, ls=ls, lw=2.5, label=label)
        p, r, _ = precision_recall_curve(y, scores)
        axes[1].plot(r, p, color=col, ls=ls, lw=2.5,
                     label=label.replace("AUC","AP").replace(
                         f"{m1['auc']:.3f}", f"{m1['ap']:.3f}").replace(
                         f"{m2['auc']:.3f}", f"{m2['ap']:.3f}"))
    axes[0].plot([0,1],[0,1],"k--",lw=0.8); axes[0].set(title="ROC Curve",
        xlabel="FPR", ylabel="TPR"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
    axes[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "1_roc_pr.png"), dpi=150)
    plt.close()
    acc_idx = np.where(y == 1)[0][:4]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Risk Momentum Effect on Per-Frame Curves\n"
                 "Top: Accident clips  |  Bottom: Momentum signal M(t)",
                 fontsize=12, fontweight="bold")
    for col, i in enumerate(acc_idx):
        ax_top = axes[0][col]
        ax_bot = axes[1][col]
        ax_top.plot(t_ax, m1_risk[i], color="#2196F3", lw=2, label="R(t) UString")
        ax_top.plot(t_ax, R_star[i],  color="#E91E63", lw=2, ls="--", label="R*(t) boosted")
        ax_top.plot(t_ax, thresholds_adaptive[i], color="#FF9800",lw=1.5, ls=":", label="theta(t) adaptive")
        ax_top.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        hit_m1 = np.where(m1_risk[i] > 0.5)[0]
        hit_m2 = np.where(R_star[i] > thresholds_adaptive[i])[0]
        if len(hit_m1): ax_top.axvline(hit_m1[0]/fps, color="#2196F3",lw=1.5, alpha=0.6)
        if len(hit_m2): ax_top.axvline(hit_m2[0]/fps, color="#E91E63",lw=1.5, alpha=0.6)
        if len(hit_m1) and len(hit_m2):
            improvement = (hit_m1[0] - hit_m2[0]) / fps
            ax_top.set_title(f"Accident #{i}\n+{improvement:.1f}s earlier", fontsize=9)
        else:
            ax_top.set_title(f"Accident #{i}", fontsize=9)
        ax_top.set_ylim([0, 1.1]); ax_top.set_xlabel("Time (s)", fontsize=8)
        if col == 0:
            ax_top.legend(fontsize=7); ax_top.set_ylabel("Risk score")
        ax_bot.plot(t_ax, M_smooth[i], color="#9C27B0", lw=1.5, label="M(t) smoothed")
        ax_bot.fill_between(t_ax, 0, M_pos[i], alpha=0.3, color="#E91E63", label="M+(t)")
        ax_bot.axhline(0, color="gray", ls="--", lw=0.8)
        ax_bot.set_ylim([-0.15, 0.25])
        ax_bot.set_xlabel("Time (s)", fontsize=8)
        if col == 0:
            ax_bot.legend(fontsize=7); ax_bot.set_ylabel("Momentum")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "2_momentum_effect_on_curves.png"), dpi=150)
    plt.close()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Average signals across positive and negative clips",fontsize=12, fontweight="bold")
    titles = ["R(t) — UString baseline", "R*(t) — momentum boosted","M(t) — momentum signal"]
    signals = [m1_risk, R_star, M_smooth]
    for ax, sig, title in zip(axes, signals, titles):
        for lbl, col, name in [(1,"#E91E63","Positive (accident)"),(0,"#2196F3","Negative (normal)")]:
            idx = np.where(y == lbl)[0]
            c = sig[idx].mean(0)
            s = sig[idx].std(0)
            ax.plot(t_ax, c, color=col, lw=2, label=name)
            ax.fill_between(t_ax, c-s, c+s, color=col, alpha=0.12)
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
        if "R" in title:
            ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set(xlabel="Time (s)", title=title)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "3_average_curves.png"), dpi=150)
    plt.close()
    metrics_show = [("Accuracy", "acc"), ("AUC-ROC", "auc"), ("Avg Precision", "ap")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("M1 vs M2 — Metric Comparison", fontsize=13, fontweight="bold")
    labels_bar = ["M1\nUString", "M2\n+Momentum"]
    colors_bar = ["#2196F3", "#E91E63"]
    for ax, (mname, mkey) in zip(axes[:3], metrics_show):
        vals = [m1[mkey], m2[mkey]]
        bars = ax.bar(labels_bar, vals, color=colors_bar, alpha=0.85,
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                    f"{v*100:.2f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")
        best_i = int(np.argmax(vals))
        bars[best_i].set_edgecolor("gold"); bars[best_i].set_linewidth(3)
        ax.set_title(mname, fontsize=11)
        ax.set_ylim(0, 1.15); ax.grid(axis="y", alpha=0.3)

    
    ax = axes[3]
    vals = [m1["tta"], m2["tta"]]
    bars = ax.bar(labels_bar, vals, color=colors_bar, alpha=0.85,edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.05,
                f"{v:.2f}s", ha="center", va="bottom",fontsize=10, fontweight="bold")
    best_i = int(np.argmax(vals))  # higher TTA = better (more seconds early)
    bars[best_i].set_edgecolor("gold"); bars[best_i].set_linewidth(3)
    ax.set_title("TTA (higher = earlier warning)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "4_metric_comparison.png"), dpi=150)
    plt.close()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Score Distributions — Positive vs Negative clips",fontsize=12, fontweight="bold")
    bins = np.linspace(0, 1, 30)
    for ax, scores, title in [
        (axes[0], m1_scores, "M1: UString (baseline)"),
        (axes[1], m2_scores, "M2: UString + Risk Momentum"),
    ]:
        for lbl, col, name in [(1,"#E91E63","Positive"),(0,"#2196F3","Negative")]:
            ax.hist(scores[np.where(y==lbl)[0]], bins=bins,
                    color=col, alpha=0.6, label=name, density=True)
        ax.axvline(0.5, color="gray", ls="--", lw=1.5, label="0.5 threshold")
        ax.set(xlabel="Score", ylabel="Density", title=title)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "5_score_distributions.png"), dpi=150)
    plt.close()
    print(f"\n  Saved 5 plots to: {out_dir}/")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--ckpt",      default=None)
    parser.add_argument("--mc_samples",type=int,   default=10)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--gamma",     type=float, default=0.3,
        help="Momentum injection strength into R*(t) (default 0.3)")
    parser.add_argument("--ema_alpha", type=float, default=0.3,
        help="EMA smoothing factor (default 0.3)")
    parser.add_argument("--delta",     type=float, default=0.15,
        help="Max threshold reduction by momentum (default 0.15)")
    parser.add_argument("--fps",       type=float, default=10.0)
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--out_dir",   type=str,   default="results_correct")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "═"*65)
    print("Risk Momentum — CORRECT Implementation")
    print("═"*65)
    print(f"  gamma={args.gamma}  ema_alpha={args.ema_alpha}  delta={args.delta}")
    print("\n  Running UString inference...")
    m1_scores, m1_risk = run_inference(model, X, args.mc_samples,args.batch, args.seed, str(device))
    m1_metrics = compute_metrics(m1_scores, m1_risk, y, args.fps,threshold=0.5, label="M1")
    print("Applying correct Risk Momentum...")
    R_star, M_smooth, M_pos, thresholds_adp, m2_scores = \
        apply_risk_momentum_correct(
            m1_risk,
            gamma         = args.gamma,
            ema_alpha     = args.ema_alpha,
            base_threshold= 0.5,
            delta         = args.delta,
        )
    m2_metrics = compute_metrics(m2_scores, R_star, y, args.fps,thresholds_adaptive=thresholds_adp, label="M2")
    print_results(m1_metrics, m2_metrics, weights_ok)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"M1": m1_metrics, "M2": m2_metrics, "config": vars(args)},f, indent=2)
    print("\n  Generating plots...")
    make_plots(m1_metrics, m2_metrics, m1_scores, m2_scores,
               m1_risk, R_star, M_smooth, M_pos, thresholds_adp,
               y, args.fps, args.out_dir)
    print("\n  Done! ✓")
if __name__ == "__main__":
    main()
