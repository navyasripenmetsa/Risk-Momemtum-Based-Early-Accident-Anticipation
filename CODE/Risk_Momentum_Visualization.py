import os, math, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from tqdm import tqdm
warnings.filterwarnings("ignore")
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] opencv-python not found. Video overlay disabled.")
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch not found. Using synthetic risk scores for demo.")
if HAS_TORCH:
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
            x_proj = self.phi_x(x)                     
            gcn_mu = self.enc_gcn1(x_proj)             
            gcn_lv = self.enc_gcn2(x_proj)
            z = self.reparameterise(gcn_mu, gcn_lv)
            feat = torch.cat([x.mean(dim=2), z, x_proj.mean(dim=2)], dim=-1)
            h  = F.relu(self.predictor['l1'](feat))
            logits = self.predictor['l2'](h)            
            return F.softmax(logits, dim=-1)[:, :, 1]   

        def forward(self, x, mc_samples=10):
            self.train()
            curves = torch.stack([self.forward_one_pass(x) for _ in range(mc_samples)])
            mean_curve = curves.mean(0)
            return mean_curve.mean(1), mean_curve


def load_model(ckpt_path, device):
    if not HAS_TORCH:
        return None, False
    model = UStringModel().to(device)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("[WARN] No checkpoint — using random weights")
        return model, False
    raw = torch.load(ckpt_path, map_location=device)
    if isinstance(raw, dict):
        for k in ("model_state_dict", "model_state", "state_dict", "model", "net"):
            if k in raw:
                raw = raw[k]
                break
    md = model.state_dict()
    matched = {k: v for k, v in raw.items() if k in md and v.shape == md[k].shape}
    md.update(matched)
    model.load_state_dict(md, strict=False)
    pct = 100 * len(matched) / len(md)
    print(f"  Checkpoint loaded: {len(matched)}/{len(md)} tensors ({pct:.0f}%)")
    return model, pct > 50

def run_inference_single(model, feat, mc_samples=20, device="cpu"):
    if not HAS_TORCH or model is None:
        curve = synthetic_risk_curve(feat.shape[0])
        return curve, np.zeros_like(curve)
    x = torch.tensor(feat[np.newaxis], dtype=torch.float32).to(device) 
    model.train()   
    all_curves = []
    with torch.no_grad():
        for s in range(mc_samples):
            torch.manual_seed(s * 31 + 7)
            curve_s = model.forward_one_pass(x)         
            all_curves.append(curve_s[0].cpu().numpy())  
    all_curves = np.array(all_curves)   
    mean_curve = all_curves.mean(axis=0) 
    std_curve  = all_curves.std(axis=0)  
    return mean_curve, std_curve


def synthetic_risk_curve(T=50, label=1, seed=7):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, T)
    if label == 1:
        base  = 0.08 + 0.72 * (1 / (1 + np.exp(-14 * (t - 0.68))))
        noise = rng.normal(0, 0.03, T)
    else:
        base  = 0.08 + 0.04 * np.sin(2 * np.pi * t * 3)
        noise = rng.normal(0, 0.02, T)
    return np.clip(base + noise, 0, 1)

def _ema(signal, alpha):
    out = np.empty_like(signal)
    out[0] = signal[0]
    for t in range(1, len(signal)):
        out[t] = alpha * signal[t] + (1.0 - alpha) * out[t - 1]
    return out

def apply_risk_momentum_advanced(
        risk_curve,
        uncertainty,
        gamma          = 0.30,
        alpha_fast     = 0.50,
        alpha_slow     = 0.10,
        alpha_baseline = 0.05,
        base_threshold = 0.50,
        delta          = 0.15,
        unc_weight     = 0.40,
        accel_weight   = 0.08,
):
    T   = len(risk_curve)
    eps = 1e-7
    discount = 1.0 / (1.0 + unc_weight * uncertainty / (risk_curve + eps))
    R_cal    = np.clip(risk_curve * discount, 0.0, 1.0)
    baseline   = _ema(R_cal, alpha_baseline)
    R_relative = np.clip(R_cal - baseline, 0.0, 1.0)
    R_cal_prev = np.concatenate([[R_cal[0]], R_cal[:-1]])
    raw_dR = R_cal - R_cal_prev
    M_fast = _ema(raw_dR, alpha_fast)
    M_slow = _ema(raw_dR, alpha_slow)
    M_pos_fast = np.clip(M_fast, 0.0, None)
    M_pos_slow = np.clip(M_slow, 0.0, None)
    M_combined = np.sqrt(M_pos_fast * M_pos_slow + eps) - np.sqrt(eps)
    M_combined = np.clip(M_combined, 0.0, None)
    raw_dM  = np.concatenate([[0.0], np.diff(M_fast)])
    M_accel = _ema(raw_dM, 0.40)
    M_accel_pos = np.clip(M_accel, 0.0, None)
    R_star = np.clip(R_cal+ gamma* M_combined+ accel_weight * M_accel_pos,0.0, 1.0)
    raw_theta = np.clip(base_threshold- delta* M_combined+ unc_weigh* uncertainty- 0.04* M_accel_pos,0.25,base_threshold + 0.10,)
    thresholds = _ema(raw_theta, 0.70) 
    return R_star, M_fast, M_slow, M_combined, M_accel, thresholds, baseline, R_relative, R_cal


def find_alarm_frame_v2(R_star, thresholds, R_relative, M_combined,rel_threshold=0.015,mom_threshold=0.003,k_gate=3):
    T = len(R_star)
    consecutive = 0
    for t in range(T):
        cond = (
            R_star[t]    > thresholds[t]
            and R_relative[t] > rel_threshold
            and M_combined[t] > mom_threshold
        )
        if cond:
            consecutive += 1
            if consecutive >= k_gate:
                return t - k_gate + 1
        else:
            consecutive = 0
    return None


def risk_color(v):
    v = float(np.clip(v, 0, 1))
    if v < 0.5:
        return (v * 2, 1.0, 0.0)
    return (1.0, 1.0 - (v - 0.5) * 2, 0.0)


def momentum_color(m):
    m = float(np.clip(m, -0.2, 0.2))
    if m >= 0:
        intensity = m / 0.2
        return (intensity, 0.15, 0.1)
    return (0.1, 0.15, abs(m) / 0.2)

def load_video_frames(video_path, n_frames=50):
    if not HAS_CV2 or not video_path or not os.path.isfile(video_path):
        return None
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, max(total - 1, 0), n_frames, dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(np.zeros((270, 480, 3), dtype=np.uint8))
    cap.release()
    return frames


def draw_hud(frame, t, T, risk, risk_star, momentum, threshold,uncertainty, alarm_frame, fps=10.0):
    if not HAS_CV2:
        return frame
    img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 130), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    time_s = t / fps
    cv2.putText(img, f"Frame {t+1:02d}/{T}  |  t={time_s:.1f}s",(10, h - 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48,(200, 200, 200), 1, cv2.LINE_AA)
    bar_x, bar_y, bar_w, bar_h = 10, h - 94, w - 20, 16
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),(55, 55, 55), -1)
    fill = int(bar_w * float(np.clip(risk, 0, 1)))
    rc   = risk_color(risk)
    col  = (int(rc[2]*255), int(rc[1]*255), int(rc[0]*255))
    if fill > 0:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), col, -1)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),(140, 140, 140), 1)
    th_x = bar_x + int(bar_w * threshold)
    cv2.line(img, (th_x, bar_y - 3), (th_x, bar_y + bar_h + 3), (255, 165, 0), 2)
    cv2.putText(img, f"P(acc)={risk:.3f}   unc={uncertainty:.3f}",
                (bar_x + 4, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"M_combined={momentum:+.4f}",
                (10, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (180, 180, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"R*(t)={risk_star:.3f}   theta(t)={threshold:.3f}",
                (10, h - 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 220, 100), 1, cv2.LINE_AA)
    alarm_triggered = alarm_frame is not None and t >= alarm_frame
    if alarm_triggered:
        if (t // 2) % 2 == 0:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        cv2.rectangle(img, (w // 4, 18), (3 * w // 4, 68), (0, 0, 180), -1)
        cv2.putText(img, "! ACCIDENT RISK DETECTED !",
                    (w // 4 + 8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.68,
                    (255, 255, 255), 2, cv2.LINE_AA)
        early_s = (T - 1 - alarm_frame) / fps
        cv2.putText(img, f"{early_s:.1f}s early warning",
                    (w // 4 + 55, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 200, 0), 1, cv2.LINE_AA)
    ub_w, ub_h = 80, 10
    ub_x, ub_y = w - ub_w - 10, 10
    cv2.rectangle(img, (ub_x, ub_y), (ub_x + ub_w, ub_y + ub_h), (55, 55, 55), -1)
    unc_fill = int(ub_w * float(np.clip(uncertainty * 4, 0, 1)))
    if unc_fill > 0:
        cv2.rectangle(img, (ub_x, ub_y), (ub_x + unc_fill, ub_y + ub_h),
                      (120, 80, 200), -1)
    cv2.putText(img, "UNC", (ub_x - 35, ub_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 140, 220), 1, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

DARK_BG  = "#0f0f14"
PANEL_BG = "#14141c"
def fig_to_rgb_array(fig):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    return buf.reshape(h, w, 4)[:, :, :3].copy()


def make_plot_frame(t, T, fps,
                    risk_curve,   
                    uncertainty,
                    R_star, M_fast, M_slow, M_combined, M_accel,
                    thresholds, baseline, R_relative,
                    alarm_m1, alarm_m2, label,
                    fig_w=10, fig_h=7):
    t_ax   = np.arange(T) / fps
    t_now  = (t + 1) / fps
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,left=0.07, right=0.97, top=0.90, bottom=0.10)
    GRID  = {"color": "#2a2a3a", "linewidth": 0.4, "linestyle": "--"}
    SPINE = "#3a3a55"
    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_color(SPINE)
        ax.tick_params(colors="#888899", labelsize=7)
        ax.set_title(title, color="#ccccdd", fontsize=8, pad=4, fontweight="bold")
        ax.set_xlim(0, (T - 1) / fps)
        ax.grid(**GRID)
        ax.axvline(t_now, color="#ffffff", lw=0.8, alpha=0.45, linestyle=":")
    xs  = t_ax[:t+1]
    rc  = risk_curve[:t+1]
    uc  = uncertainty[:t+1]
    rs  = R_star[:t+1]
    mf  = M_fast[:t+1]
    ms_ = M_slow[:t+1]
    mc_ = M_combined[:t+1]
    ma  = M_accel[:t+1]
    th  = thresholds[:t+1]
    bl  = baseline[:t+1]
    rr  = R_relative[:t+1]
    def vline_alarm(ax, frame, color, label_txt, y_txt=0.88):
        if frame is not None and frame <= t:
            ax.axvline(frame / fps, color=color, lw=1.4, ls="--", alpha=0.85)
            ax.text(frame / fps + 0.08, y_txt, label_txt,
                    color=color, fontsize=6.5, va="top")
    ax0 = fig.add_subplot(gs[0, 0])
    style_ax(ax0, "Accident Risk + Uncertainity")
    ax0.set_ylim(-0.05, 1.12)
    ax0.set_ylabel("P(accident)", color="#888899", fontsize=7)
    ax0.axhline(0.5, color="#ff9900", lw=0.8, ls="--", alpha=0.6, label="θ=0.50")
    # Uncertainty band
    if len(xs) > 1:
        ax0.fill_between(xs, np.clip(rc - uc, 0, 1), np.clip(rc + uc, 0, 1),alpha=0.20, color="#aa88ff", label="±σ")
    ax0.plot(xs, bl, color="#4488ff", lw=1.0, ls=":", alpha=0.7, label="Baseline B(t)")
    for i in range(1, len(xs)):
        ax0.plot(xs[i-1:i+1], rc[i-1:i+1], color=risk_color(rc[i]), lw=1.8)
    ax0.fill_between(xs, 0, rc, alpha=0.10, color="#3399ff")
    vline_alarm(ax0, alarm_m1, "#ff4444", "M1 ↑")
    if len(xs):
        ax0.scatter([xs[-1]], [rc[-1]], color=risk_color(rc[-1]),
                    s=35, zorder=10, edgecolors="white", linewidths=0.5)
    ax0.legend(fontsize=6, loc="upper left", facecolor=PANEL_BG,labelcolor="#888899", edgecolor=SPINE)
    ax1 = fig.add_subplot(gs[0, 1])
    style_ax(ax1, "Final Risk + Decision Threshold")
    ax1.set_ylim(-0.05, 1.12)
    ax1.set_ylabel("P*(accident)", color="#888899", fontsize=7)
    if len(xs) > 1:
        ax1.fill_between(xs, th, 0.5, alpha=0.10, color="#ff9900", label="θ zone")
    ax1.plot(xs, th, color="#ff9900", lw=1.0, ls="--", alpha=0.8, label="θ(t) adaptive")
    for i in range(1, len(xs)):
        ax1.plot(xs[i-1:i+1], rs[i-1:i+1], color=risk_color(rs[i]), lw=1.8)
    ax1.fill_between(xs, 0, rs, alpha=0.10, color="#ff6644")
    vline_alarm(ax1, alarm_m1, "#ff9933", "M1 (ref)", y_txt=0.78)
    vline_alarm(ax1, alarm_m2, "#ff3333", "M2 ↑")
    if alarm_m1 is not None and alarm_m2 is not None and alarm_m2 < alarm_m1:
        gain = (alarm_m1 - alarm_m2) / fps
        ax1.text(alarm_m2 / fps + 0.08, 0.62,
                 f"+{gain:.1f}s earlier", color="#aaffaa", fontsize=6.5)
    if len(xs):
        ax1.scatter([xs[-1]], [rs[-1]], color=risk_color(rs[-1]),
                    s=35, zorder=10, edgecolors="white", linewidths=0.5)
    ax1.legend(fontsize=6, loc="upper left", facecolor=PANEL_BG,
               labelcolor="#888899", edgecolor=SPINE)
    ax2 = fig.add_subplot(gs[0, 2])
    style_ax(ax2, "Model Uncertainity")
    uc_max = max(0.1, float(uncertainty.max()) * 1.3)
    ax2.set_ylim(0, uc_max)
    ax2.set_ylabel("σ (MC std)", color="#888899", fontsize=7)
    ax2.fill_between(xs, 0, uc, color="#aa88ff", alpha=0.45, label="σ(t)")
    ax2.plot(xs, uc, color="#cc99ff", lw=1.4)
    if len(xs):
        ax2.scatter([xs[-1]], [uc[-1]], color="#cc99ff",
                    s=35, zorder=10, edgecolors="white", linewidths=0.5)
    ax2.legend(fontsize=6, loc="upper right", facecolor=PANEL_BG,
               labelcolor="#888899", edgecolor=SPINE)

    ax3 = fig.add_subplot(gs[1, 0])
    style_ax(ax3, "Risk Trend (Fast vs Slow)")
    ax3.set_xlabel("Time (s)", color="#888899", fontsize=7)
    ax3.set_ylabel("Momentum", color="#888899", fontsize=7)
    ymax = max(0.15, float(np.abs(M_fast).max()) * 1.25)
    ax3.set_ylim(-ymax, ymax)
    ax3.axhline(0, color=SPINE, lw=0.6)
    ax3.plot(xs, mf,  color="#ff8866", lw=1.4, label="M_fast")
    ax3.plot(xs, ms_, color="#6688ff", lw=1.4, label="M_slow", ls="--")
    ax3.fill_between(xs, 0, mf, where=mf >= 0, color="#ff4444", alpha=0.20)
    ax3.fill_between(xs, 0, mf, where=mf < 0,  color="#4488ff", alpha=0.15)
    if len(xs):
        ax3.scatter([xs[-1]], [mf[-1]], color="#ff8866",
                    s=35, zorder=10, edgecolors="white", linewidths=0.5)
    ax3.legend(fontsize=6, facecolor=PANEL_BG, labelcolor="#888899", edgecolor=SPINE)

    ax4 = fig.add_subplot(gs[1, 1])
    style_ax(ax4, "Risk Momentum")
    ax4.set_xlabel("Time (s)", color="#888899", fontsize=7)
    ax4.set_ylabel("M⁺ (combined)", color="#888899", fontsize=7)
    ax4.set_ylim(-0.005, max(0.15, float(M_combined.max()) * 1.35))
    ax4.fill_between(xs, 0, mc_, color="#ff6644", alpha=0.55, label="M⁺(t)")
    ax4.plot(xs, mc_, color="#ffaa88", lw=1.5)
    ax4.axhline(0.003, color="#ffdd44", lw=0.7, ls=":", alpha=0.7, label="mom_thresh")
    vline_alarm(ax4, alarm_m2, "#ff3333", "M2 ↑", y_txt=float(M_combined.max()) * 0.9)
    if len(xs):
        ax4.scatter([xs[-1]], [mc_[-1]], color="#ffaa88",
                    s=35, zorder=10, edgecolors="white", linewidths=0.5)
    ax4.legend(fontsize=6, facecolor=PANEL_BG, labelcolor="#888899", edgecolor=SPINE)

    ax5 = fig.add_subplot(gs[1, 2])
    style_ax(ax5, "Risk Acceleration")
    ax5.set_xlabel("Time (s)", color="#888899", fontsize=7)
    ax5.set_ylabel("Acceleration", color="#888899", fontsize=7)
    ymax_a = max(0.10, float(np.abs(M_accel).max()) * 1.3)
    ax5.set_ylim(-ymax_a, ymax_a)
    ax5.axhline(0, color=SPINE, lw=0.6)
    ax5.fill_between(xs, 0, ma, where=ma >= 0, color="#ffcc44", alpha=0.40, label="A⁺(t)")
    ax5.fill_between(xs, 0, ma, where=ma < 0,  color="#4488ff", alpha=0.25, label="A⁻(t)")
    ax5.plot(xs, ma, color="#ffeeaa", lw=1.2)
    if len(xs):
        ax5.scatter([xs[-1]], [ma[-1]], color="#ffeeaa",
                    s=35, zorder=10, edgecolors="white", linewidths=0.5)
    ax5.legend(fontsize=6, facecolor=PANEL_BG, labelcolor="#888899", edgecolor=SPINE)
    clip_label = "ACCIDENT" if label == 1 else "NORMAL"
    c_lbl = "#ff5555" if label == 1 else "#55ff99"
    fig.text(0.5, 0.96,
             f"TRACT Risk Momentum — [{clip_label}]   Frame {t+1}/{T}  ({t_now:.1f}s)",
             ha="center", va="top", fontsize=9.5, color=c_lbl, fontweight="bold")
    arr = fig_to_rgb_array(fig)
    plt.close(fig)
    return arr


def make_composite_frame(t, T, fps, video_frames,
                          risk_curve, uncertainty,
                          R_star, M_fast, M_slow, M_combined, M_accel,
                          thresholds, baseline, R_relative,
                          alarm_m1, alarm_m2, label,
                          target_w=1280, target_h=720):
    plot_w = target_w if video_frames is None else target_w // 2
    plot_h = target_h
    plot_arr = make_plot_frame(
        t, T, fps,
        risk_curve, uncertainty,
        R_star, M_fast, M_slow, M_combined, M_accel,
        thresholds, baseline, R_relative,
        alarm_m1, alarm_m2, label,
        fig_w=plot_w / 100, fig_h=plot_h / 100,
    )
    if HAS_CV2:
        plot_arr = cv2.resize(plot_arr, (plot_w, plot_h), interpolation=cv2.INTER_LINEAR)
    if video_frames is None:
        return plot_arr
    vframe = video_frames[t] if t < len(video_frames) else \
        np.zeros((target_h, plot_w, 3), dtype=np.uint8)
    r_val  = float(risk_curve[t])   if t < len(risk_curve)  else 0.0
    rs_val = float(R_star[t])       if t < len(R_star)      else 0.0
    mc_val = float(M_combined[t])   if t < len(M_combined)  else 0.0
    th_val = float(thresholds[t])   if t < len(thresholds)  else 0.5
    uc_val = float(uncertainty[t])  if t < len(uncertainty) else 0.0
    vframe_hud = draw_hud(vframe, t, T, r_val, rs_val, mc_val, th_val,uc_val, alarm_m2, fps)
    if HAS_CV2:
        vframe_hud = cv2.resize(vframe_hud, (plot_w, target_h),
                                interpolation=cv2.INTER_LINEAR)
    return np.concatenate([vframe_hud, plot_arr], axis=1)

def save_summary_plot(risk_curve, uncertainty,
                       R_star, M_fast, M_slow, M_combined, M_accel,
                       thresholds, baseline, R_relative,
                       alarm_m1, alarm_m2, label, fps, out_path):
    T    = len(risk_curve)
    t_ax = np.arange(T) / fps
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor=DARK_BG)
    fig.subplots_adjust(hspace=0.42, wspace=0.35,
                        left=0.06, right=0.97, top=0.90, bottom=0.08)
    GRID  = {"color": "#2a2a3a", "linewidth": 0.5, "linestyle": "--"}
    def style(ax, title, ylabel):
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values(): sp.set_color("#3a3a55")
        ax.tick_params(colors="#888899", labelsize=8)
        ax.set_title(title, color="#ccccdd", fontsize=9, pad=5, fontweight="bold")
        ax.set_xlabel("Time (s)", color="#888899", fontsize=8)
        ax.set_ylabel(ylabel, color="#888899", fontsize=8)
        ax.grid(**GRID)
    def va(ax, frame, color, lbl):
        if frame is not None:
            ax.axvline(frame / fps, color=color, lw=1.6, ls="--", alpha=0.88,
                       label=lbl)
    clip_type = "ACCIDENT" if label == 1 else "NORMAL"
    c_title   = "#ff5555" if label == 1 else "#55ff99"
    ax = axes[0, 0]
    style(ax, "M1 — R(t) + uncertainty band + baseline B(t)", "P(accident)")
    ax.set_ylim(-0.05, 1.12)
    ax.fill_between(t_ax, np.clip(risk_curve - uncertainty, 0, 1),
                    np.clip(risk_curve + uncertainty, 0, 1),
                    alpha=0.20, color="#aa88ff", label="±σ uncertainty")
    ax.plot(t_ax, baseline, color="#4488ff", lw=1.2, ls=":", alpha=0.8, label="Baseline B(t)")
    ax.axhline(0.5, color="#ff9900", lw=1.0, ls="--", alpha=0.65, label="Fixed θ=0.50")
    for i in range(1, T):
        ax.plot(t_ax[i-1:i+1], risk_curve[i-1:i+1],
                color=risk_color(risk_curve[i]), lw=2.0)
    ax.fill_between(t_ax, 0, risk_curve, alpha=0.10, color="#3399ff")
    va(ax, alarm_m1, "#ff3333", "M1 alarm")
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="#aaaacc", edgecolor="#3a3a55")
    ax = axes[0, 1]
    style(ax, "M2 — Boosted R*(t) + adaptive θ(t)", "P*(accident)")
    ax.set_ylim(-0.05, 1.12)
    ax.fill_between(t_ax, thresholds, 0.5, alpha=0.12, color="#ff9900", label="θ zone")
    ax.plot(t_ax, thresholds, color="#ff9900", lw=1.2, ls="--", alpha=0.8, label="Adaptive θ(t)")
    for i in range(1, T):
        ax.plot(t_ax[i-1:i+1], R_star[i-1:i+1],
                color=risk_color(R_star[i]), lw=2.0)
    ax.fill_between(t_ax, 0, R_star, alpha=0.10, color="#ff6644")
    va(ax, alarm_m1, "#ff9933", "M1 (ref)")
    va(ax, alarm_m2, "#ff3333", "M2 alarm")
    if alarm_m1 is not None and alarm_m2 is not None and alarm_m2 < alarm_m1:
        gain = (alarm_m1 - alarm_m2) / fps
        ax.text(alarm_m2 / fps + 0.12, 0.72,
                f"Earlier by\n{gain:.2f}s", color="#aaffaa", fontsize=8)
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="#aaaacc", edgecolor="#3a3a55")
    ax = axes[0, 2]
    style(ax, "Epistemic uncertainty σ(t) [MC std]", "σ")
    ax.set_ylim(0, max(0.08, float(uncertainty.max()) * 1.35))
    ax.fill_between(t_ax, 0, uncertainty, color="#aa88ff", alpha=0.5, label="σ(t)")
    ax.plot(t_ax, uncertainty, color="#cc99ff", lw=1.6)
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="#aaaacc", edgecolor="#3a3a55")

   
    ax = axes[1, 0]
    style(ax, "M_fast (α=0.5) & M_slow (α=0.1)  — signed", "Momentum")
    ymax = max(0.12, float(np.abs(M_fast).max()) * 1.3)
    ax.set_ylim(-ymax, ymax)
    ax.axhline(0, color="#3a3a55", lw=0.8)
    ax.plot(t_ax, M_fast, color="#ff8866", lw=1.8, label="M_fast")
    ax.plot(t_ax, M_slow, color="#6688ff", lw=1.8, ls="--", label="M_slow")
    ax.fill_between(t_ax, 0, M_fast, where=M_fast >= 0, color="#ff4444", alpha=0.22)
    ax.fill_between(t_ax, 0, M_fast, where=M_fast < 0,  color="#4488ff", alpha=0.18)
    va(ax, alarm_m2, "#ff3333", "M2 alarm")
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="#aaaacc", edgecolor="#3a3a55")

    
    ax = axes[1, 1]
    style(ax, "M⁺(t)=√(fast⁺·slow⁺) & R_rel(t)  — TRACT drivers", "Value")
    ax.set_ylim(-0.005, max(0.15, max(float(M_combined.max()), float(R_relative.max())) * 1.35))
    ax.fill_between(t_ax, 0, M_combined, color="#ff6644", alpha=0.50, label="M⁺(t)")
    ax.plot(t_ax, M_combined, color="#ffaa88", lw=1.8)
    ax.plot(t_ax, R_relative, color="#88ffcc", lw=1.4, ls="--", label="R_rel(t)")
    ax.axhline(0.003, color="#ffdd44", lw=0.7, ls=":", alpha=0.7, label="mom_thresh=0.003")
    ax.axhline(0.015, color="#88ffcc", lw=0.7, ls=":", alpha=0.7, label="rel_thresh=0.015")
    va(ax, alarm_m2, "#ff3333", "M2 alarm")
    peak_i = int(np.argmax(M_combined))
    ax.annotate(f"Peak\n{M_combined[peak_i]:.4f}",
                xy=(t_ax[peak_i], M_combined[peak_i]),
                xytext=(t_ax[peak_i] + 0.4, M_combined[peak_i] * 0.75),
                color="#ffdd88", fontsize=7.5,
                arrowprops={"arrowstyle": "->", "color": "#ffdd88", "lw": 0.9})
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="#aaaacc", edgecolor="#3a3a55")

   
    ax = axes[1, 2]
    style(ax, "Acceleration of Risk (How Fast Risk is increasing)", "A(t)")
    ymax_a = max(0.08, float(np.abs(M_accel).max()) * 1.35)
    ax.set_ylim(-ymax_a, ymax_a)
    ax.axhline(0, color="#3a3a55", lw=0.8)
    ax.fill_between(t_ax, 0, M_accel, where=M_accel >= 0,
                    color="#ffcc44", alpha=0.45, label="A⁺(t)")
    ax.fill_between(t_ax, 0, M_accel, where=M_accel < 0,
                    color="#4488ff", alpha=0.25, label="A⁻(t)")
    ax.plot(t_ax, M_accel, color="#ffeeaa", lw=1.6)
    ax.legend(fontsize=7, facecolor=PANEL_BG, labelcolor="#aaaacc", edgecolor="#3a3a55")

    
    m1_info  = f"M1 alarm: frame {alarm_m1}" if alarm_m1 is not None else "M1: no alarm"
    m2_info  = f"M2 alarm: frame {alarm_m2}" if alarm_m2 is not None else "M2: no alarm"
    gain_str = ""
    if alarm_m1 is not None and alarm_m2 is not None:
        gain_str = f"  |  M2 fires {(alarm_m1 - alarm_m2) / fps:+.2f}s earlier"

    fig.text(0.5, 0.965,
             f"TRACT Risk Momentum — [{clip_type}]   {m1_info}   {m2_info}{gain_str}",
             ha="center", va="top", fontsize=10, color=c_title, fontweight="bold")

    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Summary PNG saved: {out_path}")


def make_visualization_video(
        risk_curve, uncertainty,
        R_star, M_fast, M_slow, M_combined, M_accel,
        thresholds, baseline, R_relative,
        video_frames, label, fps, out_path,
        target_w=1280, target_h=720,
        rel_threshold=0.015, mom_threshold=0.003, k_gate=3):
    T = len(risk_curve)
    alarm_m1 = find_alarm_frame(risk_curve, np.full(T, 0.5))
    alarm_m2 = find_alarm_frame_v2(R_star, thresholds, R_relative, M_combined,
                                    rel_threshold=rel_threshold,
                                    mom_threshold=mom_threshold,
                                    k_gate=k_gate)
    print(f"  M1 alarm at frame {alarm_m1}  |  M2 alarm (TRACT) at frame {alarm_m2}")
    if alarm_m1 is not None and alarm_m2 is not None:
        gain = (alarm_m1 - alarm_m2) / fps
        print(f"  TRACT M2 fires {gain:.2f}s {'earlier' if gain > 0 else 'later'} than M1")
    elif alarm_m2 is None:
        print("  TRACT: No alarm triggered (good for normal clips)")
    if not HAS_CV2:
        print("  [WARN] No OpenCV — saving per-frame PNGs only.")
        os.makedirs(out_path + "_frames", exist_ok=True)
        for t in tqdm(range(T), desc="Rendering"):
            arr = make_composite_frame(t, T, fps, None,
                                       risk_curve, uncertainty,
                                       R_star, M_fast, M_slow, M_combined, M_accel,
                                       thresholds, baseline, R_relative,
                                       alarm_m1, alarm_m2, label,
                                       target_w, target_h)
            plt.imsave(f"{out_path}_frames/frame_{t:03d}.png", arr)
        return alarm_m1, alarm_m2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (target_w, target_h))
    for t in tqdm(range(T), desc="Rendering video"):
        composite = make_composite_frame(
            t, T, fps, video_frames,
            risk_curve, uncertainty,
            R_star, M_fast, M_slow, M_combined, M_accel,
            thresholds, baseline, R_relative,
            alarm_m1, alarm_m2, label,
            target_w, target_h,
        )
        frame_bgr = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (target_w, target_h))
        writer.write(frame_bgr)
    writer.release()
    print(f"  Video saved: {out_path}")
    return alarm_m1, alarm_m2

def main():
    parser = argparse.ArgumentParser(
        description="TRACT Risk Momentum Visualization for CCD + UString")
    parser.add_argument("--npz",   required=True)
    parser.add_argument("--video", default=None)
    parser.add_argument("--ckpt",  default=None)
    parser.add_argument("--label", type=int, default=1,
                        help="Ground-truth: 1=accident, 0=normal")
    parser.add_argument("--mc",    type=int,   default=20,
                        help="MC samples (default 20)")
    parser.add_argument("--fps",   type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=0.30,
                        help="Momentum injection strength")
    parser.add_argument("--alpha_fast", type=float, default=0.50,
                        help="Fast EMA alpha (default 0.5)")
    parser.add_argument("--alpha_slow", type=float, default=0.10,
                        help="Slow EMA alpha (default 0.1)")
    parser.add_argument("--alpha_baseline", type=float, default=0.05,
                        help="Causal baseline EMA alpha (default 0.05)")
    parser.add_argument("--delta", type=float, default=0.15,
                        help="Adaptive threshold reduction factor")
    parser.add_argument("--unc_weight", type=float, default=0.40,
                        help="Uncertainty discounting weight")
    parser.add_argument("--k_gate", type=int, default=3,
                        help="Consecutive frames required for TRACT alarm (default 3)")
    parser.add_argument("--rel_threshold", type=float, default=0.015,
                        help="Min above-baseline deviation for alarm (default 0.015)")
    parser.add_argument("--mom_threshold", type=float, default=0.003,
                        help="Min combined momentum for alarm (default 0.003)")
    parser.add_argument("--out",    default="risk_momentum_vis.mp4")
    parser.add_argument("--width",  type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    print("\n" + "═"*68)
    print("  TRACT Risk Momentum Visualization")
    print("═"*68)
    print(f"NPZ   : {args.npz}")
    print(f"Video : {args.video or 'none (feature-only)'}")
    print(f"Ckpt  : {args.ckpt or 'none (random weights)'}")
    print(f"Label : {'ACCIDENT' if args.label == 1 else 'NORMAL'}")
    print(f"MC samples   : {args.mc}")
    print(f"gamma        : {args.gamma}")
    print(f"alpha_fast   : {args.alpha_fast}   alpha_slow: {args.alpha_slow}")
    print(f"alpha_base   : {args.alpha_baseline}")
    print(f"delta        : {args.delta}   unc_weight: {args.unc_weight}")
    print(f"k_gate       : {args.k_gate}   rel_thresh: {args.rel_threshold}"
          f"mom_thresh: {args.mom_threshold}")
    print()
    print("Loading VGG16 features...")
    npz  = np.load(args.npz, allow_pickle=True)
    feat = npz["data"].astype(np.float32)
    print(f"Feature shape: {feat.shape}")
    T = feat.shape[0]
    device = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    print(f"Device: {device}")
    model, weights_ok = load_model(args.ckpt, device)
    print("Running UString MC inference (uncertainty-aware)...")
    risk_curve, uncertainty = run_inference_single(model, feat, args.mc, device)
    print(f"Risk curve  range : [{risk_curve.min():.4f}, {risk_curve.max():.4f}]")
    print(f"Uncertainty range : [{uncertainty.min():.4f}, {uncertainty.max():.4f}]")
    print("Computing TRACT Risk Momentum...")
    (R_star, M_fast, M_slow, M_combined, M_accel,
     thresholds, baseline, R_relative, R_cal) = apply_risk_momentum_advanced(
        risk_curve, uncertainty,
        gamma          = args.gamma,
        alpha_fast     = args.alpha_fast,
        alpha_slow     = args.alpha_slow,
        alpha_baseline = args.alpha_baseline,
        base_threshold = 0.5,
        delta          = args.delta,
        unc_weight     = args.unc_weight,
    )
    alarm_m1 = find_alarm_frame(risk_curve, np.full(T, 0.5))
    alarm_m2 = find_alarm_frame_v2(
        R_star, thresholds, R_relative, M_combined,
        rel_threshold=args.rel_threshold,
        mom_threshold=args.mom_threshold,
        k_gate=args.k_gate,
    )
    print(f"  M1 alarm: frame {alarm_m1}")
    print(f"  M2 alarm: frame {alarm_m2}  (TRACT, k_gate={args.k_gate})")
    video_frames = None
    if args.video:
        print("  Loading video frames...")
        video_frames = load_video_frames(args.video, n_frames=T)
        if video_frames:
            print(f"  Loaded {len(video_frames)} frames")
        else:
            print("  [WARN] Could not read video. Feature-only mode.")

    png_path = os.path.splitext(args.out)[0] + "_summary.png"
    save_summary_plot(
        R_cal, uncertainty,
        R_star, M_fast, M_slow, M_combined, M_accel,
        thresholds, baseline, R_relative,
        alarm_m1, alarm_m2, args.label, args.fps, png_path,
    )

    print(f"\n  Rendering animated visualization → {args.out}")
    make_visualization_video(
        R_cal, uncertainty,
        R_star, M_fast, M_slow, M_combined, M_accel,
        thresholds, baseline, R_relative,
        video_frames, args.label, args.fps, args.out,
        target_w=args.width, target_h=args.height,
        rel_threshold=args.rel_threshold,
        mom_threshold=args.mom_threshold,
        k_gate=args.k_gate,
    )
    print(f"  Output video : {args.out}")
    print(f"  Summary PNG  : {png_path}")
if __name__ == "__main__":
    main()