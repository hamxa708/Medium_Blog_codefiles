############################################# WRITTEN BY M.HAMZA ####################################################

import os
# Set deterministic CuBLAS workspace configuration before importing torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Set non-interactive backend for matplotlib before importing it
import matplotlib
matplotlib.use('Agg')

import json, time, copy, random, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as Data
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy import stats
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


FEATURE_NAMES = ["T", "C", "SOC", "DOD", "SOH"]


# ---------------------------------------------------------------- optimisation
BATCH_SIZE = 256
NUM_EPOCHS = 50                   # Vanilla-NN-100%  
NUM_EPOCHS_VNN = 520              # Vanilla-NN low-data
NUM_EPOCHS_HYBRID = 800           # Hybrid two-phase budget (P1 residual + P2 joint)
LEARNING_RATE = 0.0005            # Vanilla-NN-100% 
LEARNING_RATE_LOW = 0.01          # Vanilla-NN low-data and pure log-linear
LEARNING_RATE_HYBRID = 0.001      # Hybrid
FRACTION_LOW = 0.0001             # 0.01% of the training pool
RNG_SEED = 0
VNN_PATIENCE = 60                 # early-stopping patience (epochs), vanilla low-data
HYBRID_PATIENCE = 120             # residual needs time; pure log-linear still stops early

# ---------------------------------------------------------------- evaluation protocol
# External validation (outside the labeled budget) 
N_EXTERNAL_VAL = 200
N_SEEDS = 15                      # enough for Wilcoxon p<0.05 after Holm
RUN_HPARAM_SEARCH = False
RUN_SWEEP = True
RUN_ABLATIONS = True              # warm-start / anchor / residual-reg ablations
SWEEP_FRACTIONS = [0.5 * FRACTION_LOW, FRACTION_LOW, 5 * FRACTION_LOW, 10 * FRACTION_LOW, 100 * FRACTION_LOW]
SWEEP_SEEDS = 5
N_TUNE_SEEDS = 2
LR_GRID_VNN = [1e-3, 1e-2]
LR_GRID_HYBRID = [1e-3, 1e-2]
ANCHOR_GRID = [1.0, 2.0]

# Hyperparameters used by every low-data run
HPARAMS = {
    "vanilla":   {"lr": LEARNING_RATE_LOW},
    "loglinear": {"lr": LEARNING_RATE_LOW},
    "hybrid":    {"lr": LEARNING_RATE_HYBRID, "w_phys_anchor": 2.0, "w_residual_reg": 0.05,
                  "residual_hidden": 20, "residual_mid": 10},
}

# Option C: smaller residual + stronger residual penalty (less overfit on ~233 labels).
# Both A and C are trained when SELECT_HYBRID_BY_VAL is True.
#   - "Hybrid-A" is always recorded
#   - "Hybrid" is the candidate with better validation MAPE (cannot be worse on val than A).


HYBRID_CANDIDATES = {
    "A": {"residual_hidden": 20, "residual_mid": 10, "w_residual_reg": 0.05},  # current
    "C": {"residual_hidden": 8,  "residual_mid": 4,  "w_residual_reg": 0.20},  # option C
}
SELECT_HYBRID_BY_VAL = True    # DO FALSE to go back to Option A

# ---------------------------------------------------------------- reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(RNG_SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENV = {
    "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
    "torch": torch.__version__, "device": str(device),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(json.dumps(ENV, indent=2))

# ---------------------------------------------------------------- names, colours, figure style
LOW = f"{FRACTION_LOW * 100:g}%"
NAMES = {
    "v100": "Vanilla-NN-100%",
    "vlow": f"Vanilla-NN-{LOW}",
    "loglin": f"PureLogLinear-{LOW}",
    "hybrid": f"Hybrid-NN-{LOW}",
}
SHORT = {
    "v100": "Vanilla NN (100%)", "vlow": "Vanilla NN",
    "loglin": "Log-linear", "hybrid": "Hybrid",
}

# Okabe-Ito palette
colors = {
    NAMES["v100"]: "#009E73", NAMES["vlow"]: "#7F7F7F",
    NAMES["loglin"]: "#0072B2", NAMES["hybrid"]: "#D55E00",
}
colors_short = {SHORT[k]: colors[NAMES[k]] for k in NAMES}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def plain_log_axis(ax, axis="y"):
    a_ = ax.yaxis if axis == "y" else ax.xaxis
    a_.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a_.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    a_.set_minor_formatter(ticker.NullFormatter())


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.show()


from pathlib import Path


HPC_BASE_DIR = Path("/hdd3/uspcase/hamza.uspcase")

DATA_DIR = HPC_BASE_DIR
RESULTS_DIR = HPC_BASE_DIR / "results"
FIG_DIR = HPC_BASE_DIR / "figures"

#------------------------------------------------ Create output directories 
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Data directory:    {DATA_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Figures directory: {FIG_DIR}")

# 2.------------------------------------------------ Load data
x_sample = np.load(DATA_DIR / "x_sample.npy")
y_sample = np.load(DATA_DIR / "y_sample.npy").reshape(-1)

x_test = np.load(DATA_DIR / "x_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy").reshape(-1)


#---------------------------------------------------- splits
x_train_main, x_val, y_train_main, y_val = train_test_split(x_sample, y_sample, test_size=0.2, random_state=RNG_SEED)
n_total_main = x_train_main.shape[0]


def make_low_data_split(fraction, seed):
    """Draw the labeled budget + external validation for a low-data model.
    Returns x_train, y_train, x_val, y_val.
    Validation labels come from the validation pool (outside the budget).
    """
    rng = np.random.default_rng(seed)
    n_budget = max(int(np.ceil(fraction * n_total_main)), 32)
    idx = rng.choice(n_total_main, size=n_budget, replace=False)
    v_idx = rng.choice(x_val.shape[0], size=N_EXTERNAL_VAL, replace=False)
    return (x_train_main[idx], y_train_main[idx],
            x_val[v_idx], y_val[v_idx])


x_low, y_low, x_small_val, y_small_val = make_low_data_split(FRACTION_LOW, RNG_SEED)

print(f"\nTraining pool (100% model trains on it):      {x_train_main.shape[0]:>9d} samples")
print(f"Validation pool (100% model selects on it):   {x_val.shape[0]:>9d} samples")
print(f"Low-data train split ({LOW} of the pool):      {x_low.shape[0]:>9d} samples")
print(f"Low-data validation split:                    {x_small_val.shape[0]:>9d} samples  (EXTERNAL, outside budget)")
print(f"Total labels used for training a low-data model: {x_low.shape[0]:>9d}")
print(f"Test pool (final evaluation only):            {x_test.shape[0]:>9d} samples")


#------------------------------------------------------- 3. Vanilla NN, metrics and training harness
# 
# **5-20-10-1** architecture.
# 
 


class Net(nn.Module):
    """Same architecture as the original NNBD model: 5-20-10-1, ReLU hidden,
    linear output (Fig. 2 of the paper)."""
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.hidden1 = nn.Linear(n_hidden, 10)
        self.predict = nn.Linear(10, n_output)

    def forward(self, x):
        x = nn.functional.relu(self.hidden(x))
        x = nn.functional.relu(self.hidden1(x))
        x = self.predict(x)
        return x


def make_loader(x_arr, y_arr, batch_size=BATCH_SIZE, shuffle=False):
    x_t = torch.as_tensor(x_arr, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y_arr, dtype=torch.float32, device=device)
    ds = Data.TensorDataset(x_t, y_t)
    return Data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def mape_bins(y_true, y_pred):
    """Error summary. 'mape' and 'acc_5/10/15/20' keep their original meaning
    (acc_k = share of predictions with absolute percentage error <= k%)."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    ape = np.abs(err / y_true) * 100
    out = {
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "p90_ape": float(np.percentile(ape, 90)),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "r2": float(1.0 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "frac_nonpositive": float(np.mean(y_pred <= 0)),
    }
    for k in (5, 10, 15, 20):
        out[f"acc_{k}"] = float(np.mean(ape <= k))
    return out


def val_mape(y_true, y_pred):
    """Model-selection metric shared by every model. Non-finite predictions count as infinitely bad."""
    v = float(np.mean(np.abs((y_pred - y_true) / y_true)) * 100)
    return v if np.isfinite(v) else float("inf")


def predict_np(model, x_t):
    with torch.no_grad():
        out = model(x_t)
        if isinstance(out, tuple):          # Hybrid returns (pred, phys, res)
            out = out[0]
    return out.cpu().numpy().reshape(-1)


def evaluate_model(model, x, y):
    model.eval()
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    pred = predict_np(model, x_t)
    return mape_bins(y, pred), pred


def train_vanilla_nn(x_train, y_train, x_val, y_val, x_test=None, y_test=None,
                     num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, tag="model",
                     seed=RNG_SEED, patience=None, verbose=True):
    """Train the 5-20-10-1 network with Adam + ReduceLROnPlateau.

    Checkpoints and the LR schedule follow validation MAPE (same criterion as the hybrid).
    The test set is touched once, after the best checkpoint is restored.
    """
    set_seed(seed)
    n_feat = x_train.shape[1]
    model = Net(n_feat, 20, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    train_loader = make_loader(x_train, y_train, shuffle=True)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)

    history = {"train_loss": [], "val_loss": [], "val_mape": []}
    best_val, best_epoch, best_state = float("inf"), -1, None

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        for bx, by in train_loader:
            by = by.view(-1, 1)
            loss = criterion(model(bx), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        history["train_loss"].append(total_loss / n_batches)

        model.eval()
        val_pred = predict_np(model, x_val_t)
        v_mape = val_mape(y_val, val_pred)
        history["val_loss"].append(float(np.mean((val_pred - y_val) ** 2)))
        history["val_mape"].append(v_mape)
        scheduler.step(v_mape)

        if v_mape < best_val:
            best_val, best_epoch = v_mape, epoch
            best_state = copy.deepcopy(model.state_dict())

        if verbose and ((epoch + 1) % 100 == 0 or epoch == num_epochs - 1):
            print(f"[{tag}] epoch {epoch+1:4d}/{num_epochs}  train_loss={history['train_loss'][-1]:.6f}  "
                  f"val_MAPE={v_mape:.2f}%  best_val_MAPE={best_val:.2f}% (ep {best_epoch+1})")

        if patience and (epoch - best_epoch) >= patience:
            if verbose:
                print(f"[{tag}] early stop at epoch {epoch+1}; best epoch {best_epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history["best_epoch"], history["best_val_mape"] = best_epoch, best_val

    if x_test is not None:
        history["final_test"], history["test_pred"] = evaluate_model(model, x_test, y_test)
        if verbose:
            print(f"[{tag}] FINAL (best-val checkpoint) test MAPE={history['final_test']['mape']:.2f}%  "
                  f"acc@10%={history['final_test']['acc_10']:.3f}")
    return model, history


# ## 4. Log-linear (Stress Model) hybrid NN
# 
# `LogLinear` is a single `Linear` layer wrapped in `exp(...)`. `fit_physics_prior` log-linearizes the law and solves ordinary least squares on the labeled training split, which gives the warm start. `Hybrid` multiplies the physics branch by `exp(residual_NN(x))`; the residual's last layer starts at zero, so training starts from the physics-only prediction.
# 


def fit_physics_prior(x_low, y_low, verbose=True):
    """Closed-form OLS on ln(y): returns [b0, b1, ..., b5]."""
    y = np.clip(y_low.astype(np.float64), 1e-6, None)
    design = np.concatenate(
        [np.ones((x_low.shape[0], 1)), x_low.astype(np.float64)], axis=1
    )
    target = np.log(y)
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)
    if verbose:
        print("Physics warm start (log-linear coefficients):")
        for name, val in zip(["b0 (intercept)"] + FEATURE_NAMES, coeffs):
            print(f"    {name:<15s} {val: .4f}")
    return coeffs


class LogLinear(nn.Module):
    """Generalized log-linear stress model:
       BD = exp(b0 + b1*x1 + ... + b5*x5)
    """

    def __init__(self, n_feature, warm_start_beta=None):
        super().__init__()
        self.linear = nn.Linear(n_feature, 1)
        if warm_start_beta is not None:
            b0, *b_rest = warm_start_beta
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor(b_rest, dtype=torch.float32))
                self.linear.bias.copy_(torch.tensor([b0], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(self.linear(x))  # to get back the 'BD' from 'lnBD'


class Hybrid(nn.Module):
    """Hybrid model: log-linear physics branch + a small NN residual branch
    final_prediction = physics_prediction * exp(residual_NN(x))"""

    def __init__(self, n_feature, warm_start_beta=None, hidden=20, mid=10):
        super().__init__()
        self.physics = LogLinear(n_feature, warm_start_beta=warm_start_beta)
        self.residual = nn.Sequential(
            nn.Linear(n_feature, hidden),
            nn.ReLU(),
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x):
        phys = self.physics(x)
        res = self.residual(x)
        pred = phys * torch.exp(res)
        return pred, phys, res


# ## 5. Training loop for the log-linear models
# 
# Full-batch Adam. For the **hybrid**, the loss is the **relative squared error** of the prediction, plus a penalty on the relative error of the physics branch alone (`w_phys_anchor`) and an L2 penalty on the residual (`w_residual_reg`). The pure log-linear model is fitted on the squared error of ln(BD). Checkpoints are chosen by validation MAPE and training stops after `patience` epochs without improvement.
# 


# Hybrid two-phase schedule:
#   Phase 1 — freeze physics, train residual only (forces residual off zero on every seed)
#   Phase 2 — unfreeze all, short joint fine-tune
PHASE1_FRAC = 0.75          # fraction of num_epochs spent on residual-only
PHASE2_LR_SCALE = 0.3       # smaller LR for joint fine-tune


def train_hybrid(x_low, y_low, x_val, y_val, x_test=None, y_test=None,
                 num_epochs=NUM_EPOCHS_HYBRID, lr=LEARNING_RATE_HYBRID,
                 w_phys_anchor=2.0, w_residual_reg=0.05,
                 residual_hidden=20, residual_mid=10,
                 tag="LogLinear-Hybrid",
                 is_pure_loglinear=False,
                 patience=HYBRID_PATIENCE,
                 warm_start=True,
                 seed=RNG_SEED,
                 verbose=True):

    set_seed(seed)
    warm_start_beta = fit_physics_prior(x_low, y_low, verbose=verbose)
    start = warm_start_beta if warm_start else None

    if is_pure_loglinear:
        model = LogLinear(x_low.shape[1], warm_start_beta=start).to(device)
    else:
        model = Hybrid(x_low.shape[1], warm_start_beta=start,
                       hidden=residual_hidden, mid=residual_mid).to(device)

    criterion = nn.MSELoss()
    x_low_t = torch.as_tensor(x_low, dtype=torch.float32, device=device)
    y_low_t = torch.as_tensor(y_low, dtype=torch.float32, device=device).reshape(-1, 1)
    x_val_t = torch.as_tensor(x_val, dtype=torch.float32, device=device)

    history = {"train_loss": [], "val_loss": [], "val_mape": [], "res_rms": [],
               "warm_start_beta": warm_start_beta, "phase": []}
    best_val, best_epoch, best_state = float("inf"), -1, None
    global_epoch = 0

    def _eval_and_track(loss_val, res_rms, phase_name):
        nonlocal best_val, best_epoch, best_state, global_epoch
        history["train_loss"].append(loss_val)
        history["res_rms"].append(res_rms)
        history["phase"].append(phase_name)
        model.eval()
        val_pred = predict_np(model, x_val_t)
        v_mape = val_mape(y_val, val_pred)
        history["val_loss"].append(float(np.mean((val_pred - y_val) ** 2)))
        history["val_mape"].append(v_mape)
        if v_mape < best_val:
            best_val, best_epoch = v_mape, global_epoch
            best_state = copy.deepcopy(model.state_dict())
        global_epoch += 1
        return v_mape

    # ------------------------------------------- pure log-linear:--------------------
    if is_pure_loglinear:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
        for epoch in range(num_epochs):
            model.train()
            pred = model(x_low_t)
            loss = criterion(torch.log(torch.clamp(pred, min=1e-6)),
                             torch.log(torch.clamp(y_low_t, min=1e-6)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            v_mape = _eval_and_track(loss.item(), 0.0, "loglin")
            scheduler.step(v_mape)
            if verbose and ((epoch + 1) % 100 == 0 or epoch == num_epochs - 1):
                print(f"[{tag}] epoch {epoch+1:4d}/{num_epochs}  loss={loss.item():.6f}  "
                      f"val_MAPE={v_mape:.2f}%  best_val_MAPE={best_val:.2f}% (ep {best_epoch+1})")
            if patience and (global_epoch - 1 - best_epoch) >= patience:
                if verbose:
                    print(f"[{tag}] early stop at epoch {epoch+1}; best epoch {best_epoch+1}")
                break

    # -------------------- hybrid: two-phase residual-then-joint --------------------
    else:
        n_phase1 = max(int(num_epochs * PHASE1_FRAC), 100)
        n_phase2 = max(num_epochs - n_phase1, 50)
        # Freeze physics only when warm-started; 
        freeze_physics = bool(warm_start)

        # Phase 1 — train residual (physics frozen if warm-started)
        if freeze_physics:
            for p in model.physics.parameters():
                p.requires_grad_(False)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr
        )
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

        for epoch in range(n_phase1):
            model.train()
            pred, phys, res = model(x_low_t)
            data_loss = torch.mean(((pred - y_low_t) / y_low_t) ** 2)
            res_reg = torch.mean(res ** 2)
            if freeze_physics:
                loss = data_loss + w_residual_reg * res_reg
            else:
                # no warm start: also allow physics to move in phase 1
                phys_loss = torch.mean(((phys - y_low_t) / y_low_t) ** 2)
                loss = data_loss + w_phys_anchor * phys_loss + w_residual_reg * res_reg
            res_rms = float(torch.sqrt(res_reg).detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            v_mape = _eval_and_track(loss.item(), res_rms, "residual")
            scheduler.step(v_mape)
            if verbose and ((epoch + 1) % 100 == 0 or epoch == n_phase1 - 1):
                print(f"[{tag}|P1-residual] epoch {epoch+1:4d}/{n_phase1}  loss={loss.item():.6f}  "
                      f"val_MAPE={v_mape:.2f}%  best={best_val:.2f}% (ep {best_epoch+1})  "
                      f"res_rms={res_rms:.4f}")
            # Early-stop only after residual has had a real chance to move
            min_p1 = min(150, n_phase1 // 2)
            if (epoch + 1) >= min_p1 and patience and (global_epoch - 1 - best_epoch) >= patience:
                if verbose:
                    print(f"[{tag}|P1-residual] early stop at epoch {epoch+1}; best ep {best_epoch+1}")
                break

        # Restore best weights before phase 2
        if best_state is not None:
            model.load_state_dict(best_state)

        # Phase 2 — unfreeze physics, joint fine-tune at lower LR
        for p in model.physics.parameters():
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr * PHASE2_LR_SCALE)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)

        for epoch in range(n_phase2):
            model.train()
            pred, phys, res = model(x_low_t)
            data_loss = torch.mean(((pred - y_low_t) / y_low_t) ** 2)
            phys_loss = torch.mean(((phys - y_low_t) / y_low_t) ** 2)
            res_reg = torch.mean(res ** 2)
            loss = data_loss + w_phys_anchor * phys_loss + w_residual_reg * res_reg
            res_rms = float(torch.sqrt(res_reg).detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            v_mape = _eval_and_track(loss.item(), res_rms, "joint")
            scheduler.step(v_mape)
            if verbose and ((epoch + 1) % 50 == 0 or epoch == n_phase2 - 1):
                print(f"[{tag}|P2-joint] epoch {epoch+1:4d}/{n_phase2}  loss={loss.item():.6f}  "
                      f"val_MAPE={v_mape:.2f}%  best={best_val:.2f}% (ep {best_epoch+1})  "
                      f"res_rms={res_rms:.4f}")
            if patience and (global_epoch - 1 - best_epoch) >= patience:
                if verbose:
                    print(f"[{tag}|P2-joint] early stop at epoch {epoch+1}; best ep {best_epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    history["best_epoch"], history["best_val_mape"] = best_epoch, best_val

    if x_test is not None:
        history["final_test"], history["test_pred"] = evaluate_model(model, x_test, y_test)
        if verbose:
            print(f"[{tag}] FINAL (best-val checkpoint) test MAPE={history['final_test']['mape']:.2f}%  "
                  f"acc@10%={history['final_test']['acc_10']:.3f}")

    return model, history


# ## 6. Hyperparameter selection (validation only)
# 
# Learning rate for the vanilla low-data model; learning rate and `w_phys_anchor` for the hybrid. Each configuration is scored by its best validation MAPE, averaged over `N_TUNE_SEEDS` label subsets that are not reused later (seeds 1000 and up). Disabled by default (`RUN_HPARAM_SEARCH = False`);
# 


def tune_hparams():
    """Grid search scored on the validation split; the test set is never used."""
    seeds = [1000 + i for i in range(N_TUNE_SEEDS)]
    splits = [make_low_data_split(FRACTION_LOW, s) for s in seeds]
    rows = []
    for lr in LR_GRID_VNN:
        scores = [train_vanilla_nn(xl, yl, xv, yv, num_epochs=NUM_EPOCHS_VNN, lr=lr, seed=s,
                                   patience=VNN_PATIENCE, verbose=False)[1]["best_val_mape"]
                  for (xl, yl, xv, yv), s in zip(splits, seeds)]
        rows.append({"model": "vanilla", "lr": lr, "w_phys_anchor": np.nan, "val_mape": float(np.mean(scores))})
    for lr in LR_GRID_HYBRID:
        for w in ANCHOR_GRID:
            scores = [train_hybrid(xl, yl, xv, yv, num_epochs=NUM_EPOCHS_HYBRID, lr=lr, w_phys_anchor=w,
                                   w_residual_reg=HPARAMS["hybrid"]["w_residual_reg"], seed=s,
                                   verbose=False)[1]["best_val_mape"]
                      for (xl, yl, xv, yv), s in zip(splits, seeds)]
            rows.append({"model": "hybrid", "lr": lr, "w_phys_anchor": w, "val_mape": float(np.mean(scores))})
    return pd.DataFrame(rows)


if RUN_HPARAM_SEARCH:
    print("Running hyperparameter search on validation data only...")
    tune_df = tune_hparams()
    print(tune_df.to_string(index=False))
    # Update HPARAMS from best validation scores
    for kind in ("vanilla", "hybrid"):
        sub = tune_df[tune_df.model == kind].sort_values("val_mape")
        best = sub.iloc[0]
        if kind == "vanilla":
            HPARAMS["vanilla"]["lr"] = float(best["lr"])
        else:
            HPARAMS["hybrid"]["lr"] = float(best["lr"])
            HPARAMS["hybrid"]["w_phys_anchor"] = float(best["w_phys_anchor"])
    print("Updated HPARAMS:", json.dumps(HPARAMS, indent=2))
else:
    print("RUN_HPARAM_SEARCH is False; using defaults from the working prototype.")
    print("HPARAMS:", json.dumps(HPARAMS, indent=2))


# ##---------------------------------------- 7. Representative run (seed 0)
# 
# Four models on the same label subset:
# 
# 1. `Vanilla-NN-100%`: the NNBD network on the full training pool (upper-bound reference).
# 2. `Vanilla-NN-0.01%`: the same network on the low-data budget (data-starved baseline).
# 3. `PureLogLinear-0.01%`: the physics branch alone.
# 4. `Hybrid-NN-0.01%`: physics branch × NN residual.
# 


def banner(text):
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)

banner(f"{NAMES['v100']}: NNBD network on 100% of the training pool (upper-bound reference)")
model_100, hist_100 = train_vanilla_nn(
    x_train_main, y_train_main, x_val, y_val, x_test, y_test,
    num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, tag=NAMES["v100"]
)

banner(f"{NAMES['vlow']}: same network on the low-data budget (data-starved baseline)")
model_1pct, hist_1pct = train_vanilla_nn(
    x_low, y_low, x_small_val, y_small_val, x_test, y_test,
    num_epochs=NUM_EPOCHS_VNN, lr=HPARAMS["vanilla"]["lr"], tag=NAMES["vlow"], patience=VNN_PATIENCE
)

banner(f"{NAMES['loglin']}: physics branch alone")
model_pure_log, hist_pure_log = train_hybrid(
    x_low, y_low, x_small_val, y_small_val, x_test, y_test,
    num_epochs=NUM_EPOCHS_HYBRID, lr=HPARAMS["loglinear"]["lr"],
    tag=NAMES["loglin"], is_pure_loglinear=True
)

# Seed-0 hybrid: train candidate A (always) and optionally C; pick by val MAPE
hp = HPARAMS["hybrid"]
seed0_cands = {}
for cname, ccfg in HYBRID_CANDIDATES.items():
    if cname != "A" and not SELECT_HYBRID_BY_VAL:
        continue
    banner(f"{NAMES['hybrid']} candidate {cname}: hidden={ccfg['residual_hidden']}, "
           f"mid={ccfg['residual_mid']}, w_res={ccfg['w_residual_reg']}")
    kw = dict(lr=hp["lr"], w_phys_anchor=hp["w_phys_anchor"], **ccfg)
    m, h = train_hybrid(
        x_low, y_low, x_small_val, y_small_val, x_test, y_test,
        num_epochs=NUM_EPOCHS_HYBRID, tag=f"{NAMES['hybrid']}-{cname}", **kw
    )
    seed0_cands[cname] = (m, h)
    if h.get("res_rms"):
        r0, rT = h["res_rms"][0], h["res_rms"][-1]
        print(f"[Hybrid-{cname}] val_MAPE={h['best_val_mape']:.2f}%  "
              f"test_MAPE={h['final_test']['mape']:.2f}%  "
              f"res_rms {r0:.4f}->{rT:.4f}  best_ep={h['best_epoch']+1}")

if SELECT_HYBRID_BY_VAL and len(seed0_cands) > 1:
    best_c = min(seed0_cands, key=lambda c: seed0_cands[c][1]["best_val_mape"])
else:
    best_c = "A"
model_hybrid_nn, hist_hybrid_nn = seed0_cands[best_c]
print(f"\n>>> Seed-0 selected Hybrid candidate: {best_c} "
      f"(val MAPE={hist_hybrid_nn['best_val_mape']:.2f}%)")

runs = {
    NAMES["v100"]: hist_100,
    NAMES["vlow"]: hist_1pct,
    NAMES["loglin"]: hist_pure_log,
    NAMES["hybrid"]: hist_hybrid_nn,
}


# ##------------------------------------------- 8. Training curves and diagnostics
# 


x_cdf = np.linspace(0, 40, 161)
plot_idx = np.random.default_rng(0).choice(len(y_test), size=min(4000, len(y_test)), replace=False)

fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.0))

for name, h in runs.items():
    if name != NAMES["v100"]:
        axs[0].plot(np.arange(1, len(h["val_mape"]) + 1), h["val_mape"], color=colors[name], lw=1.2)
axs[0].set(xscale="log", yscale="log", xlabel="Epoch", ylabel="Validation MAPE (%)")
plain_log_axis(axs[0], "x")
plain_log_axis(axs[0], "y")
axs[0].set_title("(a) Convergence", loc="left")

for name, h in runs.items():
    ape_sorted = np.sort(np.abs((h["test_pred"] - y_test) / y_test) * 100)
    axs[1].plot(x_cdf, np.searchsorted(ape_sorted, x_cdf, side="right") / ape_sorted.size,
                color=colors[name], lw=1.4)
axs[1].set(xlabel="Error tolerance (APE, %)", ylabel="Share within tolerance", ylim=(0, 1.0))
axs[1].set_title("(b) Accuracy vs. tolerance", loc="left")

lim = (0.0, float(y_test[plot_idx].max()) * 1.02)
for name in (NAMES["vlow"], NAMES["hybrid"]):
    axs[2].scatter(y_test[plot_idx], runs[name]["test_pred"][plot_idx], s=3, alpha=0.35,
                   color=colors[name], linewidths=0)
axs[2].plot(lim, lim, color="black", lw=0.8)
axs[2].set(xlim=lim, ylim=lim, xlabel="True BD", ylabel="Predicted BD", aspect="equal")
axs[2].set_title("(c) Predicted vs. true", loc="left")

handles = [plt.Line2D([], [], color=colors[n], lw=1.8, label=n) for n in runs]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
           fontsize=7, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save_fig(fig, "fig1_representative_run")


# ##-------------------------------------------- 9. Final comparison (single run)
# 


SINGLE_COLS = ["mape", "median_ape", "p90_ape", "acc_10", "acc_15", "acc_20", "rmse", "r2", "frac_nonpositive"]
single_run = pd.DataFrame(
    [{"model": name, **{k: h["final_test"][k] for k in SINGLE_COLS}} for name, h in runs.items()]
)
print(single_run.round(4).to_string(index=False))
single_run.to_csv(RESULTS_DIR / "single_run_seed0.csv", index=False)

print("\nPhysical consistency on the test set (degradation must be positive):")
for name, h in runs.items():
    print(f"  {name:<24s} min prediction = {h['test_pred'].min(): .4g}   "
          f"share <= 0: {h['final_test']['frac_nonpositive']:.4%}")


# ##-------------------------------------- 10. Repeated runs and statistical tests
# 
# Each seed draws a new label subset and a new initialisation; all models share the subset within a seed. Table entries are mean ± standard deviation over seeds. In the paired tests, `mean_diff` is baseline minus hybrid, so positive values favor the hybrid; the interval is a bootstrap 95% CI of the mean difference, and p-values are two-sided Wilcoxon signed-rank tests with Holm correction across baselines.
# 


def _record(hist, model, seed, fraction, n_train, n_val):
    return {"seed": seed, "fraction": fraction, "model": model, "n_train": n_train, "n_val": n_val,
            "best_epoch": hist["best_epoch"] + 1, "best_val_mape": hist["best_val_mape"],
            **hist["final_test"]}


ABL = {
    "nowarm": "Hybrid, no warm start",
    "noanchor": "Hybrid, no physics anchor",
    "noreg": "Hybrid, no residual penalty",
}


def run_all_models(fraction, seed, ablations=False):
    """Train the four core models (and optional hybrid ablations) on one label subset.

    Hybrid selection (safe for Option C):
      - Always train Hybrid-A (current architecture / reg) and record it as "Hybrid-A".
      - If SELECT_HYBRID_BY_VAL, also train Hybrid-C; report "Hybrid" = the candidate
        with lower validation MAPE. On val this cannot be worse than A alone.
    """
    xl, yl, xv, yv = make_low_data_split(fraction, seed)
    n_tr, n_va = len(xl), len(xv)
    common = dict(x_test=x_test, y_test=y_test, seed=seed, verbose=False)
    rec = []

    _, h = train_vanilla_nn(xl, yl, xv, yv, num_epochs=NUM_EPOCHS_VNN, lr=HPARAMS["vanilla"]["lr"],
                            patience=VNN_PATIENCE, **common)
    rec.append(_record(h, SHORT["vlow"], seed, fraction, n_tr, n_va))

    _, h = train_hybrid(xl, yl, xv, yv, num_epochs=NUM_EPOCHS_HYBRID, lr=HPARAMS["loglinear"]["lr"],
                        is_pure_loglinear=True, **common)
    rec.append(_record(h, SHORT["loglin"], seed, fraction, n_tr, n_va))

    # --- Hybrid candidates A (and optionally C); select by val MAPE ---
    base_hp = dict(HPARAMS["hybrid"])
    cand_hists = {}
    for cname, ccfg in HYBRID_CANDIDATES.items():
        if cname != "A" and not SELECT_HYBRID_BY_VAL:
            continue
        kw = dict(base_hp)
        kw.update(ccfg)
        _, h = train_hybrid(xl, yl, xv, yv, num_epochs=NUM_EPOCHS_HYBRID, **kw, **common)
        cand_hists[cname] = h
        # Always keep A visible under its own name
        if cname == "A":
            rec.append(_record(h, "Hybrid-A", seed, fraction, n_tr, n_va))
        elif cname == "C":
            rec.append(_record(h, "Hybrid-C", seed, fraction, n_tr, n_va))

    # Selected Hybrid for the main comparison table
    if SELECT_HYBRID_BY_VAL and len(cand_hists) > 1:
        best_c = min(cand_hists, key=lambda c: cand_hists[c]["best_val_mape"])
    else:
        best_c = "A"
    h_sel = cand_hists[best_c]
    rec_sel = _record(h_sel, SHORT["hybrid"], seed, fraction, n_tr, n_va)
    rec_sel["hybrid_candidate"] = best_c
    rec.append(rec_sel)

    if ablations:
        for name, override in [
            (ABL["nowarm"],   {"warm_start": False}),
            (ABL["noanchor"], {"w_phys_anchor": 0.0}),
            (ABL["noreg"],    {"w_residual_reg": 0.0}),
        ]:
            kw = dict(base_hp)
            kw.update(override)
            # Ablations use candidate-A architecture
            _, h = train_hybrid(xl, yl, xv, yv, num_epochs=NUM_EPOCHS_HYBRID, **kw, **common)
            rec.append(_record(h, name, seed, fraction, n_tr, n_va))
    return rec


records = []
for seed in tqdm(range(N_SEEDS), desc="seeds"):
    records += run_all_models(FRACTION_LOW, seed, ablations=RUN_ABLATIONS)
    pd.DataFrame(records).to_csv(RESULTS_DIR / "repeated_runs.csv", index=False)
df_seeds = pd.DataFrame(records)
print(f"{len(df_seeds)} model fits over {N_SEEDS} seeds saved to {RESULTS_DIR / 'repeated_runs.csv'}")


SUMMARY_METRICS = ["mape", "median_ape", "p90_ape", "acc_10", "acc_20", "rmse", "r2", "frac_nonpositive"]
METRIC_LABEL = {"mape": "MAPE (%)", "median_ape": "Median APE (%)", "p90_ape": "P90 APE (%)",
                "acc_10": "Acc@10%", "acc_20": "Acc@20%", "rmse": "RMSE", "r2": "R²",
                "frac_nonpositive": "Share ≤ 0"}
METRIC_DIGITS = {"mape": 2, "median_ape": 2, "p90_ape": 2, "acc_10": 3, "acc_20": 3,
                 "rmse": 4, "r2": 4, "frac_nonpositive": 4}


def ci95_halfwidth(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan


def summarize(df, models, metrics=SUMMARY_METRICS):
    rows = []
    for m in models:
        d = df[df.model == m]
        row = {"model": m, "n_runs": len(d)}
        for k in metrics:
            row[f"{k}_mean"] = d[k].mean()
            row[f"{k}_std"] = d[k].std(ddof=1)
            row[f"{k}_ci95"] = ci95_halfwidth(d[k])
            row[f"{k}_median"] = d[k].median()
        rows.append(row)
    return pd.DataFrame(rows)


def pretty(summary, metrics=SUMMARY_METRICS):
    out = pd.DataFrame({"Model": summary["model"], "Runs": summary["n_runs"]})
    for k in metrics:
        d = METRIC_DIGITS[k]
        cells = []
        for i in range(len(summary)):
            m, s = summary[f"{k}_mean"].iloc[i], summary[f"{k}_std"].iloc[i]
            med = summary[f"{k}_median"].iloc[i]
            if not np.isfinite(s):
                cells.append(f"{m:.{d}f}")
            elif k == "mape":
                cells.append(f"{m:.{d}f} ± {s:.{d}f} (med {med:.{d}f})")
            else:
                cells.append(f"{m:.{d}f} ± {s:.{d}f}")
        out[METRIC_LABEL[k]] = cells
    return out


def save_table(df, name):
    df.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    try:
        tex = df.to_latex(index=False, escape=True).replace("±", r"$\pm$")
        (RESULTS_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")
    except Exception as exc:
        print(f"(LaTeX export skipped for {name}: {exc})")


def paired_tests(df, ref_model, baselines, metric="mape", n_boot=10000, seed=0):
    """Paired comparison across seeds. mean_diff = baseline - ref, so positive favors ref (lower error)."""
    wide = df.pivot(index="seed", columns="model", values=metric)
    rng = np.random.default_rng(seed)
    rows = []
    for b in baselines:
        d = (wide[b] - wide[ref_model]).dropna().to_numpy()
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        try:
            p = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            p = np.nan
        rows.append({"baseline": b, "n_pairs": len(d), "mean_diff": d.mean(),
                     "ci95_lo": np.percentile(boots, 2.5), "ci95_hi": np.percentile(boots, 97.5),
                     "win_rate": float(np.mean(d > 0)), "p_wilcoxon": p})
    out = pd.DataFrame(rows)
    p = np.nan_to_num(out["p_wilcoxon"].to_numpy(), nan=1.0)
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    out["p_holm"] = adj
    return out


MAIN_MODELS = [SHORT[k] for k in ("vlow", "loglin", "hybrid")]
BASELINES = [SHORT[k] for k in ("vlow", "loglin")]

summary_main = summarize(df_seeds, MAIN_MODELS)
ref_row = {"model": SHORT["v100"], "n_runs": 1}
for k in SUMMARY_METRICS:
    ref_row[f"{k}_mean"], ref_row[f"{k}_std"], ref_row[f"{k}_ci95"] = hist_100["final_test"][k], np.nan, np.nan
summary_main = pd.concat([pd.DataFrame([ref_row]), summary_main], ignore_index=True)

table_main = pretty(summary_main)
print(f"Test-set results at {LOW} labels, mean ± std over {N_SEEDS} seeds "
      f"({SHORT['v100']}: single run on 100% of the pool)\n")
print(table_main.to_string(index=False))
save_table(table_main, "table_main_results")
summary_main.to_csv(RESULTS_DIR / "summary_main_numeric.csv", index=False)

tests_main = pd.concat(
    [paired_tests(df_seeds, SHORT["hybrid"], BASELINES, metric=m).assign(metric=m) for m in ("mape", "median_ape")],
    ignore_index=True,
)
print("\nPaired tests, Hybrid vs. each baseline (mean_diff > 0: hybrid has lower error)\n")
print(tests_main.round(4).to_string(index=False))
tests_main.to_csv(RESULTS_DIR / "tests_main.csv", index=False)

# Per-seed hybrid MAPE 
cols = ["seed", "mape", "acc_10", "best_epoch"]
if "hybrid_candidate" in df_seeds.columns:
    cols.append("hybrid_candidate")
hyb_seed = df_seeds[df_seeds.model == SHORT["hybrid"]][cols].sort_values("seed")
print("\nPer-seed selected Hybrid (best_epoch >> 1 means residual trained; candidate A=current, C=option C):")
print(hyb_seed.round(3).to_string(index=False))
hyb_seed.to_csv(RESULTS_DIR / "hybrid_per_seed.csv", index=False)


if "Hybrid-A" in set(df_seeds.model):
    a = df_seeds[df_seeds.model == "Hybrid-A"][["seed", "mape"]].rename(columns={"mape": "mape_A"})
    s = df_seeds[df_seeds.model == SHORT["hybrid"]][["seed", "mape"]].rename(columns={"mape": "mape_sel"})
    cmp = a.merge(s, on="seed")
    cmp["delta_sel_minus_A"] = cmp["mape_sel"] - cmp["mape_A"]
    print("\nSelected Hybrid vs Hybrid-A (negative delta = selection improved test MAPE vs A):")
    print(cmp.round(3).to_string(index=False))
    print(f"  mean test MAPE  A={cmp['mape_A'].mean():.2f}%  selected={cmp['mape_sel'].mean():.2f}%")
    if "hybrid_candidate" in hyb_seed.columns:
        print("  candidate wins:", hyb_seed["hybrid_candidate"].value_counts().to_dict())
    cmp.to_csv(RESULTS_DIR / "hybrid_A_vs_selected.csv", index=False)


def box_panel(ax, df, order, metric, ylabel, palette, ref=None, ref_label=None):
    """Box plot over seeds with every seed drawn as a dot."""
    data = [df.loc[df.model == m, metric].to_numpy() for m in order]
    bp = ax.boxplot(data, positions=np.arange(len(order)), widths=0.55, showfliers=False,
                    patch_artist=True, medianprops=dict(color="black", lw=1.2))
    jitter = np.random.default_rng(0)
    for i, (m, d) in enumerate(zip(order, data)):
        bp["boxes"][i].set(facecolor=palette[m], alpha=0.30, edgecolor=palette[m])
        ax.scatter(i + jitter.uniform(-0.12, 0.12, len(d)), d, s=8, color=palette[m], zorder=3, linewidths=0)
    if ref is not None:
        ax.axhline(ref, color=colors[NAMES["v100"]], ls="--", lw=1.0, label=ref_label)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel(ylabel)


fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, metric, ylabel in ((axs[0], "mape", "Test MAPE (%)"), (axs[1], "acc_10", "Test accuracy @ 10% tolerance")):
    box_panel(ax, df_seeds, MAIN_MODELS, metric, ylabel, colors_short,
              ref=hist_100["final_test"][metric], ref_label=f"{SHORT['v100']} (single run)")
axs[0].set_title(f"(a) Error, {N_SEEDS} seeds, {LOW} labels", loc="left")
axs[1].set_title("(b) Accuracy within tolerance", loc="left")
axs[0].legend(frameon=False, loc="upper right")
fig.tight_layout()
save_fig(fig, "fig2_repeated_runs")


# --------------------------------- Ablations (same seeds / label subsets as the main table) ----
if RUN_ABLATIONS:
    ABL_MODELS = [SHORT["hybrid"], ABL["nowarm"], ABL["noanchor"], ABL["noreg"], SHORT["loglin"]]
    abl_colors = {
        SHORT["hybrid"]: colors[NAMES["hybrid"]],
        ABL["nowarm"]: "#E69F00",
        ABL["noanchor"]: "#CC79A7",
        ABL["noreg"]: "#999999",
        SHORT["loglin"]: colors[NAMES["loglin"]],
    }
    summary_abl = summarize(df_seeds, ABL_MODELS)
    table_abl = pretty(summary_abl, metrics=["mape", "median_ape", "acc_10", "acc_20"])
    print("\nAblation results (mean ± std):\n")
    print(table_abl.to_string(index=False))
    save_table(table_abl, "table_ablation")

    tests_abl = paired_tests(df_seeds, SHORT["hybrid"], ABL_MODELS[1:], metric="mape")
    print("\nPaired tests vs. full Hybrid (mean_diff > 0: full hybrid has lower MAPE)\n")
    print(tests_abl.round(4).to_string(index=False))
    tests_abl.to_csv(RESULTS_DIR / "tests_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    box_panel(ax, df_seeds, ABL_MODELS, "mape", "Test MAPE (%)", abl_colors)
    ax.set_title(f"Ablation, {N_SEEDS} seeds, {LOW} labels", loc="left")
    fig.tight_layout()
    save_fig(fig, "fig3_ablation")
else:
    print("RUN_ABLATIONS is False; skipped.")


# ##------------------------------------------------------- 11. Data-efficiency sweep
# 
# Test MAPE as a function of the label budget: mean and 95% CI over `SWEEP_SEEDS` seeds, with hyperparameters held fixed across budgets. The dashed line is the Vanilla-NN-100% reference. Budgets count training labels only (validation is external). 
# 


if RUN_SWEEP:
    sweep_records = []
    for frac in SWEEP_FRACTIONS:
        if frac == FRACTION_LOW and SWEEP_SEEDS <= N_SEEDS:
            sweep_records += df_seeds[(df_seeds.seed < SWEEP_SEEDS) & df_seeds.model.isin(MAIN_MODELS)].to_dict("records")
            continue
        for seed in tqdm(range(SWEEP_SEEDS), desc=f"budget {frac * 100:g}%"):
            sweep_records += run_all_models(frac, seed)
        pd.DataFrame(sweep_records).to_csv(RESULTS_DIR / "sweep_runs.csv", index=False)
    df_sweep = pd.DataFrame(sweep_records)
    df_sweep.to_csv(RESULTS_DIR / "sweep_runs.csv", index=False)

    sweep_summary = (df_sweep.groupby(["fraction", "model"])["mape"]
                     .agg(mean="mean", std="std", n="count", ci95=ci95_halfwidth).reset_index())
    n_labels = df_sweep.groupby("fraction")["n_train"].first()
    sweep_summary["labels"] = sweep_summary["fraction"].map(n_labels)
    sweep_summary.insert(0, "budget_%", sweep_summary["fraction"] * 100)
    sweep_summary.to_csv(RESULTS_DIR / "sweep_summary.csv", index=False)
    print(sweep_summary.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    for m in MAIN_MODELS:
        s = sweep_summary[sweep_summary.model == m].sort_values("labels")
        ax.plot(s["labels"], s["mean"], marker="o", ms=3.5, lw=1.4, color=colors_short[m], label=m)
        ax.fill_between(s["labels"], (s["mean"] - s["ci95"]).clip(lower=1e-3), s["mean"] + s["ci95"],
                        color=colors_short[m], alpha=0.18, linewidth=0)
    ax.axhline(hist_100["final_test"]["mape"], color=colors[NAMES["v100"]], ls="--", lw=1.0, label=SHORT["v100"])
    ax.set(xscale="log", yscale="log", xlabel="Labeled training samples", ylabel="Test MAPE (%)")
    plain_log_axis(ax, "x")
    plain_log_axis(ax, "y")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_fig(fig, "fig4_data_efficiency")
else:
    print("RUN_SWEEP is False; skipped.")


# ##---------------------------------------- 12. Fitted coefficients and physical consistency
# 
# Coefficients are in the units of the input features, so read their signs and sizes against the dataset's scaling, not against raw physical units. The table compares the closed-form fit on the full training pool with the fit on the low-data split and with the physics branch of the trained hybrid.
# 


b_full = fit_physics_prior(x_train_main, y_train_main, verbose=False)
b_low = hist_hybrid_nn["warm_start_beta"]
lin = model_hybrid_nn.physics.linear
b_hybrid = np.concatenate([lin.bias.detach().cpu().numpy(), lin.weight.detach().cpu().numpy().ravel()])

coef_table = pd.DataFrame({
    "term": ["b0 (intercept)"] + FEATURE_NAMES,
    "OLS, full pool": b_full,
    f"OLS, {LOW} split": b_low,
    "Hybrid physics branch": b_hybrid,
})
print(coef_table.round(4).to_string(index=False))
coef_table.to_csv(RESULTS_DIR / "coefficients.csv", index=False)

# How good is the physics branch of the hybrid on its own (no residual)?
with torch.no_grad():
    phys_only = model_hybrid_nn(torch.as_tensor(x_test, dtype=torch.float32, device=device))[1].cpu().numpy().reshape(-1)
print("\nPhysics branch of the trained hybrid, evaluated alone on the test set:")
print({k: round(v, 4) for k, v in mape_bins(y_test, phys_only).items() if k in ("mape", "median_ape", "acc_10", "r2")})


# ##------------------------------------------------------------ 13. Export
# 


CONFIG = {k: v for k, v in globals().items()
          if k.isupper() and isinstance(v, (int, float, str, bool, list))}
CONFIG.update({"HPARAMS": HPARAMS, "ENV": ENV, "DATA_DIR": str(DATA_DIR)})
(RESULTS_DIR / "config.json").write_text(json.dumps(CONFIG, indent=2, default=str))

# Full model (architecture + weights). Loading it needs the Hybrid / LogLinear classes from Section 4 in scope:
#   loaded = torch.load("full_hybrid_model.pt", weights_only=False, map_location="cpu"); loaded.eval()
torch.save(copy.deepcopy(model_hybrid_nn).cpu(), RESULTS_DIR / "full_hybrid_model.pt")

# Portable checkpoint (plain tensors and numbers only, safe with weights_only=True).
torch.save(
    {"state_dict": {k: v.detach().cpu() for k, v in model_hybrid_nn.state_dict().items()},
     "feature_names": FEATURE_NAMES, "hidden": 20,
     "warm_start_beta": [float(b) for b in hist_hybrid_nn["warm_start_beta"]],
     "hparams": HPARAMS["hybrid"], "seed": RNG_SEED},
    RESULTS_DIR / "hybrid_checkpoint.pt",
)
print("Saved: full_hybrid_model.pt, config.json, hybrid_checkpoint.pt and the tables/figures in", RESULTS_DIR.resolve())
