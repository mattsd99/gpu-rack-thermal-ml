# surrogate reducer-order model for DGX H100 SuperPOD thermal prediction
# trains an MLP + CNN to predict temperature and velocity fields
# from 10 input parameters (inlet velocity, supply temp, 8 chassis loads)
#
# usage:
#   python surrogate_model_main.py --mode train
#   python surrogate_model_main.py --mode eval

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# field resolution from the CFD mesh
FIELD_H = 111
FIELD_W = 194

# physical ranges for each input parameter (used for normalization)
PARAM_RANGES = {
    'v_in_mps': (1.0, 4.0),
    'T_sup_K':  (291.0, 300.0),
    'f1': (0.25, 1.0),
    'f2': (0.25, 1.0),
    'f3': (0.25, 1.0),
    'f4': (0.25, 1.0),
    'f5': (0.25, 1.0),
    'f6': (0.25, 1.0),
    'f7': (0.25, 1.0),
    'f8': (0.25, 1.0),
}

N_INPUTS   = 10
LAMBDA_U   = 0.5   # how much to weight velocity loss vs temperature
BATCH_SIZE = 16
LR         = 1e-4
N_EPOCHS   = 500

SAVE_DIR      = './checkpoints'
RESULTS_DIR   = './results'
DATA_PATH     = './dataset.npz'
VAL_DATA_PATH = './test_dataset.npz'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------- dataset -------

class SuperPODDataset(Dataset):
    def __init__(self, params, T_fields, U_fields):
        self.params   = torch.tensor(params,   dtype=torch.float32)
        self.T_fields = torch.tensor(T_fields, dtype=torch.float32)
        self.U_fields = torch.tensor(U_fields, dtype=torch.float32)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.T_fields[idx], self.U_fields[idx]


def normalize_params(params):
    out = np.zeros_like(params, dtype=np.float32)
    for i, (lo, hi) in enumerate(PARAM_RANGES.values()):
        out[:, i] = (params[:, i] - lo) / (hi - lo)
    return out


def get_field_stats(fields):
    mean = fields.mean(axis=(0, 2, 3), keepdims=True)
    std  = fields.std(axis=(0, 2, 3),  keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return mean, std


def normalize(f, mean, std):
    return (f - mean) / std

def denormalize(f, mean, std):
    return f * std + mean


def load_data(data_path=DATA_PATH, val_path=VAL_DATA_PATH):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Training data not found: {data_path}')

    tr = np.load(data_path, allow_pickle=True)
    tr_params = normalize_params(tr['params'])
    T_tr = tr['T_fields']
    U_tr = tr['U_fields']

    # compute stats on training set only, apply to both splits
    T_mean, T_std = get_field_stats(T_tr)
    U_mean, U_std = get_field_stats(U_tr)
    norm_stats = {'T_mean': T_mean, 'T_std': T_std,
                  'U_mean': U_mean, 'U_std': U_std}

    train_set = SuperPODDataset(
        tr_params,
        normalize(T_tr, T_mean, T_std),
        normalize(U_tr, U_mean, U_std)
    )

    print(f'Loaded {data_path}: {len(tr_params)} training cases')
    print(f'  T: {T_tr.min():.1f} - {T_tr.max():.1f} K')
    print(f'  U: {U_tr.min():.2f} - {U_tr.max():.2f} m/s')

    val_set = None
    if os.path.exists(val_path):
        vd = np.load(val_path, allow_pickle=True)
        val_set = SuperPODDataset(
            normalize_params(vd['params']),
            normalize(vd['T_fields'], T_mean, T_std),
            normalize(vd['U_fields'], U_mean, U_std)
        )
        print(f'Loaded {val_path}: {len(vd["params"])} val cases')
    else:
        print(f'[warn] {val_path} not found, no validation set')

    return train_set, val_set, norm_stats


# ------- model -------

# latent seed shape -- 4:7 roughly matches the 111:194 field aspect ratio
SEED_C = 128
SEED_H = 4
SEED_W = 7


class MLPEncoder(nn.Module):
    # takes the 10 input params and maps them to a small spatial seed
    # kept simple since theres not much to learn from 10 numbers
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_INPUTS, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 256),      nn.LeakyReLU(0.2),
            nn.Linear(256, SEED_C * SEED_H * SEED_W),
        )

    def forward(self, x):
        return self.net(x).view(-1, SEED_C, SEED_H, SEED_W)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.act   = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(x + out)


class CNNDecoder(nn.Module):
    # upsamples from (128,4,7) to full field resolution
    # each stage doubles spatial size, ResBlock refines features
    def __init__(self):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.rb1 = ResBlock(128)

        self.up2 = nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1)
        self.rb2 = ResBlock(96)

        self.up3 = nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1)
        self.rb3 = ResBlock(64)

        self.up4 = nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1)
        self.rb4 = ResBlock(48)

        self.up5 = nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1)
        self.rb5 = ResBlock(32)

        self.bn = nn.BatchNorm2d

        self.T_head = nn.Conv2d(32, 1, 3, padding=1)
        self.U_head = nn.Conv2d(32, 2, 3, padding=1)

        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.act(self.up1(x)); x = self.rb1(x)
        x = self.act(self.up2(x)); x = self.rb2(x)
        x = self.act(self.up3(x)); x = self.rb3(x)
        x = self.act(self.up4(x)); x = self.rb4(x)
        x = self.act(self.up5(x)); x = self.rb5(x)

        # resize to exact field dimensions
        x = nn.functional.interpolate(x, size=(FIELD_H, FIELD_W),
                                       mode='bilinear', align_corners=False)
        return self.T_head(x), self.U_head(x)


class SuperPODSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MLPEncoder()
        self.decoder = CNNDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ------- training -------

def train(model, train_loader, val_loader, optimizer, scheduler, n_epochs=N_EPOCHS):
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    criterion = nn.MSELoss()
    best_val  = float('inf')
    train_losses = []
    val_losses   = []

    print(f'\nTraining on {DEVICE} | {model.count_params():,} params')
    print(f'Train batches: {len(train_loader)} | Val batches: {len(val_loader)}')
    print('-' * 55)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        total_T    = 0.0
        total_U    = 0.0

        for params, T_true, U_true in train_loader:
            params  = params.to(DEVICE)
            T_true  = T_true.to(DEVICE)
            U_true  = U_true.to(DEVICE)

            optimizer.zero_grad()
            T_pred, U_pred = model(params)

            loss_T = criterion(T_pred, T_true)
            loss_U = criterion(U_pred, U_true)
            loss   = loss_T + LAMBDA_U * loss_U

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_T    += loss_T.item()
            total_U    += loss_U.item()

        n = len(train_loader)
        tl = total_loss / n
        tT = total_T / n
        tU = total_U / n

        # validation
        model.eval()
        vtotal = 0.0
        vT     = 0.0
        vU     = 0.0

        with torch.no_grad():
            for params, T_true, U_true in val_loader:
                params = params.to(DEVICE)
                T_true = T_true.to(DEVICE)
                U_true = U_true.to(DEVICE)

                T_pred, U_pred = model(params)
                loss_T = criterion(T_pred, T_true)
                loss_U = criterion(U_pred, U_true)

                vtotal += (loss_T + LAMBDA_U * loss_U).item()
                vT     += loss_T.item()
                vU     += loss_U.item()

        n  = len(val_loader)
        vl = vtotal / n
        vT = vT / n
        vU = vU / n

        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:4d}/{n_epochs} | '
                  f'train: {tl:.4f} (T:{tT:.4f} U:{tU:.4f}) | '
                  f'val: {vl:.4f} (T:{vT:.4f} U:{vU:.4f}) | '
                  f'best: {best_val:.4f}')

    np.save(os.path.join(RESULTS_DIR, 'train_losses.npy'), train_losses)
    np.save(os.path.join(RESULTS_DIR, 'val_losses.npy'),   val_losses)
    plot_losses(train_losses, val_losses)
    print(f'\nBest val loss: {best_val:.6f} -> {SAVE_DIR}/best_model.pth')


def plot_losses(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(train_losses, label='train (500)')
    ax.semilogy(val_losses,   label='val (50)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('Training curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'loss_curves.png'), dpi=150)
    print(f'Loss plot saved')
    plt.show()


# ------- evaluation -------

def evaluate(model, loader, norm_stats, label='Val'):
    T_mean = norm_stats['T_mean']
    T_std  = norm_stats['T_std']
    U_mean = norm_stats['U_mean']
    U_std  = norm_stats['U_std']

    model.eval()
    T_true_all = []
    T_pred_all = []
    U_true_all = []
    U_pred_all = []

    with torch.no_grad():
        for params, T_true, U_true in loader:
            T_pred, U_pred = model(params.to(DEVICE))
            T_true_all.append(T_true.cpu().numpy())
            T_pred_all.append(T_pred.cpu().numpy())
            U_true_all.append(U_true.cpu().numpy())
            U_pred_all.append(U_pred.cpu().numpy())

    T_true_K  = denormalize(np.concatenate(T_true_all), T_mean, T_std)
    T_pred_K  = denormalize(np.concatenate(T_pred_all), T_mean, T_std)
    U_true_ms = denormalize(np.concatenate(U_true_all), U_mean, U_std)
    U_pred_ms = denormalize(np.concatenate(U_pred_all), U_mean, U_std)

    T_mae = np.mean(np.abs(T_pred_K  - T_true_K),  axis=(1, 2, 3))
    T_mse = np.mean((T_pred_K - T_true_K) ** 2,    axis=(1, 2, 3))
    T_max = np.max(np.abs(T_pred_K   - T_true_K),  axis=(1, 2, 3))
    U_mae = np.mean(np.abs(U_pred_ms - U_true_ms), axis=(1, 2, 3))

    print(f'\n=== {label} ({len(T_true_K)} cases) ===')
    print(f'T MAE:      {T_mae.mean():.3f} +/- {T_mae.std():.3f} K')
    print(f'T RMSE:     {np.sqrt(T_mse.mean()):.3f} K')
    print(f'T max err:  {T_max.mean():.3f} K mean, {T_max.max():.3f} K worst')
    print(f'U MAE:      {U_mae.mean():.4f} m/s')
    print(f'T MAE range: {T_mae.min():.3f} - {T_mae.max():.3f} K')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics = {
        'T_mae_mean_K':      float(T_mae.mean()),
        'T_mae_std_K':       float(T_mae.std()),
        'T_rmse_K':          float(np.sqrt(T_mse.mean())),
        'T_max_err_mean_K':  float(T_max.mean()),
        'T_max_err_worst_K': float(T_max.max()),
        'U_mae_mean_ms':     float(U_mae.mean()),
        'n_cases':           len(T_true_K),
    }
    tag = label.lower().replace(' ', '_').replace('(', '').replace(')', '')
    with open(os.path.join(RESULTS_DIR, f'metrics_{tag}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # plot best / median / worst by T MAE
    idx_sorted = np.argsort(T_mae)
    for name, idx in [('best',   idx_sorted[0]),
                      ('median', idx_sorted[len(idx_sorted) // 2]),
                      ('worst',  idx_sorted[-1])]:
        print(f'Plotting {name} case (idx={idx}, MAE={T_mae[idx]:.3f} K)...')
        plot_prediction(
            T_true_K[idx, 0], T_pred_K[idx, 0],
            U_true_ms[idx],   U_pred_ms[idx],
            title=f'{label} — {name} (T MAE={T_mae[idx]:.2f} K)',
            save_name=f'prediction_{tag}_{name}.png'
        )

    return metrics


def plot_prediction(T_true_K, T_pred_K, U_true, U_pred,
                    title='Prediction vs CFD', save_name='prediction.png'):
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))

    def show_row(row, true2d, pred2d, name, cmap, symmetric=False):
        if symmetric:
            m = max(abs(true2d).max(), abs(pred2d).max())
            vmin, vmax = -m, m
        else:
            vmin = min(true2d.min(), pred2d.min())
            vmax = max(true2d.max(), pred2d.max())

        kw = dict(vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        im = axes[row, 0].imshow(true2d, cmap=cmap, **kw)
        axes[row, 0].set_title(f'{name} - CFD')
        fig.colorbar(im, ax=axes[row, 0])

        im = axes[row, 1].imshow(pred2d, cmap=cmap, **kw)
        axes[row, 1].set_title(f'{name} - Predicted')
        fig.colorbar(im, ax=axes[row, 1])

        err = np.abs(pred2d - true2d)
        im = axes[row, 2].imshow(err, cmap='Reds', origin='lower', aspect='auto')
        axes[row, 2].set_title(f'{name} error (max={err.max():.3f})')
        fig.colorbar(im, ax=axes[row, 2])

    show_row(0, T_true_K - 273.15, T_pred_K - 273.15, 'T (C)',    'inferno')
    show_row(1, U_true[0],         U_pred[0],          'Ux (m/s)', 'RdBu_r', symmetric=True)
    show_row(2, U_true[1],         U_pred[1],          'Uy (m/s)', 'RdBu_r', symmetric=True)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'  saved to {out}')
    plt.show()


# ------- main -------

def main(mode='train', data_path=DATA_PATH, val_path=VAL_DATA_PATH, n_epochs=N_EPOCHS):
    print(f'Device: {DEVICE} | Mode: {mode}')

    train_set, val_set, norm_stats = load_data(data_path, val_path)
    if val_set is None:
        raise RuntimeError('No validation set found. Check test_dataset.npz path.')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SuperPODSurrogate().to(DEVICE)
    print(f'Parameters: {model.count_params():,}')

    os.makedirs(SAVE_DIR, exist_ok=True)
    np.savez(os.path.join(SAVE_DIR, 'norm_stats.npz'), **norm_stats)

    if mode == 'train':
        optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        train(model, train_loader, val_loader, optimizer, scheduler, n_epochs=n_epochs)

        # load best checkpoint and run final eval
        model.load_state_dict(torch.load(os.path.join(SAVE_DIR, 'best_model.pth'),
                                         map_location=DEVICE))
        evaluate(model, val_loader, norm_stats, label='Val (50)')

    elif mode == 'eval':
        ckpt = os.path.join(SAVE_DIR, 'best_model.pth')
        if not os.path.exists(ckpt):
            print(f'No checkpoint found at {ckpt}. Run --mode train first.')
            return
        saved = np.load(os.path.join(SAVE_DIR, 'norm_stats.npz'))
        norm_stats = {k: saved[k] for k in saved}
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        evaluate(model, val_loader, norm_stats, label='Val (50)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode',     type=str, default='train', choices=['train', 'eval'])
    p.add_argument('--data',     type=str, default=DATA_PATH)
    p.add_argument('--val_data', type=str, default=VAL_DATA_PATH)
    p.add_argument('--epochs',   type=int, default=N_EPOCHS)
    a = p.parse_args()
    main(a.mode, a.data, a.val_data, a.epochs)
