import argparse
import csv
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from wifi_dataset import XRFWiFiDataset
from model import resnet1d

# 評価する受信機の組み合わせ：3台、2台(1台欠け)、1台(2台欠け)の7通り
COMBOS = [c for n in (3, 2, 1) for c in itertools.combinations(range(3), n)]


def combo_name(c):
    return 'R' + ''.join(str(r + 1) for r in c)


def forward(model, x):
    # 配布モデルは (分類の出力, 特徴ベクトル) を返すので、分類の出力だけ使う
    out = model(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def mask_receivers(x, keep):
    # keep に含まれない受信機の90チャネルを0にする
    B, _, T = x.shape
    m = torch.zeros(1, 3, 1, 1, device=x.device)
    m[0, list(keep)] = 1
    return (x.view(B, 3, 90, T) * m).view(B, 270, T)


def drop_receivers(x, p):
    # 学習用：確率 p で、そのサンプルの受信機を1台か2台ランダムに0にする
    B, _, T = x.shape
    m = torch.ones(B, 3, device=x.device)
    for b in torch.nonzero(torch.rand(B) < p).flatten().tolist():
        k = np.random.randint(1, 3)
        m[b, np.random.choice(3, k, replace=False)] = 0
    return (x.view(B, 3, 90, T) * m.view(B, 3, 1, 1)).view(B, 270, T)


def evaluate(model, loader, device, keep=(0, 1, 2)):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = forward(model, mask_receivers(x, keep)).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return 100.0 * correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='baseline', help='実験名(結果の保存先)')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--norm', default='none', choices=['none', 'sample', 'subject'])
    p.add_argument('--recv_drop', type=float, default=0.0,
                   help='学習中に受信機を隠すサンプルの割合(0なら隠さない)')
    p.add_argument('--val_trials', type=int, default=0)
    p.add_argument('--limit', type=int, default=0, help='動作確認用。使うサンプル数を絞る')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------- データ ----------
    n_train_trials = 14 - args.val_trials
    train_set = XRFWiFiDataset(split='train', trials=set(range(1, n_train_trials + 1)), norm=args.norm)
    test_set = XRFWiFiDataset(split='test', norm=args.norm, stats=train_set.stats)
    val_set = None
    if args.val_trials > 0:
        val_set = XRFWiFiDataset(split='train', trials=set(range(n_train_trials + 1, 15)),
                                 norm=args.norm, stats=train_set.stats)
    if args.limit > 0:
        def shrink(ds):
            idx = np.linspace(0, len(ds) - 1, min(args.limit, len(ds))).astype(int)
            return Subset(ds, idx.tolist())
        train_set, test_set = shrink(train_set), shrink(test_set)
        if val_set is not None:
            val_set = shrink(val_set)

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    train_loader = make_loader(train_set, True)
    test_loader = make_loader(test_set, False)
    val_loader = make_loader(val_set, False) if val_set is not None else None
    print(f'device={device} norm={args.norm} recv_drop={args.recv_drop} seed={args.seed} '
          f'train={len(train_set)} test={len(test_set)}', flush=True)

    # ---------- モデルと最適化(論文 Sec. 4.3 の設定) ----------
    model = resnet1d.resnet18_mutual().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80, 120, 160], gamma=0.5)

    out_dir = f'runs/{args.name}/'
    os.makedirs(out_dir, exist_ok=True)
    with open(out_dir + 'log.csv', 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'train_acc', 'val_acc', 'seconds'])

    # ---------- 学習 ----------
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.recv_drop > 0:
                x = drop_receivers(x, args.recv_drop)
            out = forward(model, x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.numel()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.numel()
        scheduler.step()

        train_loss, train_acc = loss_sum / total, 100.0 * correct / total
        val_acc = evaluate(model, val_loader, device) if val_loader is not None else float('nan')
        sec = time.time() - t0
        print(f'epoch {epoch:3d} loss {train_loss:.4f} train_acc {train_acc:.2f} '
              f'val_acc {val_acc:.2f} time {sec:.0f}s', flush=True)
        with open(out_dir + 'log.csv', 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_acc, sec])

    # ---------- テストは最後に1回だけ(7通りの受信機の組み合わせ) ----------
    lines = [str(vars(args))]
    for c in COMBOS:
        acc = evaluate(model, test_loader, device, keep=c)
        print(f'TEST {combo_name(c)}: {acc:.2f}%', flush=True)
        lines.append(f'{combo_name(c)}={acc:.2f}')
    torch.save(model.state_dict(), out_dir + 'model_final.pth')
    with open(out_dir + 'result.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
