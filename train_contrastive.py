"""受信機間のcontrastive学習(案C)

段階1(事前学習): 受信機ごとに同じエンコーダーで特徴を出し、
                 同じサンプルの別の受信機どうしが近づくように学習する(ラベルは使わない)
段階2(分類):     使える受信機の特徴を平均し、55クラスの分類を学習する
--pre_epochs 0 にすると、事前学習なし(③)になる
"""
import argparse
import csv
import itertools
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from wifi_dataset import XRFWiFiDataset
from model.resnet1d import ResNetLargeBert3, BasicBlock

COMBOS = [c for n in (3, 2, 1) for c in itertools.combinations(range(3), n)]


def combo_name(c):
    return 'R' + ''.join(str(r + 1) for r in c)


class RecvNet(nn.Module):
    """受信機ごとに同じエンコーダーを通し、使える受信機の特徴を平均して分類する"""

    def __init__(self, num_classes=55, proj_dim=128):
        super().__init__()
        # resnet18_mutual と同じ構造で、入力だけを受信機1台分(90チャネル)にする
        self.encoder = ResNetLargeBert3(BasicBlock, [2, 2, 2, 2], inchannel=90, activity_num=num_classes)
        self.proj = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))
        self.head = nn.Linear(1024, num_classes)

    def features(self, x):
        # (B, 270, T) → 受信機ごとに (B*3, 90, T) → 特徴 (B, 3, 1024)
        B, _, T = x.shape
        _, f = self.encoder(x.view(B, 3, 90, T).reshape(B * 3, 90, T))
        return f.view(B, 3, -1)

    def classify(self, f, keep=(0, 1, 2)):
        # 使える受信機の特徴だけを平均して分類する
        return self.head(f[:, list(keep)].mean(dim=1))


def info_nce(z, tau):
    """z: (B, 3, d)。同じサンプルの別の受信機を正解、同じバッチのほかのサンプルを不正解とする"""
    B = z.size(0)
    target = torch.arange(B, device=z.device)
    loss, correct, n = 0.0, 0, 0
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            logits = z[:, i] @ z[:, j].T / tau
            loss = loss + F.cross_entropy(logits, target)
            correct += (logits.argmax(1) == target).sum().item()
            n += 1
    # 2つ目の値：別の受信機の特徴から、同じサンプルを正しく選べた割合
    return loss / n, 100.0 * correct / (n * B)


def evaluate(model, loader, device):
    model.eval()
    correct = {c: 0 for c in COMBOS}
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            f = model.features(x)
            for c in COMBOS:
                correct[c] += (model.classify(f, c).argmax(1) == y).sum().item()
            total += y.numel()
    return {c: 100.0 * correct[c] / total for c in COMBOS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='contrast')
    p.add_argument('--pre_epochs', type=int, default=100, help='事前学習のエポック数(0なら事前学習なし)')
    p.add_argument('--ft_epochs', type=int, default=100, help='分類の学習のエポック数')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--pre_lr', type=float, default=0.001)
    p.add_argument('--ft_lr', type=float, default=0.001)
    p.add_argument('--tau', type=float, default=0.1, help='InfoNCEの温度')
    p.add_argument('--freeze', action='store_true', help='分類の学習でエンコーダーを固定する')
    p.add_argument('--norm', default='none', choices=['none', 'sample', 'subject'])
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, default=0, help='動作確認用')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = XRFWiFiDataset(split='train', trials=set(range(1, 15)), norm=args.norm)
    test_set = XRFWiFiDataset(split='test', norm=args.norm, stats=train_set.stats)
    if args.limit > 0:
        def shrink(ds):
            idx = np.linspace(0, len(ds) - 1, min(args.limit, len(ds))).astype(int)
            return Subset(ds, idx.tolist())
        train_set, test_set = shrink(train_set), shrink(test_set)

    kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    pre_loader = DataLoader(train_set, shuffle=True, drop_last=True, **kw)
    train_loader = DataLoader(train_set, shuffle=True, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)
    print(f'device={device} pre_epochs={args.pre_epochs} ft_epochs={args.ft_epochs} '
          f'freeze={args.freeze} train={len(train_set)} test={len(test_set)}', flush=True)

    model = RecvNet().to(device)
    out_dir = f'runs/{args.name}/'
    os.makedirs(out_dir, exist_ok=True)
    log_path = out_dir + 'log.csv'
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['stage', 'epoch', 'loss', 'acc', 'seconds'])

    def log(stage, epoch, loss, acc, sec):
        print(f'{stage} epoch {epoch:3d} loss {loss:.4f} acc {acc:.2f} time {sec:.0f}s', flush=True)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([stage, epoch, loss, acc, sec])

    # ---------- 段階1：受信機間のcontrastive事前学習 ----------
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.proj.parameters()), lr=args.pre_lr)
    for epoch in range(1, args.pre_epochs + 1):
        model.train()
        t0, loss_sum, acc_sum, n = time.time(), 0.0, 0.0, 0
        for x, _ in pre_loader:
            x = x.to(device, non_blocking=True)
            z = F.normalize(model.proj(model.features(x)), dim=-1)
            loss, acc = info_nce(z, args.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum, acc_sum, n = loss_sum + loss.item(), acc_sum + acc, n + 1
        log('pre', epoch, loss_sum / n, acc_sum / n, time.time() - t0)

    # ---------- 段階2：分類の学習(3台すべての特徴の平均を使う) ----------
    if args.freeze:
        for q in model.encoder.parameters():
            q.requires_grad = False
        params = list(model.head.parameters())
    else:
        params = list(model.encoder.parameters()) + list(model.head.parameters())
    opt = torch.optim.Adam(params, lr=args.ft_lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[40, 80, 120, 160], gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.ft_epochs + 1):
        model.train()
        if args.freeze:
            model.encoder.eval()          # BatchNormの統計を変えない
        t0, loss_sum, correct, total = time.time(), 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if args.freeze:
                with torch.no_grad():
                    f = model.features(x)
            else:
                f = model.features(x)
            out = model.classify(f)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
        sched.step()
        log('ft', epoch, loss_sum / total, 100.0 * correct / total, time.time() - t0)

    # ---------- テストは最後に1回だけ(7通りの受信機の組み合わせ) ----------
    res = evaluate(model, test_loader, device)
    lines = [str(vars(args))]
    for c in COMBOS:
        print(f'TEST {combo_name(c)}: {res[c]:.2f}%', flush=True)
        lines.append(f'{combo_name(c)}={res[c]:.2f}')
    torch.save(model.state_dict(), out_dir + 'model_final.pth')
    with open(out_dir + 'result.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
