import numpy as np
import torch
from torch.utils.data import Dataset


class XRFWiFiDataset(Dataset):
    """Wi-Fi CSIだけを1ファイルずつ読むDataset

    norm : 'none'    正規化なし（ベースライン）
           'sample'  サンプルごとの標準化（1サンプル270×1000の平均と標準偏差）
           'subject' 被験者ごとの標準化（その被験者の学習用データの平均と標準偏差）
    stats: 'subject' のときに使う {被験者番号: (平均, 標準偏差)}。
           学習用データから計算し、テスト用・検証用にはそれを渡す
    """

    def __init__(self, root='./dataset/XRF_dataset/', split='train', scene='dml',
                 trials=None, norm='none', stats=None):
        # 配布コードでは、テスト用のリストが _val.txt という名前になっている
        list_file = root + scene + ('_train.txt' if split == 'train' else '_val.txt')
        data_dir = root + ('train_data' if split == 'train' else 'test_data') + '/WiFi/'
        self.norm = norm
        self.items = []
        with open(list_file) as f:
            for line in f:
                name, subject, action = line.strip().split(',')
                trial = int(name.split('_')[2])
                if trials is not None and trial not in trials:
                    continue
                # (ファイルのパス, 被験者番号, ラベル0〜54)
                self.items.append((data_dir + name + '.npy', int(subject), int(action) - 1))

        self.stats = stats
        if norm == 'subject' and stats is None:
            if split != 'train':
                raise ValueError('テスト用には、学習用データから計算した stats を渡してください')
            self.stats = self._subject_stats()

    def _subject_stats(self):
        # 被験者ごとに、値の合計・二乗の合計・個数を足し合わせ、平均と標準偏差を出す
        acc = {}
        for path, subject, _ in self.items:
            x = np.load(path)
            s, ss, n = acc.get(subject, (0.0, 0.0, 0))
            acc[subject] = (s + x.sum(), ss + (x ** 2).sum(), n + x.size)
        stats = {}
        for subject, (s, ss, n) in acc.items():
            mean = s / n
            stats[subject] = (mean, np.sqrt(ss / n - mean ** 2))
        return stats

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, subject, label = self.items[idx]
        x = np.load(path)                          # (270, 1000) float64
        if self.norm == 'sample':
            x = (x - x.mean()) / (x.std() + 1e-8)
        elif self.norm == 'subject':
            mean, std = self.stats[subject]
            x = (x - mean) / (std + 1e-8)
        return torch.from_numpy(x).float(), label
