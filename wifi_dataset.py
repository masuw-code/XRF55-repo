import numpy as np
import torch
from torch.utils.data import Dataset


class XRFWiFiDataset(Dataset):
    """Wi-Fi CSIだけを1ファイルずつ読むDataset"""

    def __init__(self, root='./dataset/XRF_dataset/', split='train', scene='dml', trials=None):
        # split : 'train'(試行1〜14) または 'test'(試行15〜20)
        # trials: 使う試行番号の集合。Noneなら全部使う
        # 配布コードでは、テスト用のリストが _val.txt という名前になっている
        list_file = root + scene + ('_train.txt' if split == 'train' else '_val.txt')
        data_dir = root + ('train_data' if split == 'train' else 'test_data') + '/WiFi/'
        self.items = []
        with open(list_file) as f:
            for line in f:
                name, subject, action = line.strip().split(',')
                trial = int(name.split('_')[2])
                if trials is not None and trial not in trials:
                    continue
                self.items.append((data_dir + name + '.npy', int(action) - 1))  # ラベルは0〜54

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        x = np.load(path)                          # (270, 1000) float64
        return torch.from_numpy(x).float(), label  # float32に変換して返す
