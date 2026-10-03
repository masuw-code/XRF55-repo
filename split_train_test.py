import os
from tqdm import tqdm


def split_train_test_new(root_ns="./dataset/Raw_dataset/", dst_wr="./dataset/XRF_dataset/", split=14):
    src_wifi = os.path.realpath(root_ns + "WiFi/")   # 元データの絶対パス
    dst_train_wifi = dst_wr + "train_data/WiFi/"
    dst_test_wifi = dst_wr + "test_data/WiFi/"
    os.makedirs(dst_train_wifi, exist_ok=True)
    os.makedirs(dst_test_wifi, exist_ok=True)

    for file in tqdm(sorted(os.listdir(src_wifi))):
        if not file.endswith(".npy"):
            continue
        filename = file.split(".")[0]              # 例: 01_01_01
        trialidx = int(filename.split("_")[2])     # 3番目の番号 = 試行番号 (1〜20)
        # 試行1〜14は学習、15〜20はテスト
        dst_dir = dst_train_wifi if trialidx <= split else dst_test_wifi
        dst = dst_dir + file
        if not os.path.lexists(dst):
            os.symlink(os.path.join(src_wifi, file), dst)   # コピーせずリンクを張る


if __name__ == '__main__':
    split_train_test_new(root_ns="./dataset/Raw_dataset/", dst_wr="./dataset/XRF_dataset/", split=14)
