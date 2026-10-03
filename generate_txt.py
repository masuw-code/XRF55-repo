import os


def generate_txt_mix(data_src_path, txt_save_path, txt_name):
    train_list = []
    val_list = []

    train_src = data_src_path + "train_data/WiFi/"
    test_src = data_src_path + "test_data/WiFi/"

    for file in sorted(os.listdir(train_src)):
        filename = file.split(".")[0]
        fulname = filename.split("_")   # 被験者, 行動, 試行
        train_list.append(filename + "," + fulname[0] + "," + fulname[1] + "\n")
    for file in sorted(os.listdir(test_src)):
        filename = file.split(".")[0]
        fulname = filename.split("_")
        val_list.append(filename + "," + fulname[0] + "," + fulname[1] + "\n")

    with open(txt_save_path + txt_name + "_train.txt", "w") as f:
        f.writelines(train_list)
        print("train_dataset len:" + str(len(train_list)))
    with open(txt_save_path + txt_name + "_val.txt", "w") as f:
        f.writelines(val_list)
        print("test_dataset len:" + str(len(val_list)))


if __name__ == '__main__':
    root_path = "./dataset/XRF_dataset/"
    data_src_path = "./dataset/XRF_dataset/"
    txt_name = "dml"
    generate_txt_mix(data_src_path, root_path, txt_name)
