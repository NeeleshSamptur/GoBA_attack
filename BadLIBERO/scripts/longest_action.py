import os
import h5py


def get_longest_action_length(folder_path):
    max_len = 0
    longest_file = None
    longest_demo = None

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(".hdf5"):
                fpath = os.path.join(root, fname)
                with h5py.File(fpath, "r") as f:
                    if "data" not in f:
                        continue
                    for demo in f["data"].keys():
                        if "actions" in f["data"][demo]:
                            actions = f["data"][demo]["actions"]
                            length = len(actions)
                            if length > max_len:
                                max_len = length
                                longest_file = fpath
                                longest_demo = demo

    print(f"最长 action 长度: {max_len}")
    print(f"所在文件: {longest_file}")
    print(f"demo 名称: {longest_demo}")
    return max_len


def print_h5_structure(file_path):
    def print_name(name, obj):
        print(name)
    with h5py.File(file_path, "r") as f:
        f.visititems(print_name)

# 用法
folder = "../Poisoned_Dataset/Poison/libero_spatial_no_noops/"
get_longest_action_length(folder)
# print_h5_structure("../Poisoned_Dataset/Poison/libero_10_no_noops/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5")
