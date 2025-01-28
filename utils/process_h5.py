import os
import random

import h5py
import numpy as np
from tqdm import tqdm

import h5py
import os
import json


def extract_h5_contents(h5_file, output_path):
    os.makedirs(output_path, exist_ok=True)

    def ensure_dir_exists(file_path):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

    def extract_dataset(name, obj):
        if isinstance(obj, h5py.Dataset):
            data = obj[()]

            if isinstance(data, np.ndarray):
                if data.dtype.char == 'S':
                    data = data.tobytes().decode('utf-8')
                elif data.dtype.kind in {'u', 'i', 'f'}:
                    data = data.tolist()

            file_path = os.path.join(output_path, name)
            ensure_dir_exists(file_path)

            if name.endswith('.jpg'):
                with open(file_path, 'wb') as f:
                    f.write(data if isinstance(data, bytes) else bytes(data))
            elif name.endswith('.json'):
                if isinstance(data, str):
                    json_data = json.loads(data)
                else:
                    json_data = data
                with open(file_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
            elif name.endswith('.csv'):
                with open(file_path, 'w') as f:
                    if isinstance(data, (int, float, str)):
                        f.write(str(data) + "\n")
                    elif isinstance(data, list):
                        if all(isinstance(row, (list, tuple)) for row in data):
                            f.writelines(",".join(map(str, row)) + "\n" for row in data)
                        else:
                            f.writelines(str(item) + "\n" for item in data)
            elif name.endswith('.bin'):
                with open(file_path, 'wb') as f:
                    f.write(data if isinstance(data, bytes) else bytes(data))
            elif name.endswith('.ini'):
                with open(file_path, 'w') as f:
                    if isinstance(data, str):
                        f.write(data)
            else:
                print(f"Unknown type：{name}")
        elif isinstance(obj, h5py.Group):
            print(f"cd {name}")

    h5_file.visititems(extract_dataset)


def delete_h5_contents(h5_file, group):
    for item in tqdm(group):
        if item in h5_file:
            del h5_file[item]
        else:
            print(f"{item} does not exist")


def get_top_level_groups(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        top_level_groups = list(h5_file.keys())
    return top_level_groups


def main(mode='delete'):
    root_path = '/media/jinpeng-yu/Data/dvg_data'
    h5_file_path = os.path.join(root_path, 'dataset_small_fpv.h5')

    top_level_groups = get_top_level_groups(h5_file_path)
    print(len(top_level_groups))

    if mode == 'delete':
        save_num = 2000
        random_delete = random.sample(top_level_groups, len(top_level_groups) - save_num)

        with h5py.File(h5_file_path, 'r+') as h5_file:
            delete_h5_contents(h5_file, random_delete)
    elif mode == 'extract':
        output_folder = os.path.join(root_path, 'dataset_small_fpv')

        with h5py.File(h5_file_path, 'r+') as h5_file:
            extract_h5_contents(h5_file, output_folder)
    else:
        return


if __name__ == "__main__":
    main(mode='none')
