# # build_split_dirs_ods_only.py
# import os
# import shutil
# import tqdm
# import pandas as pd
# from models.DynaOD.data_load import load_samples
#
# RAW_DATA_ROOT = 'data'
# OUT_DATA_ROOT = 'data_split'
# SPLIT_RATIO   = 0.7
# LLM_TAG       = 'gpt-4o-mini'
#
# # 动态目录控制
# COPY_ODS        = True   # 只拷 ods
# COPY_POI_VECS   = True
# COPY_DEMO_VECS  = True
#
#
# def mkdir(path):
#     os.makedirs(path, exist_ok=True)
#
#
# def copy_file(src, dst):
#     if os.path.exists(src):
#         shutil.copy2(src, dst)
#
#
# def build_one_split(mode, cities, dates):
#     split_dir = os.path.join(OUT_DATA_ROOT, f'data_{mode}')
#     mkdir(split_dir)
#
#     for city in tqdm.tqdm(cities, desc=f'building {mode}'):
#         city_dir = os.path.join(split_dir, city)
#         mkdir(city_dir)
#
#         # 1. 静态文件
#         static_files = ['demos.npy', 'pois.npy', 'dis.npy', 'adj.npy', 'od.npy']
#         for fname in static_files:
#             copy_file(os.path.join(RAW_DATA_ROOT, city, fname),
#                       os.path.join(city_dir, fname))
#
#         # 2. 动态目录空壳
#         ods_dir     = os.path.join(city_dir, 'ods')
#         poi_vec_dir = os.path.join(city_dir, 'poi_vecs')
#         demo_vec_dir= os.path.join(city_dir, 'demo_vecs')
#         mkdir(ods_dir); mkdir(poi_vec_dir); mkdir(demo_vec_dir)
#
#         # 3. 按需拷贝
#         for date in dates:
#             if COPY_ODS:
#                 copy_file(os.path.join(RAW_DATA_ROOT, city, 'ods', f'{date}_od.npy'),
#                           os.path.join(ods_dir, f'{date}_od.npy'))
#             if COPY_POI_VECS:
#                 copy_file(os.path.join(RAW_DATA_ROOT, city, 'poi_vecs', f'{LLM_TAG}_poi_vec_{date}.npy'),
#                           os.path.join(poi_vec_dir, f'{LLM_TAG}_poi_vec_{date}.npy'))
#             if COPY_DEMO_VECS:
#                 copy_file(os.path.join(RAW_DATA_ROOT, city, 'demo_vecs', f'{LLM_TAG}_demo_vec_{date}.npy'),
#                           os.path.join(demo_vec_dir, f'{LLM_TAG}_demo_vec_{date}.npy'))
#
#
# def main():
#     dates = pd.date_range(start='2019-01-01', end='2019-01-31').strftime('%Y_%m_%d').tolist()
#     time_split = int(len(dates) * SPLIT_RATIO)
#     date_ranges = {
#         'train': dates[:time_split],
#         'test1': dates[time_split:],
#         'test2': dates[:time_split],
#         'test3': dates[time_split:],
#     }
#
#     cities_dict = {
#         'train': load_samples(RAW_DATA_ROOT, shuffle_cities=True, split_ratio=SPLIT_RATIO, mode='train', is_simplify=True)[0],
#         'test1': load_samples(RAW_DATA_ROOT, shuffle_cities=True, split_ratio=SPLIT_RATIO, mode='test1', is_simplify=True)[0],
#         'test2': load_samples(RAW_DATA_ROOT, shuffle_cities=True, split_ratio=SPLIT_RATIO, mode='test2', is_simplify=True)[0],
#         'test3': load_samples(RAW_DATA_ROOT, shuffle_cities=True, split_ratio=SPLIT_RATIO, mode='test3', is_simplify=True)[0],
#     }
#
#     for mode in ['train', 'test1', 'test2', 'test3']:
#         build_one_split(mode, cities_dict[mode], date_ranges[mode])
#
#     print('✅ 全部 split 静态文件 + ods 动态文件 生成完毕！（poi_vecs / demo_vecs 未拷）')
#
#
# if __name__ == '__main__':
#     main()

import json
import os
import shutil
from pathlib import Path

def remove_orphan_city_dirs(run_dir: str, data_dir: str = "data"):
    """
    删除 data/ 下不在 cities_split.json 中的城市文件夹
    :param run_dir: 包含 cities_split.json 的目录
    :param data_dir: 城市文件夹所在目录，默认当前目录下的 data/
    """
    # 1. 读取合法 id
    split_file = Path(run_dir) / "cities_split.json"
    with split_file.open(encoding="utf-8") as f:
        payload = json.load(f)

    legal_ids = set(payload["cities"]) | set(payload["seen_cities"]) | set(payload["unseen_cities"])
    legal_ids = {str(cid) for cid in legal_ids}          # 统一转字符串，避免 int/str 混用

    # 2. 扫描并删除孤儿目录
    data_path = Path(data_dir)
    for sub_dir in data_path.iterdir():
        if sub_dir.is_dir() and sub_dir.name.isdigit() and sub_dir.name not in legal_ids:
            print(f"Removing orphan city dir: {sub_dir}")
            shutil.rmtree(sub_dir)

# 用法
if __name__ == "__main__":
    remove_orphan_city_dirs("ckpts")   # 把 run_dir 换成你的实际路径