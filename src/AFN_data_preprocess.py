# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# httpwww.apache.orglicensesLICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""AFN data preprocess."""

import os
import gc
import pandas as pd
from pathlib import Path


file_path1 = "../data_mov/AFN_data/"
file_path2 = "../data_mov/origin_data/"
if not os.path.exists(file_path1):
    os.makedirs(file_path1)
if not os.path.exists(file_path2):
    os.makedirs(file_path2)
headers = ["label", "user_id", "item_id", "tag_id"]
data_files = ["../data_mov/AFN_data/train.libsvm", "../data_mov/AFN_data/valid.libsvm", "../data_mov/AFN_data/test.libsvm"]

for f in data_files:
    df = pd.read_csv(f, sep=" ", names=headers)
    for col in headers[1:]:
        df[col] = df[col].apply(lambda x: x.split(':')[0])
    df.to_csv("../data_mov/AFN_data/" + Path(f).stem + ".txt", index=False)
    del df
    gc.collect()

file1 = open('../data_mov/AFN_data/train.txt')
file2 = open('../data_mov/AFN_data/valid.txt')
file3 = open('../data_mov/AFN_data/test.txt')
lines1 = file1.readlines()
lines2 = file2.readlines()
lines3 = file3.readlines()
del lines1[0], lines2[0], lines3[0]
file1.close()
file2.close()
file3.close()

file_new1 = open('../data_mov/origin_data/data.txt', 'w')
file_new1.writelines(lines1)
file_new1.writelines(lines2)
file_new1.close()

file_new2 = open('../data_mov/origin_data/test.txt', 'w')
file_new2.writelines(lines3)
file_new2.close()
