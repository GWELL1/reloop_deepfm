import pandas as pd
from pathlib import Path
import gc


headers = ["label", "user_id", "item_id", "tag_id"]
data_files = ["../data_mov/AFN_data/train.libsvm", "../data_mov/AFN_data/valid.libsvm"]
for f in data_files:
    df = pd.read_csv(f, sep=" ", names=headers)
    for col in headers[1:]:
        df[col] = df[col].apply(lambda x: x.split(':')[0])
    df.to_csv("../data_mov/AFN_data/" + Path(f).stem + ".txt", index=False)
    del df
    gc.collect()

file1 = open('../data_mov/AFN_data/train.txt')
file2 = open('../data_mov/AFN_data/valid.txt')
lines1 = file1.readlines()
lines2 = file2.readlines()
del lines1[0], lines2[0]
file1.close()
file2.close()

file_new = open('../data_mov/origin_data/data.txt', 'w')
file_new.writelines(lines1)
file_new.writelines(lines2)
file_new.close()