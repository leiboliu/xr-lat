import pandas as pd
import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix, save_npz
import ast

train_file = "../data/mimic3/full/train_data_full_raw.csv"
dev_file = "../data/mimic3/full/dev_data_full_raw.csv"
test_file = "../data/mimic3/full/test_data_full_raw.csv"

output_dir = "../data/mimic3/full-xr/"

train_data = pd.read_csv(train_file)
dev_data = pd.read_csv(dev_file)
test_data = pd.read_csv(test_file)

# text data
train_text = train_data['text']
dev_text = dev_data['text']
test_text = test_data['text']

# train_text = train_text.append(dev_text, ignore_index=True)

with open(output_dir + "train.txt", 'w', newline='\n', encoding='utf-8') as f:
    for line in train_text:
        f.write(line + "\n")

with open(output_dir + "dev.txt", 'w', newline='\n', encoding='utf-8') as f:
    for line in dev_text:
        f.write(line + "\n")

with open(output_dir + "test.txt", 'w', newline='\n', encoding='utf-8') as f:
    for line in test_text:
        f.write(line + "\n")

# feature data CSR of text: using pecos.utils.featurization.text.preprocess to generate tfidf feature

# label CSR
train_label = train_data['labels']
dev_label = dev_data['labels']
test_label = test_data['labels']

train_label = train_label.apply(ast.literal_eval).tolist()
train_label = np.array([np.array(label) for label in train_label])
sparse_label_matrix = csr_matrix(train_label)
save_npz(output_dir + "train_label.npz", sparse_label_matrix)

dev_label = dev_label.apply(ast.literal_eval).tolist()
dev_label = np.array([np.array(label) for label in dev_label])
dev_sparse_label_matrix = csr_matrix(dev_label)
save_npz(output_dir + "dev_label.npz", dev_sparse_label_matrix)

test_label = test_label.apply(ast.literal_eval).tolist()
test_label = np.array([np.array(label) for label in test_label])
test_sparse_label_matrix = csr_matrix(test_label)
save_npz(output_dir + "test_label.npz", test_sparse_label_matrix)

print('ok')
