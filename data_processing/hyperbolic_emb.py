import pickle5 as pickle
import csv
import numpy as np

# load clustering matrix to build tsv file
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors

node_prefix = [('Code', 'Category'), ('Category', 'Block'), ('Block', 'Chapter'), ('Chapter', 'ICD9')]


def generate_hierarchical_tree(cluster_chain):
    hierarchy = set()
    for i, cluster in enumerate(reversed(cluster_chain)):
        indices = np.nonzero(cluster)
        dim_names = node_prefix[i]
        dim_1 = [dim_names[0] + '-' + str(i) for i in indices[0]]
        dim_2 = [dim_names[1] + '-' + str(i) for i in indices[1]]

        hierarchy.update(list(zip(dim_1, dim_2)))

    return list(hierarchy)

with open("../data/mimic3/hct.pkl", "rb") as f:
    cluster_chain = pickle.load(f)
    hi_tree = generate_hierarchical_tree(cluster_chain)
    print("Ok")

with open("../data/mimic3/hyperbolic_emb_train.tsv", 'w', newline='') as tsvf:
    writer = csv.writer(tsvf, delimiter='\t', lineterminator='\n')
    for item in hi_tree:
        writer.writerow(list(item))

# train hyperbolic embedding model using hi_tree variable
model = PoincareModel(hi_tree)
model.train(epochs=50, print_every=200)
poincare_emb = model.kv

# with open("../../data/hi_label_tree/hyperbolic_embeddings.pkl", "wb") as f:
#     pickle.dump(poincare_emb, f)
#     print("hyperbolic embeddings dumpped")

poincare_emb.save_word2vec_format("../data/mimic3/word2vec_format_vectors")
# emb_vectors = PoincareKeyedVectors.load_word2vec_format("../../data/hi_label_tree/emb_vectors")

key2index = poincare_emb.key_to_index

chapter_emb = [(int(key[8:]), value) for key, value in key2index.items() if 'Chapter' in key]
chapter_emb.sort()
chapter_vectors = [poincare_emb.get_vector(poincare_emb.index_to_key[index]) for _, index in chapter_emb]
chapter_vectors = np.array(chapter_vectors)

block_emb = [(int(key[6:]), value) for key, value in key2index.items() if 'Block' in key]
block_emb.sort()
block_vectors = [poincare_emb.get_vector(poincare_emb.index_to_key[index]) for _, index in block_emb]
block_vectors = np.array(block_vectors)

category_emb = [(int(key[9:]), value) for key, value in key2index.items() if 'Category' in key]
category_emb.sort()
category_vectors = [poincare_emb.get_vector(poincare_emb.index_to_key[index]) for _, index in category_emb]
category_vectors = np.array(category_vectors)

code_emb = [(int(key[5:]), value) for key, value in key2index.items() if 'Code' in key]
code_emb.sort()
code_vectors = [poincare_emb.get_vector(poincare_emb.index_to_key[index]) for _, index in code_emb]
code_vectors = np.array(code_vectors)

with open("../data/mimic3/hyperbolic_vectors_all.pkl", "wb") as f:
    pickle.dump([chapter_vectors, block_vectors, category_vectors, code_vectors], f)
    print("dumped all the vectors")

print("ok")
