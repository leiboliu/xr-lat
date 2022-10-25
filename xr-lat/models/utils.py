import pickle
import random
from typing import Union

import numpy as np
import torch
import logging
from pathlib import Path
import pandas as pd

from datasets import Dataset

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc
from transformers import XLNetTokenizer, XLNetTokenizerFast, EvalPrediction

logger = logging.getLogger(__name__)


class MimicIIIDataset(Dataset):
    def __init__(self, data, C=None, M=None, Y=None, idx_padding=-1):
        """

        :param data:
        :param C: cluster chain
        :param M: M_next for current level
        :param Y: labels of the current level
        """
        self.input_ids = data["input_ids"]
        self.attention_mask = data["attention_mask"]
        self.token_type_ids = data["token_type_ids"]
        self.labels = Y
        self.C = C
        self.M = M
        self.idx_padding = idx_padding
        self.label_mat = None
        self.offset = 0.0
        self.label_width = 0

        # calculate M_next, Y_level, label_indices, label_values
        if M is None and self.labels is None:
            # self.label_width = self.labels.shape[1]
            self.label_mat = None
        elif M is not None and self.labels is None:
            self.label_width = max(np.count_nonzero(self.M, axis=1))
            self.label_mat = self.M

        elif M is None and self.labels is not None:
            self.label_width = self.labels.shape[1]
        else:
            self.label_mat = self.M + self.labels
            self.label_width = max(np.count_nonzero(self.label_mat, axis=1))
            self.offset = np.amax(self.label_mat) + 10.0
            self.label_mat[self.label_mat != 0] = self.offset
            self.label_mat += self.labels

        self.label_indices_all = []
        self.label_values_all = []

        if self.M is not None:
            for i in range(self.label_mat.shape[0]):
                no_zeros = np.nonzero(self.label_mat[i])[0]
                nr_active = len(no_zeros)
                label_indices_item = torch.zeros((self.label_width,), dtype=torch.long) + self.idx_padding
                label_indices_item[:nr_active] = torch.from_numpy(no_zeros)
                self.label_indices_all.append(label_indices_item)

                label_values_item = torch.zeros((self.label_width,), dtype=torch.float)
                curr_offset = 0.0 if self.labels is None else self.offset
                label_values_item[:nr_active] = torch.from_numpy(self.label_mat[i][no_zeros] - curr_offset)
                self.label_values_all.append(label_values_item)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        item_label_indices = None
        item_label_values = None
        if self.M is not None:
            item_label_indices = self.label_indices_all[item]
            item_label_values = self.label_values_all[item]

        return {
            "input_ids": torch.tensor(self.input_ids[item], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[item], dtype=torch.float),
            "token_type_ids": torch.tensor(self.token_type_ids[item], dtype=torch.long),
            "targets": torch.tensor(self.labels[item], dtype=torch.float) if self.M is None else item_label_values,
            "label_indices": None if item_label_indices is None else item_label_indices,
            "label_values": None if item_label_values is None else item_label_values
        }


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_data(data_file, tokenizer, cluster_chain, M, level, data_args, model_args, for_predict=False):
    """
    :param data_file:
    :param cluster_chain: cluster chain
    :param level: training level. start from 0
    :return:
    """
    # check data file format: csv only
    if not data_file.endswith(".csv"):
        raise ValueError("Data file must be csv format")
    # Input: data_file. Output: contains features: input_ids, attention_mask, token_type_ids
    # cached_file = Path(data_file).parent / Path(data_file).name \
    #     .replace(".csv", "_seq-{}_chunk-{}-{}-{}.pkl"
    #              .format(model_args.max_seq_length, model_args.num_chunks_per_document, level, str(for_predict)))
    def cache_file(tokenizer, data_file, cached_file, data_args, model_args):
        data = pd.read_csv(data_file)
        if data_args.data_segmented:
            text = data.loc[:, data.columns.str.startswith("Chunk")].fillna("").apply(
                lambda x: [seg for seg in x],
                axis=1).tolist()
            labels = data.iloc[:, 11:].apply(lambda x: [seg for seg in x], axis=1).tolist()
            results = tokenize_dataset(tokenizer, text, labels, model_args.max_seq_length)
        else:
            text = data["text"].tolist()
            import ast
            labels = data["labels"].apply(ast.literal_eval).tolist()

            results = segment_tokenize_dataset(tokenizer, text, labels,
                                               model_args.max_seq_length,
                                               model_args.num_chunks_per_document)

        with open(cached_file, 'wb') as f:
            pickle.dump(results, f)

        return results

    cached_file = Path(data_file).parent / Path(data_file).name \
        .replace(".csv", "_seq-{}_chunk-{}_model-{}.pkl"
                 .format(model_args.max_seq_length, model_args.num_chunks_per_document, model_args.model_type))
    if data_args.use_cached_datasets:
        # load data results from pickle file
        if cached_file.exists():
            with open(cached_file, "rb") as f:
                results = pickle.load(f)
        else:
            logger.info("There is no cached data file found for {}".format(data_file))
            results = cache_file(tokenizer, data_file, cached_file, data_args, model_args)
    else:
        results = cache_file(tokenizer, data_file, cached_file, data_args, model_args)

    if cluster_chain is None:
        # only one level.
        dataset = MimicIIIDataset(results, Y=np.array([np.array(target) for target in results["targets"]]))
    else:
        act_labels = np.array([np.array(target) for target in results["targets"]])
        if level != len(cluster_chain) - 1:
            for j in range(len(cluster_chain) - 1 - level):
                curr_labels = np.matmul(act_labels, cluster_chain[-1 - j])
                act_labels = curr_labels
            act_labels[act_labels != 0] = 1

        if M is not None:
            M_next = np.matmul(M, np.transpose(cluster_chain[level]))
            M_next[M_next != 0] = 1
        else:
            M_next = None

        if for_predict:
            dataset = MimicIIIDataset(results, cluster_chain, M_next, None, cluster_chain[level].shape[0])
            dataset.labels = act_labels
        else:
            dataset = MimicIIIDataset(results, cluster_chain, M_next, act_labels, cluster_chain[level].shape[0])

    return dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(p: Union[EvalPrediction, tuple], threshold=0.5):
    if isinstance(p, EvalPrediction):
        true_labels = p.label_ids
        logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    else:
        true_labels = p[0]
        logits = p[1]

    metric_scores = calculate_scores(true_labels, logits, threshold=threshold)
    micro_scores = calculate_scores(true_labels, logits, average="micro", threshold=threshold)
    metric_scores.update(micro_scores)

    return metric_scores

def post_process(predictions, num_labels, logits=False, threshold=0.5):
    preds = predictions[0] if logits else np.rint(sigmoid(predictions[0]) - threshold + 0.5)
    indices = predictions[1]
    if len(indices) == 0:
        return preds
    processed_preds = []
    for i in range(indices.shape[0]):
        tmp_preds = np.zeros(num_labels, dtype=float)
        padding_indices = np.where(indices[i] == num_labels)
        item_indices = np.delete(indices[i], padding_indices, None)
        item_preds = np.delete(preds[i], padding_indices, None)

        tmp_preds[item_indices] = item_preds
        if logits:
            tmp_preds[tmp_preds == 0.0] = -9.0
        processed_preds.append(tmp_preds)

    processed_preds = np.array(processed_preds, dtype=float)
    return processed_preds


def tokenize_inputs(text_list, tokenizer, max_seq_len=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[:max_seq_len - 2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # get token type for token_ids_0
    token_type_ids = [tokenizer.create_token_type_ids_from_sequences(x) for x in input_ids]
    # append special token to end of sentence: <sep> <cls>
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # attention mask
    attention_mask = [[1] * len(x) for x in input_ids]

    # padding to max_length
    def padding_to_max(sequence, value):
        padding_len = max_seq_len - len(sequence)
        padding = [value] * padding_len
        return sequence + padding

    input_ids = [padding_to_max(x, tokenizer.pad_token_id) for x in input_ids]
    attention_mask = [padding_to_max(x, 0) for x in attention_mask]
    token_type_ids = [padding_to_max(x, tokenizer.pad_token_type_id) for x in token_type_ids]

    return input_ids, attention_mask, token_type_ids


def tokenize_dataset(tokenizer, text, labels, max_seq_len):
    if (isinstance(tokenizer, XLNetTokenizer) or isinstance(tokenizer, XLNetTokenizerFast)):
        data = list(map(lambda t: tokenize_inputs(t, tokenizer, max_seq_len=max_seq_len), text))
        input_ids, attention_mask, token_type_ids = zip(*data)
    else:
        tokenizer.model_max_length = max_seq_len
        input_dict = tokenizer(text, padding=True, truncation=True)
        input_ids = input_dict["input_ids"]
        attention_mask = input_dict["attention_mask"]
        token_type_ids = input_dict["token_type_ids"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "targets": labels
    }


def normalise_labels(labels, n_label):
    norm_labels = []
    for label in labels:
        one_hot_vector_label = [0] * n_label
        one_hot_vector_label[label] = 1
        norm_labels.append(one_hot_vector_label)
    return np.asarray(norm_labels)


def segment_tokenize_inputs(text, tokenizer, max_seq_len, num_chunks):
    # input is full text of one document
    tokenized_texts = []
    tokens = tokenizer.tokenize(text)
    start_idx = 0
    seq_len = max_seq_len - 2
    for i in range(num_chunks):
        if start_idx > len(tokens):
            tokenized_texts.append([])
            continue
        tokenized_texts.append(tokens[start_idx:(start_idx + seq_len)])
        start_idx += seq_len

    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # get token type for token_ids_0
    token_type_ids = [tokenizer.create_token_type_ids_from_sequences(x) for x in input_ids]
    # append special token to end of sentence: <sep> <cls>
    input_ids = [tokenizer.build_inputs_with_special_tokens(x) for x in input_ids]
    # attention mask
    attention_mask = [[1] * len(x) for x in input_ids]

    # padding to max_length
    def padding_to_max(sequence, value):
        padding_len = max_seq_len - len(sequence)
        padding = [value] * padding_len
        return sequence + padding

    input_ids = [padding_to_max(x, tokenizer.pad_token_id) for x in input_ids]
    attention_mask = [padding_to_max(x, 0) for x in attention_mask]
    token_type_ids = [padding_to_max(x, tokenizer.pad_token_type_id) for x in token_type_ids]

    return input_ids, attention_mask, token_type_ids


def segment_tokenize_dataset(tokenizer, text, labels, max_seq_len, num_chunks):
    data = list(
        map(lambda t: segment_tokenize_inputs(t, tokenizer, max_seq_len, num_chunks), text))
    input_ids, attention_mask, token_type_ids = zip(*data)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "targets": labels
    }


# The following functions are modified from the relevant codes of https://github.com/aehrc/LAAT
def roc_auc(true_labels, pred_probs, average="macro"):
    if pred_probs.shape[0] <= 1:
        return

    fpr = {}
    tpr = {}
    if average == "macro":
        # get AUC for each label individually
        relevant_labels = []
        auc_labels = {}
        for i in range(true_labels.shape[1]):
            # only if there are true positives for this label
            if true_labels[:, i].sum() > 0:
                fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
                if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                    auc_score = auc(fpr[i], tpr[i])
                    if not np.isnan(auc_score):
                        auc_labels["auc_%d" % i] = auc_score
                        relevant_labels.append(i)

        # macro-AUC: just average the auc scores
        aucs = []
        for i in relevant_labels:
            aucs.append(auc_labels['auc_%d' % i])
        score = np.mean(aucs)
    else:
        # micro-AUC: just look at each individual prediction
        flat_pred = pred_probs.ravel()
        fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), flat_pred)
        score = auc(fpr["micro"], tpr["micro"])

    return score


def union_size(x, y, axis):
    return np.logical_or(x, y).sum(axis=axis).astype(float)


def intersect_size(x, y, axis):
    return np.logical_and(x, y).sum(axis=axis).astype(float)


def macro_accuracy(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (union_size(true_labels, pred_labels, 0) + 1e-10)
    return np.mean(num)


def macro_precision(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (pred_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (true_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(true_labels, pred_labels):
    prec = macro_precision(true_labels, pred_labels)
    rec = macro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def precision_at_k(true_labels, pred_probs, ks=[1, 5, 8, 10, 15]):
    # num true labels in top k predictions / k
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    for k in ks:
        topk = sorted_pred[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = len(tk)
                vals.append(num_true_in_top_k / float(denom))

        output.append(np.mean(vals))
    return output


def micro_recall(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / flat_true.sum(axis=0)


def micro_precision(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    if flat_pred.sum(axis=0) == 0:
        return 0.0
    return intersect_size(flat_true, flat_pred, 0) / flat_pred.sum(axis=0)


def micro_f1(true_labels, pred_labels):
    prec = micro_precision(true_labels, pred_labels)
    rec = micro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def micro_accuracy(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / union_size(flat_true, flat_pred, 0)


def calculate_scores(true_labels, logits, average="macro", is_multilabel=True, threshold=0.5):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    pred_probs = sigmoid(logits)
    pred_labels = np.rint(pred_probs - threshold + 0.5)

    max_size = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[: max_size]
    pred_labels = pred_labels[: max_size]
    pred_probs = pred_probs[: max_size]
    p_1 = 0
    p_5 = 0
    p_8 = 0
    p_10 = 0
    p_15 = 0
    if pred_probs is not None:
        if not is_multilabel:
            normalised_labels = normalise_labels(true_labels, len(pred_probs[0]))
            auc_score = roc_auc(normalised_labels, pred_probs, average=average)
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, average=average)
            recall = recall_score(true_labels, pred_labels, average=average)
            f1 = f1_score(true_labels, pred_labels, average=average)
        else:
            if average == "macro":
                accuracy = macro_accuracy(true_labels, pred_labels)  # categorical accuracy
                precision, recall, f1 = macro_f1(true_labels, pred_labels)
                p_ks = precision_at_k(true_labels, pred_probs, [1, 5, 8, 10, 15])
                p_1 = p_ks[0]
                p_5 = p_ks[1]
                p_8 = p_ks[2]
                p_10 = p_ks[3]
                p_15 = p_ks[4]

            else:
                accuracy = micro_accuracy(true_labels, pred_labels)
                precision, recall, f1 = micro_f1(true_labels, pred_labels)
            auc_score = roc_auc(true_labels, pred_probs, average)

    else:
        auc_score = -1

    output = {"{}_precision".format(average): precision, "{}_recall".format(average): recall,
              "{}_f1".format(average): f1, "{}_accuracy".format(average): accuracy,
              "{}_auc".format(average): auc_score, "{}_P@1".format(average): p_1, "{}_P@5".format(average): p_5,
              "{}_P@8".format(average): p_8, "{}_P@10".format(average): p_10, "{}_P@15".format(average): p_15}

    return output


