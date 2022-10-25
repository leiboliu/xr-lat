#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import pickle
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import datasets
import pandas as pd
import torch

from models.utils import load_data, count_parameters, compute_metrics, post_process
from models.model import HiLATModel
import transformers
from torch.utils.data import Subset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.file_utils import ExplicitEnum
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)

check_min_version("4.10.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "mimic3-50": ("mimic3-50"),
    "mimic3-full": ("mimic3-full"),
}


class TransformerLayerUpdateStrategy(ExplicitEnum):
    NO = "no"
    LAST = "last"
    ALL = "all"


class DocumentPoolingStrategy(ExplicitEnum):
    FLAT = "flat"
    MAX = "max"
    MEAN = "mean"


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."})

    # customized data arguments
    label_dictionary_file: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the test data file."}
    )
    # Cluster chain
    cluster_chain_file: Optional[str] = field(
        default=None,
        metadata={"help": "The pickle file name of the cluster chain file."}
    )
    ignore_keys_for_eval: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The list of keys to be ignored during evaluation process."}
    )
    use_cached_datasets: bool = field(
        default=True,
        metadata={"help": "if use cached datasets to save preprocessing time. The cached datasets were preprocessed "
                          "and saved into data folder."})
    data_segmented: bool = field(
        default=False,
        metadata={"help": "if dataset is segmented or not"})
    training_level: int = field(
        default=0,
        metadata={"help": "Current training level"})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need a training/validation file")
        elif self.label_dictionary_file is None:
            raise ValueError("label dictionary must be provided")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                    validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    transformer_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    transformer_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    transformer_tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    d_model: int = field(
        default=768,
        metadata={"help": "hidden size of model. should be the same as base transformer model"})
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout of transformer layer"})
    dropout_att: float = field(
        default=0.1,
        metadata={"help": "Dropout of label-wise attention layer"})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    num_chunks_per_document: int = field(
        default=1,
        metadata={"help": "Num of chunks per document"})
    transformer_layer_update_strategy: TransformerLayerUpdateStrategy = field(
        default="all",
        metadata={"help": "Update which transformer layers when training"})
    chunk_attention: bool = field(
        default=False,
        metadata={"help": "if use chunk attention for each label"})
    linear_init_mean: float = field(
        default=0.0,
        metadata={"help": "mean value for initializing linear layer weights"})
    linear_init_std: float = field(
        default=0.03,
        metadata={"help": "standard deviation value for initializing linear layer weights"})
    document_pooling_strategy: DocumentPoolingStrategy = field(
        default="flat",
        metadata={"help": "how to pool document representation after label-wise attention layer for each label"})
    num_labels: int = field(
        default=None,
        metadata={"help": "Num of labels"})
    bootstrapping: bool = field(
        default=False,
        metadata={"help": "if bootstrap the model using the parent model"})
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "transformer model type"},
    )
    threshold: float = field(
        default=0.5,
        metadata={"help": "threshold for prediction"})
    beam_search: bool = field(
        default=False,
        metadata={"help": "if beam search"})
    use_hyperbolic: bool = field(
        default=False,
        metadata={"help": "if use hyperbolic embeddings when initialization"})
    hyperbolic_dim: int = field(
        default=50,
        metadata={"help": "Dim of hyperbolic embeddings"})
    use_ASL: bool = field(
        default=False,
        metadata={"help": "if use asymmetric loss"})


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(training_args.output_dir, "log_{}.txt".format(now)))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), file_handler],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    transformers.utils.logging.add_handler(file_handler)

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.
    data_files = {"train": data_args.train_file,
                  "validation": data_args.validation_file,
                  "label_dict": data_args.label_dictionary_file,
                  "cluster_chain": data_args.cluster_chain_file}

    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    # when you use `do_predict` without specifying a GLUE benchmark task.
    if training_args.do_predict:
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        else:
            raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    label_dict = pd.read_csv(data_args.label_dictionary_file)

    num_dict_labels = label_dict.shape[0]
    logger.info(f"Label dictionary size:{num_dict_labels} ")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.transformer_tokenizer_name if model_args.transformer_tokenizer_name else model_args.transformer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="right"
    )

    # ==============================================================
    # xr-HiLAT
    # C. Cluster Chain
    # Chapter: C1 k1*k0 k0=1
    # Block: C2 k2*k1
    # Category: C3 k3*k2
    # Code: C4 k4*k3 k4=L (8929 for MIMIC-III-full)
    num_level = 1
    curr_cluster = None
    if data_args.cluster_chain_file is not None:
        with open(data_args.cluster_chain_file, 'rb') as f:
            curr_cluster = pickle.load(f)
            logger.info("Hierarchical label tree: {}".format([c.shape[0] for c in curr_cluster]))

        num_level = len(curr_cluster)

    print("Start training xr-HiLAT.........")
    M, eval_M, test_M = None, None, None
    parent_model = None
    for i in range(num_level):
        training_args.output_dir = training_args.output_dir + "/level_" + str(i)
        if data_args.training_level > i:
            continue
        # num_labels for current level
        num_labels = curr_cluster[i].shape[0] if data_args.cluster_chain_file is not None else num_dict_labels
        model_args.num_labels = num_labels
        # prepare datasets
        if training_args.do_train:
            train_dataset = load_data(data_args.train_file, tokenizer, curr_cluster, M, i, data_args, model_args)

        if training_args.do_eval:
            eval_dataset = load_data(data_args.validation_file, tokenizer, curr_cluster, eval_M, i, data_args, model_args)

        hilat_model = HiLATModel(model_args)
        if model_args.bootstrapping and parent_model is not None:
            hilat_model.init_transformer_from_model(parent_model)
            # assign the linear weights of other layers
            with torch.no_grad():
                hilat_model.label_wise_attention_layer.l1_linear.weight.copy_(parent_model.label_wise_attention_layer.l1_linear.weight)
                curr_cluster_tensor = torch.from_numpy(curr_cluster[i])
                hilat_model.label_wise_attention_layer.l2_linear.weight.copy_(
                    torch.matmul(curr_cluster_tensor.type(parent_model.label_wise_attention_layer.l2_linear.weight.dtype).to(training_args.device), parent_model.label_wise_attention_layer.l2_linear.weight))
                # if chunk attention is true
                if model_args.chunk_attention:
                    hilat_model.chunk_attention_layer.l1_linear.weight.copy_(parent_model.chunk_attention_layer.l1_linear.weight)
                    hilat_model.chunk_attention_layer.l2_linear.weight.copy_(
                        parent_model.chunk_attention_layer.l2_linear.weight)

                # classifier
                hilat_model.classifier_layer.weight.copy_(torch.matmul(curr_cluster_tensor.type(parent_model.classifier_layer.weight.dtype).to(training_args.device), parent_model.classifier_layer.weight))
                hilat_model.classifier_layer.bias.copy_(torch.matmul(curr_cluster_tensor.type(parent_model.classifier_layer.bias.dtype).to(training_args.device), parent_model.classifier_layer.bias))


        logger.info("Model parameters: {}".format(count_parameters(hilat_model)))
        logger.info("Level: {}, active labels: {}".format(str(i), train_dataset.label_width))

        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        # Initialize our Trainer
        trainer = Trainer(
            model=hilat_model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(
                training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint,
                                         ignore_keys_for_eval=data_args.ignore_keys_for_eval)
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

            if model_args.bootstrapping:
                parent_model = deepcopy(trainer.model)
            # generate M
            # if data_args.cluster_chain_file is not None:
            if model_args.beam_search:
                # save parent model
                train_dataset = load_data(data_args.train_file, tokenizer, curr_cluster, M, i, data_args, model_args,
                                          True)
                output = trainer.predict(train_dataset, metric_key_prefix="train_M")
                M = post_process(output.predictions, train_dataset.idx_padding)

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            metrics = trainer.evaluate(eval_dataset=eval_dataset, ignore_keys=data_args.ignore_keys_for_eval)
            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            if model_args.beam_search:
                eval_dataset = load_data(data_args.validation_file, tokenizer, curr_cluster, eval_M, i, data_args, model_args, True)
                output = trainer.predict(eval_dataset, metric_key_prefix="eval_M")
                eval_M = post_process(output.predictions, eval_dataset.idx_padding)

        if training_args.do_predict:
            logger.info("*** Predict ***")

            # output: label_ids, metrics, predictions(logits, pred_probs, attention_weights)
            predict_dataset = load_data(data_args.test_file, tokenizer, curr_cluster, test_M, i, data_args, model_args, True)
            output = trainer.predict(predict_dataset, metric_key_prefix="predict")

            metric_scores = None
            final_test_logits = None
            if model_args.beam_search:
                test_M = post_process(output.predictions, predict_dataset.idx_padding, threshold=model_args.threshold)
                final_test_logits = post_process(output.predictions, predict_dataset.idx_padding, logits=True,
                                                 threshold=model_args.threshold)

                # calculate metrics
                metric_scores = compute_metrics((predict_dataset.labels, final_test_logits),
                                                threshold=model_args.threshold)


            predict_result = {"labels": predict_dataset.labels,
                              "metrics": metric_scores if metric_scores is not None else output.metrics,
                              "predictions": output.predictions, "preds_labels": test_M,
                              "preds_logits": final_test_logits}

            logger.info(
                "Metrics on test datasets: {}".format(metric_scores if metric_scores is not None else output.metrics))
            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{data_args.task_name}.pkl")

            if trainer.is_world_process_zero():
                logger.info("Save predictions into {}".format(output_predict_file))
                with open(output_predict_file, "wb") as writer:
                    pickle.dump(predict_result, writer, pickle.HIGHEST_PROTOCOL)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
