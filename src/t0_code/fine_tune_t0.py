#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning T0 in PyTorch.

This script is adapted from
https://github.com/huggingface/transformers/blob/master/examples/pytorch/multiple-choice/run_swag_no_trainer.py
as well as
https://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization_no_trainer.py
"""

import json
import argparse
import logging
import os
import random
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union
import csv
import math

import numpy as np
import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    DataCollatorForSeq2Seq,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import PaddingStrategy

import src.metrics_utils

logger = logging.getLogger(__name__)


PROMPTS = {
    "after_reading_what_question": """{text}\nAfter reading the above EMR, what question do you have about "{trigger}"?\nQuestion:""",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning T0 in PyTorch, optionally few-shot.")

    # Data / Model
    parser.add_argument(
        "-d",
        "--dataset_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "-t",
        "--prompt_name",
        type=str,
        default=None,
        required=True,
        choices=PROMPTS.keys(),
        help="The template/prompt name from fine_tune_t0:PROMPTS.",
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help=(
            "Path to pretrained model or model identifier from huggingface.co/models. "
            "The list of T0 variants can be found on `https://huggingface.co/bigscience/T0_3B`"
        ),
    )
    parser.add_argument(
        "-il",
        "--max_length",
        type=int,
        default=1024,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "-tl",
        "--target_max_length",
        type=int,
        default=256,
        help="Target max length. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "-pml",
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Where to store the results CSV and (TODO) optionally the final model.",
    )

    # Training
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "-ep", "--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "-ms",
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "-ga",
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "-ie",
        "--input_eos",
        action="store_true",
        help=(
            "T0 was trained without EOS in its input sequences, which is the default in this script."
            "However, T5 was pretrained with EOS in its input sequences. See README for more info."
        ),
    )
    parser.add_argument(
        "-pa",
        "--parallelize",
        action="store_true",
        help=(
            "If passed, will call `model.parallelize` which splits the model on all GPUs available (model parallelism). "
            "Note that this feature is still experimental in HF Transformers."
        ),
    )
    parser.add_argument(
        "-tb",
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.01, help="Weight decay for the AdamW optimizer.")
    parser.add_argument(
        "-ls",
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "-ws", "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    # Misc
    parser.add_argument(
        "-db",
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "-wb",
        "--wandb_proj",
        type=str,
        default=None,
        help="Project name for Weights & Biases. By default, W&B is disabled.",
    )
    parser.add_argument(
        "-sd",
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k != "targets"} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id] * (max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0] * (max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # Convert to tensors
        batch = {k: torch.tensor(v) for k, v in batch.items()}

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch


def main():
    args = parse_args()
    set_seed(args.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Handle the output directory creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    raw_train_dataset = load_dataset(args.dataset_name_or_path, split="train")
    raw_eval_dataset = load_dataset(args.dataset_name_or_path, split="validation")

    # Trim a number of evaluation examples
    if args.debug:
        raw_train_dataset = raw_train_dataset.select(range(min(100, len(raw_train_dataset))))
        raw_eval_dataset = raw_eval_dataset.select(range(min(100, len(raw_eval_dataset))))

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
    )

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.pad_to_max_length:
        raise NotImplementedError(
            "Padding to max length is not implemented for the case of beginning truncation that we use here."
        )
        padding = "max_length" if args.pad_to_max_length else False
    else:
        padding = False

    template = PROMPTS[args.prompt_name]
    column_names = raw_eval_dataset.column_names

    def preprocess_batch(examples):
        bs = len(examples[column_names[0]])

        input_texts = []
        target_texts = []
        for i in range(bs):
            ex = {k: examples[k][i] for k in ["text", "trigger", "question"]}
            input_text = [template.format(text=e["text"], trigger=e["trigger"]) for e in ex]
            target_text = ex["question"]

            input_texts.append(input_text)
            target_texts.append(target_text)

        model_inputs = tokenizer(
            input_texts,
            padding=padding,
            max_length=args.max_length,
            truncation=False,
            add_special_tokens=args.input_eos,
        )

        # truncating from the beginning of the sequence
        model_inputs = {k: v[-args.max_length :] for k, v in model_inputs.items()}

        with tokenizer.as_target_tokenizer():
            tokenized_targets = tokenizer(
                target_texts,
                padding=padding,
                max_length=args.target_max_length,
                truncation=True,
                add_special_tokens=False,
            )
            model_inputs["labels"] = [
                [(t if t != tokenizer.pad_token_id else -100) for t in targets]
                for targets in tokenized_targets["input_ids"]
            ]
        return model_inputs

    with accelerator.main_process_first():
        train_dataset = raw_train_dataset.map(preprocess_batch, batched=True, remove_columns=column_names)
        eval_dataset = raw_eval_dataset.map(preprocess_batch, batched=True, remove_columns=column_names)

    # Log a few random examples:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.debug(f"Sample {index} of the training set: {train_dataset[index]}.")
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.debug(f"Sample {index} of the evaluation set: {eval_dataset[index]}.")

    # DataLoaders creation:
    seq2seq_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8 if accelerator.use_fp16 else None
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=seq2seq_collator, batch_size=args.per_device_train_batch_size
    )

    eval_dataloader = DataLoader(eval_dataset, collate_fn=seq2seq_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    if args.parallelize:
        num_gpus = torch.cuda.device_count()
        assert num_gpus > 1, "You need at least 2 GPUs to use `model.parallelize()`."
        model.parallelize()
        optimizer, train_dataloader, eval_dataloader = accelerator.prepare(optimizer, train_dataloader, eval_dataloader)
    else:
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

    # Metrics
    accuracy = load_metric("accuracy")

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_steps = 0

    if args.wandb_proj and accelerator.is_main_process:
        import wandb

        extra_metadata = {
            "template_jinja": template.jinja,
            "template_answer_choices": template.answer_choices,
            "template_reflects_original_task": template.metadata.original_task,
            "template_choices_in_prompt": template.metadata.choices_in_prompt,
            "template_comment": template.reference,
        }
        run_config = vars(args)
        run_config.update(extra_metadata)
        wandb.init(
            project=args.wandb_proj,
            config=run_config,
        )

    result_table = []
    all_predictions = None
    all_eval_input_ids = None
    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_steps += 1
                loss = loss.item()
                if accelerator.is_main_process:
                    tqdm.write(f"epoch = {epoch}, step = {global_steps}, loss = {loss}")
                if args.wandb_proj and accelerator.is_main_process:
                    wandb.log({"loss": loss}, step=global_steps)

            if global_steps >= args.max_train_steps:
                break

        # Evaluate every epoch
        total_batch_size = args.per_device_eval_batch_size * accelerator.num_processes
        logger.info("***** Running evaluation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")
        logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
        # Only show the progress bar once on each machine.  # NOTE commented out to avoid nested pbar mess
        # progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)

        all_eval_input_ids = []
        all_predictions = []
        all_targets = []
        model.eval()
        for batch in eval_dataloader:
            model_inputs = {k: batch[k] for k in ["input_ids", "attention_mask", "labels"]}
            with torch.inference_mode():
                logits = model(**model_inputs).logits

            masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
            seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
            seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
            seq_log_prob = seq_log_prob.view(
                batch["targets"].size(0), -1
            )  # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
            predictions = seq_log_prob.argmax(dim=-1)

            predictions = accelerator.gather(predictions)
            targets = accelerator.gather(batch["targets"])
            accuracy.add_batch(predictions=predictions, references=targets)

            all_predictions.append(predictions.numpy())
            all_targets.append(batch["targets"].numpy())

            if epoch == args.num_train_epochs - 1:
                # collect everything for the last epoch to log predictions on the whole eval set
                eval_input_ids = accelerator.gather(batch["input_ids"])
                all_eval_input_ids.append(eval_input_ids.numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_eval_input_ids = np.concatenate(all_eval_input_ids, axis=0)

        generation_metrics = src.metrics_utils.compute_metrics((all_predictions, all_targets), tokenizer)
        generation_metrics["accuracy"] = accuracy.compute()["accuracy"]

        result_table.append(
            {
                "dataset_name": args.dataset_name_or_path,
                "dataset_config_name": args.dataset_config_name,
                "template_name": args.template_name,
                "epoch": epoch,
                "step": global_steps,
                "metrics": generation_metrics,
            }
        )
        if args.wandb_proj and accelerator.is_main_process:
            wandb.log(generation_metrics, step=global_steps)

        # save model using accelerate
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

    # End training loop
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "results.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=result_table[0].keys())
            writer.writeheader()
            writer.writerows(result_table)

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

        predicted_texts = tokenizer.batch_decode(all_predictions, skip_special_tokens=True)
        targets = tokenizer.batch_decode(all_targets, skip_special_tokens=True)
        eval_input_ids = tokenizer.batch_decode(all_eval_input_ids, skip_special_tokens=True)

        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as f:
            for prediction, target, input_id in zip(predicted_texts, targets, eval_input_ids):
                f.write(
                    json.dumps(
                        {
                            "prediction": prediction,
                            "target": target,
                            "input_id": input_id,
                        }
                    )
                    + "\n"
                )

        if args.wandb_proj:
            wandb.finish()


if __name__ == "__main__":
    main()
