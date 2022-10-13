import argparse

import torch
from torch import nn as nn
import numpy as np
import pandas as pd

from transformers import Trainer, TrainingArguments, AutoTokenizer, BertForTokenClassification
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils import read_file, split_ids, convert_preds_to_triggers

torch.manual_seed(42)


def tokenize_df(relations_text_loc: str, tokenizer, df, max_len: int = 512, stride: int = 512):
    """Convert a pandas dataframe into a `ClassificationDataset`.
    @param relations_text_loc is the folder location of where the n2c2 discharge
    summaries are stored.
    @param tokenizer is a HF tokenizer
    @param df is a pandas dataframe in which each row contains:
            `id`: is the id to read the file from
            `start_index`: is the start index of the trigger.
            `end_index`: is the end index of the trigger.
    @param max_len is the maximum length of the document.
    @param stride is how much to shift over from last chunk taken.
    @return a ClassificationDataset instance."""
    read_files, all_tokenized_outputs = {}, {}

    # Need room for [CLS] + [SEP]
    max_len -= 2
    stride -= 2
    for index, row in df.iterrows():
        if row.id in read_files:
            text, tokenized_output, remapper = read_files[row.id]
        else:
            text, remapper = read_file(row.id, relations_text_loc)
            tokenized_output = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            read_files[row.id] = (text, tokenized_output, remapper)
            all_tokenized_outputs[row.id] = {
                "input_ids": tokenized_output.input_ids,
                "labels": [0] * len(tokenized_output.input_ids),
                "id": row.id,
            }

        # Find the offset
        new_st, new_end = remapper[row.start_index], remapper[row.end_index]
        tok_st = np.where(np.asarray(tokenized_output.offset_mapping) > new_st)[0][0]
        tok_end = np.where(np.asarray(tokenized_output.offset_mapping) > new_end)[0][0]
        for i in range(tok_st, tok_end + 1):
            all_tokenized_outputs[row.id]["labels"][i] = 1

    # Now split up data to max_len + stride.
    xs, ys, attn_mask, ids = [], [], [], []
    for id_ in all_tokenized_outputs.keys():
        for i in range(0, len(all_tokenized_outputs[id_]["input_ids"]), stride):
            pad = max_len - len(all_tokenized_outputs[id_]["input_ids"][i : i + max_len])

            # Add CLS, SEP, and PAD
            input_ = all_tokenized_outputs[id_]["input_ids"][i : i + max_len]
            input_cls_sep = [tokenizer.cls_token_id] + input_ + [tokenizer.sep_token_id]
            pad_input = input_cls_sep + [tokenizer.pad_token_id] * pad

            xs.append(pad_input)
            ys.append([0] + all_tokenized_outputs[id_]["labels"][i : i + max_len] + [0] * (1 + pad))
            ids.append(all_tokenized_outputs[id_]["id"])
            attn_mask.append([1] * (max_len - pad + 2) + [0] * pad)

    return ClassificationDataset(xs, ys, attn_mask, ids)


def load_data(
    df_loc: str,
    relations_text_loc: str,
    tokenizer,
    debug_mode: bool = False,
    max_len: int = 512,
    stride: int = 512,
    val_split: float = 0.10,
    test_split: float = 0.20,
    seed: int = 0,
):
    """Load the data and return train, val, and test sets.
    @param df_loc is the string file location of the data.
    @param relations_text_loc is the folder location of where the n2c2 discharge
    summaries are stored.
    @param tokenizer is a HF tokenizer.
    @param debug_mode is whether or not we should run with a smaller input.
    @param max_len is the maximum length of the model input.
    @param stride is how much to shift over from the last model input.
    @param val_split is what percent to allocate to the validation set.
    @param test_split is what percent to allocate to the test set.
    @param seed is the random seed to load the data with."""
    # Load the data and split it
    df = pd.read_csv(df_loc)
    ordered_ids = []
    seen = set()
    for id_ in df.id.values:
        if not (id_ in seen):
            seen.add(id_)
            ordered_ids.append(id_)

    # Create each DF
    tr_ids, val_ids, test_ids = split_ids(ordered_ids, seed=seed, val_split=val_split, test_split=test_split)
    tr_df = df[df.id.isin(tr_ids)]
    vl_df = df[df.id.isin(val_ids)]
    te_df = df[df.id.isin(test_ids)]

    # Shrink each DF if we are in debug-mode
    if debug_mode:
        tr_df = tr_df[: int(len(tr_df) * 0.05)]
        vl_df = vl_df[: int(len(vl_df) * 0.05)]
        te_df = te_df[: int(len(te_df) * 0.05)]

    # Now tokenize each df
    tr_set = tokenize_df(relations_text_loc, tokenizer, tr_df, max_len, stride)
    vl_set = tokenize_df(relations_text_loc, tokenizer, vl_df, max_len, stride=max_len)
    te_set = tokenize_df(relations_text_loc, tokenizer, te_df, max_len, stride=max_len)

    return tr_set, vl_set, te_set, df


class TriggerTrainer(Trainer):
    """Simple trainer that allows us to log and save things."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.losses = []

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss and log it."""
        pl = super().compute_loss(model, inputs, return_outputs)
        if type(pl) != type(tuple()):
            self.losses.append(torch.sum(pl).detach().item())

        return pl

    def evaluate(self, *args, **kwargs):
        """Evaluate model and log data."""
        eval_datal = self.get_eval_dataloader(self.eval_dataset)
        output = self.evaluation_loop(eval_datal, description="Eval")

        labels = torch.tensor([x["labels"] for x in self.eval_dataset]).float()
        preds = np.argmax(output.predictions, axis=-1)

        filtered_preds, filtered_labels = [], []
        for art_idx, article in enumerate(self.eval_dataset.attention_mask):
            for tk_idx, token in enumerate(article):
                if int(token) == 1:
                    filtered_preds.append(preds[art_idx][tk_idx])
                    filtered_labels.append(labels[art_idx][tk_idx])

        metrics = output.metrics
        metrics["train_loss"] = np.average(self.losses)
        metrics["eval_f1"] = f1_score(filtered_labels, filtered_preds)
        metrics["eval_recall"] = recall_score(filtered_labels, filtered_preds)
        metrics["eval_precision"] = precision_score(filtered_labels, filtered_preds)

        # Reset loss + log metrics.
        self.losses = []
        self.log(metrics)
        return metrics


class ClassificationDataset(torch.utils.data.Dataset):
    """Simple dataset to hold inputs, masks, labels, and which docs these come from.
    `encodings`: is a list of tensors for the discharge summaries.
    `labels`: is a list of tensors that is whether or not each token
     is a trigger or not.
    `attention_mask`: is a list of tensors that determines if we should
     mask the input token.
    `ids`: are which documents each of the other items came from.
    """

    def __init__(self, encodings, labels, attention_mask, ids):
        self.encodings = encodings
        self.attention_mask = attention_mask
        self.labels = labels
        self.ids = ids

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "id": self.ids[idx],
        }
        return item

    def __len__(self):
        return len(self.labels)


def train_model(model, train_dataset, validation_dataset, dir_="model_outputs/"):
    """Train the model.
    @param model to train
    @param train_dataset is the dataset to use for training.
    @param validation_dataset is the dataset to use for validation.
    @param dir_ is where to log and save."""
    training_args = TrainingArguments(
        output_dir=f"{dir_}/results",
        do_train=True,
        do_eval=True,
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=50,
        weight_decay=0.1,
        learning_rate=3e-5,
        logging_dir=f"{dir_}/logs",
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        save_total_limit=3,
        gradient_accumulation_steps=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        disable_tqdm=False,
        report_to="wandb",
    )

    trainer = TriggerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=None,
        eval_dataset=validation_dataset,
    )

    trainer.train()
    torch.save(model.state_dict(), f"{dir_}/model.weights")
    ev = trainer.evaluate()
    return model, ev


def test_model(model, test_dataset, dir_):
    """Test the model and return the labels + predictions.
    @return the model, predictions, a pd df (see `convert_preds_to_triggers`),
    and the metrics calculated."""
    training_args = TrainingArguments(
        output_dir=f"{dir_}/results",
        do_train=False,
        do_eval=True,
        overwrite_output_dir=True,
        num_train_epochs=0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.001,
        learning_rate=2e-5,
        logging_dir=f"{dir_}/logs",
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        disable_tqdm=False,
    )

    trainer = TriggerTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        compute_metrics=None,
        eval_dataset=test_dataset,
    )

    # Calculate metrics and format predictions
    metrics = trainer.evaluate()
    ev = trainer.predict(test_dataset)
    trigger_df = convert_preds_to_triggers(
        tokenizer, test_dataset.ids, test_dataset.encodings, np.argmax(ev.predictions, axis=-1)
    )

    return model, ev, trigger_df, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--df-loc", type=str, default="data/discq_questions_final.csv", help="Where is the data df file stored?"
    )
    parser.add_argument(
        "--relations-text-loc", type=str, default="data/relations_txt/", help="Where are the i2b2 documents stored?"
    )
    parser.add_argument("--debug-mode", action="store_true", help="Run with a much smaller amount of data")
    parser.add_argument("--train", action="store_true", help="To train or not to train, that is the question")
    parser.add_argument("--test", action="store_true", help="To test or not to test, that is the question.")
    parser.add_argument("--output", required=True, type=str, help="Where to save model outputs, logs, and predictions.")
    parser.add_argument(
        "--tokenizer", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="Name/Location of the tokenizer."
    )
    parser.add_argument(
        "--model-init", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="Name/Location of the QA init model."
    )
    parser.add_argument("--model-weights", type=str, help="Model weights to load.")
    parser.add_argument("--max-seq-length", required=False, default=512, type=int, help="Max length of context.")
    parser.add_argument("--doc-stride", required=False, default=512, type=int, help="Document stride.")
    parser.add_argument("--output-df", default="", type=str, help="Where to store the predicted triggers.")
    args = parser.parse_args()

    # Load the data
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
    train_cohort, validation_cohort, held_out_cohort, _ = load_data(
        args.df_loc, args.relations_text_loc, tokenizer, debug_mode=args.debug_mode
    )

    # Load the model
    model = BertForTokenClassification.from_pretrained(args.model_init)

    if args.model_weights:
        model.load_state_dict(torch.load(args.model_weights))

    if args.train:
        model, ev = train_model(model, train_cohort, validation_cohort, dir_=args.output)

    if args.test:
        model, test_preds, trigger_df, metrics = test_model(model, held_out_cohort, dir_=args.output)
        if args.output_df != "":
            trigger_df.to_csv(args.output_df, index=False)
