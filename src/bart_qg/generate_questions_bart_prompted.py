import json
import argparse

import torch
from torch import nn
import pandas as pd
import wandb

import transformers
from transformers import BartTokenizerFast
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.metrics_utils import compute_metrics
from bart_data_loader import BARTDataLoader


class LoggingTrainer(Seq2SeqTrainer):
    """Custom loss function to log error."""

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute loss and log it.
        @param model is a HF model.
        @param inputs is a list/tensor of inputs (must have `input_ids` +
        `attention_mask` + etc.) to the model.
        @param return_outputs is unused.
        @return the loss
        """
        outputs = model(**inputs)
        preds = torch.argmax(outputs["logits"], dim=-1)
        labels = inputs["labels"]
        metrics = compute_metrics((preds.cpu(), labels.cpu()), self.tokenizer)
        self.log(metrics)
        return outputs.loss


def train_model(
    model,
    tokenizer,
    train_dataset: torch.utils.data.Dataset,
    validation_dataset: torch.utils.data.Dataset,
    compute_metric_function,
    do_train: bool = True,
    do_eval: bool = True,
    dir_: str = "model_outputs/",
):
    """Train the model using HF trainer.
    @param model
    @param tokenizer
    @param train_dataset is the dataset used for training.
    @param validation_dataset is the validation dataset.
    @param compute_metric_function is the function used to compute metricss.
    @parma dir_ is the directory the model uses."""

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{dir_}/results",
        overwrite_output_dir=True,
        num_train_epochs=25,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=200,
        weight_decay=1e-6,
        learning_rate=2e-4,
        logging_dir=f"{dir_}/logs",
        adam_epsilon=1e-3,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_strategy="epoch",
        disable_tqdm=False,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        generation_num_beams=5,
        gradient_accumulation_steps=32,
        generation_max_length=184,
        report_to="wandb",
    )

    trainer = LoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metric_function,
        eval_dataset=validation_dataset,
        callbacks=[transformers.PrinterCallback],
    )

    # Train Model and save weights
    trainer.tokenizer = tokenizer
    trainer.train()
    torch.save(model.state_dict(), f"{dir_}/model.weights")

    # Save our predictions + the articles
    # Now get predictions
    preds = trainer.predict(validation_dataset)
    text = tokenizer.batch_decode(validation_dataset.input_ids, skip_special_tokens=True)
    questions = tokenizer.batch_decode(validation_dataset.questions, skip_special_tokens=True)
    ev = {
        "metrics": preds.metrics,
        "text": text,
        "qs": questions,
        "preds": tokenizer.batch_decode(preds.predictions, skip_special_tokens=True),
    }
    json.dump(ev, open(f"{dir_}/eval_preds.json", "w"))

    return model, ev


def test_model(model, test_dataset, dir_, output_file_type="eval"):
    """Test the model and return the labels + predictions."""
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{dir_}/results",
        do_train=False,
        do_eval=False,
        overwrite_output_dir=True,
        num_train_epochs=0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=200,
        weight_decay=1e-6,
        learning_rate=2e-4,
        logging_dir=f"{dir_}/logs",
        adam_epsilon=1e-3,
        max_grad_norm=1.0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        disable_tqdm=False,
        predict_with_generate=True,
        load_best_model_at_end=True,
        generation_num_beams=5,
        gradient_accumulation_steps=32,
        generation_max_length=184,
        report_to="wandb",
    )

    trainer = LoggingTrainer(
        model=model,
        tokenizer=None,
        args=training_args,
        compute_metrics=compute_metric_function,
        eval_dataset=test_dataset,
        callbacks=[transformers.PrinterCallback],
    )

    # Now get predictions
    ev = {}
    preds = trainer.predict(test_dataset)
    text = tokenizer.batch_decode(test_dataset.input_ids, skip_special_tokens=True)
    questions = tokenizer.batch_decode(test_dataset.questions, skip_special_tokens=True)
    ev["metrics"] = preds.metrics
    ev["text"] = text
    ev["qs"] = questions
    ev["preds"] = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
    ev["ids"] = test_dataset.ids
    ev["triggers"] = test_dataset.triggers

    json.dump(ev, open(f"{dir_}/{output_file_type}_preds.json", "w"))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-loc", type=str, default="data/discq_questions_final.csv", help="Where is the data df file stored?"
    )
    parser.add_argument(
        "--relations-text-loc", type=str, default="data/relations_txt/", help="Where are the n2c2 documents stored?"
    )
    parser.add_argument("--train", action="store_true", help="To train or not to train, that is the question")
    parser.add_argument("--test", action="store_true", help="To test or not to test, that is the question.")
    parser.add_argument("--debug-mode", action="store_true", help="Run with 10% of the data.")
    parser.add_argument("--final-test-file", type=str, help="Final test CSV.")
    parser.add_argument(
        "--chunk-mode",
        type=str,
        help="How to split up the discharge summaries?",
        choices=["sentence-level", "stride-level", "none"],
    )
    parser.add_argument(
        "--split-questions",
        action="store_true",
        help="Should we split the questions/triggers into individual questions?",
    )
    parser.add_argument("--output", required=True, type=str, help="Where to save model outputs, logs, and predictions.")
    parser.add_argument("--tokenizer", type=str, default="facebook/bart-large", help="Name/Location of the tokenizer.")
    parser.add_argument(
        "--model-init", type=str, default="facebook/bart-large", help="Name/Location of the QA init model."
    )
    parser.add_argument("--model-weights", type=str, help="Model weights to load.")
    parser.add_argument("--max-seq-length", required=False, default=512, type=int, help="Max length of context.")
    parser.add_argument(
        "--max-question-length", required=False, default=184, type=int, help="Max length of a question."
    )
    parser.add_argument("--max-trigger-length", required=False, default=184, type=int, help="Max length of a trigger.")
    parser.add_argument("--doc-stride", required=False, default=512, type=int, help="Document stride.")
    args = parser.parse_args()

    # Set info
    torch.manual_seed(42)
    wandb.init(project="clinical-qg", entity="clinical-qg-project")

    # Load the data
    tokenizer = BartTokenizerFast.from_pretrained(args.tokenizer)
    bdl = BARTDataLoader(
        data_loc=args.data_loc,
        relations_text_loc=args.relations_text_loc,
        tokenizer=tokenizer,
        debug_mode=args.debug_mode,
        stride=args.doc_stride,
        max_len=args.max_seq_length,
        max_q_len=args.max_question_length,
        max_trigger_len=args.max_trigger_length,
        chunk_mode=args.chunk_mode,
        split_questions=args.split_questions,
    )

    train_cohort, validation_cohort, held_out_cohort = bdl.load_data()

    # Load the model
    tokenizer.model_input_names = ["input_ids", "attention_mask", "decoder_attention_mask"]
    model = transformers.BartForConditionalGeneration.from_pretrained(args.model_init)
    compute_metric_function = lambda x: compute_metrics(x, split_questions=args.split_questions, tokenizer=tokenizer)

    if args.model_weights:
        model.load_state_dict(torch.load(args.model_weights))

    if args.train:
        train_cohort, validation_cohort, held_out_cohort = bdl.load_data()
        model, ev = train_model(
            model, tokenizer, train_cohort, validation_cohort, compute_metric_function, dir_=args.output
        )

    if args.test:
        train_cohort, validation_cohort, held_out_cohort = bdl.load_data()
        res_test = test_model(model, held_out_cohort, dir_=args.output)

    if args.final_test_file:
        df = pd.read_csv(args.final_test_file)
        test_cohort = bdl.tokenize_generated_df(df)
        decoded = tokenizer.batch_decode(test_cohort.input_ids, skip_special_tokens=True)
        test_model(model, test_cohort, dir_=args.output, output_file_type="test")
