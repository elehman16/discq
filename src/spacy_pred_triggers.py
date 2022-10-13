import json
import argparse

import spacy
import numpy as np
from scispacy.linking import EntityLinker  # it looks unused, but this import is required for scispacy_linker pipe
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from src.utils import read_file
from find_triggers_question_agnostic import load_data


def spacy_predictions(relations_text_loc: str, tokenizer, dataset, df):
    """Predict that all of the spacy entities are triggers.
    @param relations_text_loc is where the n2c2 discharge summaries are stored.
    @param tokenizer is a Huggingface tokenizer.
    @param dataset is a dataset returned from `load_data`. It should have the
    following entries:
        `labels`: token-piece level labels that determine if that token is a
        trigger or not.
        `attention_mask`: token-level attention-mask -> see HF documentation.
    @param df is a dataframe containing ids of text files. It should have the
    following columns:
        `id`: is a column full of document ids that align with the order
        given in the dataset.
    """
    labels, ids = [], []
    max_len = len(dataset[0]["input_ids"])

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        ids.append(row.id)
        text, remapper = read_file(row.id, relations_text_loc)
        tokenized_input = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tmp = nlp(text)
        preds = [0] * len(tokenized_input["input_ids"])

        for ent in tmp.ents:
            if len(ent._.umls_ents) > 0:
                tok_st = np.where(np.asarray(tokenized_input.offset_mapping) > ent.start_char)[0][0]
                tok_end = np.where(np.asarray(tokenized_input.offset_mapping) >= ent.end_char)[0][0]
                for i in range(tok_st, tok_end):
                    preds[i] = 1

        for i in range(0, len(preds), max_len - 2):
            cls_labels = [0] + preds[i : i + max_len - 2]
            pad_ = [0] * (max_len - len(cls_labels))
            cls_labels.extend(pad_)
            labels.append(cls_labels)

    # Remove any PADDED tokens. Only consider real tokens that we found in the
    # original text.
    true_labels = [x["labels"] for x in dataset]
    filtered_labels, filtered_preds = [], []
    for i in range(len(dataset)):
        for tk_idx, token in enumerate(dataset[i]["attention_mask"]):
            if int(token) == 1:
                filtered_preds.append(labels[i][tk_idx])
                filtered_labels.append(true_labels[i][tk_idx])

    metrics = {"preds": labels, "labels": true_labels, "ids": ids}
    metrics["eval_f1"] = f1_score(filtered_labels, filtered_preds)
    metrics["eval_recall"] = recall_score(filtered_labels, filtered_preds)
    metrics["eval_precision"] = precision_score(filtered_labels, filtered_preds)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-loc", type=str, required=True, help="Path to discq_questions_final.csv")
    parser.add_argument(
        "--relations-text-loc",
        type=str,
        required=True,
        help="Path to relations_txt directory with the texts of discharge summaries.",
    )
    parser.add_argument("--debug-mode", action="store_true", help="Run with a much smaller amount of data")
    parser.add_argument("--output", required=True, type=str, help="Where to save model outputs, logs, and predictions.")
    parser.add_argument(
        "--tokenizer", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="Name/Location of the tokenizer."
    )
    args = parser.parse_args()

    print(f"Starting spacy_pred_triggers.py with args: {args}")
    print("Loading spacy model...")
    # Load scispacy models
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

    # Load the data
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, do_lower_case=False)

    print("Loading data...")
    train_cohort, validation_cohort, held_out_cohort, df = load_data(
        df_loc=args.df_loc,
        relations_text_loc=args.relations_text_loc,
        tokenizer=tokenizer,
        debug_mode=args.debug_mode,
    )

    print("Predicting...")
    metrics = spacy_predictions(args.relations_text_loc, tokenizer, held_out_cohort, df)

    print(f"Metrics: {metrics}")
    with open(args.output, "w") as f:
        json.dump(metrics, f)

    print("Done!")
