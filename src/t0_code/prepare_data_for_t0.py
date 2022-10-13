# reads SQL and text files and saves a Hugginface Dataset object with fields:
#  - text: text near the trigger
#  - trigger: text of the question trigger
#  - question: text of the question
# also splits the dataset into train, dev, and test.
# Likely in the same way as in BART experiments, because random seed is fixes and = 0.
#
# based on notebooks/01_prepare_data_for_prompting.ipynb

import os
import argparse

import pandas as pd
import datasets
from tqdm.auto import tqdm

import src.utils


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Prepare data for T0")
    parser.add_argument("--df-loc", type=str, required=True, help="Path to the df file")
    parser.add_argument("--relations-text-loc", type=str, required=True, help="Path to the folder with the documents")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save Hugginface Dataset object")

    parser.add_argument("--n_prior_sentences", type=int, default=0, help="Number of sentences before the trigger")
    parser.add_argument("--n_future_sentences", type=int, default=0, help="Number of sentences after the trigger")
    parser.add_argument("--split_questions", action="store_true", help="Split questions into multiple examples")

    return parser.parse_args(args)


def extract_sentence_with_trigger(
    doc_text, trigger_span, expected_trigger_text=None, n_prior_sentences=0, n_future_sentences=0, doc_id=None
):
    """
    Extracts the sentence with the trigger.

    If expected_trigger_text is not None,
        checks if the trigger is the expected_trigger_text and tries to find it in the neighboring sentences (plus minus 20 sentences).
        If the trigger is not found, raises RuntimeError.

    Args:
        n_future_sentences: Number of sentences to extract after the trigger.
        n_prior_sentences: Number of sentences to extract before the trigger.
    """
    doc_sentences = doc_text.splitlines(keepends=True)
    # doc_sentences = [t for t in doc_text.split("\n") if len(t) > 0]
    sentence_lengths = [len(s) for s in doc_sentences]

    current_index = 0
    sentence_index = None
    for i, l in enumerate(sentence_lengths):
        if trigger_span[0] <= current_index + l:
            sentence_index = i
            break
        current_index += l

    if sentence_index is None:
        raise Exception("Could not find sentence with trigger by span")

    if expected_trigger_text is not None:
        if expected_trigger_text not in doc_sentences[sentence_index]:
            # Try to find the trigger in the neighboring sentences
            for i in range(1, 20):
                if sentence_index - i >= 0 and expected_trigger_text in doc_sentences[sentence_index - i]:
                    sentence_index = sentence_index - i
                    break
                if (
                    sentence_index + i < len(doc_sentences)
                    and expected_trigger_text in doc_sentences[sentence_index + i]
                ):
                    sentence_index = sentence_index + i
                    break
            else:
                raise RuntimeError(
                    f"Could not find expected trigger text {expected_trigger_text} in the document with name {doc_id}"
                )

    sentence_index_start = max(0, sentence_index - n_prior_sentences)
    sentence_index_end = min(len(doc_sentences), sentence_index + n_future_sentences + 1)
    sentence = " ".join(doc_sentences[sentence_index_start:sentence_index_end])
    return sentence


def build_dataset(
    df_loc,
    relations_text_loc,
    n_prior_sentences=0,
    n_future_sentences=0,
    split_questions=False,
    verbosity=1,
    val_size=0.10,
    test_size=0.20,
    seed=0,
):

    df = pd.read_csv(df_loc)

    tr_ids, val_ids, test_ids = src.utils.split_ids(
        list(set(df.id.values)), seed=seed, val_split=val_size, test_split=test_size
    )

    # Create each df
    tr_df = df[df.id.isin(tr_ids)]
    vl_df = df[df.id.isin(val_ids)]
    te_df = df[df.id.isin(test_ids)]

    dataset_dict = {}

    n_errors = 0
    for split_name, split_df in zip(["train", "validation", "test"], [tr_df, vl_df, te_df]):
        dataset = []

        for _, row in tqdm(split_df.iterrows(), total=len(df)):
            with open(os.path.join(relations_text_loc, row.id), "r") as f:
                doc_text = f.read()

            shift = len(row.id) - 1
            span = row.start_index - shift, row.end_index - shift

            try:
                sentence = extract_sentence_with_trigger(
                    doc_text=doc_text,
                    trigger_span=span,
                    n_prior_sentences=n_prior_sentences,
                    n_future_sentences=n_future_sentences,
                    expected_trigger_text=row.reasoning,
                    doc_id=row.id,
                )
            except RuntimeError as e:
                n_errors += 1
                if n_errors < 3 or verbosity > 2:
                    if verbosity:
                        print(e)
                elif n_errors == 3:
                    if verbosity > 0:
                        print("Too many errors, not printing any more")
                continue

            if split_questions:
                questions = [q.strip() + "?" for q in row.question.split("?") if len(q) > 1]
                if len(questions) > 1:
                    if verbosity > 1:
                        print("Multiple questions found:", questions)
                for question in questions:
                    dataset.append({"text": sentence, "trigger": row.reasoning, "question": question})
            else:
                dataset.append({"text": sentence, "trigger": row.reasoning, "question": row.question})

        if n_errors > 0:
            if verbosity > 0:
                print(f"Found {n_errors} errors. These examples were not added to the dataset")

        dataset = pd.DataFrame(dataset)
        dataset = datasets.Dataset.from_pandas(dataset)

        dataset_dict[split_name] = dataset

    return datasets.DatasetDict(dataset_dict)


if __name__ == "__main__":
    args = parse_args()
    dataset = build_dataset(
        args.df_loc,
        args.relations_text_loc,
        n_prior_sentences=args.n_prior_sentences,
        n_future_sentences=args.n_future_sentences,
        split_questions=args.split_questions,
        verbosity=1,
        val_size=0.10,
        test_size=0.20,
        seed=0,
    )

    dataset.save_to_disk(args.output_dir)
