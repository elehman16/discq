import os
import random
import numpy as np
import pandas as pd
from nltk import tokenize


def read_file(f_name: str, dir_path: str):
    """Simple file reader to extract offsets.
    @param f_name is the file name.
    @param dir_path is the path to the `f_name`.

    Returns:
        text: The text of the file, sentences are separated by newlines.
        remapper: A dict mapping the offsets to the new offsets.
    """
    with open(os.path.join(dir_path, f_name), encoding="utf-8-sig") as tmp:
        info = {"title": f_name, "context": tmp.read()}

    title = info["title"]
    text = tokenize.sent_tokenize(info["context"])

    # O.G. offsets do not have spaces in between sentences -> fix this
    text_seen, spaces_added = 0, 0
    remapper = {}
    for idx, sent in enumerate(text):
        for i in range(text_seen, text_seen + len(sent)):
            remapper[i] = i + spaces_added

        spaces_added += 1
        text_seen += len(sent)

    return (title + " ".join(text)).replace("\n", " "), remapper


def split_ids(ids: list, seed: int = 0, val_split: float = 0.10, test_split: float = 0.20):
    """Splits the ids based on the seed and split percents.
    @param ids is a list of document ids (str).
    @param seed is the random seed to use.
    @param val_split is the percent of data to use for validation.
    @param test_split is the percent of data to use for the test set.
    @return train_ids, val_ids, test_ids."""
    random.seed(seed)
    random.shuffle(ids)

    # Train, Val, Test IDs
    train_ids = ids[: int(len(ids) - len(ids) * (test_split + val_split))]
    train_split = int(len(ids) - len(ids) * (test_split + val_split))
    val_ids = ids[train_split : int(train_split + len(ids) * val_split)]
    test_ids = ids[int(train_split + len(ids) * val_split) :]
    return train_ids, val_ids, test_ids


def convert_preds_to_triggers(tokenizer, ids: list, input_ids: list, preds: list):
    """Convert token level predictions to continuous strings.
    @param tokenizer is a HF tokenizer.
    @param ids is a list of strings.
    @param input_ids is a list of tensors containing indicies that each corresponds
    to a discharge summary.
    @param preds is a list of tensors containing binary predictions in which a
    prediction of 1 indicates that the model has predicted that that word is a
    trigger.
    @return a DF of ids, triggers, start/end offsets, and the text."""
    all_ids, all_text, start_indices, end_indices, all_trigger_groups = [], [], [], [], []
    predicted_triggers = np.asarray(input_ids) * np.asarray(preds)

    # For each document, find where there are triggers, and convert to text
    for idx, pt in enumerate(predicted_triggers):
        start_index = [i + 1 for i, x in enumerate(pt) if (i + 1 != len(pt) and x == 0 and pt[i + 1] != 0)]
        end_index = [i + 1 for i, x in enumerate(pt) if (i + 1 != len(pt) and x != 0 and pt[i + 1] == 0)][1:]

        # Use the start + end indexes to get the triggers
        for start, end in zip(start_index, end_index):
            text = tokenizer.convert_tokens_to_string(tokenizer.batch_decode(pt[start:end]))
            before_text = tokenizer.decode(input_ids[idx][:start], skip_special_tokens=True)

            all_trigger_groups.append(text)
            start_indices.append(len(before_text))
            end_indices.append(len(before_text) + len(text))

            all_ids.append(ids[idx])
            all_text.append(tokenizer.decode(input_ids[idx], skip_special_tokens=True))

    return pd.DataFrame(
        {"id": all_ids, "trigger": all_trigger_groups, "st_ch": start_indices, "end_ch": end_indices, "text": all_text}
    )
