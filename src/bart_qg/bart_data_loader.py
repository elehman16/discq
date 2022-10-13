import copy
import torch
import spacy
import numpy as np
import pandas as pd
from src.utils import read_file, split_ids

nlp = spacy.load("en_core_sci_md")


class ClassificationDataset(torch.utils.data.Dataset):
    """Holder object for the following items:
    @param input_ids... see HF documention.
    @param attention_mask... see HF documentation.
    @param questions are the questions to generate.
    @param question_attention_mask are the attention_masks for the questions
    to generate.
    @param ids is a list of strings that correspond to the ids of the questions
    @param triggers is a separate list of triggers."""

    def __init__(self, input_ids, attention_mask, questions, question_attn_mask, ids, triggers=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.questions = questions
        self.question_attn_mask = question_attn_mask
        self.ids = ids
        self.triggers = triggers

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.questions[idx],
            "decoder_attention_mask": self.question_attn_mask[idx],
            "id": self.ids[idx],
        }

        return item

    def __len__(self):
        return len(self.input_ids)


class BARTDataLoader:
    """Data loader class for BART model.
    @param data_loc is the file location of the dataframe.
    @param relations_text_loc is the folder location of the n2c2 discharge summaries.
    @param tokenizer is the Huggingface tokenizer.
    @param max_len is the maximum length of a sequence from the passage.
    @param max_q_len is the maximum question length.
    @param chunk_mode will determine how we split up discharge summaries.
    @param split_questions if true, will take a sequence of questions for the
    same trigger and split it up into multiple labels.
    @param trigger_aware if true, will append the trigger to the beginning of
    the inputs to the model.
    @param stride is the typical stride when examining LMs.
    @param val_split is the validation split percent
    @param test_split is the test split percent.
    @param seed is the seed to split the data with."""

    def __init__(
        self,
        data_loc: str,
        relations_text_loc: str,
        tokenizer,
        debug_mode: bool = False,
        max_len: int = 512,
        max_q_len: int = 184,
        max_trigger_len: int = 184,
        chunk_mode: str = "sentence-level",
        split_questions: bool = False,
        stride: int = 512,
        val_split: int = 0.10,
        test_split: int = 0.20,
        seed: int = 0,
    ):

        self.data_loc = data_loc
        self.relations_text_loc = relations_text_loc
        self.tokenizer = tokenizer
        self.debug_mode = debug_mode
        self.max_len = max_len
        self.max_q_len = max_q_len
        self.chunk_mode = chunk_mode
        self.split_questions = split_questions
        self.max_trigger_len = max_trigger_len
        self.stride = stride
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

    def _split_data_into_sentences_more_context(self, all_tokenized_outputs):
        """Split the data into sentences and tokenize + pad it.
        @param all_tokenized_outputs is a list of dictionaries with the following input:
            @param input_ids is the input ids of the discharge summary.
            @param trigger_location is a list of 1s and 0s into @param input_ids.
            @param trigger_ids is a list of ids of the text for the trigger_ids.
            @param id is the discharge summary id.
            @param question is ids of the question asked (padded already).
            @param question_attn_mask is the attention mask of the @param question.
            @param sentence_offsets is the offset of the start of each sentence.

        @return a ClassificationDataset instance."""
        xs, questions, question_attn_mask, attn_mask, ids, triggers = [], [], [], [], [], []
        prompt = self.tokenizer.encode(
            " After reading the above EMR, what question do you have about ", add_special_tokens=False
        )
        end_prompt = self.tokenizer.encode("? Question: ", add_special_tokens=False)
        for idx, row in enumerate(all_tokenized_outputs):
            tg_st = row["trigger_location"].index(1)
            tg_end = tg_st + sum(row["trigger_location"])
            sent_onset = np.where(np.asarray(row["sentence_offsets"]) > tg_st)[0][0] - 1
            sent_end = np.where(np.asarray(row["sentence_offsets"]) > tg_end)[0][0]

            tk_onset = row["sentence_offsets"][sent_onset]
            tk_end = row["sentence_offsets"][sent_end]
            selected_sent = row["input_ids"][tk_onset:tk_end][: self.max_len]
            pad = (self.max_len + self.max_trigger_len + len(prompt) + len(end_prompt)) - (
                len(selected_sent) + len(row["trigger_ids"])
            )
            input_ = [self.tokenizer.bos_token_id] + selected_sent
            input_ += prompt + row["trigger_ids"][: self.max_trigger_len] + end_prompt + [self.tokenizer.eos_token_id]
            pad_input = input_ + [self.tokenizer.pad_token_id] * pad

            # Add input_ids, input attn mask, trigger locations, question_ids,
            # question attn map, and the article ID.
            xs.append(pad_input)
            attn_mask.append([1] * len(input_) + [0] * pad)
            questions.append(row["question"][: self.max_q_len])
            question_attn_mask.append(row["question_attn_mask"])
            ids.append(row["id"])
            triggers.append(row["trigger_str"])

        return ClassificationDataset(xs, attn_mask, questions, question_attn_mask, ids, triggers=triggers)

    def _split_data_into_strides(self, all_tokenized_outputs):
        """Split the data into @param stride sized chunks and tokenize + pad it.
        @param all_tokenized_outputs is a list of dictionaries with the following input:
            @param input_ids is the input ids of the discharge summary.
            @param trigger_location is a list of 1s and 0s into @param input_ids.
            @param trigger_ids is a list of ids of the text for the trigger_ids.
            @param id is the discharge summary id.
            @param question is ids of the question asked (padded already).
            @param question_attn_mask is the attention mask of the @param question.
            @param sentence_offsets is the offset of the start of each sentence.

        @return a ClassificationDataset instance."""
        # Now split up data to max_len + stride.
        xs, questions, question_attn_mask, attn_mask, ids, triggers = [], [], [], [], [], []
        prompt = self.tokenizer.encode(
            " After reading the above EMR, what question do you have about ", add_special_tokens=False
        )
        end_prompt = self.tokenizer.encode("? Question: ", add_special_tokens=False)
        for idx, row in enumerate(all_tokenized_outputs):
            for i in range(0, len(row["input_ids"]), self.stride):
                pad = (self.max_len + self.max_trigger_len + len(prompt) + len(end_prompt)) - (
                    len(row["input_ids"][i : i + self.max_len - 2]) + len(row["trigger_ids"])
                )
                input_ = [self.tokenizer.bos_token_id] + row["input_ids"][i : i + self.max_len - 2]
                input_ = input_ + prompt + row["trigger_ids"] + end_prompt + [self.tokenizer.eos_token_id]
                pad_input = input_ + [self.tokenizer.pad_token_id] * pad

                # If all the labels are 0, there are no questions here! Discard.
                trig = row["trigger_location"][i : i + len(input_)]
                if sum(trig) == 0:
                    continue

                # Add input_ids, input attn mask, trigger locations, question_ids,
                # question attn map, and the article ID.
                xs.append(pad_input)
                attn_mask.append([1] * len(input_) + [0] * pad)
                questions.append(row["question"])
                question_attn_mask.append(row["question_attn_mask"])
                ids.append(row["id"])
                triggers.append(row["trigger_str"])

        return ClassificationDataset(xs, attn_mask, questions, question_attn_mask, ids, triggers=triggers)

    def _split_data_into_no_inputs(self, all_tokenized_outputs):
        """Split the data into @param stride sized chunks and tokenize + pad it.
        @param all_tokenized_outputs is a list of dictionaries with the following input:
            @param input_ids is the input ids of the discharge summary.
            @param trigger_location is a list of 1s and 0s into @param input_ids.
            @param trigger_ids is a list of ids of the text for the trigger_ids.
            @param id is the discharge summary id.
            @param question is ids of the question asked (padded already).
            @param question_attn_mask is the attention mask of the @param question.
            @param sentence_offsets is the offset of the start of each sentence.

        @return a ClassificationDataset instance."""
        to_pad_length = max([len(x["trigger_ids"]) for x in all_tokenized_outputs])
        xs, questions, question_attn_mask, attn_mask, ids, triggers = [], [], [], [], [], []
        for idx, row in enumerate(all_tokenized_outputs):
            pad_tokens = to_pad_length - len(row["trigger_ids"])
            xs.append(row["trigger_ids"] + [self.tokenizer.pad_token_id] * pad_tokens)
            attn_mask.append([1] * len(row["trigger_ids"]) + [0] * pad_tokens)
            questions.append(row["question"])
            question_attn_mask.append(row["question_attn_mask"])
            ids.append(row["id"])
            triggers.append(row["trigger_str"])

        return ClassificationDataset(xs, attn_mask, questions, question_attn_mask, ids, triggers)

    def _tokenize_df(self, df, is_training):
        """Turn a dataframe into tokenized + padded inputs + labels.
        @param df is the dataframe to process.
        @param is_training is if the dataframe output is for training or evaluating."""
        all_tokenized_outputs = []
        for index, row in df.iterrows():
            text, remapper = read_file(row.id, self.relations_text_loc)
            tokenized_output = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

            # Find the offset
            labels = [0] * len(tokenized_output.input_ids)
            new_st, new_end = remapper[row.start_index], remapper[row.end_index]
            tok_st = np.where(np.asarray(tokenized_output.offset_mapping) > new_st)[0][0]
            tok_end = np.where(np.asarray(tokenized_output.offset_mapping) > new_end)[0][0]
            for i in range(tok_st, tok_end + 1):
                labels[i] = 1

            # Add to our map
            tk_q = self.tokenizer(row.question, max_length=self.max_q_len, padding="max_length")
            tk_tr = self.tokenizer(
                row.reasoning, max_length=self.max_trigger_len, truncation=True, add_special_tokens=False
            )
            sentence_ch_offsets = [x.end_char for x in nlp(text).sents]
            sentence_tk_offsets = [
                np.where(np.asarray(tokenized_output.offset_mapping) > x)[0][0] for x in sentence_ch_offsets[:-1]
            ]
            sentence_tk_offsets.append(len(tokenized_output.input_ids) - 1)

            outputs_ = {
                "input_ids": tokenized_output.input_ids,
                "trigger_location": labels,
                "trigger_ids": tk_tr["input_ids"],
                "id": row.id,
                "question": tk_q["input_ids"],
                "question_attn_mask": tk_q["attention_mask"],
                "sentence_offsets": sentence_tk_offsets,
                "trigger_str": row.reasoning,
            }

            if self.split_questions and is_training:
                tk_qs = [
                    self.tokenizer(q + "?", max_length=self.max_q_len, padding="max_length")
                    for q in row.question.split("?")[:-1]
                ]
                for q in tk_qs:
                    c_outputs_ = copy.deepcopy(outputs_)
                    c_outputs_["question"] = q["input_ids"]
                    c_outputs_["question_attn_mask"] = q["attention_mask"]
                    all_tokenized_outputs.append(c_outputs_)
            else:
                all_tokenized_outputs.append(outputs_)

        if self.chunk_mode == "sentence-level":
            return self._split_data_into_sentences_more_context(all_tokenized_outputs)
        elif self.chunk_mode == "split-level":
            return self._split_data_into_strides(all_tokenized_outputs)
        else:
            return self._split_data_into_no_inputs(all_tokenized_outputs)

    def tokenize_generated_df(self, df):
        """Turn a dataframe into tokenized + padded inputs + labels. Use this function
        for when trying to get predictions on dataset with predicted triggers.
        @param df is the dataframe to process. This DF must have a column named
        `text`."""
        all_tokenized_outputs = []
        for index, row in df.iterrows():
            text = row["text"]

            tokenized_output = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

            # Find the offset
            labels = [0] * len(tokenized_output.input_ids)
            new_st, new_end = row.st_ch, row.end_ch
            try:
                tok_st = np.where(np.asarray(tokenized_output.offset_mapping) > new_st)[0][0]
                tok_end = np.where(np.asarray(tokenized_output.offset_mapping) > new_end)[0][0]
            except:
                tok_end = len(tokenized_output.offset_mapping) - 1

            for i in range(tok_st, tok_end + 1):
                labels[i] = 1

            # Add to our map
            tk_tr = self.tokenizer(
                row.trigger, max_length=self.max_trigger_len, truncation=True, add_special_tokens=False
            )
            sentence_ch_offsets = [x.end_char for x in nlp(text).sents]
            sentence_tk_offsets = [
                np.where(np.asarray(tokenized_output.offset_mapping) > x)[0][0] for x in sentence_ch_offsets[:-1]
            ]
            sentence_tk_offsets.append(len(tokenized_output.input_ids) - 1)

            outputs_ = {
                "input_ids": tokenized_output.input_ids,
                "trigger_location": labels,
                "trigger_ids": tk_tr["input_ids"],
                "id": row.id,
                "question": [self.tokenizer.bos_token_id],
                "question_attn_mask": [1],
                "sentence_offsets": sentence_tk_offsets,
                "trigger_str": row.trigger,
            }

            all_tokenized_outputs.append(outputs_)

        if self.chunk_mode == "sentence-level":
            return self._split_data_into_sentences_more_context(all_tokenized_outputs)
        elif self.chunk_mode == "split-level":
            return self._split_data_into_strides(all_tokenized_outputs)
        else:
            return self._split_data_into_no_inputs(all_tokenized_outputs)

    def load_data(self) -> (ClassificationDataset, ClassificationDataset, ClassificationDataset):
        """Load the data and split it."""
        df = pd.read_csv(self.data_loc)
        ordered_ids = []
        seen = set()
        for id_ in list(df.id.values):
            if not (id_ in seen):
                seen.add(id_)
                ordered_ids.append(id_)

        tr_ids, val_ids, test_ids = split_ids(
            ordered_ids, seed=self.seed, val_split=self.val_split, test_split=self.test_split
        )

        # Create each df
        tr_df = df[df.id.isin(tr_ids)]
        vl_df = df[df.id.isin(val_ids)]
        te_df = df[df.id.isin(test_ids)]

        # If we are in debug mode, use a smaller percent of the data.
        if self.debug_mode:
            tr_df = tr_df[: int(len(tr_df) * 0.1)]
            vl_df = vl_df[: int(len(vl_df) * 0.1)]
            te_df = te_df[: int(len(te_df) * 0.1)]

        # Now tokenize each df
        tr_set = self._tokenize_df(tr_df, is_training=True)
        vl_set = self._tokenize_df(vl_df, is_training=False)
        te_set = self._tokenize_df(te_df, is_training=False)
        return tr_set, vl_set, te_set
