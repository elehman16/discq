import nltk
from nltk.translate.bleu_score import sentence_bleu
from datasets import load_metric
import numpy as np
import string, re

bleu = load_metric("sacrebleu")
rouge = load_metric("rouge")


def optimize_score(preds, labels, metric):
    """We assume that there is a single prediction and multiple possible labels.
    Optimize the metric.
    @param preds a list of strings.
    @param labels a list of wrapped strings (i.e. [N x 1])
    @param metric is a metric s.t. given preds + labels, output a float score.
    @return the optimized score of the metric.
    """
    add_q = lambda x: x + "?"
    labels = [list(map(add_q, x[0].split("?")[:-1])) for x in labels]

    optimized_labels = []
    for p, lol in zip(preds, labels):
        max_score, max_label = 0, ""
        for l in lol:
            score_ = metric([p], [l])
            if score_ > max_score:
                max_score = score_
                max_label = l

        optimized_labels.append(max_label)

    return metric(preds, optimized_labels)


def postprocess_text(preds, labels):
    """Simple postprocessing of predictions + labels."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds, tokenizer, split_questions=False):
    """Calculate BLEU/ROUGE/METEOR scores.
    @param eval_preds is tuple of preds, labels.
    @param tokenizer is a Huggingface tokenizer.
    @param split_questions is whether or not the questions are split or if
    the model will generate all of the questions at once."""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # labels are a list of strings
    if isinstance(labels, list) and isinstance(labels[0], str):
        decoded_labels = labels

    # labels are a list of lists of strings, multiple true answers
    elif isinstance(labels, list) and isinstance(labels[0], list) and isinstance(labels[0][0], str):
        decoded_labels = labels

    # Replace -100 in the labels as we can't decode them.
    else:
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    results = {}
    results["gen_len"] = np.mean(prediction_lens)

    #  If the questions are split, find the best possible match
    bleu_template = lambda weights: lambda c_, r_: sentence_bleu(
        [r.split() for r in r_], [c.split() for c in c_][0], weights=weights
    )
    bleu1_f = bleu_template((1.00, 0.00, 0.00, 0.00))
    bleu2_f = bleu_template((0.50, 0.50, 0.00, 0.00))
    bleu3_f = bleu_template((0.33, 0.33, 0.33, 0.33))
    bleu4_f = bleu_template((0.25, 0.25, 0.25, 0.25))

    # Diversity metrics
    number_of_unique_questions = len(set(decoded_preds))
    results["unique_questions_ratio"] = number_of_unique_questions / len(decoded_labels)

    # get the number of unique n-grams in preds
    for n in range(1, 4):
        unique_n_grams = set()
        n_n_ngrams_total = 0
        for pred in decoded_preds:
            n_grams = list(nltk.ngrams(nltk.tokenize.word_tokenize(pred), n))
            unique_n_grams.update(n_grams)
            n_n_ngrams_total += len(n_grams)
        try:
            results[f"unique_{n}_grams_ratio"] = len(unique_n_grams) / n_n_ngrams_total
        except:
            results[f"unique_{n}_grams_ratio"] = 0

    if split_questions:
        results["bleu1"] = optimize_score(decoded_preds, decoded_labels, bleu1_f)
        results["bleu2"] = optimize_score(decoded_preds, decoded_labels, bleu2_f)
        results["bleu3"] = optimize_score(decoded_preds, decoded_labels, bleu3_f)
        results["bleu4"] = optimize_score(decoded_preds, decoded_labels, bleu4_f)

        # Functions for meteor/rouge
        r = lambda p, l: rouge.compute(predictions=p, references=l)["rougeL"].mid.fmeasure
        results["rougeL"] = optimize_score(decoded_preds, decoded_labels, r)

    else:
        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        results = {"bleu": result["score"]}
        results["gen_len"] = np.mean(prediction_lens)
        results["rougeL"] = rouge_result["rougeL"].mid.fmeasure
        results["rouge1"] = rouge_result["rouge1"].mid.fmeasure
        results["rouge2"] = rouge_result["rouge2"].mid.fmeasure

    return results


# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    """Compute exact match but normalize the text first."""
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    """Compute token-f1 but normalize the text first."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2 * (prec * rec) / (prec + rec)


def get_gold_answers(example):
    """helper function that retrieves all possible true answers from a squad2.0 example"""

    gold_answers = [answer["text"] for answer in example.answers if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example -
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]

    return gold_answers
