# Learning to Ask Like a Physician
https://arxiv.org/abs/2206.02696

The code accompanying the paper Learning to Ask Like a Physician and Discharge Summary Clinical Questions (DiSCQ) dataset.

![3_14_22_pipeline_single_line](https://user-images.githubusercontent.com/2821124/179035880-add52a01-b153-496c-9132-a983fb71aec2.png)

The data has been released here: https://physionet.org/content/discq/1.0/
You will still need to get n2c2/i2b2 access to the discharge summaries.

# Citation

```
@inproceedings{lehman-etal-2022-learning,
    title = "Learning to Ask Like a Physician",
    author = "Lehman, Eric  and
      Lialin, Vladislav  and
      Legaspi, Katelyn Edelwina  and
      Sy, Anne Janelle  and
      Pile, Patricia Therese  and
      Alberto, Nicole Rose  and
      Ragasa, Richard Raymund  and
      Puyat, Corinna Victoria  and
      Tali{\~n}o, Marianne Katharina  and
      Alberto, Isabelle Rose  and
      Alfonso, Pia Gabrielle  and
      Moukheiber, Dana  and
      Wallace, Byron  and
      Rumshisky, Anna  and
      Liang, Jennifer  and
      Raghavan, Preethi  and
      Celi, Leo Anthony  and
      Szolovits, Peter",
    booktitle = "Proceedings of the 4th Clinical Natural Language Processing Workshop",
    month = jul,
    year = "2022",
    address = "Seattle, WA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.clinicalnlp-1.8",
    pages = "74--86",
}
```

## Setup

```
pip install -r requirements.txt
```

It will download all required libraries and scipacy models.

Add this directory to your PYTHONPATH

```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Download data

DISCQ dataset is available here: [physionet.org/content/discq/1.0](https://physionet.org/content/discq/1.0/)

Our dataset relates to the discharge summaries from the i2b2 2010 challenge. Specifically, you need to download `Training Data: Concept assertion relation training data` and `Test Data: Test data` from [2010 Relations Challenge Downloads](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/#collapse3).

Then combine files from `concept_assertion_relation_training_data/beth/txt` and `test_data` and place all these `.txt` files to a directory `data/relations_txt`.

## Identifying Triggers 
In our paper, we use two different ways of running extraction: (1) spaCy and (2) a ClinicalBERT model.

To run the spaCy model:
```bash
python src/spacy_pred_triggers.py \
    --df-loc "data/discq_questions_final.csv" \
    --relations-text-loc "data/relations_txt/" \
    --output output_dir/spacy_triggers.json
```

To run the ClinicalBERT model:
```bash
python src/find_triggers_question_agnostic.py \
    --train \
    --test \
    --df-loc "data/discq_questions_final.csv" \
    --relations-text-loc "data/relations_txt/" \
    --output output_dir/bart_triggers_question_agnostic \
    --output-df data/predicted_triggers.csv
```

## BART approach
To run the BART portion of the code, use the script `src/bart_qg/generate_questions_bart_prompted.py`. Usage example:
```bash
python src/bart_qg/generate_questions_bart_prompted.py \
    --train \
    --test \
    --df-loc "data/discq_questions_final.csv" \
    --relations-text-loc "data/relations_txt/" \
    --chunk-mode "none" \
    --split-questions \
    --final-test-file data/predicted_triggers.csv  \
    --output model_outputs/bart_triggers 
```

To change how the model preprocess questions or how we split up text, see the documentation of arguments given in `src/bart_qg/generate_questions_bart_prompted.py`.

## T0 approch

First, you need to preprocess the data using `prepare_data_for_t0.py`. Usage example:

```bash
python src/t0_code/prepare_data_for_t0.py \
    --df-loc "data/discq_questions_final.csv" \
    --relations-text-loc "data/relations_txt/" \
    --output_dir "data/clinical_qg_one_sentence" \
    --split_questions
```

Although still an experimental feature of 🤗 Transformers, the simplest way to train T0 is to pass the --parallelize flag as shown in the example above, which calls model.parallize() and splits the model over all visible GPUs.

To train T0 3B (`bigscience/T0_3B`), you need at least around 48GB of GPU memory in theory, which in practice usually means at least two V100s (32GB version), three RTX3090s, or a single A6000. For T0 11B, you need at least eight V100s. (If you don't need training and only need inferencing, then the VRAM requirement is about 1/4 of training, i.e., a single 3090 for T0 3B, or a single A6000 for T0 11B.)

Recommended CPU memory > 90Gb.

Before starting training, configure Accelerate via `accelerate config`.

Training script usage example:

```bash
export WANDB_START_METHOD="thread"
export TOKENIZERS_PARALLELISM=false

python src/t0_code/fine_tune_t0.py\
    --dataset_name_or_path "data/clinical_qg_3_prior_2_future_split_questions" \
    --prompt_name "after_reading_what_question" \
    --model_name_or_path "bigscience/T0pp" \
    --max_length 512 \
    --target_max_length 184 \
    --output_dir "output_dir/T0pp_split_questions_512" \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --parallelize \
    --wandb_proj "clinical-qg-T0" \
    --num_train_epochs 30 \
    --learning_rate 3e-4 \
    --num_warmup_steps 100 \

```

## Debugging

```bash
python src/t0_code/fine_tune_t0.py\
    --dataset_name_or_path "data/clinical_qg_chunk_512_Mar3_debug" \
    --prompt_name "after_reading_what_question" \
    --model_name_or_path "t5-small" \
    --max_length 512 \
    --target_max_length 184 \
    --output_dir "output_dir/debug" \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 3e-4 \
    --num_warmup_steps 10 \
```

## Predict on the test set using a trained T0 model

First, preprocess data like in `notebooks/04_prepare_predicted_triggers_forT0.ipynb`

```bash
python src/t0_code/fine_tune_t0.py \
    --predict_only \
    --dataset_name_or_path "data/predicted_triggers_200_chunked" \
    --prompt_name "after_reading_what_question" \
    --model_name_or_path "output_dir/T0pp_chunk_512_Mar3" \
    --tokenizer_name "bigscience/T0pp" \
    --max_length 512 \
    --target_max_length 184 \
    --output_dir "output_dir/T0pp_chunk_512_Mar3_predited_triggers" \
    --per_device_batch_size 2 \
    --parallelize \
    --wandb_proj "clinical-qg-T0"
```

## QA Portion
We defer to the following repository to run the ClinicalBERT models to run the QA portion of the code: https://github.com/xiangyue9607/CliniRC.
