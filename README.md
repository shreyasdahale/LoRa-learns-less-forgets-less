# LoRa Learns Less Forgets Less

## Overview

This project investigates the hypothesis **"LoRA learns less, forgets less"** by comparing Low-Rank Adaptation (LoRA) with full fine-tuning and baseline models. We focus on evaluating the performance of fine-tuned BERT and RoBERTa models across a variety of natural language understanding tasks.

## Motivation

Fine-tuning large language models often results in significant forgetting of previously learned tasks when adapted to new tasks. LoRA offers a parameter-efficient alternative that can mitigate catastrophic forgetting while maintaining competitive performance. This project evaluates whether LoRA can indeed learn efficiently and forget less compared to traditional fine-tuning methods.

## Related Work

In May 2024, Biderman et al. published a paper titled "LoRA Learns Less and Forgets Less," which explores the performance of LoRA in fine-tuning large language models. The study compares LoRA with full fine-tuning across domains like programming and mathematics, highlighting LoRA's ability to maintain the base model's performance on tasks outside the target domain. The paper is available on arXiv: [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673).

## Experiment Setup

- **Baseline models:**
  - BERT
  - RoBERTa

- **Training Approaches:**
  - **LoRA Fine-Tuning:** Utilizing low-rank adaptation to update fewer parameters.
  - **Full Fine-Tuning:** Updating all model parameters.
  - **Baseline:** Original pre-trained models without additional fine-tuning.

- **Benchmark:**
  - **GLUE Benchmark Tasks Used:** 
    - MNLI (Multi-Genre Natural Language Inference)
    - MRPC (Microsoft Research Paraphrase Corpus)
    - SST-2 (Stanford Sentiment Treebank)
    - QQP (Quora Question Pairs)
    - CoLA (Corpus of Linguistic Acceptability)
    - RTE (Recognizing Textual Entailment)

## Methodology

1. **Individual Task Fine-Tuning:**
   - Each model (BERT and RoBERTa) was fine-tuned on 6 of the GLUE tasks using both LoRA and full fine-tuning strategies.

2. **Evaluation:**
   - Post fine-tuning, the models were evaluated on all 6 GLUE tasks.
   - Metrics focused on learning efficiency (performance gains) and task retention (the extent of forgetting).

3. **Comparison:**
   - A comparative analysis was performed to understand the trade-offs between reducing the number of trainable parameters (via LoRA) and maintaining task performance.

## Links to the Fine-Tuned Models (Uploaded on Hugging Face)

Here are the fine-tuned models available from [dog-in-the-box-studio](https://huggingface.co/dog-in-the-box-studio):

1. **qpp-bert-lora**: [https://huggingface.co/dog-in-the-box-studio/qpp-bert-lora](https://huggingface.co/dog-in-the-box-studio/qpp-bert-lora)
2. **sst2-bert-full**: [https://huggingface.co/dog-in-the-box-studio/sst2-bert-full](https://huggingface.co/dog-in-the-box-studio/sst2-bert-full)
3. **sst2-roberta-full**: [https://huggingface.co/dog-in-the-box-studio/sst2-roberta-full](https://huggingface.co/dog-in-the-box-studio/sst2-roberta-full)
4. **qpp-roberta-lora**: [https://huggingface.co/dog-in-the-box-studio/qpp-roberta-lora](https://huggingface.co/dog-in-the-box-studio/qpp-roberta-lora)
5. **sst2-roberta-lora**: [https://huggingface.co/dog-in-the-box-studio/sst2-roberta-lora](https://huggingface.co/dog-in-the-box-studio/sst2-roberta-lora)
6. **sst2-bert-lora**: [https://huggingface.co/dog-in-the-box-studio/sst2-bert-lora](https://huggingface.co/dog-in-the-box-studio/sst2-bert-lora)
7. **qqp-roberta-full**: [https://huggingface.co/dog-in-the-box-studio/qqp-roberta-full](https://huggingface.co/dog-in-the-box-studio/qqp-roberta-full)
8. **qqp-bert-full**: [https://huggingface.co/dog-in-the-box-studio/qqp-bert-full](https://huggingface.co/dog-in-the-box-studio/qqp-bert-full)
9. **cola-roberta-lora**: [https://huggingface.co/dog-in-the-box-studio/cola-roberta-lora](https://huggingface.co/dog-in-the-box-studio/cola-roberta-lora)
10. **cola-roberta-full**: [https://huggingface.co/dog-in-the-box-studio/cola-roberta-full](https://huggingface.co/dog-in-the-box-studio/cola-roberta-full)
11. **mrpc-roberta-lora**: [https://huggingface.co/dog-in-the-box-studio/mrpc-roberta-lora](https://huggingface.co/dog-in-the-box-studio/mrpc-roberta-lora)
12. **mrpc-roberta-full**: [https://huggingface.co/dog-in-the-box-studio/mrpc-roberta-full](https://huggingface.co/dog-in-the-box-studio/mrpc-roberta-full)
13. **cola-bert-lora**: [https://huggingface.co/dog-in-the-box-studio/cola-bert-lora](https://huggingface.co/dog-in-the-box-studio/cola-bert-lora)
14. **cola-bert-full**: [https://huggingface.co/dog-in-the-box-studio/cola-bert-full](https://huggingface.co/dog-in-the-box-studio/cola-bert-full)
15. **mrpc-bert-lora**: [https://huggingface.co/dog-in-the-box-studio/mrpc-bert-lora](https://huggingface.co/dog-in-the-box-studio/mrpc-bert-lora)
16. **mrpc-bert-full**: [https://huggingface.co/dog-in-the-box-studio/mrpc-bert-full](https://huggingface.co/dog-in-the-box-studio/mrpc-bert-full)

## Results

- **Key Findings:**
  - **LoRA's Effectiveness:** The LoRA method demonstrated a reduction in forgetting compared to full fine-tuning.
  - **Competitive Performance:** Despite using fewer trainable parameters, LoRA maintained competitive performance on most GLUE tasks.
  - **Learning Efficiency:** The efficiency of parameter updates in LoRA supports the hypothesis that learning less (i.e., updating fewer parameters) can lead to less forgetting.

For the results, please refer to the [Results Notebook](./results/Results_Notebook.ipynb).

### Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- [Datasets (GLUE Benchmark)](https://huggingface.co/datasets/nyu-mll/glue)

### Installation

**Clone the repository:**

   ```bash
   git clone https://github.com/shreyasdahale/LoRa-learns-less-forgets-less/
   cd LoRa-Learns-Less-Forgets-Less
   ```

### Running the Experiments

1. **Download GLUE Data:**

   The GLUE dataset is available through the Hugging Face Datasets library. You can load the dataset using the following code for example:

   ```python
   from datasets import load_dataset
   dataset = load_dataset('glue', 'mrpc', split='train')
   ```

   For more details, visit the [GLUE dataset page on Hugging Face](https://huggingface.co/datasets/nyu-mll/glue).

2. **Fine-Tune Models:**

   Fine-tuning the models using the training scripts

3. **Evaluate:**

   After training, evaluate the models using the eval scripts


## Contributions

Please feel free to contribute!

---
