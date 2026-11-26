---
license: apache-2.0
base_model: TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
tags:
- generated_from_trainer
datasets:
- wikitext
metrics:
- accuracy
model-index:
- name: wikitext2-TinyLlama-1.1B-adahyc-q2-fxT-fbT-zpT-pplan-t0.1-s11
  results:
  - task:
      name: Causal Language Modeling
      type: text-generation
    dataset:
      name: wikitext wikitext-2-raw-v1
      type: wikitext
      args: wikitext-2-raw-v1
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.5636519227321772
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wikitext2-TinyLlama-1.1B-adahyc-q2-fxT-fbT-zpT-pplan-t0.1-s11

This model is a fine-tuned version of [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) on the wikitext wikitext-2-raw-v1 dataset.
It achieves the following results on the evaluation set:
- Loss: 1.9931
- Accuracy: 0.5637

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 4
- eval_batch_size: 4
- seed: 11
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.41.0
- Pytorch 2.3.0
- Datasets 2.16.1
- Tokenizers 0.19.1
