# Modified from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

import copy
import json
import logging
import math
import re
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import peft
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from datasets import load_dataset
from accelerate.utils import set_seed

from models.llama.modeling_llama import LlamaForCausalLM
from models.mistral.modeling_mistral import MistralForCausalLM

os.environ["WANDB_PROJECT"] = "hyclora-gsm8k"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."
        },
    )
    lora_init: bool = field(
        default=False,
        metadata={
            "help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."
        },
    )
    full_precision: bool = field(
        default=False,
        metadata={
            "help": "False: Use bitsandbytes Linear4bit, real quantization"
            "True: Use quantization equivalent fp16/fp32 weights."
            "Note: Set True for data parallel training"
        },
    )
    rank: int = field(
        default=16,
        metadata={
            "help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."
        },
    )
    bits: int = field(
        default=4,
        metadata={
            "help": "Bit of the backbone. LoftQ does not require this config. Used for QLoRA."
        },
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )
    replace_module: bool = field(
        default=False,
        metadata={
            "help": "True: Use transform block; False: Do not use transform block"
        },
    ),
    gradient_checkpointing_enable: bool = field(
        default=False,
        metadata={
            "help": "True: Use gradient checkpointing; False: Do not use gradient checkpointing"
        },
    )
    flash_attention: bool = field(
        default=False,
        metadata={
            "help": "True: Use Flash Attention; False: Do not use Flash Attention"
        },
    )
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    init_lora_weights: str = field(
        default="qlora",
        metadata={
            "help": "init_lora_weights (`['gaussian', 'loftq', 'pissa', 'pissa_init']`):"
        },
    )


@dataclass
class DataArguments:
    data_name: str = field(default="gsm8k", metadata={"help": "Dataset name."})
    eval_batch_size: int = field(default=32, metadata={"help": "Evaluation batch size (generation; left padded)."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    is_train: bool = field(
        default=True, metadata={"help": "True: Train the model; False: Evaluate the model"}
    )
    is_eval: bool = field(
        default=True, metadata={"help": "True: Evaluate the model; False: Train the model"}
    )
    trained_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the trained adapter. Used in evaluation or resuming from the checkpoint."},
    )

# hyper parameters about hyclora method
@dataclass
class HyCLoRAArguments:
    use_hyclora: bool = field(
        default=True,
        metadata={"help": "Whether to replace the original training module to fused training module"}
    )
    layer_type: str = field(
        default="baseline",
        metadata={"help": "Different types of fused training layers (Available: [baseline | intra | intra_inter | intra_inter_full_fuse | adahyc])"}
    )
    iteration_threshold: int = field(
        default=5,
        metadata={"help": "calibration steps"}
    )
    softmax_outlier_ratio: float = field(
        default=0.05,
        metadata={"help": "softmax outlier selection ratio"}
    )
    layernorm_outlier_ratio: float = field(
        default=0.005,
        metadata={"help": "layernorm outlier channels selection ratio"}  
    )
    q_bit: int = field(
        default=4,
        metadata={"help": "quantization bit"}
    )
    fix_outlier_bug: bool = field(
        default=False,
        metadata={"help": "intra_inter(_full_fuse): restore x with its own outlier channels in backward (released code uses x_medium's)"}
    )
    use_zero_point: bool = field(
        default=False,
        metadata={"help": "intra_inter / adahyc: asymmetric quantization with a per-channel zero point"}
    )
    bit_plan_path: Optional[str] = field(
        default=None,
        metadata={"help": "adahyc: JSON {layer_idx: {buffer: bits}}, bits in {2,4,8}; missing entries use q_bit"}
    )
    rebuild_tau: float = field(
        default=0.1,
        metadata={"help": "adahyc: rebuild x_norm_1/x_medium/x_norm_2 iff rebuild error <= (1+tau) * store error; negative disables"}
    )
    hyc_fixes: bool = field(
        default=False,
        metadata={"help": "HyC-LoRA with all upstream bug fixes (RMSNorm backward, calibration merge, batch-wide stats, outlier index, zero-point)"}
    )
    

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    # sources are questions, and targets are answers
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [
            f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT)
            for example in raw_data
        ]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        #! a tricky way to pad the input_ids and labels to 16's multiple
        # 1. find the max length of input_ids
        max_len = max([len(input_id) for input_id in input_ids])
        # 2. pad the input_ids and labels to 32's multiple
        max_len = (max_len + 31) // 32 * 32
        # 3. generate a max_len tensor
        max_len_tensor = torch.randn(max_len).to(torch.int64)
        # 4. append the max_len tensor to the input_ids and labels
        input_ids.append(max_len_tensor)
        labels.append(max_len_tensor)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # delete the max_len tensor
        input_ids = input_ids[:-1]
        labels = labels[:-1]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    train_set = dataset["train"]
    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    print("train_dataset: ", train_dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )

def train_and_eval():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, HyCLoRAArguments)
    )
    model_args, data_args, training_args, hyclora_args = parser.parse_args_into_dataclasses()
    print(f"[config] {hyclora_args}")
    set_seed(training_args.seed)  # before LoRA creation, so the adapter init depends on the seed

    if 'llama' in model_args.model_name_or_path.lower():
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            use_cache=False if model_args.gradient_checkpointing_enable else True,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            attn_implementation=(
                "flash_attention_2" if model_args.flash_attention else "eager"
            ),
        )
        model.set_fused_llama_layer(hyclora_args)
    elif 'mistral' in model_args.model_name_or_path.lower():
        model = MistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            use_cache=False if model_args.gradient_checkpointing_enable else True,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            attn_implementation=(
                "flash_attention_2" if model_args.flash_attention else "eager"
            ),
        )
        model.set_fused_mistral_layer(hyclora_args)
    else:
        raise ValueError("No implementation of model. Now support: llama/mistral")
    
    model = peft.prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=model_args.gradient_checkpointing_enable,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if model_args.gradient_checkpointing_enable:
        print("Gradient Checkpointing is enabled")

    ###################### training part ######################
    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj", "down_proj", "o_proj"]
        if model_args.rank != model_args.lora_alpha:
            raise ValueError(f"Now the hyclora code does not support {model_args.rank} != {model_args.lora_alpha}. ")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            init_lora_weights=True if model_args.init_lora_weights == "qlora" else model_args.init_lora_weights,
        )
        model = get_peft_model(model, lora_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=True,
            token=model_args.token,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            subfolder="loftq_init",
            is_trainable=True,
            token=model_args.token,
        )

    # get the model name
    for name, module in model.named_modules():
        module.name = name

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # flat layout: everything of a run goes directly into output_dir (the nested expt/model/ep/lr/seed
    # path exceeded Windows MAX_PATH when copied back)
    out_dir = training_args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    model = model.to(torch.bfloat16)

    if training_args.is_train:
        class _Bf16OnResume(transformers.TrainerCallback):
            # resume_from_checkpoint -> peft load_adapter(autocast_adapter_dtype=True) turns the LoRA weights into fp32,
            # the fused kernels need bf16. on_train_begin runs after the checkpoint (and optimizer state) is loaded;
            # .to() casts in place, so the optimizer keeps the same Parameter objects.
            # The optimizer state was cast to fp32 too (load_state_dict follows the param dtype): cast the
            # per-parameter moments back, but not the scalar `step` counter (bf16 cannot count past 256).
            def on_train_begin(self, args, state, control, model=None, optimizer=None, **kw):
                model.to(torch.bfloat16)
                for p, st in optimizer.state.items():
                    for k, v in st.items():
                        if torch.is_tensor(v) and v.dim() > 0 and v.shape == p.shape:
                            st[k] = v.to(p.dtype)
        trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=[_Bf16OnResume()], **data_module)
        # resume only when checkpointing is on (SAVE_STEPS), so stale checkpoint dirs are never picked up silently
        last_ckpt = get_last_checkpoint(out_dir) if training_args.save_strategy != "no" else None
        if last_ckpt is not None:
            print(f"[resume] from {last_ckpt}")
            if hyclora_args.use_hyclora:
                print(f"[resume] recalibrating HyC statistics during the first {hyclora_args.iteration_threshold} steps")
        torch.cuda.reset_peak_memory_stats()
        train_result = trainer.train(resume_from_checkpoint=last_ckpt)
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak memory allocated: {peak_gib:.3f} GiB | train_runtime: {train_result.metrics['train_runtime']:.1f} s | "
              f"train_samples_per_second: {train_result.metrics['train_samples_per_second']:.3f}")
        # save before evaluating, so the adapter survives a crash / lost instance during eval
        trainer.save_state()
        trainer.save_model(output_dir=os.path.join(out_dir, "adapter"))
        update_metrics(out_dir, peak_mem_gib=peak_gib, resumed_from=last_ckpt, hyclora=vars(hyclora_args),
                       seed=training_args.seed, **train_result.metrics)

    ###################### testing part ######################
    # reload the adapter (e.g. out_dir/adapter of an earlier run, with --is_train False)
    if training_args.trained_adapter_path is not None:
        load_result = model.load_adapter(training_args.trained_adapter_path, adapter_name="default")
        assert not load_result.unexpected_keys, f"adapter keys do not match the model: {load_result.unexpected_keys[:5]}"
        model.to(torch.bfloat16)  # load_adapter autocasts LoRA to fp32; keep eval-only identical to eval-after-training
        print(f"loaded adapter weights from {training_args.trained_adapter_path}")

    if training_args.is_eval:
        # Evaluation
        dataset = load_dataset(data_args.data_name, "main")
        test_set = dataset['test']
        
        logging.warning("Formatting inputs...")
        question = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
        answer = []
        
        # get numerical answer
        for example in test_set['answer']:
            ans = example.split('####')[-1]
            ans = ans.replace(',', '')  # handle numbers like 2,000
            try:
                ans = float(ans)
            except ValueError:
                ans = float("inf")
            answer.append(ans)
        
        logging.warning("Tokenizing inputs...")
        eval_step = math.ceil(len(question) / data_args.eval_batch_size)
        logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.eval_batch_size}" f"eval steps: {eval_step}")
        question_data = []
        for i in range(eval_step):
            if i < eval_step - 1:
                batch = tokenizer(
                    question[i*data_args.eval_batch_size: (i+1)*data_args.eval_batch_size],
                    return_tensors="pt",
                    padding="longest",
                )
            else:
                batch = tokenizer(
                    question[i*data_args.eval_batch_size:],
                    return_tensors="pt",
                    padding="longest",
                )
            batch['input_len'] = len(batch['input_ids'][0])
            question_data.append(batch)
            
        model.eval()
        # (a) the paper's sampled decoding (kept unchanged for comparability with the paper)
        set_seed(42)
        acc_sampled = evaluate_gsm8k(model, tokenizer, question_data, answer, out_dir, "sampled",
                                     dict(max_new_tokens=256, temperature=0.1, top_k=40, top_p=0.95, do_sample=True))
        print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*acc_sampled:.2f}% | "
            f"full precision: {model_args.full_precision}")
        print(f"GSM8K test accuracy (sampled): {100*acc_sampled:.2f}%")
        # (b) greedy decoding: deterministic, for paired comparisons between methods
        acc_greedy = evaluate_gsm8k(model, tokenizer, question_data, answer, out_dir, "greedy",
                                    dict(max_new_tokens=256, do_sample=False))
        print(f"GSM8K test accuracy (greedy): {100*acc_greedy:.2f}%")


def evaluate_gsm8k(model, tokenizer, question_data, answer, out_dir, mode, gen_kwargs):
    """Generate for every batch, save per-question predictions_<mode>.json and accuracy to metrics.json."""
    records, preds = [], []
    for step, batch in enumerate(question_data):
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=batch["input_ids"].to('cuda'), attention_mask=batch["attention_mask"].to('cuda'), **gen_kwargs)
        decoded_pred = tokenizer.batch_decode(generated_tokens[:, batch['input_len']:], skip_special_tokens=True)
        for sentence_pred in decoded_pred:
            preds.append(extract_answer_number(sentence_pred))
            records.append(make_record(len(records), preds[-1], answer[len(records)], sentence_pred))
        print(f"eval_batch {step + 1}/{len(question_data)} {mode}")
        # correct-so-far / answered-so-far
        print(f"accuracy: {compute_accuracy(preds, answer)}")
    accuracy = compute_accuracy(preds, answer)
    write_json(os.path.join(out_dir, f"predictions_{mode}.json"),
               dict(mode=mode, accuracy=accuracy, gen_kwargs=gen_kwargs, predictions=records))
    update_metrics(out_dir, **{f"gsm8k_acc_{mode}": accuracy, "gsm8k_n": len(records)})
    return accuracy


def make_record(idx, pred, gold, generation):
    finite = lambda x: None if math.isinf(x) else x  # inf = no number found; keep the JSON standard
    return dict(idx=idx, pred=finite(pred), gold=finite(gold), correct=pred == gold, generation=generation)


def write_json(path, obj):
    tmp = path + ".tmp"  # atomic: a lost instance never leaves a half-written file
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def update_metrics(out_dir, **kv):
    path = os.path.join(out_dir, "metrics.json")
    metrics = json.load(open(path)) if os.path.exists(path) else {}
    metrics.update(kv)
    write_json(path, metrics)


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        # use the last number as the answer
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)


if __name__ == "__main__":
    train_and_eval()
