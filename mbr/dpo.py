from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments

from trl import DPOTrainer

import argparse
import datetime

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="dataset name", default='XXXX-19/alpacafarm_dpo')
    parser.add_argument('--split', help="split name", default='rbon01')
    parser.add_argument('--model', help="model name", default='HuggingFaceH4/mistral-7b-sft-beta')
    parser.add_argument('--quantize', type=int, default=-1)
    parser.add_argument('--n_train', type=int, default=4000,
                        help="number of training samples")
    parser.add_argument('--n_eval', type=int, default=500,
                        help="number of evaluation samples")
    parser.add_argument('--beta', type=float, default=0.1, help='beta for DPO')
    parser.add_argument('--optim', type=str, default='rmsprop')
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--loss_type', type=str, default='sigmoid', help='loss type for DPO: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = "sigmoid')

    parser.add_argument('--bsz', type=int, default=4, help='batch size')

    parser.add_argument('--name', help="name of the repository to hub", default="auto")
    parser.add_argument('--username', help="username for hub", default="XXXX-37")

    dt_now = datetime.datetime.now()
    dtime = dt_now.strftime('%Y%m%d-%H%M%S')
    print('time=', dtime)
    
    args = parser.parse_args()

    model_name = args.model

    train_size = args.n_train
    eval_size = args.n_eval

    if args.name == 'auto':
        name = "{}_{}_{}_b-{}_r-{}_lr-{}_lt-{}".format(model_name.split('/')[1],
                                                        args.dataset.split('/')[1],
                                                        args.split,
                                                        args.beta,
                                                        args.lora_r,
                                                        args.lr,
                                                        args.loss_type[:3])
    else:
        name = args.name
        
    hub_id = args.username + '/' + name

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 torch_dtype=torch.bfloat16,
                                                 load_in_4bit=(args.quantize == 4), 
                                                 load_in_8bit=(args.quantize == 8))
    model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = args.dataset
    train_dataset = load_dataset(dataset, split='{}[:{}]'.format(args.split, train_size))
    train_dataset = train_dataset.map(lambda example: {k: (v if v is not None else '') for k, v in example.items()})

    eval_dataset = load_dataset(dataset, split='{}[{}:{}]'.format(args.split, train_size, train_size + eval_size))
    eval_dataset = eval_dataset.map(lambda example: {k: (v if v is not None else '') for k, v in example.items()})
    
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=args.bsz,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=train_size//8,
        eval_steps=train_size//4,
        output_dir="/data/dpo/test",
        optim=args.optim,
        warmup_steps=train_size//80,
        report_to='tensorboard',
        gradient_checkpointing=False,
        push_to_hub=True,
        save_steps=train_size//4,
        hub_model_id=hub_id,
        hub_strategy='checkpoint'
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=512,
        max_target_length=512,
        max_prompt_length=512,
        peft_config=peft_config,
    )

    # 6. train
    dpo_trainer.train()

    dpo_trainer.model.push_to_hub(hub_id, revision=dtime, commit_message='auto commit from dpo.py')
    
    with open('/data/model_name.txt', 'w') as f:
        f.write(name)