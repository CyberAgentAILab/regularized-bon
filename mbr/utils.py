import os

import json
from collections.abc import Iterable
from glob import glob

import numpy as np
import pandas as pd

import torch
from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList
import datasets
from peft import PeftModel

dataset_dir = './dataset'
sample_dir = './samples'
result_dir = './results'
matrix_dir = './matrix'

reward_dir = './reward'
prompt_dir = './prompts'

def load_model(dataset, torch_device, model_name, quantize=-1):

    q4 = quantize == 4
    q8 = quantize == 8

    stop_tokens = []

    model_n = os.path.basename(model_name)
    if 'Mixtral-8x7B-Instruct-ja-en' in model_n:
        # LoRA models needs to load the base model first
        base_model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    else:
        base_model_name = model_name
    
    if ('polylm' in model_n):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, legacy=False, use_fast=False, padding_size="left")
        if 'polylm' in model_n:
            stop_tokens = [213] # new line token \n           
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_size="left", use_fast=True, trust_remote_code=True)
        if 'falcon' in model_n:
            # TODO: it doesn't fully solve the problem. There seems to be random newline tokens.
            stop_tokens = [193, 1001]
        elif 'bloomz' in model_n:
            stop_tokens = [2]
        else:
            print('Stop token is not set for', model_n)
            stop_tokens = []
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if ('pythia' in base_model_name):
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|prompter|>' + message['content'] + '<|endoftext|> '}}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
        
    if ('stablelm' in base_model_name):
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|USER|>' + message['content']}}{% endfor %}{% if add_generation_prompt %}{{ '<|ASSISTANT|>' }}{% endif %}"
        
    if ('42dot' in base_model_name):
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<human>: ' + message['content'] + ' '}}{% endfor %}{% if add_generation_prompt %}{{ '<bot>: ' }}{% endif %}"

    if 'Mixtral' in model_n:
        # TODO: Flast attention is not used yet as the cuda version is old.
        model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, load_in_4bit=q4, load_in_8bit=q8, device_map="auto")
    elif ('Mistral' in base_model_name) or ('zephyr' in base_model_name) or ('mistral-7b-sft-beta' in base_model_name):
        model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, load_in_4bit=q4, load_in_8bit=q8, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True, load_in_4bit=q4, load_in_8bit=q8, device_map="auto")
    
    if 'Mixtral-8x7B-Instruct-ja-en' in model_n:
        assert not (q4 or q8) 
        model = PeftModel.from_pretrained(model=model, model_id=model_name)

    model.eval()

    return tokenizer, model, model_name, stop_tokens

def load_dataset(dataset, ref=False, raw_text=False):

    subdir_name = 'hf' # Using huggingface dataset

    if dataset == "alpaca":
        dataset = datasets.load_dataset("tatsu-lab/alpaca_eval", split='eval[:805]')
        if not ref:
            lines = []
            for d in dataset:
                # if d['input'] == "":
                #     instruct = d['instruction']
                # else:
                #     instruct = d['instruction'] + '\n\n' + d['input']
                message = [
                    {
                        "role": "user",
                        "content": d['instruction'],
                    }
                ]
                lines.append(message)
        else:
            lines = dataset['output']
    elif dataset == 'alpacafarm':
        dataset = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_human_preference", split='preference')
        if not ref:
            lines = []
            for d in dataset:
                if d['input'] == "":
                    instruct = d['instruction']
                else:
                    instruct = d['instruction'] + '\n\n' + d['input']
                message = [
                    {
                        "role": "user",
                        "content": instruct,
                    }
                ]
                lines.append(message)
        else:
            lines = []
            for d in dataset:
                if d['preference'] == 1:
                    lines.append(d['output_1'])
                else:
                    lines.append(d['output_2'])

    else:
        assert False

    assert isinstance(lines, Iterable)
    return lines

def load_kwargs(dataset):
    LLM_KWARGS = dict(max_new_tokens=256, no_repeat_ngram_size=4)
    model_kwargs = LLM_KWARGS
    return model_kwargs

def load_matrix(target_matrix_dir, filename, sim, n_samples):
    """
    Load matrix from the target_matrix_dir.
    The matrix is saved as a text file.
    The matrix stored may be larger than n_samples. In this case, it is truncated.
    If no files found, it returns None.
    """
    matrix_base = os.path.join(target_matrix_dir, filename + "_" + sim + "_")
    matrix_paths = glob(matrix_base + "*")
    
    cached_nsamples = [int(f[len(matrix_base):]) for f in matrix_paths]   
    larger_cahces = [c for c in cached_nsamples if c >= n_samples]
           
    if len(larger_cahces) == 0:
        return None
    
    min_nsamples = min(larger_cahces)

    matrix = np.loadtxt(matrix_base + str(min_nsamples))
    matrix = matrix[:n_samples, :n_samples]
    
    return matrix

    
def load_samples(dataset, sample_id, eps):
    print("utils.load_samples: not maintained")
    sample_path = os.path.join(sample_dir, dataset, "{:04d}_eps-{:.2f}".format(sample_id, eps))

    with open(sample_path) as f:
        samples = f.read().splitlines()

    return samples


def load_samples_from_file(files, epsilon, topk, topp, do_sample, diverse_k, divpen):

    filtered_files = []
    
    if do_sample:
        for filename in files:
            isnt_eps = not "eps-{:.2f}".format(epsilon) in filename
            
            # If topk is set to negative (e.g. -1), then it means that "topk" should not be in the filename.
            if topk < 0:
                isnt_topk = "topk" in filename
            else:
                isnt_topk = not "topk-{:02d}".format(topk) in filename
            
            if topp < 0:
                isnt_topp = "topp" in filename
            else:
                isnt_topp = not "topp-{:.2f}".format(topp) in filename

            if not (isnt_eps or isnt_topk or isnt_topp):
                filtered_files.append(filename)
    else:
        for filename in files:
            k_matches = "beam-{:02d}".format(diverse_k) in filename
            dp_matches = "divpen-{:.2f}".format(divpen) in filename

            if k_matches and dp_matches:
                filtered_files.append(filename)
    
    return filtered_files

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops = [], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      stop_count = 0
      for stop in self.stops:
        stop_count += (stop == input_ids[0]).sum().item()

      if stop_count >= self.ENCOUNTERS:
          return True
      return False

def list_to_text(words):
    text = words[0]
    for w in words[1:]:
        text = text + " " + w
    return text
