import os

import numpy as np
from transformers import set_seed
import torch
from tqdm import tqdm
import pandas as pd

from parser import get_mbr_parser
from utils import load_model, load_dataset, load_kwargs, StoppingCriteriaSub
from utils import sample_dir, prompt_dir

def compute_probability_s2s(sample_output):
    """
    This compute_prob function is compatible with seq2seq models.
    Doesn't work on language models.
    """
    bsz = sample_output.sequences.shape[0]
    probs = np.array([1.0] * bsz)
    # terms = [False] * bsz
    for i in range(len(sample_output.scores)):
        p = np.array([1.0] * bsz)
        for b in range(bsz):
            if hasattr(tokenizer, "pad_token_id"):
                if sample_output.sequences[b][i+1] == tokenizer.pad_token_id:
                    continue
            log_probs = torch.nn.functional.log_softmax(sample_output.scores[i][b], dim=-1)
            p[b] = torch.exp(log_probs[sample_output.sequences[b][i+1]])
        probs *= p
        # print('p=', p)
    return probs

def compute_probability_lm(model, outputs):
    """
    This compute_prob function is compatible with langauge models.
    Doesn't work on seq2seq models.
    """
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    ).cpu().to(torch.float32)
    
    seq_prob = torch.ones(transition_scores.shape[0]).to(torch.float32)
    for i in range(transition_scores.shape[1]):
        seq_prob *= np.exp(transition_scores[:,i])

    return seq_prob.numpy()

def get_texts(tokenizer, outputs, input_length):
    """
    This function is only compatible with langauge models. not for seq2seq
    """
    bsz = outputs.sequences.shape[0]
    output_texts = []
    for b in range(bsz):
        output_text = tokenizer.decode(outputs.sequences[b][input_length:], skip_special_tokens=True)
        output_texts.append(output_text)
    return output_texts

def sample(dataset, tokenizer, model, src_lines, torch_device, n_lines, start_iter, n_samples, bsz, 
           eps, topk, topp, do_sample, diversity_penalty,
           prompt, stop_tokens,
           model_n):
    n_batches = n_samples // bsz
    
    if not do_sample:
        if n_batches > 1:
            print('n_batches must be 1 for beam search. Setting n_batches to 1.')
        n_batches = 1
        if diversity_penalty < 0.000001:
            print('Running beam search as diversity penalty is zero.')

    os.makedirs(os.path.join(sample_dir, dataset, model_n), exist_ok=True)
    
    model_kwargs = load_kwargs(dataset)


    for sample_id in tqdm(range(start_iter, n_lines)):
        if sample_id > len(src_lines):
            break

        if prompt == "None":
            input_source = src_lines[sample_id]
            model_inputs = tokenizer(input_source, return_tensors='pt', truncation=True).to(torch_device)
            stopping_criteria = None
        else:
            if 'zephyr' in model_n:
                messages = [
                    {
                        "role": "system",
                        "content": prompt,
                    },
                    {
                        "role": "user",
                        "content": src_lines[sample_id],
                    }
                ]
                input_source = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if sample_id == 0:
                    print('input_source=', input_source)
            elif 'dolly' in model_n:

                INSTRUCTION_KEY = "### Instruction:"
                RESPONSE_KEY = "### Response:"
                END_KEY = "### End"
                INTRO_BLURB = (
                    "Below is an instruction that describes a task. Write a response that appropriately completes the request."
                )
                input_source = """{intro}
{instruction_key}
{instruction}
{response_key}
""".format(
                    intro=INTRO_BLURB,
                    instruction_key=INSTRUCTION_KEY,
                    instruction=src_lines[sample_id][0]["content"],
                    response_key=RESPONSE_KEY,
                )
                if sample_id == 0:
                    print('input_source=', input_source)
            elif "[[QUESTION]]" not in prompt:
                if isinstance(src_lines[sample_id], list):
                    input_source = tokenizer.apply_chat_template(src_lines[sample_id], tokenize=False, add_generation_prompt=True)
                else:
                    prompt = """USER: {}
ASSISTANT: """
                    input_source = prompt.format(src_lines[sample_id])
                if sample_id == 0:
                    print('input_source=', input_source)                
            else:
                input_source = prompt.replace("[[QUESTION]]", src_lines[sample_id])
                if 'Mistral' in model_n:
                    input_source = "[INST] " + input_source + "[/INST]"
        
            model_inputs = tokenizer(input_source, return_tensors="pt", 
                                return_token_type_ids=False).to(model.device)
            input_length = model_inputs['input_ids'].shape[1]

            bins = torch.bincount(model_inputs['input_ids'][0].to('cpu'))

            if len(stop_tokens) > 0:
                nlines = bins[stop_tokens[0]].numpy()
                stopping_criteria = StoppingCriteriaList([
                    StoppingCriteriaSub(stops=stop_tokens, encounters=nlines+1)])
            else:
                stopping_criteria = None
        
        
        set_seed(42)

        rows = []

        for i in range(n_batches):
            if do_sample:
                sample_output = model.generate(
                    **model_inputs,
                    **model_kwargs,
                    do_sample=True,
                    epsilon_cutoff=eps,
                    top_k=topk,
                    top_p=topp,
                    num_beams=1,
                    num_return_sequences=bsz,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=True,
                    forced_bos_token_id=model.config.forced_bos_token_id,
                )
            else:
                if diversity_penalty > 0.000001:
                    num_beam_groups = bsz
                else:
                    # If diversity penalty is zero, then DiverseBS is equivalent to beam search.
                    num_beam_groups = 1

                sample_output = model.generate(
                    **model_inputs,
                    **model_kwargs,
                    do_sample=False,
                    epsilon_cutoff=eps,
                    top_k=topk,
                    top_p=topp,
                    num_beams=bsz,
                    num_return_sequences=bsz,
                    num_beam_groups=num_beam_groups,
                    diversity_penalty=diversity_penalty,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=True,
                    output_scores=True,
                    forced_bos_token_id=model.config.forced_bos_token_id,
                )
                
            if prompt == "None":
                if do_sample:
                    probs = compute_probability_s2s(sample_output)
                else:
                    probs = [0.0] * bsz
                for j in range(bsz):
                    sample_text = tokenizer.decode(sample_output.sequences[j], skip_special_tokens=True)
                    rows.append((sample_text, probs[j]))
            else:
                output_prob = compute_probability_lm(model, sample_output)
                output_text = get_texts(tokenizer, sample_output, input_length)
                for j in range(bsz):
                    rows.append((output_text[j], output_prob[j]))
                    
        if do_sample:
            filename = "{:04}_eps-{:.2f}_topk-{:02d}_topp-{:.2f}".format(sample_id, eps, topk, topp)
        else:
            filename = "{:04}_beam-{:02d}_divpen-{:.2f}".format(sample_id, bsz, diversity_penalty)
        
        outfilepath = os.path.join(sample_dir, dataset, model_n, filename)

        df = pd.DataFrame(rows, columns=['text', 'probability'])
        df.to_csv(outfilepath, index=False)


if __name__ == "__main__":
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    parser = get_mbr_parser()
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    prompt_path = args.prompt
    n_lines = args.n_lines
    start_iter = args.start_iter

    n_samples = args.n_samples
    bsz = args.bsz
    eps = args.eps
    topk = args.topk
    topp = args.topp
    do_sample = args.do_sample > 0

    src_lines = load_dataset(dataset)
    tokenizer, model, model_name, stop_tokens = load_model(dataset, torch_device, model_name)

    if prompt_path == "None":
        prompt = "None"
    else:
        with open(os.path.join(prompt_dir, prompt_path), "r") as f:
            prompt = f.read()

    sample(dataset, tokenizer, model, src_lines, torch_device, n_lines, start_iter, n_samples, bsz, 
           eps, topk, topp, do_sample, 0,
           prompt, stop_tokens,
           model_n=os.path.basename(model_name))


