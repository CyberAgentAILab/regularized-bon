import argparse


def get_mbr_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="dataset name e.g. wmt19.de-en, xsum, nocaps")
    parser.add_argument('--n_lines', type=int, default=4,
                        help="number of source inputs to evaluate. default is 4 so that it can be used for debugging")
    parser.add_argument('--start_iter', type=int, default=0,
                        help="starting id")
    
    parser.add_argument('--model', default="None",
                        help="default is None which is to select predefined model for each dataset")
    parser.add_argument('--prompt', default="None",
                        help="only applicable for Language models")
    parser.add_argument('--quantize', default=-1, type=int,
                        help="quantize the input to the model. -1 is no quantization")
    
    # Sampling algorithm
    parser.add_argument('--n_samples', type=int, default=100,
                        help="For sample.py, number of samples to generate for sampling algorithm. " + \
                            "For mbr_engine.py, this is the number of samples to generate for each source input")
    parser.add_argument('--bsz', type=int, default=4, help="batch size for sampling algorithm")
    parser.add_argument('--eps', type=float, default=0.02, help="epsilon for sampling algorithm")
    parser.add_argument('--topk', type=int, default=0, help="topk for sampling algorithm")
    parser.add_argument('--topp', type=float, default=1.0, help="topp for sampling algorithm")
    
    parser.add_argument('--discard_prob', action='store_true', help="whether to discard prob for sampling algorithm")

    # MBR Algorithm
    parser.add_argument('--sample_dir', help="directory to save samples")
    parser.add_argument('--algorithm', default='None', help="mbr algorithm")
    parser.add_argument('--recompute_matrix', action='store_true', help="whether to recompute similarity matrix")

    # Utility function
    parser.add_argument('--sim', default='bertscore', help='similarity function (utility function) for MBR')
    # Evaluation function
    parser.add_argument('--eval', default='bleu', help='quality metric for evaluating the output')

    parser.add_argument('--do_sample', type=int, default=1, help="beam size for sampling algorithm")


    #################
    # Reward modeling
    parser.add_argument('--reward_model', default='OpenAssistant/reward-model-deberta-v3-large-v2', help="reward model")
    parser.add_argument('--recompute_reward', action='store_true', help="whether to recompute reward")
    parser.add_argument("--compared_dir", type=str, default=None, help="sample_dir for the model to be compared for pairwise comparison reward model")
    parser.add_argument("--c_nsamples", type=int, default=1, help="number of samples for the model to be compared for pairwise comparison reward model")

    return parser
