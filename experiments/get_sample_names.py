
def get_sample_file_pattern(epsilon, topk, topp, do_sample, diverse_k, divpen):
    if do_sample:
        if topk < 0 and topp < 0:
            return "*_eps-{:.2f}".format(epsilon)
        else:
            return "*_eps-{:.2f}_topk-{:02d}_topp-{:.2f}".format(epsilon, topk, topp)
    else:
        return "*_beam-{:02d}_divpen-{:.2f}".format(diverse_k, divpen)


if __name__ == "__main__":
    import sys
    epsilon = float(sys.argv[1])
    topk = int(sys.argv[2])
    topp = float(sys.argv[3])
    do_sample = int(sys.argv[4]) > 0
    diverse_k = int(sys.argv[5])
    divpen = float(sys.argv[6])
    print(get_sample_file_pattern(epsilon, topk, topp, do_sample, diverse_k, divpen))
