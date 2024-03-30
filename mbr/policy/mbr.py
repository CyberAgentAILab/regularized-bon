import numpy as np

def compute_score_matrix(samples, score_function, src_input=None):
    # TODO: add param ref_samples to compute the score between two different samples.
    n_samples = len(samples)
    scores = []
    for i in range(n_samples):
        score = score_function(hyp=np.array([samples[i]] * n_samples), ref=samples, src=src_input)
        scores.append(score)
    return np.array(scores)

def compute_mbr(hyp=None, compute_similatiy=None, matrix=None, weights=None, src=None, incremental=False):
    assert (compute_similatiy is not None) or (matrix is not None)
    if matrix is None:
        matrix = compute_score_matrix(hyp, compute_similarity, [src] * len(hyp))
        
    if weights is not None:
        mbr_scores = matrix @ np.transpose(weights)
    else:
        mbr_scores = np.sum(matrix, axis=1)
        
    if incremental:
        best_hyp = -1
        best_score = -np.inf
        bests = []
        for i in range(mbr_scores.shape[0]):
            if mbr_scores[i] > best_score:
                best_hyp = i
                best_score = mbr_scores[i]
            assert best_hyp >= 0
            bests.append(best_hyp)
        return bests # List of hypothesis indices.
    else:
        best_hyp = np.argmax(mbr_scores)
        
        assert best_hyp >= 0
        return best_hyp
