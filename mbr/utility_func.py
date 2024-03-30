
import utility.utility_class as utility_class


def load_similarity(sim):
    if sim == 'sentbert':
        return utility_class.SENTBERT()
    elif sim == 'cliptext':
        return utility_class.CLIPTEXT()
    elif sim == 'comet':
        return utility_class.COMET()
    elif sim == 'comet20':
        return utility_class.COMET20()
    elif sim == 'bertscore':
        return utility_class.BERTSCORE()
    elif sim == 'bleurt':
        return utility_class.BLEURT()
    else:
        assert False
