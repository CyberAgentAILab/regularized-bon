from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import llm_blender

class RewardModel:
    def __init__(self, reward_model_id):
        pass
        
    def get_reward(self, question: str, answer: str) -> float:
        pass
    
    def get_rewards(self, question: str, answers: List[str]) -> List[float]:
        scores = []
        for i in range(len(answers)):
            score = self.get_reward(question, answers[i])
            scores.append(score)
        return scores


class OASST(RewardModel):
    def __init__(self, reward_model_id):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_id)
        self.reward_model.eval()
        self.reward_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        
    def get_reward(self, question, answer):
        # TODO: Batch operation.
        inputs = self.tokenizer(question, answer, return_tensors='pt').to(self.device)
        outputs = self.reward_model(**inputs).logits[0].cpu().detach().numpy().item()
        return outputs


class StanfordNLP(RewardModel):
    def __init__(self, reward_model_id):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = T5Tokenizer.from_pretrained(reward_model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(reward_model_id).to(self.device)
        self.model.eval()
        
    def get_reward(self, question, answer):
        input_text = "POST: {} \n\n RESPONSE A: {}\n\n RESPONSE B: .\n\n Which response is better? RESPONSE".format(question.replace('\n', ' '), answer.replace('\n', ' '))
        x = self.tokenizer([input_text], return_tensors='pt').input_ids.to(self.device)
        outputs = self.model.generate(x, return_dict_in_generate=True, output_scores=True, max_new_tokens=1)
        prob_of_A = torch.exp(outputs.scores[0][:, 71]) / torch.exp(outputs.scores[0][:,:]).sum(axis=1) # index 71 corresponds to the token for 'A'
        return prob_of_A.cpu().detach().numpy().item()
    

class PairLM(RewardModel):
    def __init__(self, reward_model_id):
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM")
        self.blender.blender_config.use_tqdm = False

    def get_reward(self, question, answer):
        print('PairLM.get_reward() not implemented.')
        assert(False)
        
    def get_rewards(self, question, answers):
        ranks = self.blender.rank([question], [list(answers)], return_scores=False, batch_size=1)
        return (1 - ranks[0]).tolist()

    def get_winratio(self, question, answer, compared_answers):
        assert isinstance(question, str)
        assert isinstance(answer, str)
        assert isinstance(compared_answers, list)
        assert isinstance(compared_answers[0], str)
        
        wins = 0
        cs = list(compared_answers)
        ncs = len(cs)
        pairs = [[answer, cs[i]] for i in range(ncs)]
        ranks = self.blender.rank([question] * ncs, pairs, return_scores=False, batch_size=16)

        wins = (ranks[:, 0] < ranks[:, 1]).sum() / ncs

        return wins


def load_reward_model(reward_model_id):
    if reward_model_id == 'OpenAssistant/reward-model-deberta-v3-large-v2':
        return OASST(reward_model_id)
    elif 'stanfordnlp/SteamSHP' in reward_model_id:
        return StanfordNLP(reward_model_id)
    elif reward_model_id == 'llm-blender/PairRM':
        return PairLM(reward_model_id)
    else:
        raise ValueError('Invalid reward_model_id')
