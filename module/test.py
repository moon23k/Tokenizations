import json, math, time, torch, evaluate
from module.search import Search



class Tester:
    def __init__(self, config, model, tokenizer, test_dataloader, test_volumn=10):
        super(Tester, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = test_dataloader
        self.search = Search(config, self.model)
        
        self.path = config.path
        self.device = config.device
        self.test_volumn = test_volumn
        assert test_volumn <= len(self.dataloader)
        
        self.metric_name = 'BLEU'
        self.metric_module = evaluate.load('bleu')



    def test(self):
        self.model.eval()
        greedy_score, beam_score = 0, 0
        tot_data_len = len(self.dataloader)

        volumn_cnt = 0
        print(f"Test on {self.path} Model")
        with torch.no_grad():
            for batch in self.dataloader:

                if volumn_cnt == self.test_volumn:
                    break
                volumn_cnt += 1
               
                src = batch['src'].to(self.device)
                trg = batch['trg'].squeeze().tolist()
        
                greedy_pred = self.search.greedy_search(src).tolist()
                beam_pred = self.search.beam_search(src)
                
                greedy_score += self.metric_score(greedy_pred, trg)
                beam_score += self.metric_score(beam_pred, trg)

        greedy_score = round(greedy_score/self.test_volumn, 2)
        beam_score = round(beam_score/self.test_volumn, 2)
        
        print(f"---Greedy Score: {greedy_score}")
        print(f"--- Beam  Score: {beam_score}")
                


    def metric_score(self, pred, label):
        pred = self.tokenizer.decode(pred)
        label = self.tokenizer.decode(label)
        
        if pred == "":
            return 0.0
        else:
            score = self.metric_module.compute(predictions=[pred], references=[[label]])['bleu']
            return (score * 100)
