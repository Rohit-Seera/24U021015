from src.evaluation.metrics import score_one
def test_metric_output():
    x=score_one('a person walks',['a person walks','someone is walking']);assert {'bleu_1','bleu_4','rouge_l','meteor'}<=set(x);assert 0<=x['bleu_1']<=1
