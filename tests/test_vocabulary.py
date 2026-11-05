from src.data.vocabulary import Vocabulary
def test_encode_decode():
    v=Vocabulary().build(['a person walks']); ids=v.encode('a person walks'); assert v.decode(ids)=='a person walks'
def test_unknown():
    v=Vocabulary().build(['walk']); assert v.itos[v.encode('jump')[1]]=='<unk>'
