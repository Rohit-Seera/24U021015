import numpy as np
from src.data.dataset import HumanML3D, assert_disjoint_splits
import pytest
def test_alignment_and_splits(tmp_path):
    for d in ['new_joint_vecs','texts']:(tmp_path/d).mkdir()
    np.save(tmp_path/'new_joint_vecs'/'a.npy',np.ones((3,2),dtype='float32'))
    (tmp_path/'texts'/'a.txt').write_text('person walks#person/NOUN walk/VERB#0.0#0.0\n',encoding='utf-8')
    for s,ids in [('train','a'),('val','a'),('test','b')]: (tmp_path/f'{s}.txt').write_text(ids,encoding='utf-8')
    ds=HumanML3D(tmp_path);assert ds.captions('a')==['person walks'];assert ds.motion('a').shape==(3,2);assert set(ds.split_ids('test'))=={'b'}
def test_split_overlap_is_rejected():
    with pytest.raises(ValueError,match='overlap'): assert_disjoint_splits(['a','b'],['b','c'])
