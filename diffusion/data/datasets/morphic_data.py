from functools import partial
from numpy.random import choice
import functools
import torchvision.transforms.functional as TF
import litdata as ld



def select_sketch_sample(keys,weights):
    assert sum(weights)==1
    return choice(keys,p=weights)
    

class MorphicStreamingDataset(ld.StreamingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._keys=['sketch_refined','sketch_anime','sketch_opensketch']
        self._weights=[0.4,0.3,0.3]
        self.aspect_ratio = 1.0
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        output = {}
        sketch_sampler = functools.partial(select_sketch_sample,keys=self._keys,
                                                               weights=self._weights)
        # (TODO) Check if this preprocessing is correct
        output['image']= TF.center_crop(sample['image'],(256,256))
        output['text'] = sample['caption']
        if output['image'].shape[0]==1:
            output['image']=output['image'].repeat(3,1,1)
        output['image']= (output['image']/127.5) - 1.0
        output['sketch']= TF.to_tensor(TF.center_crop(sample[sketch_sampler()],(256,256))).reshape((1,256,256))
        return output
 