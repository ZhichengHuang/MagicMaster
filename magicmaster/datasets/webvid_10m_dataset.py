import json
import os.path
import os.path as osp
from typing import Sequence

from mmengine.registry import build_from_cfg
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from magicmaster.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class webvid10m(Dataset):

    def __init__(
        self,
        ann_file: str,
        data_root: str = '',
        pipeline: Sequence = (),
        test_mode: bool = False,
    ):
        self.data_root = data_root
        self.data_infors = self.load_annotations(osp.join(data_root, ann_file))

        self.test = test_mode


        pipeline = [build_from_cfg(p, TRANSFORMS) for p in pipeline]
        self.pipeline = Compose(pipeline)

    def load_annotations(self, data_ann):
        out=[]
        with open(data_ann,'r') as f:
            for line in f:
                filename=line.strip()
                tmp=dict()
                tmp['filename']=filename
                out.append(tmp)
        return out

    def __len__(self):
        return len(self.data_infors)

    def __getitem__(self, idx):
        """Retrieve an item based on `idx`. An item has the following format:
        {'filename': 'n02115641_23115.JPEG', 'prefix': 'train/n02115641', 'label': 541}
        """
        item = self.data_infors[idx].copy()
        item['data_root'] = self.data_root
        item = self.pipeline(item)
        return item


