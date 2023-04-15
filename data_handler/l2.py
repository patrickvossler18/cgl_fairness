"""
cgl_fairness
Copyright (c) 2022-present NAVER Corp.
MIT license
"""
import pandas as pd

from data_handler.AIF360.l2_dataset import L2Dataset
from data_handler.tabular_dataset import TabularDataset


class L2Dataset_torch(TabularDataset):
    """L2 dataset."""

    name = "l2"

    def __init__(self, root, filename="NC_bisg.csv", target_attr="black", **kwargs):

        dataset = L2Dataset(root_dir=root, filename=filename)
        if target_attr == "black":
            sen_attr_idx = 7
        else:
            raise Exception("Not allowed group")

        self.num_groups = 2
        self.num_classes = 2

        super(L2Dataset_torch, self).__init__(
            root=root, dataset=dataset, sen_attr_idx=sen_attr_idx, **kwargs
        )
