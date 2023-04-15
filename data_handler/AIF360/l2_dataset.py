"""
Original code:
    https://github.com/Trusted-AI/AIF360
"""
import os

import pandas as pd

from data_handler.AIF360.standard_dataset import StandardDataset


class L2Dataset(StandardDataset):
    def __init__(
        self,
        root_dir,
        filename,
        label_name="General_2016_11_08",
        favorable_classes=[1],
        protected_attribute_names=["black"],
        privileged_classes=[[0]],
        instance_weights_name=None,
        categorical_features=[],
        features_to_keep=[],
        features_to_drop=["prob_b", "LALVOTERID"],
        na_values=[],
        custom_preprocessing=None,
    ):

        training_cols = {
            "X": [
                "Voters_Gender",
                "Voters_Age",
                "CommercialData_EstimatedHHIncome",
                "CommercialData_EstimatedAreaMedianHHIncome",
                "CommercialData_EstHomeValue",
                "CommercialData_AreaMedianEducationYears",
                "CommercialData_AreaMedianHousingValue",
            ],
            "b": "black",
            "prob_b": "prob_black",
            "y": "General_2016_11_08",
            "std": [
                "Voters_Age",
                "CommercialData_EstimatedHHIncome",
                "CommercialData_EstimatedAreaMedianHHIncome",
                "CommercialData_EstHomeValue",
                "CommercialData_AreaMedianEducationYears",
                "CommercialData_AreaMedianHousingValue",
            ],
            "y_pred": "y_pred",
            "y_pred_proba": "y_pred_proba",
        }

        df = pd.read_csv(os.path.join(root_dir, filename))

        super(L2Dataset, self).__init__(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop,
            na_values=na_values,
            custom_preprocessing=custom_preprocessing,
        )
