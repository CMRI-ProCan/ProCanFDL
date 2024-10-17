import os
import shap
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, f1_score
from torch.optim import Adam
from torch.utils.data import DataLoader

from FedModel import FedProtNet


class WeightsAggregation:
    def __init__(self, model_paths, global_model=None, mu=0.1):
        self.models = [torch.load(model_path) for model_path in model_paths]
        self.model_dict = {}
        self.mu = mu  # Proximal term coefficient
        self.global_weights = None
        self.global_model = global_model

    def fed_avg(self):
        self.global_weights = {key: torch.zeros_like(value) for key, value in self.models[0].items()}
        num_models = len(self.models)

        for model_weights in self.models:
            for key in self.global_weights.keys():
                self.global_weights[key] += model_weights[key] / num_models

    def fed_prox(self):
        self.global_weights = {key: torch.zeros_like(value) for key, value in self.global_model.items()}
        num_models = len(self.models)

        for model_weights in self.models:
            for key in self.global_weights.keys():
                # FedProx: Incorporating the proximal term
                fed_prox_update = model_weights[key] + self.mu * (model_weights[key] - self.global_model[key])
                self.global_weights[key] += fed_prox_update / num_models

    def save_model(self, save_path):
        torch.save(self.global_weights, save_path)


if __name__ == "__main__":
    weight_agg = WeightsAggregation(
        model_paths=[f"../models/Fed/local_model_Broad_Cancer_Type_{i}.pt" for i in range(10)])
    weight_agg.fed_avg()
    weight_agg.save_model("../models/Fed/Broad_Cancer_Type_fedavg.pt")
