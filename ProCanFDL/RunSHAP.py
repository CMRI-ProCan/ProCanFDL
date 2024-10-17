import shap

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, KFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

from FedAggregateWeights import WeightsAggregation
from FedTrain import TrainFedProtNet
from utils.ProtDataset import ProtDataset
from sklearn.model_selection import GroupShuffleSplit
import pickle
import time

torch.manual_seed(0)
np.random.seed(0)

N_clients = 3
N_included_clients = 3
N_repeats = 10
N_iters = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hypers = {
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'hidden_dim': 256,
    'dropout': 0.2,
    'batch_size': 100,
    'epochs': 200,
    'target': 'cancer_type_2',
    'included_types': ['Breast_carcinoma', 'Colorectal_adenocarcinoma', 'Cutaneous_melanoma',
                       'Cutaneous_squamous_cell_carcinoma', 'Head_and_neck_squamous',
                       'Hepatocellular_carcinoma', 'Leiomyosarcoma',
                       'Liposarcoma', 'Non_small_cell_lung_adeno', 'Non_small_cell_lung_squamous',
                       'Oesophagus_adenocarcinoma', 'Pancreas_neuroendocrine',
                       'Pancreatic_ductal_adenocarcinoma', 'Prostate_adenocarcinoma']
}
print(hypers)
TARGET = hypers['target']
included_types = hypers['included_types']
FILE_NAME = f"cancer_type_2_group3of3_10times_20240815_10iter"
MODEL_PATH = f"/home/scai/VCB_E0008/models/Fed/{FILE_NAME}"

prot_id_to_name = pd.read_csv("../data/all_protein_list_mapping.csv",
                              index_col=0).to_dict()['Gene']

prot_name_to_id = pd.read_csv("../data/all_protein_list_mapping.csv",
                              index_col=3).to_dict()['UniProtID']

combined_df = pd.read_csv("../data/P10/E0008_P10_protein_averaged_log2_transformed_EB.csv", index_col=0,
                              low_memory=False)
combined_df = combined_df[combined_df['sample_tissue_descriptor'] == "Malignant"]
corr_df = pd.read_csv("../data/P10/replicate_corr_protein.csv")
low_corr_samples = corr_df[corr_df["pearsons_r"] < 0.9]["prep_lims_id"].tolist()
combined_df = combined_df[~((combined_df['prep_cohort'] != 'CPTAC') & combined_df.index.isin(low_corr_samples))]
if TARGET in ['Broad_Cancer_Type', 'cancer_type', 'cancer_type_2']:
    combined_selected_df = combined_df[combined_df[TARGET].isin(included_types)]
else:
    combined_selected_df = combined_df[
        (combined_df['Broad_Cancer_Type'] == 'Adenocarcinoma') & (combined_df[TARGET].isin(included_types))]
merged_label_map = combined_selected_df[TARGET].to_dict()

# %%
combined_vcb_df = combined_selected_df[combined_selected_df['prep_cohort'] == "E0008-P01"]
combined_rest_df = combined_selected_df[combined_selected_df['prep_cohort'] != "E0008-P01"]
combined_rest_df = combined_rest_df[
    (combined_rest_df['prep_cohort'] != "E0006-P02") | (combined_rest_df['prep_general_sample_info'] == "Primary")]

combined_rest_cohort_df = combined_rest_df[['prep_cohort', TARGET]].drop_duplicates()
combined_rest_cohort_df = combined_rest_cohort_df.sort_values(by=[TARGET, 'prep_cohort'])
combined_rest_cohort_df = combined_rest_cohort_df.drop_duplicates(subset=['prep_cohort'], keep='last').reset_index(
    drop=True)
META_COL_NUMS = 73

meta_vcb = pd.read_excel(
    "../data/sample_info/sample_metadata_path_noHek_merged_replicates_Adel_EB_2.0_no_Mucinous.xlsx",
    engine='openpyxl')
meta_vcb = meta_vcb.drop_duplicates(subset=['LIMS-ID1'])
meta_vcb = meta_vcb[meta_vcb["Include_Y_N"] == 'Yes']
included_vcb_lims = meta_vcb['LIMS-ID1'].tolist()
combined_vcb_df = combined_vcb_df[combined_vcb_df.index.isin(included_vcb_lims)]

le = LabelEncoder()
le.fit(combined_selected_df[TARGET])
# train_rest_df, test_rest_df = train_test_split(combined_rest_df, stratify=combined_rest_df[TARGET],
#                                                test_size=0.1, random_state=0)
unique_cohorts = combined_rest_df['prep_cohort'].unique()
train_rest_df = []
test_rest_df = []
for c in unique_cohorts:
    print(f"{c}: {combined_rest_df[combined_rest_df['prep_cohort'] == c][TARGET].value_counts()}")
    tmp_df = combined_rest_df[combined_rest_df['prep_cohort'] == c]

    gss = GroupShuffleSplit(n_splits=1, test_size=.1, random_state=42)
    i, (train_index, test_index) = next(enumerate(
        gss.split(tmp_df, y=tmp_df[TARGET], groups=tmp_df['subject_collaborator_patient_id'])))
    train_rest_cohort_df = tmp_df.iloc[train_index]
    test_rest_cohort_df = tmp_df.iloc[test_index]
    train_rest_df.append(train_rest_cohort_df)
    test_rest_df.append(test_rest_cohort_df)
train_rest_df = pd.concat(train_rest_df)
test_rest_df = pd.concat(test_rest_df)

print(f"test shape of samples: {test_rest_df.shape[0]}")
test_rest_df = test_rest_df.drop_duplicates(subset=['subject_collaborator_patient_id'], keep='last')
print(f"test shape of unique samples: {test_rest_df.shape[0]}")

print(train_rest_df[TARGET].value_counts())
train_combined_df = pd.concat([combined_vcb_df, train_rest_df])
best_scores = []
print(train_combined_df.groupby([TARGET, 'prep_cohort']).size())
print(test_rest_df[TARGET].value_counts())
# print(test_rest_df.groupby([TARGET, 'prep_cohort']).size())

# traditional
test_dataset = ProtDataset(
    test_rest_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
    le.transform(test_rest_df[TARGET].to_numpy()),
    prot_id_to_name,
    prot_name_to_id,
    merged_label_map
)

all_dataset = ProtDataset(
    combined_selected_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
    le.transform(combined_selected_df[TARGET].to_numpy()),
    prot_id_to_name,
    prot_name_to_id,
    merged_label_map
)

# load model
repeat = 9
model_training = TrainFedProtNet(test_dataset, test_dataset, hypers,
                                 load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg_rep{repeat}.pt")
model = model_training.model
model.eval()

# batch_size = len(test_dataset)
# data_explain = DataLoader(
#     test_dataset,
#     batch_size=batch_size,
#     shuffle=True,
# )

batch_size = len(all_dataset)
data_explain = DataLoader(
    all_dataset,
    batch_size=batch_size,
    shuffle=False,
)

data = next(iter(data_explain))
X, labels = data
X = X.to(device)
e = shap.GradientExplainer(model, X)
start = time.time()
shap_values = e.shap_values(X)
shap_obj = e(X)
end = time.time()
print(f"Time taken: {(end - start) / 60:.2f} minutes")
pickle.dump(shap_values, open(f"{MODEL_PATH}/{hypers['target']}_shap_values_rep{repeat}_all.pkl", "wb"))
pickle.dump(shap_obj, open(f"{MODEL_PATH}/{hypers['target']}_shap_values_rep{repeat}_all_obj.pkl", "wb"))
