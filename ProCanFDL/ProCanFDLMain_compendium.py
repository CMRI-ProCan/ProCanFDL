# Description: This script is used to run the federated learning experiments on the ProCan compendium dataset (Figure 3).

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    StratifiedGroupKFold,
    KFold,
)
from sklearn.preprocessing import LabelEncoder
from FedAggregateWeights import WeightsAggregation
from FedTrain import TrainFedProtNet
from utils.ProtDataset import ProtDataset
from sklearn.model_selection import GroupShuffleSplit
import config

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

N_clients = 3
N_included_clients = 3
N_repeats = 10
N_iters = 10

if __name__ == "__main__":

    # Load hyperparameters from config
    hypers = config.DEFAULT_HYPERPARAMETERS.copy()
    hypers.update(
        {
            "target": config.DEFAULT_TARGET,
            "included_types": config.DEFAULT_CANCER_TYPES,
        }
    )

    print(hypers)
    TARGET = hypers["target"]
    included_types = hypers["included_types"]
    FILE_NAME = f"{hypers['target']}_group{N_included_clients}of{N_clients}_{N_repeats}times_20240815_{N_iters}iter"
    MODEL_PATH = config.get_model_path(FILE_NAME)

    prot_id_to_name = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=0).to_dict()[
        "Gene"
    ]

    prot_name_to_id = pd.read_csv(config.PROTEIN_MAPPING_FILE, index_col=3).to_dict()[
        "UniProtID"
    ]

    combined_df = pd.read_csv(
        config.PROCAN_DATA_FILE,
        index_col=0,
        low_memory=False,
    )
    combined_df = combined_df[combined_df["sample_tissue_descriptor"] == "Malignant"]
    combined_df = combined_df[combined_df["sample_sample_type"] != "Cell lines"]
    corr_df = pd.read_csv(config.REPLICATE_CORR_FILE)
    low_corr_samples = corr_df[corr_df["pearsons_r"] < config.QUALITY_THRESHOLD][
        "prep_lims_id"
    ].tolist()
    combined_df = combined_df[~combined_df.index.isin(low_corr_samples)]
    if TARGET in ["Broad_Cancer_Type", "cancer_type", "cancer_type_2"]:
        combined_selected_df = combined_df[combined_df[TARGET].isin(included_types)]
    else:
        combined_selected_df = combined_df[
            (combined_df["Broad_Cancer_Type"] == "Adenocarcinoma")
            & (combined_df[TARGET].isin(included_types))
        ]
    merged_label_map = combined_selected_df[TARGET].to_dict()

    # %%
    combined_vcb_df = combined_selected_df[
        combined_selected_df["prep_cohort"] == "E0008-P01"
    ]
    combined_rest_df = combined_selected_df[
        combined_selected_df["prep_cohort"] != "E0008-P01"
    ]
    combined_rest_df = combined_rest_df[
        (combined_rest_df["prep_cohort"] != "E0006-P02")
        | (combined_rest_df["prep_general_sample_info"] == "Primary")
    ]

    combined_rest_cohort_df = combined_rest_df[
        ["prep_cohort", TARGET]
    ].drop_duplicates()
    combined_rest_cohort_df = combined_rest_cohort_df.sort_values(
        by=[TARGET, "prep_cohort"]
    )
    combined_rest_cohort_df = combined_rest_cohort_df.drop_duplicates(
        subset=["prep_cohort"], keep="last"
    ).reset_index(drop=True)
    META_COL_NUMS = config.META_COL_NUMS
    # combined_selected_meta_df = combined_selected_df.iloc[:, :META_COL_NUMS]
    # combined_selected_meta_df['prep_cohort_number'] = combined_selected_meta_df['prep_cohort_number'] + 1
    # combined_selected_meta_df.to_excel(config.RESULTS_DIR / "Samples_for_Figure3.xlsx")

    meta_vcb = pd.read_excel(
        config.SAMPLE_METADATA_FILE,
        engine="openpyxl",
    )
    meta_vcb = meta_vcb.drop_duplicates(subset=["LIMS-ID1"])
    meta_vcb = meta_vcb[meta_vcb["Include_Y_N"] == "Yes"]
    included_vcb_lims = meta_vcb["LIMS-ID1"].tolist()
    combined_vcb_df = combined_vcb_df[combined_vcb_df.index.isin(included_vcb_lims)]

    le = LabelEncoder()
    le.fit(combined_selected_df[TARGET])
    # train_rest_df, test_rest_df = train_test_split(combined_rest_df, stratify=combined_rest_df[TARGET],
    #                                                test_size=0.1, random_state=0)
    unique_cohorts = combined_rest_df["prep_cohort"].unique()
    train_rest_df = []
    test_rest_df = []
    for c in unique_cohorts:
        print(
            f"{c}: {combined_rest_df[combined_rest_df['prep_cohort'] == c][TARGET].value_counts()}"
        )
        tmp_df = combined_rest_df[combined_rest_df["prep_cohort"] == c]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        i, (train_index, test_index) = next(
            enumerate(
                gss.split(
                    tmp_df,
                    y=tmp_df[TARGET],
                    groups=tmp_df["subject_collaborator_patient_id"],
                )
            )
        )
        train_rest_cohort_df = tmp_df.iloc[train_index]
        test_rest_cohort_df = tmp_df.iloc[test_index]
        train_rest_df.append(train_rest_cohort_df)
        test_rest_df.append(test_rest_cohort_df)
    train_rest_df = pd.concat(train_rest_df)
    test_rest_df = pd.concat(test_rest_df)

    print(f"test shape of samples: {test_rest_df.shape[0]}")
    test_rest_df = test_rest_df.drop_duplicates(
        subset=["subject_collaborator_patient_id"], keep="last"
    )
    print(f"test shape of unique samples: {test_rest_df.shape[0]}")

    print(train_rest_df[TARGET].value_counts())
    train_combined_df = pd.concat([combined_vcb_df, train_rest_df])
    best_scores = []
    print(train_combined_df.groupby([TARGET, "prep_cohort"]).size())
    print(test_rest_df[TARGET].value_counts())
    # print(test_rest_df.groupby([TARGET, 'prep_cohort']).size())

    # traditional
    test_dataset = ProtDataset(
        test_rest_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
        le.transform(test_rest_df[TARGET].to_numpy()),
        prot_id_to_name,
        prot_name_to_id,
        merged_label_map,
    )
    if N_included_clients == N_clients:
        train_dataset = ProtDataset(
            train_combined_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
            le.transform(train_combined_df[TARGET].to_numpy()),
            prot_id_to_name,
            prot_name_to_id,
            merged_label_map,
        )
    elif N_included_clients < N_clients:
        skf = StratifiedKFold(n_splits=N_clients, shuffle=True, random_state=0)
        included_idx = []
        for fold, (train_index, val_index) in enumerate(
            skf.split(combined_rest_cohort_df, combined_rest_cohort_df[TARGET])
        ):
            if fold < N_included_clients:
                included_idx.extend(val_index)
            else:
                break
        selected_cohorts = combined_rest_cohort_df.iloc[included_idx][
            "prep_cohort"
        ].tolist()
        train_df = pd.concat(
            [
                combined_vcb_df,
                train_rest_df[train_rest_df["prep_cohort"].isin(selected_cohorts)],
            ]
        )
        train_dataset = ProtDataset(
            train_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
            le.transform(train_df[TARGET].to_numpy()),
            prot_id_to_name,
            prot_name_to_id,
            merged_label_map,
        )
    else:
        raise ValueError("N_included_clients should be less than or equal to N_clients")

    print(f"running traditional: {len(train_dataset)} {len(test_dataset)}")
    model_training = TrainFedProtNet(
        train_dataset,
        test_dataset,
        hypers,
        save_path=f"{MODEL_PATH}/local_model_{hypers['target']}_traditional.pt",
    )
    score_dict = model_training.run_train_val()
    score_dict["fold"] = "traditional"
    best_scores.append(score_dict)
    confs_df = pd.DataFrame(model_training.get_pred_for_cm(), columns=le.classes_)
    confs_df["pred"] = confs_df.idxmax(axis=1)
    confs_df["true"] = test_rest_df[TARGET].to_numpy()
    confs_df.index = test_rest_df.index
    confs_df.to_csv(config.get_results_path() / f"centralised_pred_{FILE_NAME}.csv")

    # start Fed training
    best_rep = 0
    best_rep_f1 = 0
    all_site_sizes_df = []
    all_confs_local_sites_df = []

    all_confs_df = []
    for repeat in range(N_repeats):
        # initailize global model
        model = TrainFedProtNet(
            train_dataset,
            test_dataset,
            hypers,
            save_path=f"{MODEL_PATH}/{hypers['target']}_fedavg_rep{repeat}.pt",
        )
        model.save_model()

        # all the clients
        model_paths = [
            f"{MODEL_PATH}/local_model_{hypers['target']}_vcb_rep{repeat}.pt"
        ] + [
            f"{MODEL_PATH}/local_model_{hypers['target']}_d2_{i}_rep{repeat}.pt"
            for i in range(N_included_clients)
        ]

        for fed_iter in range(N_iters):
            print(f"Running Fed Iteration {fed_iter}")
            # Train vcb
            print(combined_vcb_df.groupby([TARGET, "prep_cohort"]).size())
            train_dataset = ProtDataset(
                combined_vcb_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
                le.transform(combined_vcb_df[TARGET].to_numpy()),
                prot_id_to_name,
                prot_name_to_id,
                merged_label_map,
            )
            model_training = TrainFedProtNet(
                train_dataset,
                test_dataset,
                hypers,
                load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg_rep{repeat}.pt",
                save_path=f"{MODEL_PATH}/local_model_{hypers['target']}_vcb_rep{repeat}.pt",
            )

            score_dict = model_training.run_train_val()
            score_dict["model"] = "vcb_local"
            score_dict["iter"] = fed_iter
            score_dict["repeat"] = repeat
            best_scores.append(score_dict)

            if fed_iter == 0:
                confs = model_training.get_pred_for_cm()
                confs_df = pd.DataFrame(confs, columns=le.classes_)
                confs_df["pred"] = confs_df.idxmax(axis=1)
                confs_df["true"] = test_rest_df[TARGET].to_numpy()
                confs_df.index = test_rest_df.index
                confs_df.to_csv(
                    config.get_results_path() / f"vcblocal_pred_{FILE_NAME}.csv"
                )

            # Train rest
            # skf = StratifiedKFold(n_splits=N_clients, shuffle=True, random_state=repeat)
            kf = KFold(n_splits=N_clients, shuffle=True, random_state=repeat)
            combined_rest_cohort_tmp_df = combined_rest_cohort_df.sample(
                frac=1, random_state=repeat
            ).reset_index(drop=True)
            # combined_rest_cohort_tmp_df = combined_rest_cohort_df
            for fold, (train_index, val_index) in enumerate(
                kf.split(combined_rest_cohort_tmp_df)
            ):
                selected_cohorts = combined_rest_cohort_tmp_df.iloc[val_index][
                    "prep_cohort"
                ].tolist()
                print(f"running repeat {repeat} fold {fold}: {selected_cohorts}")
                train_fold_df = train_rest_df[
                    train_rest_df["prep_cohort"].isin(selected_cohorts)
                ]
                # train_fold_df = train_rest_df.iloc[val_index]
                # print(train_fold_df.groupby([TARGET, 'prep_cohort']).size())
                if fed_iter == 0:
                    site_size_df = (
                        train_fold_df.groupby([TARGET, "prep_cohort"])
                        .size()
                        .reset_index()
                    )
                    site_size_df["site"] = f"simulated_local_{fold}"
                    site_size_df["repeat"] = repeat
                    all_site_sizes_df.append(site_size_df)

                train_dataset = ProtDataset(
                    train_fold_df.iloc[:, META_COL_NUMS:].fillna(0).to_numpy(),
                    le.transform(train_fold_df[TARGET].to_numpy()),
                    prot_id_to_name,
                    prot_name_to_id,
                    merged_label_map,
                )

                model_training = TrainFedProtNet(
                    train_dataset,
                    test_dataset,
                    hypers,
                    load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg_rep{repeat}.pt",
                    save_path=f"{MODEL_PATH}/local_model_{hypers['target']}_d2_{fold}_rep{repeat}.pt",
                )

                score_dict = model_training.run_train_val()
                score_dict["model"] = f"simulated_local_{fold}"
                score_dict["iter"] = fed_iter
                score_dict["repeat"] = repeat
                best_scores.append(score_dict)

                if fed_iter == N_iters - 1:
                    confs = model_training.get_pred_for_cm()
                    confs_df = pd.DataFrame(confs, columns=le.classes_)
                    confs_df["pred"] = confs_df.idxmax(axis=1)
                    confs_df["true"] = test_rest_df[TARGET].to_numpy()
                    confs_df["repeat"] = repeat
                    confs_df["fold"] = fold
                    confs_df.index = test_rest_df.index
                    all_confs_local_sites_df.append(confs_df)

                if fold + 1 >= N_included_clients:
                    break

            # Aggregate weights
            weight_agg = WeightsAggregation(model_paths=model_paths)
            weight_agg.fed_avg()
            weight_agg.save_model(
                f"{MODEL_PATH}/{hypers['target']}_fedavg_rep{repeat}.pt"
            )
            model_training = TrainFedProtNet(
                test_dataset,
                test_dataset,
                hypers,
                load_model=f"{MODEL_PATH}/{hypers['target']}_fedavg_rep{repeat}.pt",
            )
            score_dict = model_training.predict()
            score_dict["model"] = "fedavg"
            score_dict["iter"] = fed_iter
            score_dict["repeat"] = repeat
            best_scores.append(score_dict)
            print(score_dict)

            if fed_iter == N_iters - 1:
                confs = model_training.get_pred_for_cm()
                confs_df = pd.DataFrame(confs, columns=le.classes_)
                confs_df["pred"] = confs_df.idxmax(axis=1)
                confs_df["true"] = test_rest_df[TARGET].to_numpy()
                confs_df["repeat"] = repeat
                confs_df.index = test_rest_df.index
                all_confs_df.append(confs_df)

        if best_scores[-1]["test_f1"] > best_rep_f1:
            best_rep = repeat
            best_rep_f1 = score_dict["test_f1"]

    all_site_sizes_df = pd.concat(all_site_sizes_df)
    all_site_sizes_df.to_csv(
        config.get_results_path() / f"site_sizes_{FILE_NAME}.csv", index=False
    )
    best_scores_df = pd.DataFrame(best_scores)
    best_scores_df.to_csv(
        config.get_results_path() / f"best_scores_{FILE_NAME}.csv", index=False
    )

    all_confs_local_sites_df = pd.concat(all_confs_local_sites_df)
    all_confs_local_sites_df.to_csv(
        config.get_results_path() / f"local_sites_pred_{FILE_NAME}.csv"
    )

    all_confs_df = pd.concat(all_confs_df)
    all_confs_df.to_csv(config.get_results_path() / f"fedavg_pred_{FILE_NAME}.csv")
