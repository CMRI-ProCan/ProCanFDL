from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def run_model(model,
              X_train,
              y_train, X_val, y_val, return_preds=False):
    y_conf = []
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    model.fit(X_train, y_train)
    conf = model.predict_proba(X_val)
    # conf = (conf - np.mean(conf)) / np.std(conf)
    y_conf.append(conf)

    y_conf = np.average(np.stack(y_conf), axis=0)
    y_pred = le.inverse_transform(np.argmax(y_conf, axis=1))

    # auc_scores = roc_auc_score(y_val, y_conf, multi_class='ovr', average=None)
    # f1_scores = f1_score(y_val, y_pred, average=None)

    res = {
        # 'AUROC': roc_auc_score(y_val, y_conf, multi_class='ovr'),
        'F1': f1_score(y_val, y_pred, average='macro'),
        'Accuracy': accuracy_score(y_val, y_pred),
        'Top-3 Accuracy': top_k_accuracy_score(y_val, y_conf, k=3, labels=le.classes_),
    }

    # for i in range(len(le.classes_)):
    #     res['AUROC_' + le.classes_[i]] = auc_scores[i]
    #     res['F1_' + le.classes_[i]] = f1_scores[i]

    if not return_preds:
        return res
    else:
        all_preds = {}
        all_preds['LIMS-ID1'] = X_val.index.values

        for i in range(len(le.classes_)):
            all_preds[le.classes_[i]] = y_conf[:, i]
        all_preds['y_true'] = y_val
        all_preds['y_pred'] = y_pred
        all_preds = pd.DataFrame(all_preds)
        # all_preds['Model'] = model_name
        # all_preds['Imputation'] = imputation
        return res, all_preds


def run_online_learning(seed, X_base, X_test, y_test, X_online, y_online, merged_label_map, N_SPLITS=10):
    res_online_df = []
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed).split(X_online, y_online)
    X_train_online = X_base
    first_pred = None
    last_pred = None
    for size in np.arange(0, 1.01, 1 / N_SPLITS):
        model = LogisticRegression(class_weight='balanced', random_state=42)

        if 0 < size < 1:
            train_idx, test_idx = next(skf)
            X_train_online = pd.concat([X_train_online, X_online.iloc[test_idx, :]]).sort_index()
        elif size == 1:
            X_train_online = pd.concat([X_base, X_online]).sort_index()

        y_train_online = X_train_online.index.map(merged_label_map)
        res, all_pred = run_model(model, X_train_online, y_train_online, X_test,
                                  y_test, return_preds=True)
        if size == 0:
            first_pred = all_pred
        elif size == 1:
            last_pred = all_pred
        res['size'] = size
        res_online_df.append(res)
    return pd.DataFrame(res_online_df), first_pred, last_pred
