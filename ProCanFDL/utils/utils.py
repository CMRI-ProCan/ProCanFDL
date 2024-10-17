def remove_by_missing_rate(X_train, X_test, threshold):
    missing_rate_train_df = X_train.isna().mean().to_frame(name="missing_rate")
    missing_rate_test_df = X_test.isna().mean().to_frame(name="missing_rate")
    selected_proteins = missing_rate_train_df[missing_rate_train_df["missing_rate"] < threshold].index.tolist()
    selected_proteins = list(set(selected_proteins).intersection(
        missing_rate_test_df[missing_rate_test_df["missing_rate"] < threshold].index.tolist()))
    return selected_proteins