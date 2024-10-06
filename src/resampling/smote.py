from imblearn.over_sampling import SMOTE
import numpy as np


def apply_smote(X_train_q1, X_train_q2, y_train):
    # Concatenate the two question columns side by side
    X_train_combined = np.hstack((X_train_q1, X_train_q2))

    # Apply SMOTE on the combined data
    smote = SMOTE()
    X_resampled_combined, y_resampled = smote.fit_resample(X_train_combined, y_train)

    # Split the resampled combined data back into q1 and q2
    X_resampled_q1 = X_resampled_combined[:, :X_train_q1.shape[1]]
    X_resampled_q2 = X_resampled_combined[:, X_train_q1.shape[1]:]

    return X_resampled_q1, X_resampled_q2, y_resampled
