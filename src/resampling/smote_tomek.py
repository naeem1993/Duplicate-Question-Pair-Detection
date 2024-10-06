from imblearn.combine import SMOTETomek
import numpy as np


def apply_smote_tomek(X_train_q1, X_train_q2, y_train):
    # Concatenate the two question columns side by side
    X_train_combined = np.hstack((X_train_q1, X_train_q2))

    # Apply SMOTE-Tomek on the combined data
    smote_tomek = SMOTETomek()
    X_resampled_combined, y_resampled = smote_tomek.fit_resample(X_train_combined, y_train)

    # Split the resampled combined data back into q1 and q2
    X_resampled_q1 = X_resampled_combined[:, :X_train_q1.shape[1]]
    X_resampled_q2 = X_resampled_combined[:, X_train_q1.shape[1]:]

    return X_resampled_q1, X_resampled_q2, y_resampled
''