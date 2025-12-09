import numpy as np

def add_temporal_context(X):
    """
    Append features from t-1 and t+1 to each row t.
    Input:
        X : np.array of shape (n_samples, n_features)
    Output:
        X_new : np.array of shape (n_samples-2, 3*n_features)
    """
    # shifted versions
    X_prev = X[:-2]      # t-1
    X_curr = X[1:-1]     # t
    X_next = X[2:]       # t+1

    # concatenate along feature axis
    X_new = np.concatenate([X_prev, X_curr,X_next], axis=1)
    return X_new