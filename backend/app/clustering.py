"""Clustering as a recipe step: find groups of similar rows, and give every row (new ones too) its group.

Standalone on purpose (no app imports): the exported pipeline script includes these exact functions,
so it assigns clusters exactly as the app does.

learn_clusters looks at the TRAINING rows only and returns plain numbers (each column's mean and
spread, and the cluster centres); assign_clusters puts any rows (test rows, a new customer) into the
nearest cluster using only those numbers. Nothing is ever refitted on other rows.

  kmeans  k groups around centres: the classic. Fast, round-ish groups of similar size
  gmm     Gaussian mixture: groups of different shapes and spreads, each row to its likeliest group
"""

import numpy as np
import pandas as pd

AUTO_RANGE = range(2, 9)  # when no number of clusters is given, the best of 2..8 by silhouette
SAMPLE = 3000             # rows used to choose that number (silhouette is slow on big tables)


def _standardise(X, means, stds):
    return ((X - means) / stds).fillna(0.0).to_numpy(dtype=float)  # a blank sits at the column's mean


def learn_clusters(train, columns, method="kmeans", clusters=None, seed=42):
    """Fit on the training rows' `columns`; returns everything needed to assign rows later (plain JSON)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    X = train[columns].astype(float)
    means, stds = X.mean(), X.std().replace(0, 1.0).fillna(1.0)
    Z = _standardise(X, means, stds)
    rng = np.random.default_rng(seed)
    sample = Z[rng.choice(len(Z), size=min(SAMPLE, len(Z)), replace=False)]

    def fit(k):
        if method == "gmm":
            cov = "full" if len(columns) <= 20 else "diag"  # full covariances grow with columns squared
            return GaussianMixture(n_components=k, covariance_type=cov, random_state=seed).fit(Z)
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)

    tried = {}
    if clusters is None:
        for k in AUTO_RANGE:
            if k < len(sample):
                labels = fit(k).predict(sample)
                if len(set(labels)) > 1:
                    tried[k] = float(silhouette_score(sample, labels))
        clusters = max(tried, key=tried.get) if tried else 2
    model = fit(int(clusters))
    labels = model.predict(Z)
    learned = {"columns": list(columns), "method": method, "clusters": int(clusters),
               "means": means.tolist(), "stds": stds.tolist(),
               "sizes": [int((labels == c).sum()) for c in range(int(clusters))], "tried": tried}
    if len(set(labels)) > 1:
        picked = rng.choice(len(Z), size=min(SAMPLE, len(Z)), replace=False)
        learned["silhouette"] = round(float(silhouette_score(Z[picked], labels[picked])), 3)
    if method == "gmm":
        learned.update(weights=model.weights_.tolist(), centres=model.means_.tolist(),
                       covariances=np.asarray(model.covariances_).tolist(), covariance_type=model.covariance_type)
    else:
        learned["centres"] = model.cluster_centers_.tolist()
    return learned


def assign_clusters(df, learned):
    """Each row's cluster (0, 1, ...), from the learned numbers only. Missing columns or blanks count
    as the training average."""
    columns = learned["columns"]
    X = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df else np.nan for c in columns},
                     index=df.index)
    Z = _standardise(X, pd.Series(learned["means"], index=columns), pd.Series(learned["stds"], index=columns))
    centres = np.asarray(learned["centres"])
    if learned["method"] != "gmm":
        return ((Z[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    weights, covariances = np.asarray(learned["weights"]), np.asarray(learned["covariances"])
    scores = np.empty((len(Z), len(weights)))
    for c in range(len(weights)):
        diff = Z - centres[c]
        if learned["covariance_type"] == "diag":
            var = covariances[c]
            log_density = -0.5 * (np.log(2 * np.pi * var).sum() + (diff ** 2 / var).sum(axis=1))
        else:
            sign, logdet = np.linalg.slogdet(covariances[c])
            inverse = np.linalg.inv(covariances[c])
            log_density = -0.5 * (len(columns) * np.log(2 * np.pi) + logdet + np.einsum("ij,jk,ik->i", diff, inverse, diff))
        scores[:, c] = np.log(weights[c]) + log_density
    return scores.argmax(axis=1)
