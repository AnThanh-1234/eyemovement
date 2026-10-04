import os

# Prevent nested BLAS threads when using multiple processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import glob
import logging
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "testdataset")
RESULTS_DIR = os.path.join(BASE_DIR, "dataset", "results")
TOTAL_CONFIGURATIONS = 108
DEFAULT_WORKERS = 8

logging.getLogger("hmmlearn").setLevel(logging.ERROR)


def kmeans_sse(X, k, max_iter, restarts, rng):
    best = np.inf
    for _ in range(restarts):
        mu = X[rng.choice(len(X), k, replace=False)]
        for _ in range(max_iter):
            distances = ((X[:, None, :] - mu[None]) ** 2).sum(-1)
            clusters = distances.argmin(1)
            new_mu = np.array([
                X[clusters == i].mean(0) if (clusters == i).any() else mu[i]
                for i in range(k)
            ])
            if np.allclose(new_mu, mu):
                break
            mu = new_mu
        sse = sum(((X[clusters == i] - mu[i]) ** 2).sum() for i in range(k))
        best = min(best, sse)
    return best


def find_elbow(ks, sse):
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (sse - sse.min()) / (sse.max() - sse.min() + 1e-12)
    distance = np.abs(
        (y[-1] - y[0]) * x
        - (x[-1] - x[0]) * y
        + x[-1] * y[0]
        - y[-1] * x[0]
    )
    return int(ks[distance.argmax()])


def step0_optimal_k(df):
    rng = np.random.default_rng(42)
    X = df[["x", "y"]].values
    ks = np.arange(1, min(9, len(X) - 1) + 1)
    sse = np.array([kmeans_sse(X, k, 30, 3, rng) for k in ks])
    return find_elbow(ks, sse)


def fit_gmmhmm(X, n_states, config, seed):
    for n_mix in dict.fromkeys([config["N_MIX"], 1]):
        best = None
        for restart in range(config["N_RESTARTS"]):
            try:
                model = GMMHMM(
                    n_components=n_states,
                    n_mix=n_mix,
                    covariance_type=config["COV_TYPE"],
                    n_iter=config["N_ITER"],
                    tol=config["TOL"],
                    min_covar=1e-3,
                    random_state=seed + restart,
                )
                model.fit(X)
                score = model.score(X)
                if np.isfinite(score) and (best is None or score > best[1]):
                    best = (model, score)
            except Exception:
                continue
        if best is not None:
            return best[0]
    return None


def classify_file(job):
    file_path, config = job
    filename = os.path.basename(file_path)
    base_name = os.path.splitext(filename)[0]
    result_path = os.path.join(RESULTS_DIR, f"{base_name}_results.csv")

    arr = np.loadtxt(file_path)
    labels = pd.read_csv(result_path)["manually results"].values.astype(int)
    df = pd.DataFrame(
        {
            "x": arr[:, 0],
            "y": arr[:, 1],
            "v": arr[:, 2],
        }
    )
    if len(arr) != len(labels):
        raise ValueError(f"{filename}: {len(arr)} samples but {len(labels)} labels")

    k = step0_optimal_k(df)
    scale = StandardScaler()
    X1 = scale.fit_transform(df[config["L1_FEATURES"]].values) if config["STANDARDIZE"] else df[config["L1_FEATURES"]].values
    model1 = fit_gmmhmm(X1, k, config, 42)
    if model1 is None:
        raise RuntimeError(f"{filename}: level-1 model did not converge")
    segment_ids = model1.decode(X1, algorithm="viterbi")[1]

    values = df["v"].values
    local = np.full(len(df), -1)
    segment_stats = []
    tiny_segments = []
    for segment in np.unique(segment_ids):
        indexes = np.where(segment_ids == segment)[0]
        if len(indexes) < config["MIN_SEG_LEN"]:
            tiny_segments.append(indexes)
            continue
        X2 = df.loc[indexes, config["L2_FEATURES"]].values
        if config["STANDARDIZE"]:
            X2 = StandardScaler().fit_transform(X2)
        model2 = fit_gmmhmm(X2, 3, config, 142)
        if model2 is None:
            tiny_segments.append(indexes)
            continue
        states = model2.decode(X2, algorithm="viterbi")[1]
        local[indexes] = states
        for state in np.unique(states):
            segment_stats.append((segment, state, values[indexes][states == state].mean()))

    if not segment_stats:
        raise RuntimeError(f"{filename}: no valid segment")
    log_values = np.log10(np.array([item[2] for item in segment_stats]) + 1e-6).reshape(-1, 1)
    kmeans = KMeans(3, n_init=10, random_state=42).fit(log_values)
    order = np.argsort(kmeans.cluster_centers_.ravel())
    ranks = {int(cluster): rank for rank, cluster in enumerate(order)}
    centers = kmeans.cluster_centers_.ravel()[order]
    state_labels = {
        (segment, state): ranks[int(cluster)]
        for (segment, state, _), cluster in zip(segment_stats, kmeans.labels_)
    }

    predictions = np.full(len(df), -1)
    for index in range(len(df)):
        if local[index] >= 0:
            predictions[index] = state_labels.get((segment_ids[index], local[index]), 0)
    for indexes in tiny_segments:
        distances = np.abs(
            np.log10(values[indexes] + 1e-6)[:, None] - centers[None]
        )
        predictions[indexes] = distances.argmin(1)
    return filename, float(np.mean(predictions == labels))


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not files:
        raise FileNotFoundError(f"No input files found in {DATA_DIR}")
    missing = [
        path for path in files
        if not os.path.exists(
            os.path.join(
                RESULTS_DIR,
                f"{os.path.splitext(os.path.basename(path))[0]}_results.csv",
            )
        )
    ]
    if missing:
        raise FileNotFoundError(f"Missing result CSV for {len(missing)} input files")

    config = {
        "L1_FEATURES": ["x", "y", "v"],
        "COV_TYPE": "full",
        "N_MIX": 2,
        "MIN_SEG_LEN": 15,
        "STANDARDIZE": True,
        "L2_FEATURES": ["v"],
        "N_ITER": 25,
        "TOL": 1e-4,
        "N_RESTARTS": 2,
    }
    workers = min(DEFAULT_WORKERS, len(files))
    print(f"Benchmark: {len(files)} files, {workers} workers, 1 configuration")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(classify_file, [(path, config) for path in files]))
    elapsed = time.perf_counter() - start

    print(f"Completed: {len(results)}/{len(files)} files")
    print(f"Benchmark time: {elapsed / 60:.2f} minutes")
    print(f"Estimated full grid ({TOTAL_CONFIGURATIONS} configs): {elapsed * TOTAL_CONFIGURATIONS / 3600:.2f} hours")
    print(f"Mean accuracy in benchmark: {np.mean([item[1] for item in results]):.4f}")


if __name__ == "__main__":
    main()
