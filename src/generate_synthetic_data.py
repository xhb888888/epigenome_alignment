
import numpy as np
import json
import os

np.random.seed(42)

def save_dataset(folder, A):
    os.makedirs(folder, exist_ok=True)
    B = A.copy()

    A_intervals = np.diff(A)
    B_intervals = np.diff(B)

    np.save(os.path.join(folder,"A_coords.npy"), A)
    np.save(os.path.join(folder,"B_coords.npy"), B)
    np.save(os.path.join(folder,"A_intervals.npy"), A_intervals)
    np.save(os.path.join(folder,"B_intervals.npy"), B_intervals)

    meta = {"n_points": int(len(A)), "format":"synthetic DP input"}
    with open(os.path.join(folder,"meta.json"),"w") as f:
        json.dump(meta,f,indent=2)


def make_clean(base):
    return base.copy()

def make_missing(base, rate=0.1):
    mask = np.random.rand(len(base)) > rate
    mask[0] = True
    return base[mask]

def make_extra(base, rate=0.1):
    extra = []
    for i in range(len(base)-1):
        extra.append(base[i])
        if np.random.rand() < rate:
            new = base[i] + (base[i+1]-base[i])//2
            extra.append(new)
    extra.append(base[-1])
    return np.array(sorted(extra))

def make_big_indel(base):
    shift = base.copy()
    shift[len(base)//2:] += 5000
    return shift

if __name__ == "__main__":
    base = np.cumsum(np.random.randint(100,500,size=2000))

    datasets = {
        "clean": make_clean(base),
        "missing_sites": make_missing(base),
        "extra_sites": make_extra(base),
        "big_indel": make_big_indel(base)
    }

    for name, A in datasets.items():
        folder = os.path.join("synthetic_data", name)
        save_dataset(folder, A)

    print("Synthetic datasets generated in ./synthetic_data")
