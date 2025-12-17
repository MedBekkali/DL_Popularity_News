import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Mock config class
class Config:
    test_size = 0.2
    val_size = 0.2
    random_state = 42


def split_A_same_indices(X: pd.DataFrame, y_class: np.ndarray, cfg: Config):
    idx = np.arange(len(X))

    idx_train, idx_test = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y_class
    )

    val_relative = cfg.val_size / (1.0 - cfg.test_size)

    idx_train, idx_val = train_test_split(
        idx_train,
        test_size=val_relative,
        random_state=cfg.random_state,
        stratify=y_class[idx_train],
    )

    return {"train": idx_train, "val": idx_val, "test": idx_test}


def split_B_independent(X: pd.DataFrame, y_class: np.ndarray, Y_reg: np.ndarray, cfg: Config):
    idx = np.arange(len(X))
    val_relative = cfg.val_size / (1.0 - cfg.test_size)

    # classification split
    idxc_train, idxc_test = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y_class
    )
    idxc_train, idxc_val = train_test_split(
        idxc_train,
        test_size=val_relative,
        random_state=cfg.random_state,
        stratify=y_class[idxc_train],
    )

    # regression split (stratify by binned y1 for stability)
    bins = pd.qcut(Y_reg[:, 0], q=10, duplicates="drop")
    idxr_train, idxr_test = train_test_split(
        idx, test_size=cfg.test_size, random_state=cfg.random_state, stratify=bins
    )
    idxr_train, idxr_val = train_test_split(
        idxr_train,
        test_size=val_relative,
        random_state=cfg.random_state,
        stratify=bins[idxr_train],
    )

    return {
        "class": {"train": idxc_train, "val": idxc_val, "test": idxc_test},
        "reg": {"train": idxr_train, "val": idxr_val, "test": idxr_test},
    }


def analyze_splits():
    # Create synthetic data similar to your real data
    n_samples = 10000
    idx = np.arange(n_samples)

    # Synthetic classification labels (25% positive like your data)
    y_class = np.zeros(n_samples)
    y_class[:int(0.25 * n_samples)] = 1
    np.random.shuffle(y_class)

    # Synthetic regression targets
    Y_reg = np.column_stack([
        np.random.randn(n_samples) * 0.5 + 0.5,  # y1
        np.random.randn(n_samples) * 5000 + 5000,  # y2
        np.random.randn(n_samples) * 2 + 5  # y3
    ])

    cfg = Config()

    # Create dummy DataFrame
    X_dummy = pd.DataFrame(np.random.randn(n_samples, 10))

    print("=" * 60)
    print("SPLIT DIAGNOSTIC ANALYSIS")
    print("=" * 60)

    # Get splits
    split_A = split_A_same_indices(X_dummy, y_class, cfg)
    split_B = split_B_independent(X_dummy, y_class, Y_reg, cfg)

    # 1. Check if classification test sets are identical
    a_test = split_A["test"]
    b_class_test = split_B["class"]["test"]

    are_identical = np.array_equal(np.sort(a_test), np.sort(b_class_test))
    overlap_count = len(set(a_test) & set(b_class_test))
    overlap_pct = overlap_count / len(a_test) * 100

    print(f"1. Classification Test Sets Comparison:")
    print(f"   - Identical (exact same indices): {are_identical}")
    print(f"   - Overlap: {overlap_count}/{len(a_test)} ({overlap_pct:.1f}%)")

    # 2. Check overlap between classification and regression test sets in Protocol B
    b_reg_test = split_B["reg"]["test"]
    b_class_reg_overlap = len(set(b_class_test) & set(b_reg_test))
    b_class_reg_pct = b_class_reg_overlap / len(b_class_test) * 100

    print(f"\n2. Protocol B Internal Comparison:")
    print(
        f"   - Overlap between class_test and reg_test: {b_class_reg_overlap}/{len(b_class_test)} ({b_class_reg_pct:.1f}%)")

    # 3. Check if validation sets also overlap
    a_val = split_A["val"]
    b_class_val = split_B["class"]["val"]
    b_reg_val = split_B["reg"]["val"]

    print(f"\n3. Validation Sets:")
    print(
        f"   - Protocol A val vs Protocol B class val: {len(set(a_val) & set(b_class_val))}/{len(a_val)} samples overlap")
    print(
        f"   - Protocol B class val vs reg val: {len(set(b_class_val) & set(b_reg_val))}/{len(b_class_val)} samples overlap")

    # 4. Check distribution of y_class in test sets
    print(f"\n4. Class Distribution in Test Sets:")
    print(f"   - Protocol A test: {y_class[a_test].mean():.3f} positive rate")
    print(f"   - Protocol B class test: {y_class[b_class_test].mean():.3f} positive rate")
    print(f"   - Protocol B reg test: {y_class[b_reg_test].mean():.3f} positive rate")

    # 5. Check if any training samples leak into test
    print(f"\n5. Train-Test Leakage Check:")
    print(f"   - Protocol A: Train-Test overlap: {len(set(split_A['train']) & set(a_test))}")
    print(f"   - Protocol B Class: Train-Test overlap: {len(set(split_B['class']['train']) & set(b_class_test))}")
    print(f"   - Protocol B Reg: Train-Test overlap: {len(set(split_B['reg']['train']) & set(b_reg_test))}")

    # 6. Check the critical issue: same random_state with different stratification
    print(f"\n6. Random State Analysis:")
    print(f"   - Using same random_state ({cfg.random_state}) for different stratifications")
    print(f"   - This causes inconsistent behavior!")

    # Run a test with different random states
    print(f"\n7. Testing with Different Random States:")
    cfg2 = Config()
    cfg2.random_state = cfg.random_state + 1  # Different seed
    split_B2 = split_B_independent(X_dummy, y_class, Y_reg, cfg2)
    b2_class_test = split_B2["class"]["test"]

    overlap_with_B1 = len(set(b_class_test) & set(b2_class_test))
    print(
        f"   - With different random_state, overlap between class test sets: {overlap_with_B1}/{len(b_class_test)} samples")

    # The key test: what happens when we use same random_state but different stratification?
    print(f"\n8. KEY ISSUE - Same Random State, Different Stratification:")
    print(f"   In Protocol B, you use the SAME random_state for BOTH splits,")
    print(f"   but DIFFERENT stratification (y_class vs bins).")
    print(f"   This means train_test_split will produce DIFFERENT indices,")
    print(f"   but due to same random_state, the RNG sequence is shared.")
    print(f"   This leads to unpredictable and potentially overlapping splits!")


if __name__ == "__main__":
    analyze_splits()