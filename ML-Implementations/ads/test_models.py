"""
Test and compare all advertising models on the same dataset.
Currently implemented: Wide & Deep Learning (WDL), ESMM
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import pandas as pd
import time

from wdl import WDL
from esmm import ESMM
from dcn import DCN
from din import DIN
from mmoe import MMoE
from ple import PLE
from dien import DIEN
from deepfm import DeepFM
from autoint import AutoInt


def generate_esmm_dataset(n_samples=30000, feature_dim=30):
    """
    Generate synthetic impression-space data for ESMM.

    The funnel:  impression → click (~15% CTR) → conversion (~12% CVR given click)
    So CTCVR ≈ 0.15 × 0.12 = ~1.8% of all impressions end in conversion.

    Signals are encoded as logits with a fixed bias so probabilities stay spread out
    (no post-hoc calibration that would compress the signal variance).

    Labels:
        y_ctr   : 1 if clicked
        y_ctcvr : 1 if clicked AND converted  (y_ctcvr[i] <= y_ctr[i] always)
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, feature_dim).astype(np.float32)

    # CTR logit: bias ≈ logit(0.15) ≈ -1.73 gives ~15% average CTR
    # Strong coefficients keep signal-to-noise ratio high
    ctr_logit = (
        1.2 * X[:, 0] + 0.9 * X[:, 1] - 0.7 * X[:, 2]
        + 0.6 * X[:, 0] * X[:, 1]   # interaction
        - 1.73                        # bias → ~15% mean CTR
    )
    p_ctr = 1 / (1 + np.exp(-ctr_logit))
    y_ctr = (np.random.rand(n_samples) < p_ctr).astype(np.float32)

    # CVR logit: uses different features (purchase intent ≠ click intent)
    # bias ≈ logit(0.12) ≈ -1.99 gives ~12% average CVR given click
    cvr_logit = (
        1.4 * X[:, 3] + 1.0 * X[:, 4] - 0.8 * X[:, 5]
        + 0.7 * X[:, 3] * X[:, 4]   # interaction
        - 1.99                        # bias → ~12% mean CVR
    )
    p_cvr = 1 / (1 + np.exp(-cvr_logit))
    y_cvr_given_click = (np.random.rand(n_samples) < p_cvr).astype(np.float32)

    # Conversion only happens if there was a click
    y_ctcvr = (y_ctr * y_cvr_given_click).astype(np.float32)

    actual_ctr   = y_ctr.mean()
    actual_ctcvr = y_ctcvr.mean()
    actual_cvr   = y_ctcvr.sum() / max(y_ctr.sum(), 1)
    print(f"CTR: {actual_ctr:.2%}  |  CVR (given click): {actual_cvr:.2%}  |  CTCVR: {actual_ctcvr:.2%}")

    return X, y_ctr, y_ctcvr


def generate_dataset(n_samples=10000, wide_dim=10, deep_dim=20, target_ctr=0.08):
    """
    Generate synthetic CTR-like dataset.

    Args:
        n_samples: Number of samples
        wide_dim: Wide feature dimension
        deep_dim: Deep feature dimension
        target_ctr: Target click-through rate (default: 8%, realistic for ads)
    """
    np.random.seed(42)

    # Wide: sparse binary features (one-hot encoded categories)
    X_wide = np.random.binomial(1, 0.3, size=(n_samples, wide_dim)).astype(np.float32)

    # Deep: dense continuous features (embeddings, numerical)
    X_deep = np.random.randn(n_samples, deep_dim).astype(np.float32)

    # Create scores with patterns
    wide_score = np.sum(X_wide[:, :5] * [0.5, -0.3, 0.4, -0.2, 0.6], axis=1)
    deep_score = 0.3 * X_deep[:, 0] * X_deep[:, 1] + 0.4 * np.sin(X_deep[:, 2])

    # Combine scores and adjust threshold for realistic CTR
    combined_score = wide_score + deep_score

    # Find threshold that gives target CTR
    probabilities = 1 / (1 + np.exp(-combined_score))
    threshold = np.percentile(probabilities, (1 - target_ctr) * 100)

    y = (probabilities > threshold).astype(np.float32)

    # Verify CTR
    actual_ctr = y.mean()
    print(f"Generated CTR: {actual_ctr:.2%} (target: {target_ctr:.2%})")

    return X_wide, X_deep, y


def evaluate_model(model, X_wide, X_deep, y, name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_wide, X_deep).flatten()
    y_pred_class = (y_pred >= 0.5).astype(int)

    return {
        'Model': name,
        'Accuracy': accuracy_score(y, y_pred_class),
        'Precision': precision_score(y, y_pred_class),
        'Recall': recall_score(y, y_pred_class),
        'F1': f1_score(y, y_pred_class),
        'AUC': roc_auc_score(y, y_pred),
        'LogLoss': log_loss(y, y_pred)
    }




def main():
    """Test all models on the same dataset."""
    print("="*70)
    print("Testing Advertising Models")
    print("="*70)

    # Generate data
    print("\nGenerating dataset...")
    X_wide, X_deep, y = generate_dataset(n_samples=10000, wide_dim=15, deep_dim=25)
    X_wide_train, X_wide_test, X_deep_train, X_deep_test, y_train, y_test = train_test_split(
        X_wide, X_deep, y, test_size=0.2, random_state=42
    )
    print(f"Train: {len(y_train)}, Test: {len(y_test)}, Positive rate: {y_train.mean():.1%}")

    results = []

    # ============================================================================
    # Wide & Deep Learning
    # ============================================================================
    print("\n" + "-"*70)
    print("1. Wide & Deep (Adam)")
    print("-"*70)
    start = time.time()
    model = WDL(wide_input_dim=15, deep_input_dim=25, deep_hidden_units=[128, 64, 32])
    model.fit(X_wide_train, X_deep_train, y_train, epochs=10, batch_size=64, verbose=0)
    train_time = time.time() - start

    metrics = evaluate_model(model, X_wide_test, X_deep_test, y_test, 'WDL (Adam)')
    metrics['TrainTime'] = f"{train_time:.1f}s"
    results.append(metrics)
    print(f"AUC: {metrics['AUC']:.4f}, Accuracy: {metrics['Accuracy']:.4f}, Time: {train_time:.1f}s")

    # ============================================================================
    # Wide & Deep with Dual Optimizers (FTRL + Adagrad) - Experimental
    # ============================================================================
    print("\n" + "-"*70)
    print("2. Wide & Deep (FTRL + Adagrad) - Experimental")
    print("-"*70)
    print("NOTE: Dual optimizers designed for production with highly sparse features.")
    print("      Adam optimizer is recommended for most use cases.")
    print("-"*70)
    start = time.time()
    model_dual = WDL(wide_input_dim=15, deep_input_dim=25, deep_hidden_units=[128, 64, 32], learning_rate=0.001)
    model_dual.fit_with_dual_optimizers(X_wide_train, X_deep_train, y_train,
                                       epochs=10, batch_size=64, verbose=0)
    train_time = time.time() - start

    metrics = evaluate_model(model_dual, X_wide_test, X_deep_test, y_test, 'WDL (Dual)')
    metrics['TrainTime'] = f"{train_time:.1f}s"
    results.append(metrics)
    print(f"AUC: {metrics['AUC']:.4f}, Accuracy: {metrics['Accuracy']:.4f}, Time: {train_time:.1f}s")

    # ============================================================================
    # ESMM — Entire Space Multi-Task Model (CTR + CVR)
    # ============================================================================
    print("\n" + "="*70)
    print("ESMM: Entire Space Multi-Task Model")
    print("="*70)
    print("Generating impression-space dataset (CTR + CVR funnel)...")
    FEAT_DIM = 30
    X_esmm, y_ctr_all, y_ctcvr_all = generate_esmm_dataset(
        n_samples=30000, feature_dim=FEAT_DIM
    )
    (X_tr, X_te,
     y_ctr_tr, y_ctr_te,
     y_ctcvr_tr, y_ctcvr_te) = train_test_split(
        X_esmm, y_ctr_all, y_ctcvr_all, test_size=0.2, random_state=42
    )
    print(f"Train: {len(y_ctr_tr)}, Test: {len(y_ctr_te)}")

    start = time.time()
    esmm = ESMM(
        input_dim=FEAT_DIM,
        shared_units=[],           # no shared bottleneck; let towers specialise freely
        ctr_tower_units=[128, 64],
        cvr_tower_units=[128, 64],
        dropout_rate=0.1,
        learning_rate=0.001,
    )
    esmm.fit(X_tr, y_ctr_tr, y_ctcvr_tr, epochs=20, batch_size=256, verbose=0)
    train_time = time.time() - start

    preds = esmm.predict(X_te)
    p_ctr_pred   = preds["p_ctr"].flatten()
    p_cvr_pred   = preds["p_cvr"].flatten()
    p_ctcvr_pred = preds["p_ctcvr"].flatten()

    ctr_auc   = roc_auc_score(y_ctr_te, p_ctr_pred)
    ctcvr_auc = roc_auc_score(y_ctcvr_te, p_ctcvr_pred)

    # CVR AUC is evaluated only on clicked samples (where CVR label is meaningful)
    clicked_mask = y_ctr_te == 1
    y_cvr_te = y_ctcvr_te[clicked_mask]   # conversion given click
    p_cvr_clicked = p_cvr_pred[clicked_mask]
    cvr_auc = roc_auc_score(y_cvr_te, p_cvr_clicked) if y_cvr_te.sum() > 0 else float("nan")

    esmm_results = {
        "Model": "ESMM",
        "CTR AUC": round(ctr_auc, 4),
        "CVR AUC (on clicks)": round(cvr_auc, 4),
        "CTCVR AUC": round(ctcvr_auc, 4),
        "TrainTime": f"{train_time:.1f}s",
    }
    print(f"\nCTR  AUC : {ctr_auc:.4f}")
    print(f"CVR  AUC : {cvr_auc:.4f}  (evaluated on clicked samples only)")
    print(f"CTCVR AUC: {ctcvr_auc:.4f}")
    print(f"Train time: {train_time:.1f}s")

    # ============================================================================
    # DCN — Deep & Cross Network
    # DCN takes a single stacked feature vector (wide + deep concatenated).
    # ============================================================================
    print("\n" + "="*70)
    print("DCN: Deep & Cross Network")
    print("="*70)
    # Concatenate wide and deep features into a single vector for DCN
    X_dcn_train = np.concatenate([X_wide_train, X_deep_train], axis=1)
    X_dcn_test  = np.concatenate([X_wide_test,  X_deep_test],  axis=1)
    INPUT_DIM = X_dcn_train.shape[1]  # 15 + 25 = 40

    dcn_configs = [
        dict(version='v2', structure='parallel', num_cross_layers=3, label='DCN-v2 Parallel'),
        dict(version='v2', structure='stacked',  num_cross_layers=3, label='DCN-v2 Stacked'),
        dict(version='v1', structure='parallel', num_cross_layers=3, label='DCN-v1 Parallel'),
    ]

    for cfg in dcn_configs:
        label = cfg.pop('label')
        print(f"\n--- {label} ---")
        start = time.time()
        dcn = DCN(
            input_dim=INPUT_DIM,
            deep_units=[128, 64, 32],
            dropout_rate=0.1,
            learning_rate=0.001,
            **cfg,
        )
        dcn.fit(X_dcn_train, y_train, epochs=10, batch_size=64, verbose=0)
        train_time = time.time() - start

        y_pred = dcn.predict(X_dcn_test).flatten()
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, (y_pred >= 0.5).astype(int))
        metrics = {
            'Model': label,
            'Accuracy': acc,
            'Precision': precision_score(y_test, (y_pred >= 0.5).astype(int)),
            'Recall':    recall_score(y_test, (y_pred >= 0.5).astype(int)),
            'F1':        f1_score(y_test, (y_pred >= 0.5).astype(int)),
            'AUC':       auc,
            'LogLoss':   log_loss(y_test, y_pred),
            'TrainTime': f"{train_time:.1f}s",
        }
        results.append(metrics)
        print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}, Time: {train_time:.1f}s")

    # ============================================================================
    # DeepFM — Factorization Machine + DNN with shared embeddings.
    # Uses a categorical dataset (user/item/category fields + dense features)
    # where pairwise FM interactions are the dominant signal. This tests whether
    # the FM second-order term captures user×category affinity without any
    # manual feature engineering.
    # ============================================================================
    print("\n" + "="*70)
    print("DeepFM: Deep Factorization Machine")
    print("="*70)

    # Dataset design: user×category interaction as the primary CTR signal.
    # Key constraint: need enough samples per (user, preferred_cat) pair for
    # the FM to learn the interaction. With N_CATS=5 (P(match)=20%) and
    # N_USERS=200 (150 samples/user), each user gets ~30 match samples — sufficient.
    N_USERS_FM, N_ITEMS_FM, N_CATS_FM = 200, 200, 5
    DENSE_DIM_FM = 5
    N_FM = 30000

    np.random.seed(42)

    # Each user has a fixed preferred category; each item belongs to one category.
    user_pref = np.random.randint(0, N_CATS_FM, N_USERS_FM)
    item_cat  = np.random.randint(0, N_CATS_FM, N_ITEMS_FM + 1)  # index 0 unused

    user_ids_fm = np.random.randint(0,         N_USERS_FM,      N_FM).astype(np.int32)
    item_ids_fm = np.random.randint(1,         N_ITEMS_FM + 1,  N_FM).astype(np.int32)
    cat_ids_fm  = item_cat[item_ids_fm].astype(np.int32)
    dense_fm    = np.random.randn(N_FM, DENSE_DIM_FM).astype(np.float32)

    # CTR signal: user×category match (P(match) = 1/5 = 20%).
    # match → sigmoid(2) ≈ 88%,  no match → sigmoid(−3) ≈ 5%.
    # Large CTR gap matches DIN's oracle (~0.89) so AUC is meaningful.
    is_match   = (user_pref[user_ids_fm] == cat_ids_fm).astype(np.float32)
    ctr_logit  = 5.0 * is_match + 0.3 * dense_fm[:, 0] - 3.0
    p_ctr_fm   = 1.0 / (1.0 + np.exp(-ctr_logit))
    y_fm       = (np.random.rand(N_FM) < p_ctr_fm).astype(np.float32)

    match_ctr    = y_fm[is_match == 1].mean()
    no_match_ctr = y_fm[is_match == 0].mean()
    print(f"CTR: {y_fm.mean():.2%}  |  CTR if user×category matches: {match_ctr:.2%}"
          f"  |  if not: {no_match_ctr:.2%}")

    (uid_tr, uid_te,
     iid_tr, iid_te,
     cid_tr, cid_te,
     dense_tr, dense_te,
     y_fm_tr, y_fm_te) = train_test_split(
        user_ids_fm, item_ids_fm, cat_ids_fm, dense_fm, y_fm,
        test_size=0.2, random_state=42,
    )
    print(f"Train: {len(y_fm_tr)}, Test: {len(y_fm_te)}")

    start = time.time()
    deepfm = DeepFM(
        field_dims=[N_USERS_FM, N_ITEMS_FM, N_CATS_FM],
        embed_dim=16,
        dense_feat_dim=DENSE_DIM_FM,
        dnn_units=[256, 128, 64],
        dropout_rate=0.1,
        learning_rate=0.001,
    )
    deepfm.fit(
        [uid_tr, iid_tr, cid_tr], y_fm_tr,
        dense_feats=dense_tr,
        epochs=25, batch_size=256, verbose=0,
    )
    train_time = time.time() - start

    y_pred_fm = deepfm.predict([uid_te, iid_te, cid_te], dense_feats=dense_te).flatten()
    fm_auc = roc_auc_score(y_fm_te, y_pred_fm)
    fm_acc = accuracy_score(y_fm_te, (y_pred_fm >= 0.5).astype(int))
    results.append({
        'Model':     'DeepFM',
        'Accuracy':  fm_acc,
        'Precision': precision_score(y_fm_te, (y_pred_fm >= 0.5).astype(int)),
        'Recall':    recall_score(y_fm_te, (y_pred_fm >= 0.5).astype(int)),
        'F1':        f1_score(y_fm_te, (y_pred_fm >= 0.5).astype(int)),
        'AUC':       fm_auc,
        'LogLoss':   log_loss(y_fm_te, y_pred_fm),
        'TrainTime': f"{train_time:.1f}s",
    })
    print(f"AUC: {fm_auc:.4f}, Accuracy: {fm_acc:.4f}, Time: {train_time:.1f}s")

    # ============================================================================
    # AutoInt — Automatic Feature Interaction via Self-Attentive Neural Networks
    # Reuses the DeepFM categorical dataset for a direct comparison.
    # AutoInt replaces explicit FM / cross terms with multi-head self-attention
    # over the stacked field embeddings; each layer attends over all n fields.
    # ============================================================================
    print("\n" + "="*70)
    print("AutoInt: Automatic Feature Interaction (Self-Attention)")
    print("="*70)
    print("(Reusing DeepFM dataset for a direct comparison)")

    start = time.time()
    autoint = AutoInt(
        field_dims=[N_USERS_FM, N_ITEMS_FM, N_CATS_FM],
        embed_dim=16,
        dense_feat_dim=DENSE_DIM_FM,
        num_heads=2,
        att_dim=8,       # per-head dim; total = 2×8 = 16 = embed_dim → no W_res
        num_layers=3,
        use_residual=True,
        dnn_units=[],    # pure AutoInt (no parallel DNN)
        dropout_rate=0.1,
        learning_rate=0.001,
    )
    autoint.fit(
        [uid_tr, iid_tr, cid_tr], y_fm_tr,
        dense_feats=dense_tr,
        epochs=25, batch_size=256, verbose=0,
    )
    train_time = time.time() - start

    y_pred_ai = autoint.predict([uid_te, iid_te, cid_te], dense_feats=dense_te).flatten()
    ai_auc = roc_auc_score(y_fm_te, y_pred_ai)
    ai_acc = accuracy_score(y_fm_te, (y_pred_ai >= 0.5).astype(int))
    results.append({
        'Model':     'AutoInt',
        'Accuracy':  ai_acc,
        'Precision': precision_score(y_fm_te, (y_pred_ai >= 0.5).astype(int)),
        'Recall':    recall_score(y_fm_te, (y_pred_ai >= 0.5).astype(int)),
        'F1':        f1_score(y_fm_te, (y_pred_ai >= 0.5).astype(int)),
        'AUC':       ai_auc,
        'LogLoss':   log_loss(y_fm_te, y_pred_ai),
        'TrainTime': f"{train_time:.1f}s",
    })
    print(f"AUC: {ai_auc:.4f}, Accuracy: {ai_acc:.4f}, Time: {train_time:.1f}s")
    print(f"\nComparison on the same categorical dataset:")
    print(f"  DeepFM  AUC: {fm_auc:.4f}")
    print(f"  AutoInt AUC: {ai_auc:.4f}")

    # ============================================================================
    # DIN — Deep Interest Network
    # DIN requires user behavior sequences and a target item ID.
    # ============================================================================
    print("\n" + "="*70)
    print("DIN: Deep Interest Network")
    print("="*70)

    N_ITEMS     = 200    # item vocabulary size
    N_CATS      = 5     # fewer categories → P(target in fav) = 1/5 = 20%, cleaner signal
    MAX_SEQ_LEN = 15    # padded sequence length
    OTHER_DIM   = 8     # other dense features (user profile, context)
    N_DIN       = 30000

    np.random.seed(42)

    # Items evenly distributed across categories: 40 items per category
    items_per_cat = N_ITEMS // N_CATS
    item_to_cat = np.zeros(N_ITEMS + 1, dtype=np.int32)
    item_to_cat[1:] = np.arange(N_ITEMS) // items_per_cat
    cat_to_items = [np.where(item_to_cat == c)[0] for c in range(N_CATS)]

    # Each user has a latent favorite category the model must infer from the sequence.
    user_fav_cat = np.random.randint(0, N_CATS, size=N_DIN)

    # Behavior sequences: 90% items from fav category → clear interest signal.
    seq_lens = np.random.randint(5, MAX_SEQ_LEN + 1, size=N_DIN)
    item_seq = np.zeros((N_DIN, MAX_SEQ_LEN), dtype=np.int32)
    for i, l in enumerate(seq_lens):
        fav_items = cat_to_items[user_fav_cat[i]]
        n_fav   = max(1, round(l * 0.9))
        n_noise = l - n_fav
        seq = np.concatenate([
            np.random.choice(fav_items, n_fav,  replace=True),
            np.random.randint(1, N_ITEMS + 1,    size=n_noise),
        ])
        np.random.shuffle(seq)
        item_seq[i, :l] = seq

    target_items = np.random.randint(1, N_ITEMS + 1, size=N_DIN).astype(np.int32)
    other_feats  = np.random.randn(N_DIN, OTHER_DIM).astype(np.float32)

    # Large CTR gap: target in fav → ~88%, target not in fav → ~5%.
    # P(target in fav) = 1/5 = 20% → oracle AUC ≈ 0.89.
    target_cat = item_to_cat[target_items]
    is_fav = (target_cat == user_fav_cat).astype(np.float32)

    ctr_logit = (
        5.0 * is_fav                # is_fav=1: logit=2 → p≈88%; is_fav=0: logit=-3 → p≈5%
        + 0.4 * other_feats[:, 0]   # minor user signal
        - 3.0
    )
    p_ctr = 1 / (1 + np.exp(-ctr_logit))
    y_din = (np.random.rand(N_DIN) < p_ctr).astype(np.float32)
    click_if_fav     = y_din[is_fav == 1].mean()
    click_if_not_fav = y_din[is_fav == 0].mean()
    print(f"CTR: {y_din.mean():.2%}  |  "
          f"CTR if target=fav: {click_if_fav:.2%}  |  "
          f"CTR if target≠fav: {click_if_not_fav:.2%}")

    (seq_tr, seq_te,
     tgt_tr, tgt_te,
     oth_tr, oth_te,
     y_din_tr, y_din_te) = train_test_split(
        item_seq, target_items, other_feats, y_din, test_size=0.2, random_state=42
    )
    print(f"Train: {len(y_din_tr)}, Test: {len(y_din_te)}")

    start = time.time()
    din = DIN(
        n_items=N_ITEMS,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=16,
        other_feat_dim=OTHER_DIM,
        attention_units=(32,),    # keep attention MLP small for stability
        dnn_units=[128, 64],
        use_dice=False,           # ReLU: simpler and more stable for this task
        dropout_rate=0.1,
        learning_rate=0.001,
    )
    din.fit(seq_tr, tgt_tr, y_din_tr, other_features=oth_tr,
            epochs=25, batch_size=256, verbose=0)
    train_time = time.time() - start

    y_pred_din = din.predict(seq_te, tgt_te, other_features=oth_te).flatten()
    din_auc = roc_auc_score(y_din_te, y_pred_din)
    din_acc = accuracy_score(y_din_te, (y_pred_din >= 0.5).astype(int))
    din_metrics = {
        'Model': 'DIN',
        'Accuracy': din_acc,
        'Precision': precision_score(y_din_te, (y_pred_din >= 0.5).astype(int)),
        'Recall':    recall_score(y_din_te, (y_pred_din >= 0.5).astype(int)),
        'F1':        f1_score(y_din_te, (y_pred_din >= 0.5).astype(int)),
        'AUC':       din_auc,
        'LogLoss':   log_loss(y_din_te, y_pred_din),
        'TrainTime': f"{train_time:.1f}s",
    }
    results.append(din_metrics)
    print(f"AUC: {din_auc:.4f}, Accuracy: {din_acc:.4f}, Time: {train_time:.1f}s")

    # ============================================================================
    # DIEN — Deep Interest Evolution Network
    # Extends DIN with a two-stage RNN: GRU extracts interest states (with
    # auxiliary next-click supervision), then AUGRU evolves them conditioned
    # on the target item. Reuses the exact same dataset as DIN for direct
    # comparison. neg_seq provides random negatives for the auxiliary loss.
    # ============================================================================
    print("\n" + "="*70)
    print("DIEN: Deep Interest Evolution Network")
    print("="*70)
    print("(Reusing DIN dataset for a direct comparison)")

    # Random negative samples for auxiliary loss: uniform over item vocab.
    # In production, negatives would be sampled outside the user's history.
    neg_seq_tr = np.random.randint(1, N_ITEMS + 1, size=seq_tr.shape).astype(np.int32)
    neg_seq_te = np.zeros_like(seq_te)   # ignored during inference

    start = time.time()
    dien = DIEN(
        n_items=N_ITEMS,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=16,           # gru_units defaults to embed_dim (16) → H = D
        other_feat_dim=OTHER_DIM,
        attention_units=(32,),
        dnn_units=[128, 64],
        aux_loss_weight=0.5,
        dropout_rate=0.1,
        learning_rate=0.001,
    )
    dien.fit(seq_tr, tgt_tr, y_din_tr,
             neg_seq=neg_seq_tr, other_features=oth_tr,
             epochs=25, batch_size=256, verbose=0)
    train_time = time.time() - start

    y_pred_dien = dien.predict(seq_te, tgt_te, other_features=oth_te).flatten()
    dien_auc = roc_auc_score(y_din_te, y_pred_dien)
    dien_acc = accuracy_score(y_din_te, (y_pred_dien >= 0.5).astype(int))
    dien_metrics = {
        'Model': 'DIEN',
        'Accuracy': dien_acc,
        'Precision': precision_score(y_din_te, (y_pred_dien >= 0.5).astype(int)),
        'Recall':    recall_score(y_din_te, (y_pred_dien >= 0.5).astype(int)),
        'F1':        f1_score(y_din_te, (y_pred_dien >= 0.5).astype(int)),
        'AUC':       dien_auc,
        'LogLoss':   log_loss(y_din_te, y_pred_dien),
        'TrainTime': f"{train_time:.1f}s",
    }
    results.append(dien_metrics)
    print(f"AUC: {dien_auc:.4f}, Accuracy: {dien_acc:.4f}, Time: {train_time:.1f}s")
    print(f"\nComparison with DIN on the same dataset:")
    print(f"  DIN  AUC: {din_auc:.4f}")
    print(f"  DIEN AUC: {dien_auc:.4f}")

    # ============================================================================
    # MMoE — Multi-gate Mixture of Experts (joint CTR + CVR)
    # Reuses the ESMM dataset: same X, y_ctr, y_ctcvr for a direct comparison.
    # Unlike ESMM (which links tasks via p_ctcvr = p_ctr × p_cvr), MMoE treats
    # both tasks as independent outputs of a shared expert pool.
    # ============================================================================
    print("\n" + "="*70)
    print("MMoE: Multi-gate Mixture of Experts")
    print("="*70)
    print("(Reusing ESMM dataset for a direct multi-task comparison)")

    start = time.time()
    mmoe = MMoE(
        input_dim=FEAT_DIM,
        num_experts=8,
        expert_units=[128, 64],
        task_names=["ctr", "cvr"],   # "cvr" here means CTCVR (same label as ESMM)
        tower_units=[64, 32],
        dropout_rate=0.1,
        learning_rate=0.001,
    )
    mmoe.fit(
        X_tr,
        labels={"ctr": y_ctr_tr, "cvr": y_ctcvr_tr},
        epochs=20, batch_size=256, verbose=0,
    )
    train_time = time.time() - start

    mmoe_preds = mmoe.predict(X_te)
    p_ctr_mmoe   = mmoe_preds["ctr"].flatten()
    p_ctcvr_mmoe = mmoe_preds["cvr"].flatten()

    mmoe_ctr_auc   = roc_auc_score(y_ctr_te, p_ctr_mmoe)
    mmoe_ctcvr_auc = roc_auc_score(y_ctcvr_te, p_ctcvr_mmoe)

    print(f"\nCTR   AUC : {mmoe_ctr_auc:.4f}")
    print(f"CTCVR AUC : {mmoe_ctcvr_auc:.4f}")
    print(f"Train time: {train_time:.1f}s")
    print("\nComparison with ESMM on the same dataset:")
    print(f"  ESMM  — CTR AUC: {ctr_auc:.4f}  CTCVR AUC: {ctcvr_auc:.4f}")
    print(f"  MMoE  — CTR AUC: {mmoe_ctr_auc:.4f}  CTCVR AUC: {mmoe_ctcvr_auc:.4f}")

    # ============================================================================
    # PLE — Progressive Layered Extraction (joint CTR + CVR)
    # Same dataset as MMoE / ESMM.  Key difference from MMoE: each extraction
    # layer has both task-specific experts AND shared experts.  Task k's gate
    # selects only over [task-k experts + shared experts], insulating task-k's
    # specific experts from the pull of other tasks.
    # ============================================================================
    print("\n" + "="*70)
    print("PLE: Progressive Layered Extraction")
    print("="*70)
    print("(Reusing ESMM dataset for a direct multi-task comparison)")

    start = time.time()
    ple = PLE(
        input_dim=FEAT_DIM,
        num_task_experts=3,       # task-specific experts per task
        num_shared_experts=3,     # shared experts across all tasks
        expert_units=[128, 64],
        num_extraction_layers=2,  # L=1 would be plain CGC
        task_names=["ctr", "cvr"],
        tower_units=[64, 32],
        dropout_rate=0.1,
        learning_rate=0.001,
    )
    ple.fit(
        X_tr,
        labels={"ctr": y_ctr_tr, "cvr": y_ctcvr_tr},
        epochs=20, batch_size=256, verbose=0,
    )
    train_time = time.time() - start

    ple_preds = ple.predict(X_te)
    p_ctr_ple   = ple_preds["ctr"].flatten()
    p_ctcvr_ple = ple_preds["cvr"].flatten()

    ple_ctr_auc   = roc_auc_score(y_ctr_te, p_ctr_ple)
    ple_ctcvr_auc = roc_auc_score(y_ctcvr_te, p_ctcvr_ple)

    print(f"\nCTR   AUC : {ple_ctr_auc:.4f}")
    print(f"CTCVR AUC : {ple_ctcvr_auc:.4f}")
    print(f"Train time: {train_time:.1f}s")
    print("\nComparison on the same dataset:")
    print(f"  ESMM  — CTR AUC: {ctr_auc:.4f}  CTCVR AUC: {ctcvr_auc:.4f}")
    print(f"  MMoE  — CTR AUC: {mmoe_ctr_auc:.4f}  CTCVR AUC: {mmoe_ctcvr_auc:.4f}")
    print(f"  PLE   — CTR AUC: {ple_ctr_auc:.4f}  CTCVR AUC: {ple_ctcvr_auc:.4f}")

    # ============================================================================
    # Results
    # ============================================================================
    print("\n" + "="*70)
    print("Results")
    print("="*70)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv('results.csv', index=False)
    print("\nSaved to results.csv")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Run tests
    main()
