import logging
import time
from copy import deepcopy
import math
import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score
import os
import sys

# Ensure project root (one level up from this file's folder) is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.kgrs import KGRS


def nDCG(sorted_items, pos_item, train_pos_item, k=5):
    dcg = 0
    train_pos_item = set(train_pos_item)
    filter_item = set(filter(lambda item: item not in train_pos_item, pos_item))
    max_correct = min(len(filter_item), k)

    train_hit_num = 0
    valid_num = 0
    recommended_items = set()
    for index in range(len(sorted_items)):
        if sorted_items[index] in train_pos_item:
            train_hit_num += 1
        else:
            valid_num += 1

        if (sorted_items[index] in filter_item) and (sorted_items[index] not in recommended_items):
            # Rank starts from 0
            dcg += 1 / math.log2(index - train_hit_num + 2)
            recommended_items.add(sorted_items[index])

        if valid_num >= k:
            break

    idcg = sum([1 / math.log2(i + 2) for i in range(max_correct)])
    return dcg / idcg


def pretty_print_result(auc: float, ndcg5: float, times: tuple[float, float, float, float], header: str = "--- Result ---"):
    init_time, train_time, ctr_time, topk_time = times
    total_time = init_time + train_time + ctr_time + topk_time
    print(f"\n{header}")
    print(f"AUC Score : {auc:.4f}")
    print(f"nDCG@5 Score : {ndcg5:.4f}")
    print("------------------------")
    print(f"⏱ Initialization Time: {init_time:.2f} seconds")
    print(f"⏱ Training Time : {train_time:.2f} seconds")
    print(f"⏱ CTR Evaluation Time: {ctr_time:.2f} seconds")
    print(f"⏱ TopK Evaluation Time: {topk_time:.2f} seconds")
    print("------------------------")
    print(f"⏱ Execution Time: {total_time:.2f} seconds")


def load_data():
    train_pos = np.load("data/train_pos.npy")
    train_neg = np.load("data/train_neg.npy")
    np.random.shuffle(train_pos)
    np.random.shuffle(train_neg)

    all_users = set(train_pos[:, 0]) | set(train_neg[:, 0])
    all_items = set(train_pos[:, 1]) | set(train_neg[:, 1])
    n_user = max(all_users) + 1
    n_item = max(all_items) + 1

    train_pos_len = int(len(train_pos) * 0.8)
    train_neg_len = int(len(train_neg) * 0.8)
    test_pos = train_pos[train_pos_len:]
    test_neg = train_neg[train_neg_len:]
    train_pos = train_pos[:train_pos_len]
    train_neg = train_neg[:train_neg_len]

    return train_pos, train_neg, test_pos, test_neg, n_user, n_item


def get_user_pos_items(train_pos, test_pos):
    user_pos_items = {}
    user_train_pos_items = {}

    for record in train_pos:
        user, item = record[0], record[1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        user_train_pos_items[user].add(item)

    for record in test_pos:
        user, item = record[0], record[1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        if user not in user_pos_items:
            user_pos_items[user] = set()
        user_pos_items[user].add(item)

    return user_pos_items, user_train_pos_items


def evaluate(config: dict | None = None):
    train_pos, train_neg, test_pos, test_neg, n_user, n_item = load_data()
    user_pos_items, user_train_pos_items = get_user_pos_items(train_pos=train_pos, test_pos=test_pos)

    logging.disable(logging.INFO)
    torch.set_num_threads(8)

    auc, ndcg5 = 0, 0
    init_timeout, train_timeout, ctr_timeout, topk_timeout = False, False, False, False
    start_time, init_time, train_time, ctr_time, topk_time = time.time(), 0, 0, 0, 0

    kgrs = KGRS(
        train_pos=deepcopy(train_pos),
        train_neg=deepcopy(train_neg),
        kg_lines=open('data/kg.txt', encoding='utf-8').readlines(),
        n_user=n_user,
        n_item=n_item,
        config=config,
        device_index=-1,
    )

    init_time = time.time() - start_time
    kgrs.training()
    train_time = time.time() - start_time - init_time

    test_data = np.concatenate((deepcopy(test_neg), deepcopy(test_pos)), axis=0)
    np.random.shuffle(test_data)
    test_label = test_data[:, 2]
    test_data = test_data[:, :2]

    kgrs.eval_ctr(test_data)
    scores = kgrs.eval_ctr(test_data=test_data)
    auc = roc_auc_score(y_true=test_label, y_score=scores)
    ctr_time = time.time() - start_time - init_time - train_time

    users = list(user_pos_items.keys())
    user_item_lists = kgrs.eval_topk(users=users)
    ndcg5 = np.mean([
        nDCG(user_item_lists[index], user_pos_items[user], user_train_pos_items[user])
        for index, user in enumerate(users)
    ])

    topk_time = time.time() - start_time - init_time - train_time - ctr_time

    return (
        auc,
        ndcg5,
        init_timeout,
        train_timeout,
        ctr_timeout,
        topk_timeout,
        init_time,
        train_time,
        ctr_time,
        topk_time,
    )


def evaluate_with_config(config: dict):
    auc, ndcg5, _, _, _, _, init_time, train_time, ctr_time, topk_time = evaluate(config=config)
    return {
        "auc": auc,
        "ndcg5": ndcg5,
        "times": (init_time, train_time, ctr_time, topk_time),
    }


def hyperparam_search():
    candidates = [
        {"emb_dim": 32, "l1": False, "margin": 10, "neg_rate": 3.0, "learning_rate": 1e-3, "batch_size": 256, "epoch_num": 35, "eval_batch_size": 1024, "weight_decay": 5e-4},
        {"emb_dim": 64, "l1": True,  "margin": 30, "neg_rate": 2.0, "learning_rate": 5e-4, "batch_size": 256, "epoch_num": 35, "eval_batch_size": 1024, "weight_decay": 5e-4},
        {"emb_dim": 32, "l1": True,  "margin": 30, "neg_rate": 2.0, "learning_rate": 1e-3, "batch_size": 512, "epoch_num": 35, "eval_batch_size": 1024, "weight_decay": 5e-4},
        {"emb_dim": 32, "l1": True,  "margin": 30, "neg_rate": 2.5, "learning_rate": 1e-3, "batch_size": 256, "epoch_num": 50, "eval_batch_size": 1024, "weight_decay": 5e-4},
        {"emb_dim": 48, "l1": False, "margin": 15, "neg_rate": 3.0, "learning_rate": 1e-3, "batch_size": 256, "epoch_num": 35, "eval_batch_size": 1024, "weight_decay": 1e-4},
        {"emb_dim": 24, "l1": True,  "margin": 30, "neg_rate": 3.0, "learning_rate": 1.5e-3, "batch_size": 256, "epoch_num": 35, "eval_batch_size": 1024, "weight_decay": 5e-4},
    ]

    results = []
    for i, cfg in enumerate(candidates, 1):
        print(f"\n=== Trial {i}/{len(candidates)} ===")
        print(f"Config: {cfg}")
        out = evaluate_with_config(cfg)
        # Tampilkan ringkas per trial (tanpa breakdown waktu)
        print(f"AUC={out['auc']:.4f}, nDCG@5={out['ndcg5']:.4f}")
        results.append((out["auc"], out["ndcg5"], cfg, out["times"]))

    best_auc = max(results, key=lambda x: x[0])
    best_ndcg = max(results, key=lambda x: x[1])

    print("\n=== Best by AUC ===")
    print(f"Config: {best_auc[2]}")
    pretty_print_result(best_auc[0], best_auc[1], best_auc[3], header="--- Result ---")

    print("=== Best by nDCG@5 ===")
    print(f"Config: {best_ndcg[2]}")
    pretty_print_result(best_ndcg[0], best_ndcg[1], best_ndcg[3], header="--- Result ---")


if __name__ == '__main__':
    seed_everything(1088, workers=True)

    if len(sys.argv) > 1 and sys.argv[1] == "--tune":
        hyperparam_search()
        sys.exit(0)

    print("Training started...")
    start_time = time.time()

    (
        auc,
        ndcg5,
        init_timeout,
        train_timeout,
        ctr_timeout,
        topk_timeout,
        init_time,
        train_time,
        ctr_time,
        topk_time,
    ) = evaluate()

    print("\n--- Result ---")
    print(f"AUC Score : {auc:.4f}")
    print(f"nDCG@5 Score : {ndcg5:.4f}")
    print("------------------------")
    print(f"⏱ Initialization Time: {init_time:.2f} seconds")
    print(f"⏱ Training Time : {train_time:.2f} seconds")
    print(f"⏱ CTR Evaluation Time: {ctr_time:.2f} seconds")
    print(f"⏱ TopK Evaluation Time: {topk_time:.2f} seconds")
    print("------------------------")
    total_time = time.time() - start_time
    print(f"⏱ Execution Time: {total_time:.2f} seconds")