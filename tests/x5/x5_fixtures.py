"""Deterministic, tiny X5-shaped test data. TEST ONLY: never a substitute for the real dataset.

Column names match the real files (verified from the downloaded CSV headers on 2026-11-25).
The generated values are random and carry no information about the real data.
"""

from __future__ import annotations

import csv
import gzip
import random
from datetime import datetime, timedelta
from pathlib import Path

CLIENT_COLUMNS = ["client_id", "first_issue_date", "first_redeem_date", "age", "gender"]
TRAIN_COLUMNS = ["client_id", "treatment_flg", "target"]
TEST_COLUMNS = ["client_id"]
PRODUCT_COLUMNS = [
    "product_id",
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "segment_id",
    "brand_id",
    "vendor_id",
    "netto",
    "is_own_trademark",
    "is_alcohol",
]
PURCHASE_COLUMNS = [
    "client_id",
    "transaction_id",
    "transaction_datetime",
    "regular_points_received",
    "express_points_received",
    "regular_points_spent",
    "express_points_spent",
    "purchase_sum",
    "store_id",
    "product_id",
    "product_quantity",
    "trn_sum_from_iss",
    "trn_sum_from_red",
]


def _hex_id(rng: random.Random, prefix: str = "") -> str:
    return (prefix + "".join(rng.choice("0123456789abcdef") for _ in range(10)))[:10]


def build_tiny_x5(
    out_dir: Path,
    *,
    n_clients: int = 200,
    n_products: int = 40,
    n_stores: int = 6,
    seed: int = 7,
    gzip_files: bool = False,
) -> dict[str, Path]:
    """Write the five logical tables as CSV (or ``.csv.gz``) into ``out_dir``; return their paths.

    Properties relied on by tests:
    * client IDs are 10-char hex strings, some with leading zeros;
    * half of the clients are labeled (uplift_train), half are unlabeled (uplift_test);
    * ~10% of clients have no purchases;
    * transactions have several product rows that repeat ``purchase_sum`` and the points columns,
      and ``trn_sum_from_iss`` values that sum to ``purchase_sum`` (by construction here).
    """
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    client_ids: list[str] = []
    seen: set[str] = set()
    while len(client_ids) < n_clients:
        prefix = "0000" if len(client_ids) % 5 == 0 else ""
        cid = _hex_id(rng, prefix)
        if cid not in seen:
            seen.add(cid)
            client_ids.append(cid)
    client_ids.sort()

    products = [_hex_id(rng) for _ in range(n_products)]
    stores = [_hex_id(rng) for _ in range(n_stores)]

    clients_rows = []
    for i, cid in enumerate(client_ids):
        issue = datetime(2017, 1, 1) + timedelta(days=rng.randint(0, 600), seconds=rng.randint(0, 86399))
        redeem = "" if i % 4 == 0 else (issue + timedelta(days=rng.randint(1, 100))).strftime("%Y-%m-%d %H:%M:%S")
        age = rng.choice([18, 25, 33, 41, 52, 67, 79, 120, -5])
        clients_rows.append([cid, issue.strftime("%Y-%m-%d %H:%M:%S"), redeem, age, rng.choice("FMU")])

    labeled = client_ids[::2]
    unlabeled = client_ids[1::2]
    train_rows = [[cid, rng.randint(0, 1), rng.randint(0, 1)] for cid in labeled]
    test_rows = [[cid] for cid in unlabeled]

    product_rows = [
        [
            pid,
            _hex_id(rng),
            _hex_id(rng),
            _hex_id(rng),
            _hex_id(rng),
            float(rng.randint(1, 300)),
            _hex_id(rng),
            _hex_id(rng),
            round(rng.random() * 2, 3),
            rng.randint(0, 1),
            0,
        ]
        for pid in products
    ]

    purchase_rows: list[list] = []
    tx_counter = 0
    start = datetime(2018, 9, 1, 8, 0, 0)
    for i, cid in enumerate(client_ids):
        if i % 10 == 9:
            continue  # clients with no purchase history
        for _ in range(rng.randint(1, 5)):
            tx_counter += 1
            tx_id = _hex_id(rng)
            ts = start + timedelta(days=rng.randint(0, 150), seconds=rng.randint(0, 86399))
            if tx_counter % 7 == 0:
                ts = start + timedelta(days=3)  # deliberate timestamp ties across clients
            store = rng.choice(stores)
            n_items = rng.randint(1, 4)
            amounts = [float(rng.randint(10, 300)) for _ in range(n_items)]
            total = float(sum(amounts))
            received = float(rng.choice([0.0, 2.0, 5.1, 10.0]))
            spent = float(rng.choice([0.0, 0.0, 0.0, -20.0]))
            for amount in amounts:
                purchase_rows.append(
                    [
                        cid,
                        tx_id,
                        ts.strftime("%Y-%m-%d %H:%M:%S"),
                        received,
                        0.0,
                        spent,
                        0.0,
                        total,
                        store,
                        rng.choice(products),
                        float(rng.randint(1, 3)),
                        amount,
                        "" if rng.random() < 0.8 else 5.0,
                    ]
                )

    tables = {
        "clients": (CLIENT_COLUMNS, clients_rows),
        "uplift_train": (TRAIN_COLUMNS, train_rows),
        "uplift_test": (TEST_COLUMNS, test_rows),
        "products": (PRODUCT_COLUMNS, product_rows),
        "purchases": (PURCHASE_COLUMNS, purchase_rows),
    }
    paths: dict[str, Path] = {}
    for name, (columns, rows) in tables.items():
        path = out_dir / (f"{name}.csv.gz" if gzip_files else f"{name}.csv")
        opener = gzip.open if gzip_files else open
        with opener(path, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(columns)
            writer.writerows(rows)
        paths[name] = path
    return paths


def write_landing_config(tmp_path: Path, *, bucket: str | None = None, expectations: dict | None = None) -> Path:
    """Write a raw-landing YAML pointing at ``tmp_path/data`` and return its path."""
    import yaml

    config = {
        "dataset": {"name": "x5_test", "tables": ["uplift_train", "uplift_test", "clients", "products", "purchases"]},
        "paths": {
            "data_root": str(tmp_path / "data"),
            "downloads_subdir": "downloads",
            "extracted_subdir": "extracted",
            "provenance_file": "download_provenance.json",
        },
        "gcp": {"project": None, "bucket": bucket, "pubsub_topic": "x5-transactions"},
        "raw_landing": {
            "object_prefix": "raw/batch",
            "manifests_subdir": "manifests",
            "upload_chunk_bytes": 262144,
            "required_columns": {
                "uplift_train": TRAIN_COLUMNS,
                "uplift_test": TEST_COLUMNS,
                "clients": CLIENT_COLUMNS,
                "products": PRODUCT_COLUMNS,
                "purchases": PURCHASE_COLUMNS,
            },
            "expectations": expectations or {},
        },
    }
    path = tmp_path / "landing.yaml"
    path.write_text(yaml.safe_dump(config))
    return path
