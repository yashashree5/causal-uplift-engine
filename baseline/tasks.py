"""The four reproduction tasks, built exactly as in the authors' notebooks/qini.ipynb."""

from __future__ import annotations

SEED = 82718
N_FOLDS = 5

# Expected rows per task (paper Appendix D.6 and the Hillstrom arm sizes). Used for a warning only.
EXPECTED_ROWS = {
    "hill_womens_vs_none": 42_693,
    "hill_mens_vs_none": 42_613,
    "criteo_sub50k": 50_000,
    "criteo_full": 2_500_000,  # "2.5M stratified subset from sklift"
}

# Task name → (dataset, constructor kwargs, paper column)
TASKS = {
    # Code's "Hill (1)" and "Hill (2)". The paper text labels the arms the other way round,
    # so we always report by arm name.
    "hill_womens_vs_none": (
        "hillstrom",
        {"outcome_col": "visit", "control_arm": "No E-Mail", "treatment_arm": "Womens E-Mail"},
        "Hill(1)",
    ),
    "hill_mens_vs_none": (
        "hillstrom",
        {"outcome_col": "visit", "control_arm": "No E-Mail", "treatment_arm": "Mens E-Mail"},
        "Hill(2)",
    ),
    "criteo_sub50k": (
        "criteo",
        {"outcome_col": "visit", "treatment_col": "treatment", "subsample_max_rows": 50_000},
        "Criteo (Table 2, 50k)",
    ),
    "criteo_full": (
        "criteo",
        {"outcome_col": "visit", "treatment_col": "treatment"},
        "Criteo (Table 7, full)",
    ),
}

DATASET_TASKS = {
    "hillstrom": ["hill_womens_vs_none", "hill_mens_vs_none"],
    "criteo": ["criteo_sub50k", "criteo_full"],
}

# Normalized Qini from the paper. Best method in each column = 1.000.
PAPER = {
    "table2": {
        "hill_womens_vs_none": {
            "causalpfn": 0.992,
            "x_learner": 0.975,
            "s_learner": 1.000,
            "da_learner": 0.985,
            "t_learner": 0.991,
        },
        "hill_mens_vs_none": {
            "causalpfn": 0.968,
            "x_learner": 0.980,
            "s_learner": 1.000,
            "da_learner": 0.964,
            "t_learner": 0.972,
        },
        "criteo_sub50k": {
            "causalpfn": 0.859,
            "x_learner": 1.000,
            "s_learner": 0.881,
            "da_learner": 0.626,
            "t_learner": 0.701,
        },
    },
    "table7": {
        "criteo_full": {
            "causalpfn": 0.939,
            "x_learner": 0.994,
            "s_learner": 1.000,
            "da_learner": 0.955,
            "t_learner": 0.902,
        },
    },
}
TOLERANCE = 0.03


def paper_scores(task: str) -> dict[str, float]:
    for table in PAPER.values():
        if task in table:
            return table[task]
    return {}


def build(task: str):
    """Return the authors' dataset object (iterable over 5 folds) for a task."""
    from baseline.authors import datasets

    hillstrom_cls, criteo_cls = datasets()
    dataset, kwargs, _ = TASKS[task]
    cls = hillstrom_cls if dataset == "hillstrom" else criteo_cls
    return cls(seed=SEED, n_folds=N_FOLDS, **kwargs)
