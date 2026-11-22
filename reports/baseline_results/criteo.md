# Criteo Uplift v2.1 baseline results

The **50,000-row** and **2,500,000-row** benchmarks use separate stratified samples we drew from Criteo's verified **Uplift v2.1** archive. The authors' scikit-uplift 10% archive returned HTTP 404. The paper scores below are reference values measured on the authors' cohorts. [Official Criteo v2.1 archive](https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz) · [Paper Tables 2 and 7](https://arxiv.org/html/2506.07918v2)

[Aggregate scores for both cohorts (CSV)](criteo.csv)

The gap is the run's normalized Qini minus the published normalized Qini. PASS means the absolute gap is at most 0.03, using the unrounded run score. TEA-6 sets this tolerance for CausalPFN. Comparator statuses are shown for context. [TEA-6](https://linear.app/298a-team4-causal-inference-and/issue/TEA-6)

## 50,000 rows from Criteo v2.1

The [executed Criteo notebook](../../notebooks/baseline_criteo.ipynb) ran all 25 method-folds in **124.6 seconds** after data preparation. Each fold had 40,000 training and 10,000 held-out rows. CausalPFN ranked **4 out of 5** with normalized Qini **0.841598**. The top method was **X-learner**.

| Method | Qini | Normalized Qini | Table 2 reference | Gap | ±0.03 |
| --- | ---: | ---: | ---: | ---: | :---: |
| X-learner | 0.767801 | 1.000000 | 1.000 | +0.000000 | PASS |
| DA-learner | 0.767416 | 0.999499 | 0.626 | +0.373499 | FAIL |
| T-learner | 0.720935 | 0.938960 | 0.701 | +0.237960 | FAIL |
| CausalPFN | 0.646180 | 0.841598 | 0.859 | -0.017402 | PASS |
| S-learner | 0.434844 | 0.566349 | 0.881 | -0.314651 | FAIL |

CausalPFN passes the numerical comparison with Table 2 on our v2.1 sample.

We used the authors' stratified subsampling procedure with the full v2.1 archive as its sampling frame. The notebook's root seed 82718 generated sampling random state 973774917 and fold random state 4112872775. [Authors' sampling and fold code](https://github.com/vdblm/CausalPFN/blob/7da4afa35affea0f1fb58893e9f41e87dafdc7eb/benchmarks/base.py#L96-L121)

## 2,500,000 rows from Criteo v2.1

CausalPFN led the five methods with normalized Qini **1.000000**. Its raw Qini of **0.820425** was ahead of the X-learner at **0.810781**. Each method used five folds with 2,000,000 training and 500,000 held-out rows per fold. All 25 method-folds were verified. The benchmark took **2 hours and 9 seconds**, excluding sample preparation.

| Method | Qini | Normalized Qini | Table 7 reference | Gap | ±0.03 |
| --- | ---: | ---: | ---: | ---: | :---: |
| CausalPFN | 0.820425 | 1.000000 | 0.939 | +0.061000 | FAIL |
| X-learner | 0.810781 | 0.988246 | 0.994 | -0.005754 | PASS |
| DA-learner | 0.706372 | 0.860983 | 0.955 | -0.094017 | FAIL |
| T-learner | 0.646337 | 0.787808 | 0.902 | -0.114192 | FAIL |
| S-learner | 0.458563 | 0.558934 | 1.000 | -0.441066 | FAIL |

CausalPFN exceeds the Table 7 reference by 0.061, outside tolerance. It led on our stratified v2.1 cohort, so its normalized score is 1.000. The S-learner led on the authors' cohort and provides the normalization denominator in Table 7.

We sampled without replacement in each treatment-by-visit group using NumPy PCG64 seed `20261120`. We allocated exactly 2,500,000 rows proportionally to the four group sizes by largest remainder and kept the selected rows in source order. We froze the cohort before forming five shuffled stratified folds using random state `973774917`, the first draw from the paper's root seed `82718`.

The [cohort preparation module](../../baseline/prepare_criteo_v21.py) and [GPU benchmark module](../../baseline/run_criteo_v21.py) reproduce the sampling and run procedure. Regenerating the cohort matched the archived SHA-256 checksums for the features, treatment, outcome, source row IDs, and fold assignments. [Setup and commands](../../baseline/README.md#reproduce-the-25m-criteo-v21-run)

## Tied uplift predictions

The S-learner predicted zero uplift for **47,676 of 50,000** rows in the smaller run and for **every row** in the 2.5M run. With all rows tied in the 2.5M run, its Qini is entirely determined by row ordering. Ordering tied rows by ascending or descending cohort row ID changed its raw Qini from **0.233** to **0.881** using the same predictions. The relative order of the DA-learner and T-learner also changed at 2.5M. The X-learner still led at 50k and CausalPFN still led at 2.5M under both checks. The tables use the authors' scoring implementation and the tie order from NumPy `1.26.4`. [Scoring code](../../eval/qini.py)

## Run details

- Both cohorts use randomized `treatment` and website `visit`, following the authors' notebook. The paper's prose describes conversion. The shared v2.1 source has 13,979,592 rows and SHA-256 `2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc`. [Authors' notebook](https://github.com/vdblm/CausalPFN/blob/7da4afa35affea0f1fb58893e9f41e87dafdc7eb/notebooks/qini.ipynb) · [Source archive and checksum](https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz)
- The two runs use the benchmark method implementations from team commit `2766650`, upstream CausalPFN `7da4afa`, the published checkpoint, and GPU scoring on an RTX 4080 SUPER. The 2.5M run reused repeated FAISS queries in bounded chunks. We checked the retrieved neighbor labels against the pinned FAISS search on test cases that included historical Criteo weak-effect queries and tied values.
- The 50k run executed the classical methods in branch order (X, S, DA, T). The 2.5M run used the authors' notebook order for these methods (T, S, X, DA). That order affects the random-number sequence used by the comparators.
- Qini is calculated from the pooled held-out predictions. Normalized Qini divides each score by the best score among the five methods in that cohort. The ranking change between runs can reflect differences in sampled rows and comparator execution order as well as cohort size. [Scoring code](../../eval/qini.py)
