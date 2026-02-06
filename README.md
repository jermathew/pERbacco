# Entity Resolution via Batched Oracle Queries

This repository contains the source code accompanying the paper:

**Entity Resolution via Batched Oracle Queries**

The code supports the experimental evaluation presented in the paper and allows reproducing the reported results.

---

## Contents

The repository includes implementations of the proposed and baseline algorithms, as well as scripts for computing theoretical bounds used in the analysis.

---

## Computing Bounds on Φ

To compute the upper and lower bounds of the function Φ defined in Equation (3) of the paper, for all datasets reported in Table 2, run:

```bash
python compute_bounds_phi.py
```

---

## Running Batched Entity Resolution Algorithms

To run the batched entity resolution algorithms **pERbacco**, **pERbac**, and **Online** on a given dataset with batch size equal to 10, execute:

```bash
python multiple_pERbacco.py --dataset datasetname
```

where `datasetname` specifies the target dataset.

**Note:** For `datasetname = "cora"` it should take a few minutes.

---

## Running Experiments on All Datasets

To apply **pERbacco**, **pERbac**, and **Online** to all datasets reported in Table 2 with batch size equal to 10, run:

```bash
python multiple_pERbacco.py --dataset all
```

**Note:** This experiment is computationally expensive and not parallelized. Running it may take several days.

---

## Running Individual Algorithms

The script `perbacco.py` allows running individual algorithms with fine-grained control over parameters.

### pERbac

To run **pERbac** on a given dataset with batch size `batch_size`, execute:

```bash
python perbacco.py \
  --dataset datasetname \
  --batch_size batch_size \
  --alg_community "False" \
  --lambda_w "False" \
  --mu_benefit "brmean" \
  --optimal "False" \
  --synth_precision "False"
```

### Online

To run **Online** on a given dataset with batch size `batch_size`, execute:

```bash
python perbacco.py \
  --dataset datasetname \
  --batch_size batch_size \
  --alg_community "False" \
  --lambda_w "False" \
  --mu_benefit "brmax" \
  --optimal "False" \
  --synth_precision "False"
```

### pERbacco (Proposed Method)

To run **pERbacco** with the parameter configuration described in the paper, execute:

```bash
python perbacco.py \
  --dataset datasetname \
  --batch_size batch_size \
  --alg_community "louvain" \
  --lambda_w "0.05" \
  --mu_benefit "brmean" \
  --optimal "False" \
  --synth_precision "False"
```

### Suboptimal

To run **SubOptimal** with batch size `batch_size`, execute:

```bash
python perbacco.py \
  --dataset datasetname \
  --batch_size batch_size \
  --alg_community "False" \
  --lambda_w "False" \
  --mu_benefit "brmax" \
  --optimal "True" \
  --synth_precision "False"
```

---

## Plot Generation

To generate the plots corresponding to Figure 3 and Figure 4 of the paper for a given dataset and batch size, run:

```bash
python make_plot.py --dataset datasetname --batch_size batch_size
```

---

## Reproducibility

All experiments reported in the paper can be reproduced using the scripts provided in this repository with the same parameter settings described in the experimental evaluation section.
