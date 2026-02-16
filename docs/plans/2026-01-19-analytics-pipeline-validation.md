# Analytics Pipeline Validation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate end-to-end pipeline from cloud experiment results to DoubleML causal analysis outputs.

**Architecture:** Download GCS results → Place in local structure → Run estimation scripts → Verify outputs (plots, tables, logs).

**Tech Stack:** gsutil, Python (doubleml, lightgbm, pandas, matplotlib)

---

## Current State

**Cloud results available:**
```
gs://collusion-exp-1768782033-collusion-experiments/results/exp1/20260119-090340/exp1/data.csv
gs://collusion-exp-1768782033-collusion-experiments/results/exp1/20260119-090340/exp1/param_mappings.json
```

**Estimation scripts expect:**
```
results/exp1/data.csv  ← est1.py reads this
results/exp2/data.csv  ← est2.py reads this
results/exp3/data.csv  ← est3.py reads this
```

**Estimation outputs:**
```
results/exp1/analysis_stdout.txt   # Full analysis log
results/exp1/gate_plots/*.png      # GATE plots
results/exp1/cate_plots/*.png      # CATE plots
```

---

### Task 1: Download Cloud Results to Local

**Files:**
- Create: `results/exp1/data.csv` (from GCS)
- Create: `results/exp1/param_mappings.json` (from GCS)

**Step 1: Download exp1 results from GCS**

Run:
```bash
gsutil cp gs://collusion-exp-1768782033-collusion-experiments/results/exp1/20260119-090340/exp1/data.csv results/exp1/data.csv
gsutil cp gs://collusion-exp-1768782033-collusion-experiments/results/exp1/20260119-090340/exp1/param_mappings.json results/exp1/param_mappings.json
```

Expected: Files downloaded, ~2KB each

**Step 2: Verify data.csv has expected columns**

Run:
```bash
head -1 results/exp1/data.csv
```

Expected: Headers include `avg_rev_last_1000,time_to_converge,avg_regret_of_seller,...`

**Step 3: Verify row count**

Run:
```bash
wc -l results/exp1/data.csv
```

Expected: 11 lines (header + 10 data rows from quick test)

---

### Task 2: Run Estimation Script for Exp1

**Files:**
- Read: `results/exp1/data.csv`
- Create: `results/exp1/analysis_stdout.txt`
- Create: `results/exp1/gate_plots/*.png`
- Create: `results/exp1/cate_plots/*.png`

**Step 1: Run est1.py**

Run:
```bash
cd /Users/pranjal/Code/algorithmic-collusion && python src/estimation/est1.py
```

Expected: "Analysis complete. All logs saved to 'results/exp1/analysis_stdout.txt'."

**Step 2: Verify analysis log created**

Run:
```bash
ls -la results/exp1/analysis_stdout.txt
```

Expected: File exists, size > 1KB

**Step 3: Verify GATE plots created**

Run:
```bash
ls results/exp1/gate_plots/
```

Expected: Files like `gate_plots_avg_rev_last_1000.png`, etc.

**Step 4: Verify CATE plots created**

Run:
```bash
ls results/exp1/cate_plots/
```

Expected: Files like `cate_plots_avg_rev_last_1000.png`, etc.

**Step 5: Check analysis log for ATE results**

Run:
```bash
grep -A5 "Final ATE Estimates" results/exp1/analysis_stdout.txt
```

Expected: Table with ATE coefficients for each outcome

---

### Task 3: Verify Analysis Quality

**Files:**
- Read: `results/exp1/analysis_stdout.txt`

**Step 1: Check for errors in log**

Run:
```bash
grep -i "error\|exception\|warning" results/exp1/analysis_stdout.txt | head -10
```

Expected: No critical errors (some warnings OK)

**Step 2: Check ATE statistical significance**

Run:
```bash
grep -A10 "ATE for outcome" results/exp1/analysis_stdout.txt | head -30
```

Expected: p-values shown for each outcome

**Step 3: Verify plots are valid PNGs**

Run:
```bash
file results/exp1/gate_plots/*.png results/exp1/cate_plots/*.png | head -10
```

Expected: All files identified as "PNG image data"

---

### Task 4: Add Download Command to CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Add section for downloading and running analysis**

Add after "Full Status Check" section:
```markdown
### Running Analytics After Cloud Completion

```bash
# 1. Download results
gsutil cp gs://${PROJECT}-collusion-experiments/results/exp1/TIMESTAMP/exp1/data.csv results/exp1/data.csv

# 2. Run DoubleML estimation
python src/estimation/est1.py
python src/estimation/est2.py
python src/estimation/est3.py

# 3. Check outputs
ls results/exp1/analysis_stdout.txt
ls results/exp1/gate_plots/
ls results/exp1/cate_plots/
```
```

---

### Task 5: Test Full Pipeline with Quick Local Run (Optional)

**Purpose:** Verify pipeline works end-to-end without cloud dependency

**Step 1: Run quick local experiment**

Run:
```bash
python scripts/run_experiment.py --exp 1 --quick
```

Expected: Completes, creates `results/exp1/quick_test/data.csv`

**Step 2: Copy to expected location**

Run:
```bash
cp results/exp1/quick_test/data.csv results/exp1/data.csv
```

**Step 3: Run estimation**

Run:
```bash
python src/estimation/est1.py
```

Expected: Analysis completes, plots generated

---

## Success Criteria

1. Cloud results downloaded successfully
2. est1.py runs without errors
3. GATE plots generated for all outcomes
4. CATE plots generated for all outcomes
5. ATE estimates present in analysis log
6. CLAUDE.md updated with analytics workflow

---

## Notes

- Cloud results use timestamp directories to avoid overwrites
- Estimation scripts require minimum ~10 rows for DoubleML cross-validation
- Paper figures (`e1_reg_bidder.png`, etc.) come from separate/manual process
- For full experiments, increase `n_folds` in estimation scripts
