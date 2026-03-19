.PHONY: smoke smoke1 smoke2 smoke3 smoke3a smoke3b smoke4a smoke4b \
        experiments exp1 exp2 exp3 exp3a exp3b exp4a exp4b \
        analyze analyze1 analyze2 analyze3 analyze3a analyze3b analyze4a analyze4b \
        robust robust1 robust2 robust3 robust3a robust3b robust4a robust4b \
        sensitivity sensitivity1 sensitivity2 sensitivity3a sensitivity3b sensitivity4a sensitivity4b \
        traces tables pdf paper all clean help \
        dive1 dive2 dive3 dive3a dive3b dive4a dive4b \
        check check-freshness discretization budget-robust \
        verify verify-quick calibrate calibrate-multi calibrate-exp3 calibrate-exp4b

PYTHON  := python3
PY      := PYTHONPATH=src $(PYTHON)
REPS    := 2
SEED    := 42
WORKERS :=

# Workers flag (empty = use default)
ifdef WORKERS
  _W := --workers $(WORKERS)
else
  _W :=
endif

help:
	@echo "Usage:"
	@echo "  make smoke          Quick-test all experiments (16 cells, 1 rep)"
	@echo "  make smoke1         Quick-test experiment 1 only"
	@echo "  make exp1           Run experiment 1 (REPS=$(REPS), SEED=$(SEED))"
	@echo "  make experiments    Run all experiments"
	@echo "  make analyze        Run factorial ANOVA for all experiments"
	@echo "  make analyze1       Run factorial ANOVA for experiment 1"
	@echo "  make robust         Run robustness checks for all experiments"
	@echo "  make robust1        Run robustness checks for experiment 1"
	@echo "  make sensitivity    Run global sensitivity analysis for all experiments"
	@echo "  make sensitivity1   Run sensitivity analysis for experiment 1"
	@echo "  make traces         Generate single-run trace plots for paper"
	@echo "  make tables         Generate LaTeX tables + copy figures"
	@echo "  make pdf            Generate standalone results PDF"
	@echo "  make paper          Compile main paper (pdflatex)"
	@echo "  make all            Full pipeline: experiments -> analyze -> robust -> tables -> pdf -> paper"
	@echo "  make check          Verify paper numbers match data"
	@echo "  make discretization Discretization robustness (Exp1-3)"
	@echo "  make budget-robust  Budget robustness (Exp4a)"
	@echo ""
	@echo "Options:"
	@echo "  REPS=5              Replicates per cell (default: 2)"
	@echo "  SEED=123            Random seed (default: 42)"
	@echo "  WORKERS=8           Parallel workers (default: cpu_count/2)"

# ── Smoke Tests ──────────────────────────────────────────────
smoke: smoke1 smoke2 smoke3a smoke3b smoke4a smoke4b

smoke1:
	$(PYTHON) scripts/run_experiment.py --exp 1 --quick --parallel $(_W)

smoke2:
	$(PYTHON) scripts/run_experiment.py --exp 2 --quick --parallel $(_W)

smoke3: smoke3a smoke3b

smoke3a:
	$(PYTHON) scripts/run_experiment.py --exp 3a --quick --parallel $(_W)

smoke3b:
	$(PYTHON) scripts/run_experiment.py --exp 3b --quick --parallel $(_W)

smoke4a:
	$(PYTHON) scripts/run_experiment.py --exp 4a --quick --parallel $(_W)

smoke4b:
	$(PYTHON) scripts/run_experiment.py --exp 4b --quick --parallel $(_W)

# ── Full Experiments ─────────────────────────────────────────
experiments: exp1 exp2 exp3a exp3b exp4a exp4b

exp1:
	$(PYTHON) scripts/run_experiment.py --exp 1 --parallel --replicates $(REPS) --seed $(SEED) $(_W)

exp2:
	$(PYTHON) scripts/run_experiment.py --exp 2 --parallel --replicates $(REPS) --seed $(SEED) $(_W)

exp3: exp3a exp3b

exp3a:
	$(PYTHON) scripts/run_experiment.py --exp 3a --parallel --replicates $(REPS) --seed $(SEED) $(_W)

exp3b:
	$(PYTHON) scripts/run_experiment.py --exp 3b --parallel --replicates $(REPS) --seed $(SEED) $(_W)

exp4a:
	$(PYTHON) scripts/run_experiment.py --exp 4a --parallel --replicates $(REPS) --seed $(SEED) $(_W)

exp4b:
	$(PYTHON) scripts/run_experiment.py --exp 4b --parallel --replicates $(REPS) --seed $(SEED) $(_W)

# ── Factorial Analysis ───────────────────────────────────────
analyze: analyze1 analyze2 analyze3a analyze3b analyze4a analyze4b

analyze1:
	$(PY) src/estimation/est1.py

analyze2:
	$(PY) src/estimation/est2.py

analyze3: analyze3a analyze3b

analyze3a:
	$(PY) src/estimation/est3a.py

analyze3b:
	$(PY) src/estimation/est3b.py

analyze4a:
	$(PY) src/estimation/est4a.py

analyze4b:
	$(PY) src/estimation/est4b.py

# ── Robustness Checks ───────────────────────────────────────
# Note: est*.py now runs robustness automatically after factorial ANOVA.
# These targets are for standalone re-runs of robustness only.
robust: robust1 robust2 robust3a robust3b robust4a robust4b

robust1:
	$(PY) src/estimation/robust_analysis.py --exp 1

robust2:
	$(PY) src/estimation/robust_analysis.py --exp 2

robust3: robust3a robust3b

robust3a:
	$(PY) src/estimation/robust_analysis.py --exp 3a

robust3b:
	$(PY) src/estimation/robust_analysis.py --exp 3b

robust4a:
	$(PY) src/estimation/robust_analysis.py --exp 4a

robust4b:
	$(PY) src/estimation/robust_analysis.py --exp 4b

# ── Sensitivity Analysis ───────────────────────────────────
# Note: est*.py now runs sensitivity automatically after robustness.
# These targets are for standalone re-runs of sensitivity only.
sensitivity: sensitivity1 sensitivity2 sensitivity3a sensitivity3b sensitivity4a sensitivity4b

sensitivity1:
	$(PY) src/estimation/sensitivity_analysis.py --exp 1

sensitivity2:
	$(PY) src/estimation/sensitivity_analysis.py --exp 2

sensitivity3a:
	$(PY) src/estimation/sensitivity_analysis.py --exp 3a

sensitivity3b:
	$(PY) src/estimation/sensitivity_analysis.py --exp 3b

sensitivity4a:
	$(PY) src/estimation/sensitivity_analysis.py --exp 4a

sensitivity4b:
	$(PY) src/estimation/sensitivity_analysis.py --exp 4b

# ── Trace Plots ─────────────────────────────────────────────
traces:
	$(PY) scripts/generate_trace_plots.py

# ── Figures ─────────────────────────────────────────────────
sensitivity-heatmap:
	$(PYTHON) scripts/generate_sensitivity_heatmap.py

forest-plot:
	$(PYTHON) scripts/generate_forest_plot.py

figures: forest-plot

# ── Tables, PDF, Paper ──────────────────────────────────────
tables:
	$(PYTHON) scripts/generate_tables.py

pdf:
	$(PYTHON) scripts/generate_results.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex && cd ..

# ── Full Pipeline ────────────────────────────────────────────
all: experiments analyze robust sensitivity traces tables pdf paper

# ── Deep Dive ────────────────────────────────────────────────
dive1:
	$(PY) scripts/deep_dive.py --exp 1

dive2:
	$(PY) scripts/deep_dive.py --exp 2

dive3a:
	$(PY) scripts/deep_dive.py --exp 3a

dive3b:
	$(PY) scripts/deep_dive.py --exp 3b

dive4a:
	$(PY) scripts/deep_dive.py --exp 4a

dive4b:
	$(PY) scripts/deep_dive.py --exp 4b

# ── Consistency Check ───────────────────────────────────────
check:
	$(PYTHON) scripts/check_consistency.py

# ── Data Freshness Check ───────────────────────────────────
check-freshness:
	$(PY) -c "from estimation.factorial_analysis import check_all_freshness; check_all_freshness()"

# ── Discretization Robustness ──────────────────────────────
discretization:
	$(PY) scripts/discretization_robustness.py --exp 1
	$(PY) scripts/discretization_robustness.py --exp 2
	$(PY) scripts/discretization_robustness.py --exp 3

# ── Budget Robustness (Exp4a) ──────────────────────────────
budget-robust:
	$(PY) scripts/budget_robustness.py

# ── Mathematical Verification ────────────────────────────────
verify:
	$(PYTHON) scripts/verification/run_all.py

verify-quick:
	$(PYTHON) scripts/verification/run_all.py --quick

# ── Exploration Calibration ────────────────────────────────────
calibrate:
	$(PYTHON) scripts/calibration_exploration.py --output-dir results/calibration

calibrate-multi:
	$(PYTHON) scripts/calibration_exploration.py --mode multi --seeds 5 --output-dir results/calibration

calibrate-exp3:
	$(PYTHON) scripts/calibration_exploration.py --mode exp3 --seeds 5 --output-dir results/calibration

calibrate-exp4b:
	$(PY) scripts/calibrate_exp4b.py

# ── Cleanup ──────────────────────────────────────────────────
clean:
	rm -f main.aux main.log main.out
	rm -f paper/*.aux paper/*.log paper/*.out paper/*.bbl paper/*.blg paper/*.toc
