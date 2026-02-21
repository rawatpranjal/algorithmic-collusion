.PHONY: smoke smoke1 smoke2 smoke3 smoke4 \
        experiments exp1 exp2 exp3 exp4 \
        analyze analyze1 analyze2 analyze3 analyze4 \
        robust robust1 robust2 robust3 robust4 \
        traces tables pdf paper all clean help \
        dive1 dive2 dive3 dive4

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
	@echo "  make traces         Generate single-run trace plots for paper"
	@echo "  make tables         Generate LaTeX tables + copy figures"
	@echo "  make pdf            Generate standalone results PDF"
	@echo "  make paper          Compile main paper (pdflatex)"
	@echo "  make all            Full pipeline: experiments -> analyze -> robust -> tables -> pdf -> paper"
	@echo ""
	@echo "Options:"
	@echo "  REPS=5              Replicates per cell (default: 2)"
	@echo "  SEED=123            Random seed (default: 42)"
	@echo "  WORKERS=8           Parallel workers (default: cpu_count/2)"

# ── Smoke Tests ──────────────────────────────────────────────
smoke: smoke1 smoke2 smoke3 smoke4

smoke1:
	$(PYTHON) scripts/run_experiment.py --exp 1 --quick --parallel $(_W)

smoke2:
	$(PYTHON) scripts/run_experiment.py --exp 2 --quick --parallel $(_W)

smoke3:
	$(PYTHON) scripts/run_experiment.py --exp 3 --quick --parallel $(_W)

smoke4:
	$(PYTHON) scripts/run_experiment.py --exp 4 --quick --parallel $(_W)

# ── Full Experiments ─────────────────────────────────────────
experiments: exp1 exp2 exp3 exp4

exp1:
	$(PYTHON) scripts/run_experiment.py --exp 1 --parallel --replicates $(REPS) --seed $(SEED) $(_W)

exp2:
	$(PYTHON) scripts/run_experiment.py --exp 2 --parallel --replicates $(REPS) --seed $(SEED) $(_W)

exp3:
	$(PYTHON) scripts/run_experiment.py --exp 3 --parallel --replicates $(REPS) --seed $(SEED) $(_W)

exp4:
	$(PYTHON) scripts/run_experiment.py --exp 4 --parallel --replicates $(REPS) --seed $(SEED) $(_W)

# ── Factorial Analysis ───────────────────────────────────────
analyze: analyze1 analyze2 analyze3 analyze4

analyze1:
	$(PY) src/estimation/est1.py

analyze2:
	$(PY) src/estimation/est2.py

analyze3:
	$(PY) src/estimation/est3.py

analyze4:
	$(PY) src/estimation/est4.py

# ── Robustness Checks ───────────────────────────────────────
# Note: est*.py now runs robustness automatically after factorial ANOVA.
# These targets are for standalone re-runs of robustness only.
robust: robust1 robust2 robust3 robust4

robust1:
	$(PY) src/estimation/robust_analysis.py --exp 1

robust2:
	$(PY) src/estimation/robust_analysis.py --exp 2

robust3:
	$(PY) src/estimation/robust_analysis.py --exp 3

robust4:
	$(PY) src/estimation/robust_analysis.py --exp 4

# ── Trace Plots ─────────────────────────────────────────────
traces:
	$(PY) scripts/generate_trace_plots.py

# ── Tables, PDF, Paper ──────────────────────────────────────
tables:
	$(PYTHON) scripts/generate_tables.py

pdf:
	$(PYTHON) scripts/generate_results.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && cd ..

# ── Full Pipeline ────────────────────────────────────────────
all: experiments analyze robust traces tables pdf paper

# ── Deep Dive ────────────────────────────────────────────────
dive1:
	$(PY) scripts/deep_dive.py --exp 1

dive2:
	$(PY) scripts/deep_dive.py --exp 2

dive3:
	$(PY) scripts/deep_dive.py --exp 3

dive4:
	$(PY) scripts/deep_dive.py --exp 4

# ── Cleanup ──────────────────────────────────────────────────
clean:
	rm -f main.aux main.log main.out
	rm -f paper/*.aux paper/*.log paper/*.out
