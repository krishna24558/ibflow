# EDAC + IBM solver - common tasks.  Run `make help` for the list.
PYTHON ?= python3
STEPS  ?=            # override run length, e.g.  make cylinder STEPS=50000
RE     ?=            # override Reynolds number, e.g.  make cavity RE=100

steps_arg = $(if $(strip $(STEPS)),--steps $(STEPS),)
re_arg    = $(if $(strip $(RE)),--re $(RE),)

.PHONY: help install test quick cavity channel cylinder airfoil validate plots strip-notebook clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install Python dependencies
	$(PYTHON) -m pip install -r requirements.txt

test:  ## Run the fast unit/sanity tests (seconds)
	$(PYTHON) -m pytest -q tests

quick:  ## Short smoke run of every case (a few minutes) - checks the pipeline end to end
	$(PYTHON) cases/lid_cavity.py    --steps 2000 --out results/quick_cavity.npz
	$(PYTHON) cases/channel.py       --steps 2000 --out results/quick_channel.npz
	$(PYTHON) cases/external_body.py --body cylinder --steps 1000 --out results/quick_cylinder.npz
	$(PYTHON) cases/external_body.py --body naca0012 --steps 1000 --out results/quick_naca0012.npz

cavity:  ## Lid-driven cavity (default Re=400, original grid and time step)
	$(PYTHON) cases/lid_cavity.py $(re_arg) $(steps_arg)
	$(PYTHON) plot.py cavity results/lid_cavity.npz

channel:  ## Developing laminar channel flow
	$(PYTHON) cases/channel.py $(re_arg) $(steps_arg)
	$(PYTHON) plot.py channel results/channel.npz

cylinder:  ## Flow past a circular cylinder with IBM (default Re=100)
	$(PYTHON) cases/external_body.py --body cylinder $(re_arg) $(steps_arg)
	$(PYTHON) plot.py field  results/cylinder.npz
	$(PYTHON) plot.py forces results/cylinder.npz

airfoil:  ## Flow past a NACA 0012 airfoil with IBM (default Re=100)
	$(PYTHON) cases/external_body.py --body naca0012 $(re_arg) $(steps_arg)
	$(PYTHON) plot.py field  results/naca0012.npz
	$(PYTHON) plot.py forces results/naca0012.npz

validate:  ## Cavity at Re=100 and 400 vs Ghia et al. (80x80 grid, ~3 min each)
	$(PYTHON) cases/lid_cavity.py --re 100 --dt 1e-4 --steps 200000 --n-fine 41 --n-coarse 41 --print-every 20000 --out results/val_cavity_re100.npz
	$(PYTHON) cases/lid_cavity.py --re 400 --dt 1e-4 --steps 200000 --n-fine 41 --n-coarse 41 --print-every 20000 --out results/val_cavity_re400.npz
	$(PYTHON) plot.py cavity results/val_cavity_re100.npz
	$(PYTHON) plot.py cavity results/val_cavity_re400.npz

plots:  ## Re-plot everything already in results/
	@for f in results/*.npz; do \
	  case $$f in \
	    *cavity*)  $(PYTHON) plot.py cavity $$f ;; \
	    *channel*) $(PYTHON) plot.py channel $$f ;; \
	    *)         $(PYTHON) plot.py field $$f && $(PYTHON) plot.py forces $$f ;; \
	  esac; \
	done

strip-notebook:  ## Remove outputs from notebooks before committing
	$(PYTHON) -m jupyter nbconvert --clear-output --inplace notebooks/*.ipynb

clean:  ## Delete results and caches
	rm -rf results/*.npz results/*.png results/*.log .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
