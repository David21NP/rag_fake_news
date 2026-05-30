import os
from pathlib import Path

from linguistic import run_NC_LBFV

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_FILE_PATH = os.path.join(RESULTS_PATH, "results.csv")
Path(RESULTS_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)

nc_lbfv = run_NC_LBFV()
