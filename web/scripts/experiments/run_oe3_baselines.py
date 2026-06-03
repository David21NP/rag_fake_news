import os
from pathlib import Path

from linguistic import run_NC_LBFV
import common.utils

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results")
RESULTS_FILE_PATH = os.path.join(RESULTS_PATH, "results.csv")
Path(RESULTS_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)

def run_compares():
    # TODO: Fix subset selection and other algo
    df_test = common.utils.get_df_test()
    df_test_subset = df_test[0:500]
    nc_lbfv = run_NC_LBFV(df_test_subset)
