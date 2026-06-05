from pathlib import Path

from linguistic import run_NC_LBFV
import common.utils

RESULTS_PATH = Path(__file__).parent.parent / "results"
RESULTS_FILE_PATH = RESULTS_PATH / "oe3_ablation.csv"

def run_oe3():
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    # TODO: Fix subset selection and other algo
    df_test = common.utils.get_df_test()
    df_test_subset = df_test[0:500]
    nc_lbfv = run_NC_LBFV(df_test_subset)


if __name__ == "__main__":
    run_oe3()
