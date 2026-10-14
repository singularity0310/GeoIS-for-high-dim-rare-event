from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parent))
from experiments import run_case

if __name__ == "__main__":
    run_case("upper_exterior_ellipsoid", CASE_DIR)
