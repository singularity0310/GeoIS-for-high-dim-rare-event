from pathlib import Path
import sys

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parent))
from plotting import plot_case_cli

if __name__ == "__main__":
    plot_case_cli("portfolio", CASE_DIR)
