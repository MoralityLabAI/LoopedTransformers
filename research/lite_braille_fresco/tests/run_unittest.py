"""Run legacy unittest modules without importing pytest-only LBF-19 tests.

LBF-19's original pytest files are exercised by lbf19-expensive-verifier.yml.
Keep their paths and bytes intact for the registered test-file hashes.
"""
from pathlib import Path
import unittest


def main() -> int:
    directory = Path(__file__).resolve().parent
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    pytest_only = {
        "test_lbf19_expensive_verifier.py",
        "test_lbf19_learned_transfer.py",
        "test_lbf19_tree_automata.py",
    }
    for path in sorted(directory.glob("test_*.py")):
        if path.name not in pytest_only:
            suite.addTests(loader.discover(str(directory), pattern=path.name))
    if not suite.countTestCases():
        raise RuntimeError("No legacy unittest tests discovered")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
