import subprocess
import sys
import unittest
from pathlib import Path


class PipelineImportTests(unittest.TestCase):
    def test_run_stage_zero_imports_in_fresh_python_process(self):
        repository_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from src.pipeline import run_stage_zero; "
                    "assert callable(run_stage_zero); "
                    "assert run_stage_zero.__module__ == 'src.pipeline'"
                ),
            ],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
