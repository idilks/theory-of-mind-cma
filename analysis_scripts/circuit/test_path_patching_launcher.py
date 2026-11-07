"""Run the real launcher with local substitutes for the Andes Conda hook, flock and model cache."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
REVISION = "a" * 40


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        root = self.root
        plan = root / "results/aggregated/path_patching_setup/plan.json"
        plan.parent.mkdir(parents=True)
        plan.write_text(json.dumps({"revision": REVISION}))
        for stub in ("codebase/tasks/causal_analysis/cma_config.py", "scripts/condition_registry.py",
                     "scripts/circuit/path_patching.py", "scripts/circuit/path_patching_study.py"):
            (root / stub).parent.mkdir(parents=True, exist_ok=True)
            (root / stub).write_text("# v1\n")
        (root / "codebase/results/big.pt").parent.mkdir(parents=True)
        (root / "codebase/results/big.pt").write_text("excluded")
        self.cache = root / "models/cache"
        self.calls = root / "calls.txt"
        bin_dir = root / "bin"
        bin_dir.mkdir()
        # Environment checks pass; the plan-revision read runs in real Python; study calls are logged.
        (bin_dir / "python").write_text('#!/bin/bash\n'
            'test "$MKL_INTERFACE_LAYER" = activated || exit 81\n'
            'if [ "$1" = "-c" ]; then case "$2" in *revision*) exec "$REAL_PYTHON" "$@";; *) exit 0;; esac; fi\n'
            'printf "%s\\n" "$@" >> "$CALL_LOG"\n'
            'printf "HF_HUB_CACHE=%s\\n" "$HF_HUB_CACHE" >> "$CALL_LOG"\n')
        (bin_dir / "flock").write_text("#!/bin/bash\nexit 0\n")
        for tool in ("python", "flock"):
            (bin_dir / tool).chmod(0o755)
        self.bin_dir = bin_dir

    def tearDown(self):
        self.tmp.cleanup()

    def run_launcher(self, activation_fails=False):
        root = self.root
        conda = root / "conda.sh"
        # Reproduce a real activation-hook hazard: an unset-variable read.
        conda.write_text('conda() {\n'
            '  if [ "$MKL_INTERFACE_LAYER" = "" ]; then export MKL_INTERFACE_LAYER=activated; fi\n'
            + ('  return 23\n' if activation_fails else '  return 0\n') + '}\n')
        text = (ROOT / "slurm_scripts/path_patching.sh").read_text()
        text = text.replace("/optnfs/common/miniconda3/etc/profile.d/conda.sh", str(conda))
        launcher = root / "launcher.sh"
        launcher.write_text(text)
        env = dict(os.environ)
        env.pop("MKL_INTERFACE_LAYER", None)
        env.update(PROJECT_DIR=str(root), SLURM_ARRAY_TASK_ID="0", PAIRS_PER_CELL="10",
                   OUTPUT_ROOT=str(root / "output"), MODEL_CACHE=str(self.cache),
                   CALL_LOG=str(self.calls), REAL_PYTHON=sys.executable,
                   PATH=str(self.bin_dir) + os.pathsep + env["PATH"])
        result = subprocess.run(["bash", "-u", str(launcher)], env=env,
                                text=True, capture_output=True)
        calls = self.calls.read_text().splitlines() if self.calls.exists() else []
        self.calls.unlink(missing_ok=True)
        return result, calls

    def add_snapshot(self, base):
        (base / f"models--Qwen--Qwen2.5-14B-Instruct/snapshots/{REVISION}").mkdir(parents=True)

    def test_unset_mkl_during_activation_still_launches_full_study_from_snapshot(self):
        self.add_snapshot(self.cache)
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        snapshot = self.root / "output/code_snapshot"
        self.assertIn(str(snapshot / "scripts/circuit/path_patching_study.py"), calls)
        self.assertEqual(calls[calls.index("--plan") + 1],
                         str(snapshot / "results/aggregated/path_patching_setup/plan.json"))
        self.assertEqual(calls[calls.index("--task-id") + 1], "0")
        self.assertEqual(calls[calls.index("--pairs-per-cell") + 1], "10")
        self.assertIn(f"HF_HUB_CACHE={self.cache}", calls)
        self.assertTrue((snapshot / "codebase/tasks/causal_analysis/cma_config.py").exists())
        self.assertFalse((snapshot / "codebase/results").exists())

    def test_snapshot_is_reused_after_source_changes(self):
        self.add_snapshot(self.cache)
        self.assertEqual(self.run_launcher()[0].returncode, 0)
        (self.root / "scripts/circuit/path_patching.py").write_text("# v2 pushed mid-run\n")
        result, _ = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        frozen = self.root / "output/code_snapshot/scripts/circuit/path_patching.py"
        self.assertEqual(frozen.read_text(), "# v1\n")

    def test_hub_subdirectory_cache_layout(self):
        self.add_snapshot(self.cache / "hub")
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"HF_HUB_CACHE={self.cache / 'hub'}", calls)

    def test_missing_pinned_revision_stops_before_study(self):
        (self.cache / "models--Qwen--Qwen2.5-14B-Instruct/snapshots/other").mkdir(parents=True)
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn("not found", result.stderr)
        self.assertIn("snapshots/other", result.stderr)
        self.assertEqual(calls, [])

    def test_failed_activation_stops_before_python(self):
        result, calls = self.run_launcher(activation_fails=True)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
