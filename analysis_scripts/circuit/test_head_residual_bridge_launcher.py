"""Exercise the real SLURM launcher with isolated Conda/cache/Python substitutes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "slurm_scripts/head_residual_bridge.sh"
REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"
REQUIRED = (
    "scripts/circuit/stage2_steering.py",
    "results/aggregated/circuit/stage1/directions_L41.npz",
    "results/rsa_head_outputs/activations_code_position_full/Qwen2.5-14B-Instruct/code_position.pt",
    "results/circuit/stage1_capture/Qwen2.5-14B-Instruct/weights.pt",
    "results/circuit/stage1_capture/Qwen2.5-14B-Instruct/food_truck.pt",
    "results/circuit/stage1_capture/Qwen2.5-14B-Instruct/hair_styling.pt",
    "results/circuit/stage1_capture/Qwen2.5-14B-Instruct/library_book.pt",
)


class BridgeLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cache = self.root / "cache"
        self.calls = self.root / "calls.jsonl"
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        for relative in REQUIRED:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        # Evaluate the launcher's real GPU assertion, substituting only hardware.
        self.fake_modules = self.root / "fake_modules"
        self.fake_modules.mkdir()
        (self.fake_modules / "torch.py").write_text(
            "import json, os\n"
            "class cuda:\n"
            "    names = json.loads(os.environ['TEST_GPU_NAMES'])\n"
            "    @staticmethod\n"
            "    def is_available(): return bool(cuda.names)\n"
            "    @staticmethod\n"
            "    def device_count(): return len(cuda.names)\n"
            "    @staticmethod\n"
            "    def get_device_name(i): return cuda.names[i]\n"
        )
        stub = self.bin_dir / "python"
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, subprocess, sys\n"
            "assert os.environ['MKL_INTERFACE_LAYER'] == 'activated'\n"
            "if sys.argv[1] == '-c':\n"
            "    if 'Versions:' in sys.argv[2]:\n"
            "        sys.exit(int(os.environ.get('TEST_IMPORT_EXIT', '0')))\n"
            "    sys.exit(subprocess.run([sys.executable, *sys.argv[1:]]).returncode)\n"
            "with open(os.environ['CALL_LOG'], 'a') as f:\n"
            "    f.write(json.dumps({'args':sys.argv[1:], 'cache':os.environ['HF_HUB_CACHE'],\n"
            "        'offline':os.environ['HF_HUB_OFFLINE'], 'transformers_offline':os.environ['TRANSFORMERS_OFFLINE']}) + '\\n')\n"
            "if '--bridge_preflight' in sys.argv:\n"
            "    sys.exit(int(os.environ.get('TEST_PREFLIGHT_EXIT', '0')))\n"
        )
        stub.chmod(0o755)

    def snapshot(self, base=None, revision=REVISION):
        ((base or self.cache) / f"models--Qwen--Qwen2.5-14B-Instruct/snapshots/{revision}").mkdir(parents=True)

    def launch(self, activation_exit=0, names=None, phase="pilot", arguments=(), **extra_env):
        conda = self.root / "conda.sh"
        conda.write_text(
            'conda() {\n'
            '    if [ "$MKL_INTERFACE_LAYER" = "" ]; then export MKL_INTERFACE_LAYER=activated; fi\n'
            '    export CONDA_PREFIX="$2"\n'
            f'    return {activation_exit}\n'
            '}\n'
        )
        launcher = self.root / "launcher.sh"
        launcher.write_text(LAUNCHER.read_text().replace(
            "/optnfs/common/miniconda3/etc/profile.d/conda.sh", str(conda)))
        env = dict(os.environ)
        env.pop("MKL_INTERFACE_LAYER", None)
        env.update(
            PROJECT_DIR=str(self.root), MODEL_CACHE=str(self.cache), PHASE=phase,
            CALL_LOG=str(self.calls), PYTHONPATH=str(self.fake_modules),
            TEST_GPU_NAMES=json.dumps(["NVIDIA H200"] if names is None else names),
            PATH=str(self.bin_dir) + os.pathsep + env["PATH"], **extra_env,
        )
        result = subprocess.run(["bash", "-u", str(launcher), *arguments], env=env,
                                text=True, capture_output=True)
        calls = [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []
        self.calls.unlink(missing_ok=True)
        return result, calls

    def test_pilot_preflight_then_existing_runner_and_pinned_offline_cache(self):
        self.snapshot()
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)
        self.assertIn("--bridge_preflight", calls[0]["args"])
        args = calls[1]["args"]
        self.assertEqual(args[0], "scripts/circuit/stage2_steering.py")
        self.assertIn("--head_residual_bridge", args)
        self.assertNotIn("--bridge_preflight", args)
        for flag, expected in {
            "--bridge_phase": "pilot", "--bridge_pairs": "2", "--bridge_seed": "20261023",
            "--bridge_revision": REVISION, "--bridge_dtype": "bfloat16",
            "--bridge_output": str(self.root / "results/circuit/head_residual_bridge/v1_pilot"),
        }.items():
            self.assertEqual(args[args.index(flag) + 1], expected)
        self.assertEqual(calls[1]["cache"], str(self.cache))
        self.assertEqual(calls[1]["offline"], "1")
        self.assertEqual(calls[1]["transformers_offline"], "1")
        # The launcher must leave run state and checkpoint ownership to the runner.
        self.assertFalse((self.root / "results/circuit/head_residual_bridge").exists())

    def test_full_phase_and_hub_cache_fallback(self):
        self.snapshot(self.cache / "hub")
        result, calls = self.launch(phase="full")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("full", calls[-1]["args"])
        self.assertIn(str(self.root / "results/circuit/head_residual_bridge/v1_full"), calls[-1]["args"])
        self.assertEqual(calls[-1]["cache"], str(self.cache / "hub"))

    def test_failed_activation_stops_even_with_nounset_at_entry(self):
        result, calls = self.launch(activation_exit=23)
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertEqual(calls, [])

    def test_missing_pinned_revision_stops(self):
        self.snapshot(revision="wrong-revision")
        result, calls = self.launch()
        self.assertEqual(result.returncode, 1)
        self.assertIn("Pinned Qwen revision", result.stderr)
        self.assertEqual(calls, [])

    def test_missing_or_empty_input_stops_before_runner(self):
        self.snapshot()
        for relative in REQUIRED:
            with self.subTest(path=relative):
                path = self.root / relative
                path.write_text("")
                result, calls = self.launch()
                self.assertEqual(result.returncode, 1)
                self.assertIn(relative, result.stderr)
                self.assertEqual(calls, [])
                path.write_text("fixture\n")

    def test_dependency_or_preflight_failure_stops_before_main(self):
        self.snapshot()
        result, calls = self.launch(TEST_IMPORT_EXIT="27")
        self.assertEqual(result.returncode, 27)
        self.assertEqual(calls, [])
        result, calls = self.launch(TEST_PREFLIGHT_EXIT="29")
        self.assertEqual(result.returncode, 29)
        self.assertEqual(len(calls), 1)
        self.assertIn("--bridge_preflight", calls[0]["args"])

    def test_a100_is_accepted(self):
        self.snapshot()
        result, calls = self.launch(names=["NVIDIA A100-SXM4-80GB"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 2)

    def test_all_visible_devices_must_be_a100_or_h200(self):
        self.snapshot()
        for names in ([], ["NVIDIA V100"], ["NVIDIA H200", "NVIDIA V100"]):
            with self.subTest(names=names):
                result, calls = self.launch(names=names)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(len(calls), 1)
                self.assertIn("CUDA unavailable" if not names else "A100 or H200 required", result.stderr)

    def test_invalid_phase_or_unrecognized_arguments_fail(self):
        for kwargs in ({"phase": "unexpected"}, {"arguments": ("--other",)}):
            result, calls = self.launch(**kwargs)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(calls, [])

    def test_shell_syntax_and_scheduler_resources(self):
        result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = LAUNCHER.read_text()
        for directive in ("--partition=h200_preemptable,a100_preemptable,gpuq,preempt_wager,preempt_lsong", "--gres=gpu:1",
                          "--cpus-per-task=2", "--mem=90GB", "--time=07:59:00"):
            self.assertIn("#SBATCH " + directive, text)


if __name__ == "__main__":
    unittest.main()
