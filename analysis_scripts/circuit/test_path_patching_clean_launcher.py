"""Exercise the actual clean-control launcher with local HPC substitutes."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import test_path_patching_launcher as fixture


class CleanLauncherTests(unittest.TestCase):
    root: Path
    bin_dir: Path
    cache: Path
    calls: Path
    tmp: tempfile.TemporaryDirectory
    def setUp(self):
        fixture.LauncherTests.setUp(self)
        self.source = self.root / 'results/circuit/path_patching/union41_v2'
        self.source.mkdir(parents=True)
        (self.source / 'study.json').write_text('{"plan":{"revision":"' + fixture.REVISION + '"}}')
        (self.root / 'analysis_scripts/circuit/path_patching_clean_control.py').write_text('# frozen control v1\n')
        (self.root / 'slurm_scripts').mkdir()
        self.script = (fixture.ROOT / 'slurm_scripts/path_patching_clean_control.sh').read_text()
        (self.root / 'slurm_scripts/path_patching_clean_control.sh').write_text(self.script)
        # Run path overlap and revision checks in real Python, mock only GPU check.
        (self.bin_dir / 'python').write_text('#!/bin/bash\n'
            'test "$MKL_INTERFACE_LAYER" = activated || exit 81\n'
            'if [ "$1" = "-c" ]; then case "$2" in *cuda*) exit 0;; *) exec "$REAL_PYTHON" "$@";; esac; fi\n'
            'printf "%s\\n" "$@" >> "$CALL_LOG"\n'
            'printf "HF_HUB_CACHE=%s\\n" "$HF_HUB_CACHE" >> "$CALL_LOG"\n')

    def tearDown(self):
        self.tmp.cleanup()

    def run_launcher(self, activation_fails=False, output=None):
        conda = self.root / 'conda.sh'
        conda.write_text('conda() {\n'
            'if [ "$MKL_INTERFACE_LAYER" = "" ]; then export MKL_INTERFACE_LAYER=activated; fi\n'
            + ('return 23\n' if activation_fails else 'return 0\n') + '}\n')
        launcher = self.root / 'launcher.sh'
        launcher.write_text(self.script.replace('/optnfs/common/miniconda3/etc/profile.d/conda.sh', str(conda)))
        env = dict(os.environ)
        env.pop('MKL_INTERFACE_LAYER', None)
        env.update(PROJECT_DIR=str(self.root), SOURCE_ROOT=str(self.source), SLURM_ARRAY_TASK_ID='0',
            OUTPUT_ROOT=str(output or self.root/'output'), MODEL_CACHE=str(self.cache),
            CALL_LOG=str(self.calls), REAL_PYTHON=sys.executable,
            PATH=str(self.bin_dir) + os.pathsep + env['PATH'])
        result = subprocess.run(['bash', '-u', str(launcher)], env=env, text=True, capture_output=True)
        calls = self.calls.read_text().splitlines() if self.calls.exists() else []
        self.calls.unlink(missing_ok=True)
        return result, calls

    def test_mkl_activation_flat_cache_and_frozen_script_resume(self):
        fixture.LauncherTests.add_snapshot(self, self.cache)
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.root/'output/code_snapshot/analysis_scripts/circuit/path_patching_clean_control.py'), calls)
        self.assertIn(str(self.source), calls)
        self.assertIn(f'HF_HUB_CACHE={self.cache}', calls)
        (self.root/'analysis_scripts/circuit/path_patching_clean_control.py').write_text('# pushed edit\n')
        result, _ = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root/'output/code_snapshot/analysis_scripts/circuit/path_patching_clean_control.py').read_text(), '# frozen control v1\n')

    def test_hub_cache(self):
        fixture.LauncherTests.add_snapshot(self, self.cache/'hub')
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f'HF_HUB_CACHE={self.cache / "hub"}', calls)

    def test_missing_model_and_failed_activation_stop(self):
        result, calls = self.run_launcher()
        self.assertEqual(result.returncode, 1)
        self.assertIn('missing', result.stderr)
        self.assertEqual(calls, [])
        result, calls = self.run_launcher(activation_fails=True)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(calls, [])

    def test_original_output_protected(self):
        result, calls = self.run_launcher(output=self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('must be separate', result.stderr)
        self.assertEqual(calls, [])
        self.assertFalse((self.source/'code_snapshot').exists())


if __name__ == '__main__':
    unittest.main()
