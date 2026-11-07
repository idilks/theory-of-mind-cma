"""Exercise repair launchers through bash with real GPU guards and mocked inference."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
import test_path_patching_clean_launcher as fixture

class RepairShellTests(unittest.TestCase):
    root: Path
    source: Path
    bin_dir: Path
    cache: Path
    calls: Path
    tmp: tempfile.TemporaryDirectory
    def setUp(self):
        fixture.CleanLauncherTests.setUp(self)
        self.driver=fixture.fixture.ROOT/'scripts/circuit/path_patching_h200_repair.py'
        (self.root/'scripts/circuit/path_patching_h200_repair.py').write_bytes(self.driver.read_bytes())
        (self.root/'scripts/circuit/torch.py').write_text('import os\nclass cuda:\n @staticmethod\n def device_count(): return 1\n @staticmethod\n def get_device_name(i): return os.environ["FAKE_GPU"]\n')
        (self.bin_dir/'python').write_text('#!/bin/bash\n'
            'test "$MKL_INTERFACE_LAYER" = activated || exit 81\n'
            'if [ "$1" = "-c" ]; then exec "$REAL_PYTHON" "$@"; fi\n'
            'printf "%s\\n" "$@" >> "$CALL_LOG"\n'
            'printf "HF_HUB_CACHE=%s\\n" "$HF_HUB_CACHE" >> "$CALL_LOG"\n')
        self.conda=self.root/'conda.sh'
        self.conda.write_text('conda() { if [ "$MKL_INTERFACE_LAYER" = "" ]; then export MKL_INTERFACE_LAYER=activated; fi; }\n')

    def tearDown(self):self.tmp.cleanup()

    def launch(self,kind='path',task=1,gpu='NVIDIA H200 NVL',output=None):
        name='path_patching_h200_repair.sh' if kind=='path' else 'path_patching_clean_h200_repair.sh'
        content=(fixture.fixture.ROOT/'slurm_scripts'/name).read_text().replace('/optnfs/common/miniconda3/etc/profile.d/conda.sh',str(self.conda))
        file=self.root/'launcher.sh';file.write_text(content)
        env=dict(os.environ);env.pop('MKL_INTERFACE_LAYER',None)
        env.update(PROJECT_DIR=str(self.root),SOURCE_ROOT=str(self.source),CONTROL_SOURCE_ROOT=str(self.root/'old_controls'),
            OUTPUT_ROOT=str(output or self.root/'output'),MODEL_CACHE=str(self.cache),
            CALL_LOG=str(self.calls),REAL_PYTHON=sys.executable,FAKE_GPU=gpu,SLURM_ARRAY_TASK_ID=str(task),
            PATH=str(self.bin_dir)+os.pathsep+env['PATH'])
        result=subprocess.run(['bash','-u',str(file)],env=env,capture_output=True,text=True)
        calls=self.calls.read_text() if self.calls.exists() else ''
        self.calls.unlink(missing_ok=True)
        return result,calls

    def test_wrong_gpu_and_unrequested_task_stop_before_writing(self):
        cases: list[tuple[dict[str,Any],str]] = [({'gpu':'NVIDIA L40S'},'H200'),({'task':0},'repair task')]
        for options,message in cases:
            result,calls=self.launch(**options)
            self.assertNotEqual(result.returncode,0)
            self.assertIn(message,result.stderr)
            self.assertFalse((self.root/'output').exists())
            self.assertEqual(calls,'')

    def test_both_original_roots_are_protected(self):
        for original in (self.source,self.root/'old_controls'):
            result,calls=self.launch(output=original)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('separate',result.stderr)
            self.assertEqual(calls,'')

    def test_path_and_control_launch_from_frozen_driver_flat_and_hub_cache(self):
        for kind,task,hub in [('path',1,self.cache),('control',2,self.cache/'hub')]:
            with self.subTest(kind=kind):
                # Isolate the two repair roots, as the real launchers do.
                output=self.root/('repair_'+kind)
                snapshot=hub/f'models--Qwen--Qwen2.5-14B-Instruct/snapshots/{fixture.fixture.REVISION}'
                snapshot.mkdir(parents=True)
                result,calls=self.launch(kind,task,output=output)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn(str(output/'repair_driver_snapshot/path_patching_h200_repair.py'),calls)
                self.assertIn('--kind\n'+kind,calls)
                self.assertIn(f'HF_HUB_CACHE={hub}',calls)
                frozen=(output/'repair_driver_snapshot/path_patching_h200_repair.py').read_bytes()
                (self.root/'scripts/circuit/path_patching_h200_repair.py').write_bytes(self.driver.read_bytes()+b'\n# synchronized edit\n')
                result,_=self.launch(kind,task,output=output)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual((output/'repair_driver_snapshot/path_patching_h200_repair.py').read_bytes(),frozen)
                snapshot.rmdir()

if __name__=='__main__':unittest.main()
