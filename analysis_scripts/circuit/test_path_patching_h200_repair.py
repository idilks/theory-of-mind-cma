"""Repair-specific regressions: never regenerate inputs or overwrite accepted results."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
DRIVER=ROOT/'scripts/circuit/path_patching_h200_repair.py'

class RepairTests(unittest.TestCase):
    def module(self):
        self.assertTrue(DRIVER.exists(), 'Missing H200 repair driver')
        spec=importlib.util.spec_from_file_location('repair_under_test',DRIVER)
        assert spec is not None and spec.loader is not None
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        return mod

    def test_runtime_rejects_non_h200_even_if_scheduler_overridden(self):
        r=self.module()
        r.validate_gpu_names(['NVIDIA H200 NVL'])
        for names in ([],['NVIDIA L40S'],['NVIDIA A100-SXM4-80GB'],['NVIDIA H200 NVL','NVIDIA L40S']):
            with self.assertRaisesRegex(ValueError,'H200'):
                r.validate_gpu_names(names)

    def test_only_missing_tasks_are_allowed(self):
        r=self.module()
        self.assertEqual(r.TASKS,{'path':(1,3,6,7),'control':(2,5,8,13)})
        for kind in r.TASKS:
            for i in range(14):
                if i not in r.TASKS[kind]:
                    with self.assertRaisesRegex(ValueError,'repair task'):
                        r.validate_task(kind,i)

    def test_root_guards_protect_both_original_studies_and_symlinks(self):
        r=self.module()
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);a=root/'original';b=root/'controls'
            a.mkdir();b.mkdir();(root/'alias').symlink_to(a,target_is_directory=True)
            for out in (a,b,root,a/'nested',root/'alias'):
                with self.assertRaisesRegex(ValueError,'separate'):
                    r.validate_output(out,a,b)
            r.validate_output(root/'new',a,b)

    def test_copy_preserves_exact_bytes_and_refuses_changed_inputs(self):
        r=self.module()
        with tempfile.TemporaryDirectory() as t:
            source=Path(t)/'source';target=Path(t)/'target'
            source.write_bytes(b'original prompts\n')
            r.copy_verified(source,target)
            self.assertEqual(source.read_bytes(),target.read_bytes())
            r.copy_verified(source,target)
            source.write_bytes(b'changed prompts\n')
            with self.assertRaisesRegex(ValueError,'Changed'):
                r.copy_verified(source,target)
            self.assertEqual(target.read_bytes(),b'original prompts\n')

    def test_preparation_reuses_real_snapshot_prompts_and_no_old_checkpoints(self):
        r=self.module()
        source=ROOT/'results/circuit/path_patching/union41_v2'
        controls=ROOT/'results/circuit/path_patching/union41_v2_clean_control_v1'
        before={str(p):r.sha(p) for base in (source,controls) for p in base.glob('*/*/pair_*.json')}
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'repair'
            info=r.inspect('path',1,source,controls)
            self.assertEqual(info['cell'],'C1_food_truck_ABB')
            r.prepare('path',1,source,controls,out,info)
            self.assertEqual((out/info['cell']/'pairs.jsonl').read_bytes(),(source/info['cell']/'pairs.jsonl').read_bytes())
            self.assertFalse(list(out.glob('*/*/pair_*.json')))
            self.assertEqual(r.sha(out/'code_snapshot/scripts/circuit/path_patching.py'),r.sha(source/'code_snapshot/scripts/circuit/path_patching.py'))
            sentinel=out/info['cell']/'existing_candidates/pair_000000.json'
            sentinel.parent.mkdir();sentinel.write_text('completed repair checkpoint')
            r.prepare('path',1,source,controls,out,info)
            self.assertEqual(sentinel.read_text(),'completed repair checkpoint')
            self.assertEqual(set(p.name for p in out.iterdir() if p.is_dir()),{'code_snapshot',info['cell']})
        self.assertTrue(all(r.sha(Path(p))==h for p,h in before.items()))

    def test_launchers_request_h200_and_exact_arrays(self):
        for name,ids in [('path_patching_h200_repair.sh','1,3,6,7%4'),('path_patching_clean_h200_repair.sh','2,5,8,13%4')]:
            path=ROOT/'slurm_scripts'/name
            self.assertTrue(path.exists(),f'Missing {name}')
            content=path.read_text()
            self.assertIn('#SBATCH --partition=h200_preemptable,preempt_wager\n',content)
            self.assertIn('#SBATCH --array='+ids+'\n',content)
            self.assertIn('validate_gpu_names',content)

    def test_original_launchers_also_reject_other_gpu_types(self):
        for name in ('path_patching.sh','path_patching_clean_control.sh'):
            text=(ROOT/'slurm_scripts'/name).read_text()
            self.assertIn('#SBATCH --partition=h200_preemptable,preempt_wager\n',text)
            self.assertIn('validate_gpu_names',text)

    def test_real_study_dispatch_uses_saved_pairs_and_original_settings(self):
        import torch
        r=self.module(); original_load=r.load
        source=ROOT/'results/circuit/path_patching/union41_v2'
        controls=ROOT/'results/circuit/path_patching/union41_v2_clean_control_v1'
        versions=r.read(source/'study.json')['versions']
        calls=[]
        def load(path,name):
            module=original_load(path,name)
            if name=='original_repair_core':
                module.provenance=lambda:{'versions':versions}
            if name=='frozen_path_repair_study':
                module.provenance=lambda:{'versions':versions}
                def record_run(args,model):
                    calls.append(args)
                    return model
                module.run=record_run
                module.export_pairs=lambda *a: self.fail('Must never regenerate prompts')
            return module
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'repair'
            args=Namespace(kind='path',task_id=1,source_root=source,control_source_root=controls,output_root=out,preflight=False)
            with patch.object(r,'load',side_effect=load),patch.object(torch.cuda,'device_count',return_value=1),patch.object(torch.cuda,'get_device_name',return_value='NVIDIA H200 NVL'):
                r.execute(args)
            self.assertEqual(len(calls),2)
            self.assertEqual(r.read(out/'study.json'),r.read(source/'study.json'))
            for call in calls:
                self.assertEqual(call.batch_size,8)
                self.assertEqual(call.dtype,'bfloat16')
                self.assertEqual(r.sha(Path(call.pairs)),r.sha(source/'C1_food_truck_ABB/pairs.jsonl'))
                self.assertTrue(Path(call.output).is_relative_to(out.resolve()))

if __name__=='__main__':unittest.main()
