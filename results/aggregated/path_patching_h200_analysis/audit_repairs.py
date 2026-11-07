"""Read-only validation and extraction of the repaired 14-cell path study."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
SOURCE = ROOT/'results/circuit/path_patching/union41_v2'
CLEAN = ROOT/'results/circuit/path_patching/union41_v2_clean_control_v1'
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
label = lambda h: 'L%dH%d' % tuple(h)


def audit():
    study = read(SOURCE/'study.json'); plan = study['plan']
    controlstudy = read(CLEAN/'study.json'); tol = controlstudy['tolerance']
    selection = read(ROOT/'results/aggregated/path_patching_setup/h200_repair_sources.json')
    assert controlstudy['source_study_sha256'] == sha(SOURCE/'study.json')
    assert controlstudy['source_plan'] == plan and controlstudy['versions'] == study['versions']
    hashes = {}
    def record(p):
        hashes[str(p.relative_to(ROOT))] = sha(p)
        return hashes[str(p.relative_to(ROOT))]
    def verify_map(base, mapping):
        for name, digest in mapping.items():
            assert record(base/name) == digest, (base, name)
    record(SOURCE/'study.json'); record(CLEAN/'study.json')
    verify_map(SOURCE/'code_snapshot', study['source_sha256'])
    assert record(CLEAN/'code_snapshot/scripts/circuit/path_patching_clean_control.py') == controlstudy['control_script_sha256']
    assert len(selection['cells']) == 14
    assign = {label(p['target']): dict(control=label(p['control']), membership=p['membership']) for p in plan['sender_pairs']}
    assert len(assign) == 41
    metadata = None; arrays = {}; auditrows = []; cleanrows = []; differences = []
    for cell, selected in zip(plan['cells'], selection['cells']):
        cid = cell['id']; taskid = selected['task_id']
        assert selected['cell'] == cid and plan['cells'][taskid] == cell
        pathroot = ROOT/selected['path_root']; ctrlroot = ROOT/selected['control_root']
        path_repaired = taskid in selection['path_repair_tasks']
        ctrl_repaired = taskid in selection['control_repair_tasks']
        assert (pathroot != SOURCE) == path_repaired
        assert (ctrlroot != CLEAN) == ctrl_repaired
        assert not (path_repaired and ctrl_repaired)
        for base, kind, repaired in [(pathroot,'path',path_repaired),(ctrlroot,'control',ctrl_repaired)]:
            if not repaired: continue
            rm = read(base/'repair_manifest.json'); record(base/'repair_manifest.json')
            assert rm['kind'] == kind and rm['gpu_family'] == 'H200'
            assert rm['task_ids'] == selection['path_repair_tasks' if kind=='path' else 'control_repair_tasks']
            assert rm['source_study_sha256'] == sha(SOURCE/'study.json')
            assert rm['control_study_sha256'] == sha(CLEAN/'study.json')
            assert record(base/'repair_driver_snapshot/path_patching_h200_repair.py') == rm['driver_sha256']
            rt = read(base/cid/'repair_runtime.json'); record(base/cid/'repair_runtime.json')
            assert rt['gpu_names'] and all('H200' in name for name in rt['gpu_names'])
            assert rt['versions'] == study['versions'] and rt['driver_sha256'] == rm['driver_sha256']
            inputs = read(base/cid/'repair_inputs.json'); record(base/cid/'repair_inputs.json')
            assert inputs['cell']==cid and inputs['task_id']==taskid and inputs['kind']==kind
            assert inputs['pairs']==10 and inputs['selected_heads']==41 and inputs['batch_size']==8
            verify_map(SOURCE,inputs['source_sha256']); verify_map(CLEAN,inputs['control_source_sha256'])
            if kind=='path':
                assert read(base/'study.json') == study
                verify_map(base/'code_snapshot',study['source_sha256'])
            else:
                assert read(base/'study.json') == controlstudy
                assert record(base/'code_snapshot/scripts/circuit/path_patching_clean_control.py') == controlstudy['control_script_sha256']
        folder=ctrlroot/cid; rawfolder=pathroot/cid
        record(folder/'manifest.json'); record(folder/'runtime.json')
        manifest=read(folder/'manifest.json'); runtime=read(folder/'runtime.json')
        verify_map(SOURCE,manifest['source_sha256'])  # Controls truthfully reference original path files.
        assert runtime['provenance']['versions']==study['versions'] and 'H200' in runtime['gpu']
        if path_repaired:
            path_gpu=read(rawfolder/'repair_runtime.json')['gpu_names'][0]
            path_host=read(rawfolder/'repair_runtime.json')['hostname']
        else:
            log=ROOT/f'slurm_logs/path_union41_9923275_{taskid}.out'
            record(log); path_host=log.read_text().splitlines()[0].split(' on ')[1]
            assert path_host.startswith('h200')
            path_gpu='H200 (original node assignment)'
        task=read(rawfolder/'task.json'); assert task == read(SOURCE/cid/'task.json')
        record(rawfolder/'task.json')
        assert (rawfolder/'pairs.jsonl').read_bytes()==(SOURCE/cid/'pairs.jsonl').read_bytes()
        assert record(rawfolder/'pairs.jsonl')==read(rawfolder/'pairs.meta.json')['pairs_sha256']
        pairs=[json.loads(s) for s in (rawfolder/'pairs.jsonl').read_text().splitlines() if s]
        assert len(pairs)==10 and len(set(p['pair_id'] for p in pairs))==10
        failures=nonzero=single_drift=batch_drift=counterexamples=0
        maxerr=0.; maxeffect=0.; maxpatherr=0.; rowcount=0; controlcount=0
        mats=[]; ctrls=[]; cellmeta=[]
        for group in task['task']['groups']:
            scope=group['scope']; per_sender=54 if scope=='existing_candidates' else 56
            expected=4212 if scope=='existing_candidates' else 224
            files=sorted((folder/scope).glob('pair_*.json')); assert len(files)==10
            assert [p.name for p in files]==[p.name for p in sorted((rawfolder/scope).glob('pair_*.json'))]
            xs=[]; cs=[]; groupmeta=None
            for i, file in enumerate(files):
                saved=read(file); rawfile=rawfolder/scope/file.name; raw=read(rawfile)
                oldfile=SOURCE/cid/scope/file.name; old=read(oldfile)
                record(file); record(rawfile)
                assert saved['source_sha256']==record(oldfile)
                assert saved['pair']==raw['pair']==old['pair']==pairs[i]
                assert len(raw['rows'])==len(old['rows'])==expected
                assert saved['clean_and_frozen_receiver_caches_identical']
                assert saved['frozen_clean_max_logit_error']==0
                assert saved['single_clean_margin_drift_from_original']==saved['single_clean_margin']-saved['original_single_clean_margin']
                assert saved['original_single_clean_margin']==old['rows'][0]['clean_margin']
                single_drift += saved['single_clean_margin'] != raw['rows'][0]['clean_margin']
                mapping={}
                for cr in saved['rows']:
                    fields=['control_effect','max_logit_error','clean_cache_control_effect','clean_cache_max_logit_error','batch_row_max_logit_error','untouched_reference_max_logit_error','batch_vs_single_max_logit_error']
                    passed=max(abs(cr[k]) for k in fields)<=tol
                    assert passed==cr['passes_tolerance']
                    failures+=not passed; nonzero+=cr['control_effect']!=0
                    maxerr=max(maxerr,*(abs(cr[k]) for k in fields));maxeffect=max(maxeffect,abs(cr['control_effect']))
                    oldm=[];newm=[]
                    for ref in cr['source_rows']:
                        idx=ref['row_index']; assert idx not in mapping
                        rr=raw['rows'][idx]; oldrr=old['rows'][idx]
                        assert ref['sender']==rr['senders']==oldrr['senders']
                        assert all(rr[k]==cr[k]==oldrr[k] for k in ('receivers','receiver_aliases','component','routes'))
                        ordinal=idx%per_sender
                        assert cr['batch_row']==ordinal%study['batch_size']+1
                        assert cr['batch_width']==min(8,per_sender-ordinal//8*8)+1
                        mapping[idx]=cr;oldm.append(oldrr['batch_clean_margin']);newm.append(rr['batch_clean_margin'])
                    assert sorted(set(oldm))==cr['original_batch_clean_margins']
                    drift=any(m!=cr['batch_clean_margin'] for m in newm);batch_drift+=drift
                    cleanrows.append(dict(cell=cid,scope=scope,pair=i,component=cr['component'],receiver=cr['receivers'],batch_row=cr['batch_row'],batch_width=cr['batch_width'],passes=passed,max_logit_error=cr['max_logit_error'],control_effect=cr['control_effect'],selected_path_margin_drift=drift,mapped_rows=len(cr['source_rows'])))
                assert set(mapping)==set(range(expected))
                rowcount+=expected;controlcount+=len(saved['rows'])
                lookup={}
                for rr in raw['rows']:
                    assert len(rr['routes'])==1
                    r=rr['routes'][0];key=(rr['senders'],rr['component'],r['layer'],r['hook_index'])
                    assert key not in lookup
                    assert rr['path_effect']==rr['path_margin']-rr['batch_clean_margin']
                    assert rr['clean_margin']==saved['single_clean_margin']
                    diagnostics=['sender_self_max_logit_error','receiver_noop_max_logit_error','self_path_max_logit_error','batch_row_max_logit_error','batch_vs_single_max_logit_error']
                    maxpatherr=max(maxpatherr,*(abs(rr[k]) for k in diagnostics))
                    counterexamples+=all(d==0 for d in rr['receiver_delta_l2']) and rr['path_effect']!=0
                    lookup[key]=rr['path_effect']
                metas=[];xx=[];cc=[]
                for rr in raw['rows']:
                    h=rr['senders']
                    if h not in assign: continue
                    r=rr['routes'][0];comp=rr['component'];l=r['layer'];ix=r['hook_index']
                    metas.append(dict(sender=h,**assign[h],scope=scope,component=comp,receiver_layer=l,receiver_index=ix,receiver=f'L{l}'+('H' if comp=='q' else 'KV')+str(ix),aliases=';'.join(rr['receiver_aliases'])))
                    xx.append(rr['path_effect']);cc.append(lookup[(assign[h]['control'],comp,l,ix)])
                if groupmeta is None:groupmeta=metas
                assert metas==groupmeta
                xs.append(xx);cs.append(cc)
                if path_repaired:
                    delta=np.array([r['path_effect'] for r in raw['rows']])-np.array([r['path_effect'] for r in old['rows']])
                    differences.append(dict(cell=cid,scope=scope,pair=i,changed_rows=int((delta!=0).sum()),max_abs_change=float(abs(delta).max())))
            mats.append(np.array(xs));ctrls.append(np.array(cs));cellmeta+=groupmeta
        x=np.concatenate(mats,1);c=np.concatenate(ctrls,1)
        assert x.shape==c.shape==(10,2218) and np.isfinite(x).all() and np.isfinite(c).all()
        if metadata is None:metadata=cellmeta
        assert metadata==cellmeta
        arrays[cid+'__target']=x;arrays[cid+'__control']=c
        passed=failures==nonzero==single_drift==batch_drift==counterexamples==0 and maxerr==maxpatherr==0
        auditrows.append(dict(task_id=taskid,cell=cid,domain=cell['condition'],path_repaired=path_repaired,control_repaired=ctrl_repaired,path_gpu=path_gpu,path_host=path_host,control_gpu=runtime['gpu'],control_rows=controlcount,failed_controls=failures,nonzero_control_effects=nonzero,max_control_error=maxerr,max_path_diagnostic_error=maxpatherr,single_margin_mismatches=single_drift,batch_margin_mismatches=batch_drift,zero_delta_nonzero_effect_rows=counterexamples,path_rows=rowcount,passed=passed,path_root=selected['path_root'],control_root=selected['control_root']))
        print(cid,'PASS' if passed else 'FAIL','controls',controlcount,'path rows',rowcount,'max error',maxerr,'single/batch mismatches',single_drift,batch_drift,flush=True)
    a=pd.DataFrame(auditrows);a.to_csv(OUT/'control_audit_by_cell.csv',index=False)
    pd.DataFrame(cleanrows).to_csv(OUT/'clean_controls.csv',index=False)
    pd.DataFrame(differences).to_csv(OUT/'old_vs_repaired_path_changes.csv',index=False)
    assert a.passed.all() and a.control_rows.sum()==15400 and a.path_rows.sum()==621040
    pd.DataFrame(metadata).to_csv(OUT/'connections.csv',index=False)
    np.savez_compressed(OUT/'validated_effects.npz',**arrays)
    validated=dict(selection)
    validated['status']='validated_all_14_cells'
    validated['validation_report']='results/aggregated/path_patching_h200_analysis/REPORT.md'
    validated['note']='Validated combination of original and repaired arms; original source hashes are preserved. See audit.json for provenance and control_audit_by_cell.csv for all numerical checks.'
    validated['cells']=[dict(c,status='validated_compatible') for c in selection['cells']]
    (OUT/'validated_sources.json').write_text(json.dumps(validated,indent=2))
    summary=dict(all_14_cells_passed=True,repair_cells_passed=8,control_rows=15400,path_rows=621040,connections=2218,heads=41,tolerance=tol,source_hashes=hashes,audit_script_sha256=sha(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2))
    print('AUDIT PASSED: 8 repaired arms; 14 compatible cells; 15400 clean controls; 621040 path rows',flush=True)

if __name__=='__main__':audit()
