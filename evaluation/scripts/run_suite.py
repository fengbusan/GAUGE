"""Run isolated SDK processes serially; record every failure and timeout."""
import argparse
import json
import os
import subprocess
import sys
from gauge_eval.paths import ROOT, SOURCE

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',choices=['xiephysic','isaac','genesis','newton'],required=True)
    p.add_argument('--python',default=sys.executable,help='SDK Python or Isaac python.sh')
    p.add_argument('--case',action='append');p.add_argument('--smoke',action='store_true');p.add_argument('--retry',action='store_true')
    p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--profile',default='xie_sdk_v1',help='Xie scene profile; native peers use native_sdk_v1')
    p.add_argument('--result-key',default=None,help='Separate method result namespace, e.g. xie_sap')
    p.add_argument('--record-state',action='store_true')
    a=p.parse_args();manifest=json.loads((SOURCE/'recipes/manifest.json').read_text())
    cases=[c for c in manifest['cases'] if not a.case or c['case_id'] in a.case]
    if a.case and len(cases)!=len(set(a.case)):p.error('Unknown case ID')
    output='results/smoke' if a.smoke else 'results'
    env={**os.environ,'GAUGE_WORKSPACE':str(ROOT),'GAUGE_NATIVE_PROFILE':'native_sdk_v1','THIRDPARTY_OUTPUT_ROOT':output,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    failures=0
    for case in cases:
        cid=case['case_id'];key=a.result_key or a.engine;folder=ROOT/output/key/cid;folder.mkdir(parents=True,exist_ok=True);status=folder/'status.json'
        if status.exists() and not a.retry and json.loads(status.read_text()).get('status') in ['completed','smoke_passed']:
            print('SKIP completed',a.engine,cid,flush=True);continue
        adapter={'rigid':'rigid','textile':'cloth','deformable':'foam'}[case['regime']]
        script=SOURCE/'scripts/run_xie.py' if a.engine=='xiephysic' else SOURCE/f'scripts/thirdparty/{a.engine}_{adapter}.py'
        cmd=[a.python,str(script),cid]+(['--smoke'] if a.smoke else [])
        if a.engine=='xiephysic':cmd+=['--profile',a.profile,'--result-key',key]+(['--record-state'] if a.record_state else [])
        print('START',a.engine,cid,flush=True)
        # Clear the previous attempt's receipt before launching, so a crash cannot
        # leave an old completed result looking current.
        if status.exists():status.unlink()
        with (folder/'run.log').open('w') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            timed_out=False
            try:child.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                import signal
                timed_out=True;os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        data=json.loads(status.read_text()) if status.exists() else {}
        logtext=(folder/'run.log').read_text(errors='replace')
        data.update(engine=key,engine_family=a.engine,case_id=cid,exit_code=child.returncode,frames_required=case['rollout_frames'])
        if timed_out:data.update(status='timeout',limit_seconds=a.timeout)
        elif any(x in logtext for x in ['PhysX Internal CUDA error','PhysX ABORT error','contact buffer overflow detected']):data.update(status='native_error',error='PhysX fatal error or buffer overflow')
        elif 'status'not in data or (child.returncode!=0 and data['status']in ['completed','smoke_passed','running']):data['status']='process_error'
        status.write_text(json.dumps(data,indent=2))
        if data['status'] not in ['completed','smoke_passed']:failures+=1
        print('END',a.engine,cid,data['status'],flush=True)
    print('Finished',len(cases),'cases;',failures,'unsuccessful (all receipts retained)')
    return int(failures>0)

if __name__=='__main__':raise SystemExit(main())
