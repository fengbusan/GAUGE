"""Audit real baselines, score SDK trajectories, compare local/published peers."""
import argparse
import hashlib
import json
import numpy as np
from gauge_eval.paths import ROOT, DATA, SOURCE
from gauge_eval.observables import observe
from gauge_eval.scoring import METRICS, real_reference, matches_printed, rank_summary, validate_reused_evaluation

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--references-only',action='store_true');p.add_argument('--case',action='append')
    p.add_argument('--xie-profile',default='xie_sdk_v1');p.add_argument('--xie-result-key',default='xiephysic');p.add_argument('--report-name',default='score_report.json')
    args=p.parse_args();public=json.loads((SOURCE/'recipes/published_scores.json').read_text());manifest=json.loads((SOURCE/'recipes/manifest.json').read_text())
    cases=[c for c in manifest['cases'] if not args.case or c['case_id'] in args.case]
    if args.case and len(cases)!=len(set(args.case)):p.error('Unknown case ID')
    rows=[];matched=0;count=0
    for case in cases:
        cid=case['case_id'];ep=ROOT/'configs'/args.xie_profile/cid/'evaluation.json';e=json.loads(ep.read_text());special=e['metric']in ['pendulum','cradle']
        flags=e['comparison_flags'];row={**case,'comparison_flags':flags,'published':public.get(cid,{}),'engines':{},'ranking_eligible':not special and not set(flags)&{'material_label_alias','material_label_cross_match','event_window_differs','marker_outlier'},'official_equivalence_verified':False}
        if special:
            row.update(ranking_eligible=False,score_status='withheld_unverified_special_metric')
            for engine in [] if args.references_only else ['xiephysic','isaac','genesis','newton']:
                sp=ROOT/'results'/(args.xie_result_key if engine=='xiephysic' else engine)/cid/'status.json'
                result=json.loads(sp.read_text())if sp.exists()else {'status':'not_run'}
                result.pop('scores',None);result['score_status']='withheld_unverified_special_metric';row['engines'][engine]=result
            rows.append(row);continue
        reference,baseline=real_reference(e);row['baseline']=baseline;row['baseline_audit']={}
        for metric,b in baseline.items():
            tokens=public[cid][metric]['baseline_printed'].replace(')','').split(' (')
            audit={'computed_mean':b['mean'],'computed_std_ddof1':b['std'],'published_mean':tokens[0],'published_std':tokens[1],'mean_matches':matches_printed(b['mean'],tokens[0]),'std_matches':matches_printed(b['std'],tokens[1])}
            row['baseline_audit'][metric]=audit;count+=1;matched+=int(audit['mean_matches']and audit['std_matches'])
        for engine in [] if args.references_only else ['xiephysic','isaac','genesis','newton']:
            folder=ROOT/'results'/(args.xie_result_key if engine=='xiephysic' else engine)/cid;sp=folder/'status.json'
            result=json.loads(sp.read_text()) if sp.exists() else {'status':'not_run'}
            result.pop('scores',None)  # Recompute here; never trust embedded SDK scores.
            if result['status']=='completed':
                profile=result.get('profile',args.xie_profile) if engine=='xiephysic' else 'native_sdk_v1'
                frozen=json.loads((ROOT/'configs'/profile/cid/'evaluation.json').read_text())
                try:
                    if engine=='xiephysic' and profile!=args.xie_profile:
                        # Reuse an immutable recording only if physical scenes
                        # are identical and all observable/driver/window inputs
                        # match. Keep its original receipt and profile identity.
                        validate_reused_evaluation(frozen,e)
                    scene_path=ROOT/'configs'/profile/cid/'scene.json'
                    if hashlib.sha256(scene_path.read_bytes()).hexdigest()!=frozen['scene_sha256'] or result.get('source_scene_sha256')!=frozen['scene_sha256']:raise ValueError('Rollout does not match current scene receipt')
                    if result.get('source_evaluation_sha256')!=hashlib.sha256((ROOT/'configs'/profile/cid/'evaluation.json').read_bytes()).hexdigest():raise ValueError('Rollout does not match current driver/evaluation inputs')
                    pp=folder/'marker_positions_m.npy';points=np.load(pp,allow_pickle=False)
                    if len(points)!=e['rollout_frames'] or not np.isfinite(points).all():raise ValueError('Incomplete or nonfinite full-process recording')
                    if e['regime']!='rigid':
                        target=np.asarray(json.loads((DATA/'metadata'/e['regime']/(e['task']+'.json')).read_text())['markers']['translation'])
                        if points.ndim!=3 or points.shape[1:]!=target.shape or np.max(np.linalg.norm(points[0]-target,axis=-1))>=.01:raise ValueError('Initial native marker mapping exceeds 1 cm')
                    obs=observe(points[:e['frames']]*1000,e,'published_table')
                    result['scores']={metric:{'raw':fn(obs,reference),'real_baseline':baseline[metric]['mean'],'normalized':fn(obs,reference)/baseline[metric]['mean']}for metric,fn in METRICS.items() if baseline[metric]['mean']>0}
                    result['trajectory_sha256']=hashlib.sha256(pp.read_bytes()).hexdigest()
                except (ValueError,FileNotFoundError) as ex:result.update(status='invalid_recording',score_error=str(ex))
            row['engines'][engine]=result
        rows.append(row);print('SCORED',cid,flush=True)
    report={'scoring_version':'published_table_reproduction_v3','xie_profile':args.xie_profile,'xie_result_key':args.xie_result_key,'official_equivalence_verified':False,'baseline_validation':{'matched_mean_and_std':matched,'generic_metric_rows':count},'rows':rows,'rankings':{source:rank_summary(rows,source)for source in ['published','local']},'ranking_policy':'Per metric and common eligible cases only; half-up rounding to least precise published peer token; competition ranks 1,1,3; no aggregate score; failed/blocked/withheld cases disclosed, never assigned zero.'}
    out=ROOT/'results';out.mkdir(exist_ok=True);(out/args.report_name).write_text(json.dumps(report,indent=2,ensure_ascii=False));print(report['baseline_validation'])
    return report

if __name__=='__main__':main()
