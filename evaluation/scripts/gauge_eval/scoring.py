"""Shared baseline computation and conservative ranking; no engine fitting."""
import json
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import numpy as np
from .paths import DATA, verify_dataset_paths
from .metrics import rmse, dtw
from .observables import observe

METRICS={'RMSE':rmse,'DTW':dtw}

def validate_reused_evaluation(actual, target):
    """Profile metadata may differ; immutable physics/observation inputs may not."""
    bookkeeping={'profile','source_base_profile','source_base_scene_sha256','sap_scene_policy','rigid_topology_normalization'}
    if actual['scene_sha256']!=target['scene_sha256'] or {k:v for k,v in actual.items()if k not in bookkeeping}!={k:v for k,v in target.items()if k not in bookkeeping}:
        raise ValueError('Reused recording has different physical/evaluation inputs')

def matches_printed(value, token):
    token=Decimal(str(token))
    if not np.isfinite(value) or not token.is_finite():return False
    return abs(Decimal(str(value))-token)<=Decimal(5).scaleb(token.as_tuple().exponent-1)

def real_reference(e,profile='published_table'):
    verify_dataset_paths(e['trial_paths'])
    arrays=[]
    for i,name in enumerate(e['trial_paths']):
        raw=json.loads((DATA/name).read_text())[e['body']]
        def xyz(node):return np.stack([node[k]for k in 'xyz'],axis=-1)[:e['frames']]
        p=xyz(raw) if e['regime']=='rigid' else np.stack([xyz(raw[k])for k in e['raw_marker_mappings'][i]['metadata_to_raw']],axis=1)
        if len(p)!=e['frames']:raise ValueError('Reference trial shorter than score window')
        arrays.append(observe(p,e,profile))
    arrays=np.asarray(arrays);reference=arrays.mean(0);n=len(arrays)
    if n<2:raise ValueError('At least two real repeats are required for sample std')
    baseline={}
    for metric,fn in METRICS.items():
        errors=[fn(a,reference if profile!='published_table' or e['regime']=='rigid' else (arrays.sum(0)-a)/(n-1))for a in arrays]
        baseline[metric]={'mean':float(np.mean(errors)),'std':float(np.std(errors,ddof=1))}
    return reference,baseline

def competition_ranks(values,decimal_places):
    quantum=Decimal(1).scaleb(-decimal_places)
    rounded={k:Decimal(str(v)).quantize(quantum,rounding=ROUND_HALF_UP)for k,v in values.items()}
    if not all(v.is_finite()for v in rounded.values()):raise ValueError('Rank values must be finite')
    return {k:1+sum(other<v for other in rounded.values())for k,v in rounded.items()}

def rank_summary(rows,source):
    engines=['xiephysic','isaac','genesis','newton'];public_ids={'isaac':'IsaacSim','genesis':'Genesis','newton':'Newton'}
    output={}
    for metric in METRICS:
        output[metric]={}
        for group in ['all','rigid','textile','deformable']:
            cohort=[]
            for row in rows:
                if group!='all' and row['regime']!=group:continue
                audit=row.get('baseline_audit',{}).get(metric)
                if not row['ranking_eligible'] or not audit or not audit['mean_matches'] or not audit['std_matches']:continue
                tokens=row['published'][metric]['engines']
                try:
                    decimals=[Decimal(str(tokens[k]))for k in public_ids.values()]
                    if not all(v.is_finite()for v in decimals):continue
                    # The public results display all engine errors with two
                    # decimals. Numeric serialization drops trailing zeroes
                    # (1.00 -> '1'); that must not turn an entire row into an
                    # integer-precision tie. Ranking never changes raw scores.
                    places=2
                    if source=='published':
                        own=row['engines']['xiephysic']['scores'][metric]['normalized']
                        values={'xiephysic':own,**{k:float(tokens[v])for k,v in public_ids.items()}}
                    else:values={k:row['engines'][k]['scores'][metric]['normalized']for k in engines}
                except (KeyError,ValueError,TypeError,InvalidOperation):continue
                cohort.append({'case_id':row['case_id'],'ranks':competition_ranks(values,places),'decimal_places':places})
            n=len(cohort)
            output[metric][group]={'n':n,'cases':cohort,'engines':{k:{'first':sum(c['ranks'][k]==1 for c in cohort),'second':sum(c['ranks'][k]==2 for c in cohort),'first_pct':100*sum(c['ranks'][k]==1 for c in cohort)/n if n else None,'second_pct':100*sum(c['ranks'][k]==2 for c in cohort)/n if n else None}for k in engines}}
    return output
