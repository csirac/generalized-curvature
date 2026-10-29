"""Generate paper experiment tables and summaries from immutable cell JSONs."""
from release_paths import RESULTS, FIGURES, ORIGINAL
import collections
import json
import subprocess
import time
from pathlib import Path
import numpy as np
from run_camera_ready import clean, write_json
from curvature import guarantee_curvature

ROOT = (RESULTS / 'camera_ready')
ALGS = ['GP', 'DG', 'SG', 'RG', 'LS', 'MCG']
NAMES = dict(design='Design', coverage='Coverage', features='Features',
             planted_distractor='Planted', erdos_renyi='ER')


def mean(values):
    vs = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vs)) if vs else None


def fmt(value, digits=3):
    return '--' if value is None else f'{value:.{digits}f}'


def load():
    rows = []
    for p in sorted(ROOT.glob('*.json')):
        d = json.loads(p.read_text())
        if 'algorithms' in d:
            d['_file'] = str(p)
            rows.append(d)
    return rows


def group(rows, keys):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[tuple(r['parameters'][k] for k in keys)].append(r)
    return dict(sorted(groups.items()))


def table(caption, label, header, body, spec=None):
    spec = spec or 'l' + 'r' * (len(header)-1)
    return '\n'.join([r'\begin{table}[htbp]', r'\centering\scriptsize',
        '\\caption{' + caption + '}', '\\label{' + label + '}',
        '\\begin{tabular}{@{}' + spec + '@{}}', r'\toprule',
        ' & '.join(header) + r' \\', r'\midrule',
        *[' & '.join(row) + r' \\' for row in body],
        r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])


def replace_table(tex, label, replacement):
    return tex+'\n'+replacement


def block(tex, name, content, anchor):
    return tex+'\n'+content


def main():
    start=time.perf_counter();rows=load();tex=''
    FIGURES.mkdir(parents=True,exist_ok=True)
    additional = [r'\subsection{Additional Baselines and Query Counts}']
    additional += [r'''Tables below report means of per-instance ratios, excluding
zero-reference instances (their raw values remain in the JSON). Original
results are retained without overwriting. MCG/RG are heuristic on signed
objectives. For baseline curvature comparisons, the JSON reports the same
set-ratio diagnostic along each discrete trajectory, with nonpositive
denominators marked undefined. For MCG it reports only the rounded terminal
set diagnostic, not a discrete greedy curvature or a fractional guarantee.
At scale even these diagnostics use a best-known reference, not OPT.
The Multi-News rerun and original-selection reproduction checks are reported
in Appendix~\ref{apx:multinews-baselines}.''']
    for stage in ['tier1', 'tier2', 'features']:
        selected = [r for r in rows if r['parameters']['stage'] == stage]
        if not selected: continue
        body = []
        for (app, cost), rs in group(selected, ['app', 'cost_scale']).items():
            body.append([NAMES[app], f'{cost:g}'] + [fmt(mean([r['algorithms'].get(a, {}).get('ratio') for r in rs])) for a in ALGS])
        desc = {'tier1': 'Tier 1 setups, ten seeds and exact OPT.',
                'tier2': 'Tier 2, five seeds and non-certified best-known normalization.',
                'features': 'Redesigned features: three groups, correlation 0.9, ten seeds, exact OPT.'}[stage]
        additional.append(table(desc + ' RG uses the corrected fixed pool.',
            'tab:cr-' + stage, ['Setup', 'Cost'] + ALGS, body))
    legacy = [r for r in rows if r['parameters']['stage'] == 'tier1' and 'legacy_SG' in r]
    if legacy:
        body = []
        for (app, cost), rs in group(legacy, ['app', 'cost_scale']).items():
            body.append([NAMES[app], f'{cost:g}', fmt(mean([r['legacy_SG']['ratio'] for r in rs])),
                fmt(mean([r['legacy_RG_shrinking_pool']['ratio_mean'] for r in rs]))])
        additional.append(table('Reused submitted Tier 1 ratios. Legacy RG uses the shrinking pool and is not BFNS Algorithm 1.',
            'tab:cr-legacy', ['Setup', 'Cost', 'Legacy SG', 'Legacy RG'], body))

    query_body=[]; diagnostic_body=[]; summary=[]
    for (stage, app, n, k), rs in group(rows, ['stage','app','n','k']).items():
        alg_stats={a:dict(ratio=mean([r['algorithms'].get(a,{}).get('ratio') for r in rs]),
                          value=mean([r['algorithms'].get(a,{}).get('value') for r in rs]),
                          queries={q:mean([r['algorithms'].get(a,{}).get('queries',{}).get(q) for r in rs]) for q in ['f','g','ell']}) for a in ALGS}
        summary.append(dict(stage=stage,app=app,n=n,k=k,instances=len(rs),algorithms=alg_stats,
            certificate=mean([r['certificate']['guarantee'] for r in rs if r['certificate']]),
            gp_bound=n*k+2*k*k,gp_max_queries=max(r['algorithms']['GP']['queries']['f'] for r in rs)))
        title=f'{stage}/{NAMES[app]}/{n}'
        queries=[fmt(alg_stats[a]['queries']['f'],0) for a in ALGS if a!='DG']
        query_body.append([title]+queries+[fmt(alg_stats['DG']['queries']['g'],0),fmt(alg_stats['DG']['queries']['ell'],0),str(n*k+2*k*k)])
        diagnostic_body.append([title]+[fmt(mean([r['algorithms'].get(a,{}).get('diagnostic_curvature') for r in rs])) for a in ALGS])
    # Split at 18 rows to fit an ordinary page without changing the preamble.
    for i in range(0,len(query_body),18):
        additional.append(table(r'Mean per-run oracle counts over cost levels and instance seeds. DG columns count $g$ and $\ell$ separately; its $f$ calls for output/trace are recorded in JSON. Bound is $nk+2k^2$.',
            'tab:cr-queries-'+str(i),['Stage/setup/$n$','GP','SG','RG','LS','MCG',r'DG $g$',r'DG $\ell$','Bound'],query_body[i:i+18]))
        additional.append(table('Mean trajectory diagnostics, excluding undefined values; MCG uses its rounded terminal set only. These are not guarantees for the baselines.',
            'tab:cr-diagnostics-'+str(i),['Stage/setup/$n$']+ALGS,diagnostic_body[i:i+18]))
    scale=[r for r in rows if r['parameters']['stage']=='scale']
    if scale:
        body=[]
        for (app,n,k),rs in group(scale,['app','n','k']).items():
            body.append([NAMES[app],str(n),str(k)]+[fmt(mean([r['algorithms'][a]['ratio'] for r in rs])) for a in ALGS]+[fmt(mean([r['certificate']['guarantee'] for r in rs]))])
        additional.append(table('Scale grid: means over five cost levels and three seeds, with non-certified best-known normalization. All cells include MCG ($T=50,m=16$). Cert. is the GP removal certificate for all three applications.',
            'tab:cr-scale',['Setup','$n$','$k$']+ALGS+['Cert.'],body))
        additional.append(r'''For $n=500,1000,2000$ we use $k=20,30,50$ respectively.
D-optimal design uses $d=10$ and old costs $0,0.02,0.05,0.10,0.15$, multiplied by the frozen size-specific $r$; coverage uses $2n$
items, edge probability $0.05$, and costs $0,1.5,5,8,12$; features use ten
groups, within-group correlation $0.7$, and costs $0,0.08,0.25,0.4,0.6$.
Seeds are $0,1,2$. The generator parameters and costs were fixed before the
scale runs; the covariance's existing positive-definiteness shift is retained.
Standalone LS is capped at 200 accepted moves; these capped-run values
need not be local optima. Zero-reference ratios are undefined, and the cost
grid is never retuned after seeing results.
Per-cost values, timings, counts and certificates are recorded in the JSON.''')
    maxcut=[r for r in rows if r['parameters']['stage']=='maxcut']
    if len(maxcut)==240:
        body=[]
        for (app,n,k),rs in group(maxcut,['app','n','k']).items():
            wins=sum(r['algorithms']['GP']['value']>r['algorithms']['SG']['value']+1e-10 for r in rs)
            body.append([NAMES[app],str(n),str(k)]+[fmt(mean([r['algorithms'][a]['ratio'] for r in rs])) for a in ['SG','GP','RG','LS','MCG']]+[str(wins)+'/20'])
        tex=replace_table(tex,'tab:maxcut',table(r'MaxCut: mean ratios to exact OPT over 20 instance seeds. RG uses fixed top-$k$ pools (ten sub-seeds); MCG uses $T=m=50$ (three sub-seeds). LS uses swap/add/delete search. The submitted shrinking-pool random results remain in the original JSON.',
            'tab:maxcut',['Instance','$n$','$k$','SG','GP','RG','LS','MCG',r'GP $>$ SG'],body))
    tex = block(tex, 'CAMERA READY TABLES', '\n\n'.join(additional),
                r'\section{Multilinear Extension, DMCG-P, and MCG}')

    t2=[r for r in rows if r['parameters']['stage']=='tier2']
    if len(t2)==105:
        choices=dict(design=[.02,.05,.10,.15],coverage=[1.5,5,8,12],features=[.08,.25,.4,.6])
        body=[]
        for app in ['design','coverage','features']:
            for cost in choices[app]:
                rs=[r for r in t2 if r['parameters']['app']==app and r['parameters'].get('old_cost_scale',r['parameters']['cost_scale'])==cost]
                body.append([NAMES[app],f"{rs[0]['parameters']['cost_scale']:.4g}",fmt(mean([r['algorithms']['GP']['value'] for r in rs]),2),
                    fmt(mean([r['algorithms']['DG']['value'] for r in rs]),2),fmt(mean([r['certificate']['c_g_cert'] for r in rs]),2),
                    fmt(mean([r['certificate']['guarantee'] for r in rs]),2),fmt(mean([r['harshaw'] for r in rs]),2)])
        tex=replace_table(tex,'tab:tier2',table(r'Tier 2 means over five seeds. Cert. uses removal marginals and the clamp. All three applications are submodular; design uses D-optimality. HFWK with a best-known denominator is also diagnostic. Infinite curvature entries are omitted from the curvature mean, but contribute zero to the certificate mean.',
            'tab:tier2',['Application','Cost',r'$f_{\rm GP}$',r'$f_{\rm DG}$',r'$\hat c_g$','Cert.','HFWK'],body))
        plot_tier2(t2)
    (ROOT/'tables.txt').write_text(tex)
    write_json(ROOT/'summary.json',dict(parameters=dict(operation='aggregate all camera-ready cell files',files=[r['_file'] for r in rows]),
        seeds=sorted(set(r['parameters']['seed'] for r in rows)),git_commit=subprocess.run(['git','rev-parse','HEAD'],text=True,capture_output=True).stdout.strip() or 'uncommitted',
        wall_time=time.perf_counter()-start,summary=summary))
    print('Summarized',len(rows),'instances; GP violations:',sum(not r['gp_query_bound_pass'] for r in rows))


def plot_tier2(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for app in ['design','coverage']:
        groups=group([r for r in rows if r['parameters']['app']==app],['cost_scale'])
        costs=[key[0] for key in groups]
        fig,axes=plt.subplots(1,2,figsize=(10,3.4))
        for alg in ALGS:
            axes[0].plot(costs,[mean([r['algorithms'][alg]['value'] for r in rs]) for rs in groups.values()],marker='.',label=alg)
        axes[0].set(xlabel='Cost scale',ylabel='Objective value',title=NAMES[app]);axes[0].legend(fontsize=7,ncol=2)
        axes[1].plot(costs,[mean([r['certificate']['guarantee'] for r in rs]) for rs in groups.values()],marker='o',label='Removal certificate')
        axes[1].plot(costs,[mean([r['harshaw'] for r in rs]) for rs in groups.values()],linestyle='--',label='HFWK diagnostic')
        axes[1].set(xlabel='Cost scale',ylabel='Ratio');axes[1].legend(fontsize=7)
        fig.tight_layout();fig.savefig(str(FIGURES / 'fig_t2_')+app+'.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
