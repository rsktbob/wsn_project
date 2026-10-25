"""Small initial-state diagnostic, not a lifetime or optimizer comparison.
Run with wsnenv from any directory; writes JSON under artifacts only.
"""
from pathlib import Path
import os,sys,json,hashlib,contextlib,io,random,csv,statistics
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'codes/2027WSN';sys.path.insert(0,str(P));os.chdir(P)
import numpy as np
from Problem.Problem import Problem
from Problem.services.fitness_servicev2 import FitnessServiceV2
from State.SensorEncoding import SensorEncoding
from Algorithm.se.SA_SETS import SA_SETS
from Algorithm.se.Ring_SETS import Ring_SETS
SEED=20261002;COUNT=128
out={'date':'2026-10-02','seed':SEED,'sample_count_per_map_method':COUNT,'scope':'six fixed maps; initial F=10; no optimizer, no lifetime, no movement; selected one-gene non-noop mutations; initial candidate distributions only','cases':[]}
def sig(s):return (tuple(s.levels),tuple(s.next_hops),tuple(s.tx_load))
for k in range(1,7):
 random.seed(SEED+k);np.random.seed(SEED+k)
 with contextlib.redirect_stdout(io.StringIO()):p=Problem(B=100,S=100,T=100,F=10,FILE=f'MAP{k:02}',sensing_mode='discrete')
 p.fitness_service=FitnessServiceV2(p)
 a=SA_SETS(p,n=8,h=4,w=2,mu=.4,seed=SEED+k);a.identity_sensors=a.select_identity_sensors(p)
 rng=np.random.default_rng(SEED+k)
 row={'map':f'MAP{k:02}','identity_original_ids':[int(p.sensor_original_ids[i]) for i in a.identity_sensors],'methods':{}}
 seeds=[a.create_candidate(p) for _ in range(COUNT)]
 ring=Ring_SETS(p,n=8,h=4,w=2,mu=.4,seed=SEED+k)
 ring_candidates=[ring.create_candidate(p) for _ in range(COUNT)]
 row['SA_Ring_initial_codes_equal']=int(sum(np.array_equal(x.code,y.code) for x,y in zip(seeds,ring_candidates)))
 for name,cs in [('uniform_level_priority',[SensorEncoding.random(p.SENSOR_NUMBER,p.radius_option_counts,rng=rng) for _ in range(COUNT)]),('SA_CCS_before_alignment',seeds),('Ring_CCS',ring_candidates)]:
  mutation_rng=np.random.default_rng(SEED+1000+k)
  full=feasible=neutral=0;ss=[]
  for c in cs:
   s=c.decode(p);score=float(np.sum(p.evaluate_state(s)));ss.append(sig(s));feasible+=score>=0
   full+=bool(np.all(np.any(np.asarray(p.coverage_table)[:,np.arange(p.SENSOR_NUMBER),s.levels]>0,axis=1)))
   d=c.copy();idx=int(mutation_rng.integers(len(d.code)));bound=10 if idx%2 else p.sensing_option_count(idx//2)
   d.code[idx]=(int(d.code[idx])+int(mutation_rng.integers(1,bound)))%bound
   neutral+=sig(d.decode(p))==sig(s)
  row['methods'][name]={'full_coverage':int(full),'fitness_feasible':int(feasible),'unique_physical_states':len(set(ss)),'one_gene_neutral':int(neutral)}
 confusion=[];align_unchanged=[]
 for region in range(4):
  counts=[0]*4;unchanged=0
  for c in seeds:
   orig=c.decode(p);d=c.copy();a.align_region(p,d,region);s=d.decode(p)
   label=sum((int(s.levels[i])>0)<<j for j,i in enumerate(a.identity_sensors))
   counts[label]+=1;unchanged+=sig(orig)==sig(s)
  confusion.append(counts);align_unchanged.append(unchanged)
 row['identity_requested_to_decoded_counts']=confusion
 row['alignment_physical_unchanged']=align_unchanged
 out['cases'].append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
out['source_sha256']={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),P/'State/SensorEncoding.py',P/'Problem/Problem.py',P/'Problem/services/evaluation_kernels.py',P/'Problem/services/fitness_servicev2.py',P/'Algorithm/se/SA_SETS.py',P/'Algorithm/se/Ring_SETS.py']}
(ROOT/'artifacts/ring_sets_evidence_audit_20261002.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print('Saved diagnostic JSON',flush=True)
