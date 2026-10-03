"""Capture deterministic A2 traces before/after SE flow refactoring."""
import sys, json, hashlib
from pathlib import Path
import numpy as np
CALLER_CWD = Path.cwd()
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_algorithms import parse_args, build_problem, build_algorithm

def plain(x):
    if isinstance(x,np.ndarray): return plain(x.tolist())
    if isinstance(x,np.generic):return x.item()
    if hasattr(x,'code'):return plain(x.code)
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    return x

def digest(x):return hashlib.sha256(json.dumps(plain(x),sort_keys=True,allow_nan=True).encode()).hexdigest()

def main():
    from Algorithm.se.BaseSE import BaseSE
    from Algorithm.se.SA_SETS import SA_SETS
    from Algorithm.se.SI_SETS import SI_SETS
    from Algorithm.se.SI_SETSv2 import SI_SETSv2
    from Algorithm.se.Ring_SETS import Ring_SETS
    from Algorithm.se.parallel_market import ParallelSEMarket
    assert not hasattr(ParallelSEMarket, "search")
    assert not issubclass(SI_SETS, SA_SETS)
    assert not issubclass(Ring_SETS, SI_SETS)
    for cls in (SA_SETS, SI_SETS, SI_SETSv2, Ring_SETS):
        assert cls.vision_search is BaseSE.vision_search
    output=(CALLER_CWD / sys.argv[1]).resolve(); data={}
    for name in ['sa_sets','si_setsv2']:
        for seed in [7,11]:
            args=parse_args(['--algorithm',name,'--maps','maps/maps_100100100_a2.json','--mode','single','--runs','1','--evaluate','10000'])
            args._current_map=args.map_specs[0]
            problem=build_problem(args,seed); algorithm=build_algorithm(name,problem,args,seed)
            rows=[]
            def record(**event):
                row={'evals':algorithm.evatime,'fitness':algorithm.fitness,
                     'searchers':digest(algorithm.searchers),'searcher_fitness':digest(algorithm.searcher_fitness),
                     'regions':plain(algorithm.selected_regions),'ta':plain(algorithm.ta),'tb':plain(algorithm.tb),
                     'best_code':digest(algorithm.best_candidate),
                     'best_state':digest([algorithm.best_state.levels,algorithm.best_state.next_hops,algorithm.best_state.tx_load])}
                if name=='si_setsv2':
                    row.update(goods=digest(algorithm.goods),goods_fitness=digest(algorithm.goods_fitness),quality=digest(algorithm.investment_quality))
                rows.append(row)
            algorithm.set_iteration_callback(record)
            result=algorithm.run(problem,budget=10000,max_iteration=800)
            data[f'{name}:{seed}']={'rounds':rows,'history':digest(result.history),'evaluations':result.evaluations,'fitness':plain(result.best_fitness)}
            print(name,seed,result.evaluations,algorithm.iterations_completed,plain(result.best_fitness),flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data,indent=2))
    if len(sys.argv) > 2:
        baseline = json.loads((CALLER_CWD / sys.argv[2]).read_text())
        assert data == baseline, "A2 deterministic traces differ from baseline"
        print("A2 traces match baseline exactly")
if __name__=='__main__':main()
