"""Four small finite models: same demand and action feasible sets for all methods."""
from core import *
import json
from pathlib import Path

def main():
    rows=[]
    for L in [1,2]:
        for p in [.7,.9]:
            c=Config(L=L,T=6,p=p,mu=1,sigma=np.sqrt(.5),co=1,h=1,b=3,alpha=.95)
            actions=np.arange(5);support=[0,1,2];prob=[.25,.5,.25]
            opt,dp,cache=finite_dp(c,support,prob,actions)
            check=finite_policy_value(c,dp,support,prob,actions)
            assert np.isclose(opt,check)
            row=dict(L=L,p=p,DP=opt,cached_states=cache().currsize)
            for name in ['LIR','WNH']:
                pol=heuristic(c,name)
                def projected(I,Q,t,pol=pol):return actions[np.abs(pol(I,Q,t)[:,None]-actions).argmin(1)]
                v=finite_policy_value(c,projected,support,prob,actions)
                assert v>=opt-1e-8
                row[name]=v;row[name+'_gap_percent']=100*(v/opt-1)
            rows.append(row)
    out=Path('results/teaching/dp_study.json');out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
