"""Audited inventory simulator. Period index t=0 corresponds to paper t=1."""
from dataclasses import dataclass, asdict
from itertools import product
from functools import lru_cache
import numpy as np
from scipy.stats import norm

@dataclass(frozen=True)
class Config:
    L:int=2
    T:int=50
    p:float=.7
    mu:float=10.
    sigma:float=2.
    co:float=2.
    h:float=1.
    b:float=9.
    alpha:float=.99
    demand:str='normal'
    scale:float=1.6
    def __post_init__(self):
        if self.L<1 or self.T<=self.L: raise ValueError('Require T>L>=1')
        if not .5<self.p<=1: raise ValueError('Target approximation requires .5<p<=1')
        if min(self.mu,self.sigma,self.co,self.h,self.b)<0 or self.h+self.b<=0: raise ValueError('Invalid costs/moments')
        if self.demand not in ('normal','gamma'): raise ValueError('Unknown demand')
        if not 0<self.alpha<=1 or self.scale<=0: raise ValueError('Invalid discount/scale')
    @property
    def variance(self):
        return self.sigma**2 if self.demand=='normal' else self.mu*self.scale
    @property
    def target(self):
        rho2=(1-self.p)/self.p
        v=(self.L+1)*self.variance+max(self.L,1)*rho2/(1-rho2)*(self.mu**2+self.variance)
        return (self.L+1)*self.mu+norm.ppf(self.b/(self.b+self.h))*np.sqrt(v)

def scenarios(c):
    # Column i=0 is oldest outstanding order; binary index j=sum(2**i*y_i).
    m=2**(c.L-1)
    B=((np.arange(m)[:,None]>>np.arange(c.L-1))&1).astype(float)
    count=B.sum(1)
    w=c.p**count*(1-c.p)**(c.L-1-count)
    return B,w

def generate(c,n,seed):
    ss=np.random.SeedSequence(seed).spawn(2)
    rd,ry=map(np.random.default_rng,ss)
    d=rd.normal(c.mu,c.sigma,(n,c.T)) if c.demand=='normal' else rd.gamma(c.mu/c.scale,c.scale,(n,c.T))
    return np.maximum(d,0), (ry.random((n,c.T))<c.p).astype(float)

def heuristic(c,name):
    B,w=scenarios(c)
    def policy(I,Q,t):
        if name=='LIR': return np.maximum(c.target-I-c.p*Q.sum(1),0)
        if name=='WNH': return np.maximum(c.target-I[:,None]-Q@B.T,0)@w
        raise ValueError(name)
    return policy

def rollout(c,d,y,policy,trace=False):
    d,y=np.asarray(d,float),np.asarray(y,float)
    if d.shape!=y.shape or d.ndim!=2 or d.shape[1]!=c.T: raise ValueError('Bad paths')
    if np.any(d<0) or not np.isin(y,[0,1]).all(): raise ValueError('Invalid exogenous paths')
    n=len(d); I=np.zeros(n); Q=np.zeros((n,c.L)); total=np.zeros(n); records=[]
    for t in range(c.T):
        arriving=Q[:,0].copy(); received=arriving*y[:,t]; I=I+received
        # Q contains L slots before arrival; only L-1 remain unknown after arrival.
        pending=Q[:,1:]
        q=np.asarray(policy(I,pending,t)) if t<c.T-c.L else np.zeros(n)
        q=np.broadcast_to(q,(n,)).copy()
        if not np.isfinite(q).all() or np.any(q<0): raise ValueError('Infeasible action')
        end=I-d[:,t]
        purchase=c.co*received; holding=c.h*np.maximum(end,0); backlog=c.b*np.maximum(-end,0)
        stage=purchase+holding+backlog; total+=c.alpha**t*stage
        if trace:
            for k in range(n): records.append(dict(path=k,period=t+1,demand=d[k,t],yield_success=y[k,t],arriving=arriving[k],received=received[k],inventory_before=I[k],order=q[k],inventory_end=end[k],purchase=purchase[k],holding=holding[k],backlog=backlog[k],cost=stage[k],discounted_cost=c.alpha**t*stage[k]))
        I=end; Q=np.concatenate([pending,q[:,None]],axis=1)
    return (total,records) if trace else total

def paired_summary(cost,reference):
    from scipy.stats import t as student
    cost,reference=np.asarray(cost),np.asarray(reference)
    diff=cost-reference; se=diff.std(ddof=1)/np.sqrt(len(diff))
    z=student.ppf(.975,len(diff)-1)
    return dict(mean=float(cost.mean()),gap_percent=float(100*(cost.mean()/reference.mean()-1)),paired_difference=float(diff.mean()),ci95_difference=[float(diff.mean()-z*se),float(diff.mean()+z*se)],n=len(diff))

def finite_dp(c,support,probabilities,actions):
    """Exact recursion for finite demand/action alphabets, NOT continuous original.
    No inventory clipping: every reachable integer state is memoized.
    """
    support=tuple(support); probs=tuple(probabilities); actions=tuple(actions)
    if not np.isclose(sum(probs),1) or min(probs)<0: raise ValueError('Invalid probabilities')
    if any(int(a)!=a or a<0 for a in actions+support) or 0 not in actions: raise ValueError('Nonnegative integer alphabets, including zero action, required')
    @lru_cache(None)
    def value(t,I,Q):
        if t==c.T: return 0.
        choices=actions if t<c.T-c.L else (0,)
        vals=[]
        for q in choices:
            val=0.
            for d,pr in zip(support,probs):
                end=I-d
                nextQ=Q[1:]+(q,)
                immediate=c.h*max(end,0)+c.b*max(-end,0)
                # Arrival at next period. Current purchase is sunk at decision time.
                future=0.
                if t+1<c.T:
                    for y,py in [(0,1-c.p),(1,c.p)]:
                        rec=nextQ[0]*y
                        future+=py*(c.co*rec+value(t+1,end+rec,nextQ))
                val+=pr*(immediate+c.alpha*future)
            vals.append(val)
        return min(vals)
    def policy(I,Q,t):
        # Q passed here excludes current arrival; append placeholder for sunk slot.
        out=[]
        for stock,pending in zip(I,Q):
            full=(0,)+tuple(map(int,pending)); choices=actions if t<c.T-c.L else (0,)
            vals=[]
            for q in choices:
                v=0.
                for d,pr in zip(support,probs):
                    end=int(stock)-d; nq=full[1:]+(q,)
                    f=0.
                    if t+1<c.T:
                        for y,py in [(0,1-c.p),(1,c.p)]:
                            rec=nq[0]*y; f+=py*(c.co*rec+value(t+1,end+rec,nq))
                    v+=pr*(c.h*max(end,0)+c.b*max(-end,0)+c.alpha*f)
                vals.append(v)
            out.append(choices[int(np.argmin(vals))])
        return np.array(out)
    return value(0,0,(0,)*c.L),policy,value.cache_info

def experiment_grid():
    for L,p,demand,ratio in product([2,5],[.55,.7,.9],['normal','gamma'],[.9,.95]):
        yield Config(L=L,p=p,demand=demand,b=ratio/(1-ratio))

def finite_policy_value(c,policy,support,probabilities,actions):
    """Exact expectation of a deterministic feasible policy on finite alphabets."""
    support=tuple(support); probabilities=tuple(probabilities); actions=tuple(actions)
    @lru_cache(None)
    def value(t,I,Q):
        if t==c.T:return 0.
        q=int(policy(np.array([I],float),np.array([Q[1:]],float),t)[0]) if t<c.T-c.L else 0
        if q not in actions:raise ValueError('Policy action outside DP action set')
        total=0.
        for d,pr in zip(support,probabilities):
            end=I-d;nq=Q[1:]+(q,);f=0.
            if t+1<c.T:
                for y,py in [(0,1-c.p),(1,c.p)]:
                    rec=nq[0]*y;f+=py*(c.co*rec+value(t+1,end+rec,nq))
            total+=pr*(c.h*max(end,0)+c.b*max(-end,0)+c.alpha*f)
        return total
    return value(0,0,(0,)*c.L)
