"""Differentiable sample-path optimization; explicit feasibility modification."""
from dataclasses import asdict
from pathlib import Path
import copy,json,random
import numpy as np
import torch
from torch import nn
from core import scenarios,generate,rollout,heuristic

class Weights(nn.Module):
    def __init__(self,c,kind='DRL',delta=None):
        super().__init__(); self.c=c; self.kind=kind
        self.delta=delta if delta is not None else (.0884 if c.L==5 else .25)
        B,w=scenarios(c)
        self.register_buffer('B',torch.tensor(B,dtype=torch.float32))
        self.register_buffer('teacher',torch.tensor(w,dtype=torch.float32))
        m=len(w)
        if kind=='GDSH': self.raw=nn.Parameter(torch.log(torch.expm1(self.teacher)))
        elif kind=='DRL': self.net=nn.Sequential(nn.Linear(c.L+1,8),nn.ReLU(),nn.Linear(8,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU(),nn.Linear(16,2*m))
        else: raise ValueError(kind)
    def forward(self,I,Q,t):
        gaps=torch.relu(self.c.target-I[:,None]-Q@self.B.T)
        if self.kind=='GDSH':
            # Explicit additional constraint: nonnegative weights, NOT sum-to-one.
            w=torch.nn.functional.softplus(self.raw)
        else:
            state=torch.cat([torch.full_like(I[:,None],self.c.target),I[:,None],Q],1)
            eta=self.net(state).reshape(-1,len(self.teacher),2).softmax(-1)
            bounds=torch.stack([self.teacher-self.delta,self.teacher+self.delta],-1)
            w=(eta*bounds).sum(-1)
        return torch.relu((gaps*w).sum(-1))  # added q>=0 feasibility projection

def costs(c,d,y,model):
    n=len(d); I=torch.zeros(n,device=d.device,dtype=d.dtype); Q=torch.zeros(n,c.L,device=d.device,dtype=d.dtype); total=torch.zeros_like(I)
    for t in range(c.T):
        rec=Q[:,0]*y[:,t]; I=I+rec; pending=Q[:,1:]
        q=model(I,pending,t) if t<c.T-c.L else torch.zeros_like(I)
        end=I-d[:,t]
        total=total+c.alpha**t*(c.co*rec+c.h*torch.relu(end)+c.b*torch.relu(-end))
        I=end; Q=torch.cat([pending,q[:,None]],1)
    return total

def train(c,kind='DRL',seed=0,n=500,epochs=300,device='cpu',out='results/run',delta=None,resume=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root=Path(out); root.mkdir(parents=True,exist_ok=True)
    split=dict(train=100000+seed,validation=200000+seed,test=300000)
    d,y=generate(c,n,split['train']); vd,vy=generate(c,100,split['validation'])
    base=torch.tensor(rollout(c,d,y,heuristic(c,'WNH')),device=device,dtype=torch.float32)
    dt,yt=[torch.tensor(a,device=device,dtype=torch.float32) for a in [d,y]]
    vt,vt_y=[torch.tensor(a,device=device,dtype=torch.float32) for a in [vd,vy]]
    model=Weights(c,kind,delta).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    history=[]; best=float('inf'); best_state=None; start=0
    checkpoint=root/'checkpoint.pt'
    if resume and checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        expected=dict(config=asdict(c),kind=kind,seed=seed,n=n,delta=delta)
        if ck['identity']!=expected: raise ValueError('Checkpoint configuration mismatch')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer']); history=ck['history'];start=ck['epoch']+1;best=ck['best'];best_state=ck['best_state']
    for epoch in range(start,epochs):
        for group in optimizer.param_groups: group['lr']=max(.0001,.003*.98**epoch)
        optimizer.zero_grad(); ct=costs(c,dt,yt,model)
        # GDSH uses mean total cost; DRL uses mean of per-trajectory exponential losses.
        loss=ct.mean() if kind=='GDSH' else torch.exp(100*(ct-base)/base).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite original exponential loss; do not silently alter objective')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()): raise FloatingPointError('Nonfinite gradient')
        optimizer.step()
        # Round-2 audit fix: finite gradients do not imply finite Adam moments.
        for state in optimizer.state.values():
            if any(torch.is_tensor(x) and not torch.isfinite(x).all() for x in state.values()):
                raise FloatingPointError('Nonfinite optimizer state after Adam step; see round-2 precision audit')
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('Nonfinite parameter after Adam step')
        with torch.no_grad(): validation=float(costs(c,vt,vt_y,model).mean())
        if validation<best: best=validation;best_state=copy.deepcopy(model.state_dict())
        history.append(dict(epoch=epoch+1,loss=float(loss.detach()),train_cost=float(ct.mean().detach()),validation_cost=validation,lr=optimizer.param_groups[0]['lr']))
        if (epoch+1)%10==0 or epoch+1==epochs:
            torch.save(dict(identity=dict(config=asdict(c),kind=kind,seed=seed,n=n,delta=delta),model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,history=history,best=best,best_state=best_state),checkpoint)
            (root/'history.json').write_text(json.dumps(history,indent=2))
    torch.save(model.state_dict(),root/'last.pt');torch.save(best_state,root/'best_validation.pt')
    (root/'metadata.json').write_text(json.dumps(dict(config=asdict(c),kind=kind,seed=seed,n=n,epochs=epochs,splits=split,delta=model.delta,torch=torch.__version__,device=device,training_data='fixed paths',selection='last model returned; best validation saved separately'),indent=2))
    return model,history

def numpy_policy(model):
    device=next(model.parameters()).device
    def policy(I,Q,t):
        with torch.no_grad(): return model(torch.tensor(I,dtype=torch.float32,device=device),torch.tensor(Q,dtype=torch.float32,device=device),t).cpu().numpy()
    return policy
