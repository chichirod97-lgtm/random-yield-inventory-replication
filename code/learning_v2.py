"""Round 2: unchanged exponential objective; precision and controlled factors."""
from pathlib import Path
from dataclasses import asdict
import json,math,copy,time,random
import numpy as np
import torch
from torch import nn
from core import Config,generate,rollout,heuristic
from learning import Weights,costs

VARIANTS={
    'precision64':dict(teacher_init=False,scaled=False),
    'teacher64':dict(teacher_init=True,scaled=False),
    'scaled64':dict(teacher_init=False,scaled=True),
    'teacher_scaled64':dict(teacher_init=True,scaled=True),
}

class DiagnosticWeights(Weights):
    def __init__(self,c,variant='precision64',dtype=torch.float64):
        super().__init__(c,'DRL')
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;self.options=VARIANTS[variant];self.target=float(c.target)
        # Identical float32 parameter draws to v1, then cast. Only arithmetic precision changes.
        self.to(dtype=dtype)
        if self.options['teacher_init']:
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,I,Q,t):
        gaps=torch.relu(self.target-I[:,None]-Q@self.B.T)
        state=torch.cat([torch.full_like(I[:,None],self.target),I[:,None],Q],1)
        if self.options['scaled']:state=state/max(abs(self.target),1.)
        eta=self.net(state).reshape(-1,len(self.teacher),2).softmax(-1)
        bounds=torch.stack([self.teacher-self.delta,self.teacher+self.delta],-1)
        w=(eta*bounds).sum(-1)
        return torch.relu((gaps*w).sum(1))

def check_numerics(model,optimizer,stage):
    for name,p in model.named_parameters():
        if not torch.isfinite(p).all():raise FloatingPointError(f'{stage}: nonfinite parameter {name}')
        if p.grad is not None and not torch.isfinite(p.grad).all():raise FloatingPointError(f'{stage}: nonfinite gradient {name}')
    for index,state in optimizer.state.items():
        for name,x in state.items():
            if torch.is_tensor(x) and not torch.isfinite(x).all():
                raise FloatingPointError(f'{stage}: nonfinite optimizer {name}')

def loss_exponential(cost,teacher):
    if not (teacher>0).all():raise ValueError('Teacher cost must be positive')
    z=100*(cost-teacher)/teacher
    return torch.exp(z).mean(),z

def atomic_torch_save(payload,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)

def atomic_json_write(payload,path):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2))
    temporary.replace(path)

def fit(c,variant,seed,out,n=500,epochs=300,device='cpu',resume=True):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    model=DiagnosticWeights(c,variant).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    d,y=generate(c,n,100000+seed);vd,vy=generate(c,100,200000+seed)
    teacher=torch.tensor(rollout(c,d,y,heuristic(c,'WNH')),dtype=torch.float64,device=device)
    td,ty,tvd,tvy=[torch.tensor(a,dtype=torch.float64,device=device) for a in [d,y,vd,vy]]
    identity=dict(config=asdict(c),variant=variant,seed=seed,n=n,dtype='float64',objective='mean_exp_percent_gap',schedule='0.003*0.98**epoch; floor .0001')
    history=[];start=0;best=float('inf');best_state=None;elapsed=0.;checkpoint=root/'checkpoint.pt'
    if resume and checkpoint.exists():
        ck=torch.load(checkpoint,map_location=device,weights_only=False)
        if ck['identity']!=identity:raise ValueError('Checkpoint mismatch')
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);history=ck['history'];start=ck['epoch']+1;best=ck['best'];best_state=ck['best_state'];elapsed=ck['elapsed_training_seconds']
    status='completed';reason=None;failed_epoch=None;begin=time.perf_counter()
    try:
        check_numerics(model,optimizer,'resume')
        for epoch in range(start,epochs):
            for group in optimizer.param_groups:group['lr']=max(.0001,.003*.98**epoch)
            optimizer.zero_grad();ct=costs(c,td,ty,model);loss,z=loss_exponential(ct,teacher)
            if not torch.isfinite(loss):raise FloatingPointError('loss is nonfinite')
            loss.backward();check_numerics(model,optimizer,'before_step')
            optimizer.step();check_numerics(model,optimizer,'after_step')
            with torch.no_grad():val=float(costs(c,tvd,tvy,model).mean())
            if not math.isfinite(val):raise FloatingPointError('validation cost is nonfinite')
            if val<best:best=val;best_state=copy.deepcopy(model.state_dict())
            history.append(dict(epoch=epoch+1,loss=float(loss.detach()),exponent_max=float(z.max().detach()),train_cost=float(ct.mean().detach()),validation_cost=val))
            if (epoch+1)%10==0 or epoch+1==epochs:
                atomic_torch_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,epoch=epoch,best=best,best_state=best_state,elapsed_training_seconds=elapsed+time.perf_counter()-begin),checkpoint)
                atomic_json_write(history,root/'history.json')
    except FloatingPointError as error:
        status='failed';reason=str(error);failed_epoch=locals().get('epoch',start)+1
        torch.save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,failed_epoch=failed_epoch),root/'failure_checkpoint.pt')
    metadata=dict(**identity,status=status,reason=reason,failed_epoch=failed_epoch,completed_epochs=len(history),elapsed_training_seconds=elapsed+time.perf_counter()-begin,selection='last model; best validation retained but not used in headline',torch=torch.__version__,device=device)
    atomic_json_write(metadata,root/'metadata.json');atomic_json_write(history,root/'history.json')
    if status=='completed':
        atomic_torch_save(model.state_dict(),root/'last.pt');atomic_torch_save(best_state,root/'best_validation.pt')
        atomic_torch_save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,epoch=len(history)-1,best=best,best_state=best_state,elapsed_training_seconds=metadata['elapsed_training_seconds']),checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['epoch']!=len(history)-1 or len(saved['history'])!=len(history):raise IOError('Final checkpoint verification failed')
    return model,metadata

@torch.no_grad()
def evaluate(model,c,n=1000,seed=400000,device='cpu'):
    d,y=generate(c,n,seed)
    return costs(c,torch.tensor(d,dtype=torch.float64,device=device),torch.tensor(y,dtype=torch.float64,device=device),model).cpu().numpy()
