"""New submission runner. Default is a cheap plan; --execute starts training.

Grid IDs are canonical IDs introduced by this runner, NOT verified historical
instance IDs. Outputs are deliberately isolated from the submitted results.
"""
import argparse
import itertools
import json
import platform
import time
from pathlib import Path


def grid():
    return [dict(L=L, p=p, demand=demand, b=9. if ratio==.90 else 19., h=1.,
                 T=50, mu=10., sigma=2., co=2., alpha=.99, scale=1.6)
            for L, p, demand, ratio in itertools.product(
                [2, 5], [.55, .70, .90], ['normal', 'gamma'], [.90, .95])]


def train_mean(c, seed, out, device, epochs, n):
    # Newly reconstructed extension: not recovered historical source code.
    import random
    import numpy as np
    import torch
    from learning import costs
    from learning_v2 import DiagnosticWeights, check_numerics, atomic_torch_save
    from core import generate
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = DiagnosticWeights(c, 'teacher_scaled64').to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    d, y = generate(c, n, 100000+seed)
    d, y = [torch.tensor(x, dtype=torch.float64, device=device) for x in (d, y)]
    history = []
    for epoch in range(epochs):
        for group in opt.param_groups:
            group['lr'] = max(.0001, .003*.98**epoch)
        opt.zero_grad()
        loss = costs(c, d, y, model).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite mean loss')
        loss.backward(); check_numerics(model, opt, 'before_step')
        opt.step(); check_numerics(model, opt, 'after_step')
        history.append(dict(epoch=epoch+1, loss=float(loss.detach())))
    atomic_torch_save(model.state_dict(), out/'last.pt')
    (out/'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--instances', type=int, nargs='+', default=list(range(24)))
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--methods', nargs='+', choices=['mean_loss','exponential_loss','GDSH'],
                        default=['mean_loss','exponential_loss','GDSH'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train-paths', type=int, default=500)
    parser.add_argument('--test-paths', type=int, default=1000)
    parser.add_argument('--test-seed', type=int, default=1100000)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, default=Path('runs/submission_rerun'))
    args = parser.parse_args()
    if any(i not in range(24) for i in args.instances): parser.error('IDs must be 0..23')
    if min(args.epochs,args.train_paths,args.test_paths,args.threads)<=0: parser.error('Counts must be positive')
    plan = dict(configurations=[dict(canonical_id=i, **grid()[i]) for i in args.instances],
                seeds=args.seeds, methods=args.methods, epochs=args.epochs,
                train_paths=args.train_paths, validation_paths=100,
                train_seed='100000 + seed', validation_seed='200000 + seed',
                test_paths=args.test_paths, test_seed=args.test_seed,
                model_selection='last epoch', device=args.device, threads=args.threads,
                provenance='New runner and reconstructed mean-loss trainer; historical grid-ID mapping unverified')
    print(json.dumps(plan, indent=2))
    count=len(args.instances)*len(args.seeds)*len(args.methods)
    print(f'{count} training runs; default full grid: 216 runs / 64800 epochs. Duration unmeasured; budget hours to days, calibrate a pilot first.')
    if not args.execute: return
    import numpy as np
    import torch
    from core import Config, generate, rollout, heuristic
    from learning import train, costs
    from learning_v2 import fit, evaluate
    torch.set_num_threads(args.threads)
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    # Refuse overwrites; a failed run is retained for diagnosis, never silently retried.
    args.output.mkdir(parents=True, exist_ok=False)
    plan.update(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    records=[]
    for i in args.instances:
        c=Config(**grid()[i]); d,y=generate(c,args.test_paths,args.test_seed)
        refs={name:float(rollout(c,d,y,heuristic(c,name)).mean()) for name in ['LIR','WNH']}
        for seed in args.seeds:
            for method in args.methods:
                out=args.output/f'config_{i:02d}'/f'seed_{seed}'/method
                out.mkdir(parents=True)
                start=time.perf_counter()
                if method=='mean_loss':
                    model=train_mean(c,seed,out,args.device,args.epochs,args.train_paths)
                elif method=='exponential_loss':
                    model,meta=fit(c,'teacher_scaled64',seed,out,n=args.train_paths,
                                   epochs=args.epochs,device=args.device,resume=False)
                    if meta['status']!='completed': raise RuntimeError(meta)
                else:
                    model,_=train(c,'GDSH',seed,args.train_paths,args.epochs,args.device,str(out))
                if method=='GDSH':
                    with torch.no_grad():
                        values=costs(c,*[torch.tensor(x,dtype=torch.float32,device=args.device) for x in (d,y)],model).cpu().numpy()
                else:
                    values=evaluate(model,c,args.test_paths,args.test_seed,args.device)
                np.save(out/'test_costs.npy',values)
                record=dict(canonical_id=i,seed=seed,method=method,config=grid()[i],
                            mean_cost=float(values.mean()),seconds=time.perf_counter()-start,
                            **{f'{name}_mean':v for name,v in refs.items()})
                records.append(record)
                (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
                print(json.dumps(record),flush=True)


if __name__=='__main__': main()
