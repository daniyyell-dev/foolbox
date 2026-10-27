"""Compare only the upstream and corrected L2 projection on the same held-out real digits."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import eagerpy as ep
import sys
sys.path.insert(0, str(Path.cwd()))
import foolbox as fb
assert Path(fb.__file__).resolve().parent.parent == Path.cwd().resolve(), "Run from the Foolbox checkout"
from foolbox.devutils import flatten, atleast_kd


def upstream_projection(self, x, x0, epsilon):
    norms = flatten(x).norms.l2(axis=-1)
    norms = ep.maximum(norms, 1e-12)
    factor = ep.minimum(1, norms / norms)
    return x0 + (x - x0) * atleast_kd(factor, x.ndim)


torch.set_num_threads(1)
torch.manual_seed(42)
x, y = load_digits(return_X_y=True)
x = (x / 16).astype(np.float32)
train, test = train_test_split(np.arange(len(y)), test_size=0.25, random_state=42, stratify=y)
fitted = MLPClassifier(hidden_layer_sizes=(32,), activation='tanh', solver='lbfgs', max_iter=1000, random_state=42).fit(x[train], y[train])
model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 10)).eval()
with torch.no_grad():
    for layer, w, b in zip([model[0], model[2]], fitted.coefs_, fitted.intercepts_):
        layer.weight.copy_(torch.from_numpy(w.T.astype(np.float32)))
        layer.bias.copy_(torch.from_numpy(b.astype(np.float32)))
fmodel = fb.PyTorchModel(model, bounds=(0, 1), device='cpu')
clean = fmodel(torch.from_numpy(x[test])).argmax(1).numpy()
selected = test[clean == y[test]][:100]
inputs, labels = x[selected], y[selected]
arrays = {'x': inputs, 'y': labels, 'indices': selected}
results = {'dataset': 'sklearn load_digits', 'train_samples': len(train), 'test_samples': len(test),
           'clean_correct': int((clean == y[test]).sum()), 'attacked_samples': len(selected),
           'selection': 'first correctly classified held-out samples; stratified seed 42', 'iterations': 100,
           'upstream_base': '2513a9a8675d7017e5266d3b0ed89124cb436ec5', 'results': []}
fixed_projection = fb.attacks.L2FMNAttack.project
for name, projection in [('before', upstream_projection), ('after', fixed_projection)]:
    for targeted in (False, True):
        targets = (labels + 1) % 10 if targeted else labels
        criterion = fb.TargetedMisclassification if targeted else fb.Misclassification
        with patch.object(fb.attacks.L2FMNAttack, 'project', projection):
            adv = fb.attacks.L2FMNAttack(steps=100, max_stepsize=1, min_stepsize=0.01).run(
                fmodel, torch.from_numpy(inputs), criterion(torch.from_numpy(targets))).numpy()
        predictions = fmodel(torch.from_numpy(adv)).argmax(1).numpy()
        success = predictions == targets if targeted else predictions != labels
        norms = np.linalg.norm(adv - inputs, axis=1)
        key = name + ('_targeted' if targeted else '_untargeted')
        arrays[key] = adv
        arrays[key + '_predictions'] = predictions
        results['results'].append({'mode': key, 'successes': int(success.sum()),
                                  'median_norm_successful': float(np.median(norms[success])) if success.any() else None,
                                  'within_bounds': bool(np.all((adv >= 0) & (adv <= 1)))})
results['matched_successes'] = []
for mode in ['untargeted', 'targeted']:
    before = arrays['before_' + mode + '_predictions']
    after = arrays['after_' + mode + '_predictions']
    targets = (labels + 1) % 10
    common = (before == targets) & (after == targets) if mode == 'targeted' else (before != labels) & (after != labels)
    before_norms = np.linalg.norm(arrays['before_' + mode] - inputs, axis=1)
    after_norms = np.linalg.norm(arrays['after_' + mode] - inputs, axis=1)
    results['matched_successes'].append({'mode': mode, 'samples': int(common.sum()),
        'median_before': float(np.median(before_norms[common])), 'median_after': float(np.median(after_norms[common]))})
results['versions'] = {'foolbox': fb.__version__, 'torch': torch.__version__, 'numpy': np.__version__}
results['device'] = 'cpu'
root = Path(__file__).resolve().parent
(root/'digits-results.json').write_text(json.dumps(results, indent=2)+'\n')
np.savez_compressed(root/'digits-samples.npz', **arrays)
print(json.dumps(results, indent=2))
