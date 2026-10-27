`L2FMNAttack.project` currently computes `min(1, norms / norms)`, which always yields one, and computes `norms` from `x` instead of `x - x0`. Consequently it can return a perturbation outside the requested L2 ball.

## Reproduce

```python
import eagerpy as ep
import numpy as np
from foolbox.attacks import L2FMNAttack

x0 = ep.astensor(np.array([[0., 0.]], dtype=np.float32))
x = ep.astensor(np.array([[3., 4.]], dtype=np.float32))
epsilon = ep.astensor(np.array([0.5], dtype=np.float32))
projected = L2FMNAttack().project(x, x0, epsilon)
print(projected.numpy())
print(np.linalg.norm(projected.numpy() - x0.numpy()))
# Current upstream: [[3., 4.]], distance 5.0.
# Expected: [[0.3, 0.4]], distance 0.5.
```

Reproduced on upstream `2513a9a8675d7017e5266d3b0ed89124cb436ec5`, NumPy 2.5.4 / Python 3.13.13, and NumPy 1.26.4 / Python 3.10.20.

A [two-line correction and regression tests are prepared](https://github.com/daniyyell-dev/foolbox/tree/fix/fmn-l2-projection). The tests cover translated centres, scalar/per-sample radii, vector/image shapes, interior points and zero radii; all three new cases fail on upstream and pass after the correction. [Review and real-digit examples](https://github.com/daniyyell-dev/foolbox/tree/47f6405bdfc976fe12ab046c0b00d92ebc89b7d3/contribution-review/fmn-l2) include the change in attack behavior and its limitations.

## Reference

Maura Pintor, Fabio Roli, Wieland Brendel and Battista Biggio. **Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints.** NeurIPS 2021. [Published paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a709909b1ea5c2bee24248203b1728a5-Abstract.html).
