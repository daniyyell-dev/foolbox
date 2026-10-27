`L2FMNAttack.project` ignored its radius because `norms / norms` was always one. For centre `[0,0]`, candidate `[3,4]` and radius `0.5`, it returned distance `5.0`. The correction computes the displacement norm and scales it by `min(1, epsilon / norm)`, returning `[0.3,0.4]` at distance `0.5`.

Fixes https://github.com/bethgelab/foolbox/issues/741

Adds backend-parameterized regression cases for translated centres, scalar and per-sample radii, vector/image shapes, interior points, zero perturbations/radii and projection idempotence. The three cases fail before the change and pass afterwards.

Validation: NumPy FMN/distance/criteria/devutils tests (40 passed, 162 skipped); CPU PyTorch projection and existing simple-model FMN tests (23 passed, 20 skipped); Python 3.10/NumPy 1.26 projection tests (3 passed). Pinned Black, flake8, targeted mypy and whitespace checks pass. TensorFlow/JAX, GPU and full-suite validation were not run locally.

A 100-sample held-out real-digit comparison is included in the [review evidence](https://github.com/daniyyell-dev/foolbox/tree/47f6405bdfc976fe12ab046c0b00d92ebc89b7d3/contribution-review/fmn-l2). On the same 82 untargeted successes, the median L2 perturbation changed from 1.238255 to 0.502980. Success counts decreased from 100 to 82 under the corrected radius constraint at the same 100-iteration budget; this fix does not claim to improve every attack's success rate. Targeted results and all failures are also recorded.

![First four selected real digits](https://raw.githubusercontent.com/daniyyell-dev/foolbox/47f6405bdfc976fe12ab046c0b00d92ebc89b7d3/contribution-review/fmn-l2/digit-samples.png)

Reference: Maura Pintor, Fabio Roli, Wieland Brendel and Battista Biggio. **Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints.** NeurIPS 2021. [Published paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a709909b1ea5c2bee24248203b1728a5-Abstract.html).
