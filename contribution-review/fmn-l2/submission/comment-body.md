Submitted the correction in https://github.com/bethgelab/foolbox/pull/742. The three projection regressions fail before the fix and pass afterwards, with NumPy and CPU PyTorch validation.

The review includes [real-digit results and limitations](https://github.com/daniyyell-dev/foolbox/tree/47f6405bdfc976fe12ab046c0b00d92ebc89b7d3/contribution-review/fmn-l2), including lower success counts when the adaptive radius is enforced. The first four selected samples are shown below.

![Real-digit before/after examples](https://raw.githubusercontent.com/daniyyell-dev/foolbox/47f6405bdfc976fe12ab046c0b00d92ebc89b7d3/contribution-review/fmn-l2/digit-samples.png)

Reference: Maura Pintor, Fabio Roli, Wieland Brendel and Battista Biggio. **Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints.** NeurIPS 2021. [Published paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a709909b1ea5c2bee24248203b1728a5-Abstract.html).
