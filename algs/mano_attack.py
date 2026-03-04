"""Adversarial attacks on the MaNo evaluation metric.

1. Norm Maximization Attack (MaNoMaxAttack):
   Minimizes CE(f(x+d), k*) to push predictions toward one-hot vectors.
   Effect: inflates Lp norm -> MaNo overestimates accuracy.

2. Norm Minimization Attack (MaNoMinAttack):
   Minimizes CE(sigma(f(x+d)), T) where T is near-uniform with margin gamma.
   Effect: flattens Lp norm -> MaNo underestimates accuracy.

Both attacks use PGD with L-inf constraint and preserve the original predicted class.
"""

import torch
import torch.nn.functional as F
from algs.mano import MaNo

# ImageNet normalization (used by all datasets in this project)
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


def _pgd(model, inputs, eps, alpha, num_steps, opt_targets):
    """Core PGD optimization (gradient descent on cross-entropy).

    Args:
        model: neural network.
        inputs: normalized input tensor (B, C, H, W).
        eps: L-inf budget in pixel space [0, 1].
        alpha: step size in pixel space [0, 1].
        num_steps: number of PGD iterations.
        opt_targets: hard labels (LongTensor of shape B) or
                     soft labels (FloatTensor of shape (B, K)).

    Returns:
        Adversarial inputs tensor (detached).
    """
    device = inputs.device
    std = torch.tensor(_NORM_STD, device=device).view(1, 3, 1, 1)
    mean = torch.tensor(_NORM_MEAN, device=device).view(1, 3, 1, 1)
    # Convert pixel-space budget to normalized space (per-channel)
    eps_n = eps / std
    alpha_n = alpha / std
    # Valid input range in normalized space
    lower = (0.0 - mean) / std
    upper = (1.0 - mean) / std

    was_training = model.training
    model.eval()

    # Freeze model params during PGD for memory efficiency
    orig_rg = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad_(False)

    delta = torch.zeros_like(inputs, requires_grad=True)

    for _ in range(num_steps):
        logits = model(inputs + delta)
        loss = F.cross_entropy(logits, opt_targets)
        loss.backward()

        with torch.no_grad():
            # Gradient descent step
            delta.data -= alpha_n * delta.grad.sign()
            # Project onto L-inf ball
            delta.data = torch.max(torch.min(delta.data, eps_n), -eps_n)
            # Clamp to valid pixel range
            delta.data = torch.max(torch.min(inputs + delta.data, upper), lower) - inputs
            delta.grad.zero_()

    # Restore model state
    for n, p in model.named_parameters():
        p.requires_grad_(orig_rg[n])
    if was_training:
        model.train()

    return (inputs + delta).detach()


def pgd_maximize_norm(model, inputs, num_classes, eps, alpha, num_steps):
    """Norm Maximization: minimize CE(f(x+d), k*) -> one-hot prediction."""
    model.eval()
    with torch.no_grad():
        k_star = model(inputs).argmax(dim=1)
    return _pgd(model, inputs, eps, alpha, num_steps, k_star)


def pgd_minimize_norm(model, inputs, num_classes, eps, alpha, num_steps, gamma):
    """Norm Minimization: minimize CE(sigma(f(x+d)), T) -> near-uniform prediction.

    Target distribution T:
        T_{k*} = 1/K + gamma
        T_{j != k*} = (1 - T_{k*}) / (K - 1)
    """
    model.eval()
    B, K = inputs.shape[0], num_classes
    with torch.no_grad():
        k_star = model(inputs).argmax(dim=1)

    T_k = 1.0 / K + gamma
    T_other = (1.0 - T_k) / (K - 1)
    T = torch.full((B, K), T_other, device=inputs.device)
    T.scatter_(1, k_star.unsqueeze(1), T_k)

    return _pgd(model, inputs, eps, alpha, num_steps, T)


class _MaNoAttackBase(MaNo):
    """Shared evaluate() logic for both attack variants."""

    def _attack_fn(self, inputs):
        raise NotImplementedError

    def evaluate(self):
        self.base_model.train()
        self.phi = self.uniform_cross_entropy()

        score_list = []
        adv_correct, adv_total = 0, 0

        for batch_idx, batch_data in enumerate(self.val_loader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            inputs = self._gpu_resize(inputs)

            # PGD attack
            adv_inputs = self._attack_fn(inputs)

            # MaNo scoring on adversarial inputs (train mode for BatchNorm)
            self.base_model.train()
            with torch.no_grad():
                outputs = self.base_model(adv_inputs)

                # Adversarial accuracy
                _, predicted = outputs.max(1)
                adv_total += len(predicted)
                adv_correct += predicted.eq(labels).sum().item()

                # Lp norm score
                scaled = self.scaling_method(outputs)
                score = torch.norm(scaled, p=self.args['norm_type']) / (
                    (scaled.shape[0] * scaled.shape[1]) ** (1 / self.args['norm_type']))
            score_list.append(score)

        self.adv_acc = 100.0 * adv_correct / adv_total if adv_total > 0 else 0.0
        scores = torch.Tensor(score_list).numpy()
        return scores.mean()


class MaNoMaxAttack(_MaNoAttackBase):
    """Norm Maximization Attack: inflates MaNo score."""

    def _attack_fn(self, inputs):
        return pgd_maximize_norm(
            self.base_model, inputs,
            num_classes=self.args['num_classes'],
            eps=self.args.get('attack_eps', 8 / 255),
            alpha=self.args.get('attack_alpha', 2 / 255),
            num_steps=self.args.get('attack_steps', 20),
        )


class MaNoMinAttack(_MaNoAttackBase):
    """Norm Minimization Attack: deflates MaNo score."""

    def _attack_fn(self, inputs):
        return pgd_minimize_norm(
            self.base_model, inputs,
            num_classes=self.args['num_classes'],
            eps=self.args.get('attack_eps', 8 / 255),
            alpha=self.args.get('attack_alpha', 2 / 255),
            num_steps=self.args.get('attack_steps', 20),
            gamma=self.args.get('attack_gamma', 0.05),
        )


# ── Analytical attacks (zero-cost logit temperature scaling) ─────────

class _MaNoAnalyticalBase(MaNo):
    """Zero-cost attack via logit temperature scaling.

    No input perturbation or gradient computation. Same speed as clean MaNo.
    Directly scales logits before the scoring function:
      - Low temperature (tau -> 0): softmax -> one-hot  (max attack)
      - High temperature (tau -> inf): softmax -> uniform (min attack)
    Class ordering (argmax) is always preserved.
    """

    def _scale_logits(self, logits):
        raise NotImplementedError

    def evaluate(self):
        self.base_model.train()
        self.phi = self.uniform_cross_entropy()

        score_list = []
        adv_correct, adv_total = 0, 0

        for batch_idx, batch_data in enumerate(self.val_loader):
            inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)
            inputs = self._gpu_resize(inputs)

            with torch.no_grad():
                logits = self.base_model(inputs)

                # Accuracy (based on original logits — class order preserved)
                _, predicted = logits.max(1)
                adv_total += len(predicted)
                adv_correct += predicted.eq(labels).sum().item()

                # MaNo score on temperature-scaled logits
                scaled = self.scaling_method(self._scale_logits(logits))
                score = torch.norm(scaled, p=self.args['norm_type']) / (
                    (scaled.shape[0] * scaled.shape[1]) ** (1 / self.args['norm_type']))
            score_list.append(score)

        self.adv_acc = 100.0 * adv_correct / adv_total if adv_total > 0 else 0.0
        scores = torch.Tensor(score_list).numpy()
        return scores.mean()


class MaNoMaxAttackFast(_MaNoAnalyticalBase):
    """Analytical norm maximization: logits / tau (tau -> 0) -> one-hot."""

    def _scale_logits(self, logits):
        tau = self.args.get('attack_tau', 0.01)
        return logits / tau


class MaNoMinAttackFast(_MaNoAnalyticalBase):
    """Analytical norm minimization: logits * tau (tau -> 0) -> uniform."""

    def _scale_logits(self, logits):
        tau = self.args.get('attack_tau', 0.01)
        return logits * tau
