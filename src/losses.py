"""Physics-informed loss functions for gravitational-lensing super-resolution."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeSRLoss(nn.Module):
    """Physics-informed composite loss: L1 + flux consistency + back-projection.

    L_total = L_L1(SR, HR) + λ_flux · L_flux(SR, HR) + λ_bp · L_bp(SR, LR)

    Components
    ----------
    L1 : pixel-wise MAE.
        Sharper than L2 which over-smooths by penalising large errors
        quadratically, pushing predictions toward the conditional mean.
    Flux consistency : |Σ SR − Σ HR| / N_pixels.
        Preserves total integrated intensity — gravitational lensing conserves
        photon flux, so the super-resolved image must have the same total
        brightness as the ground truth.
    Back-projection : L1(downsample(SR), LR).
        Ensures the SR image is consistent with the LR observation when
        downsampled through the degradation model.  Only requires knowledge of
        the degradation operator, not paired data — a key insight for
        unsupervised SR.
    """

    def __init__(self, lambda_flux: float = 0.05, lambda_bp: float = 0.1) -> None:
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_flux = lambda_flux
        self.lambda_bp = lambda_bp

    def forward(
        self, sr: torch.Tensor, hr: torch.Tensor, lr: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Pixel-wise L1
        l1_loss = self.l1(sr, hr)

        # Flux consistency
        sr_flux = sr.sum(dim=(1, 2, 3))  # per-image flux
        hr_flux = hr.sum(dim=(1, 2, 3))
        n_pixels = sr.shape[-1] * sr.shape[-2]
        flux_loss = torch.mean(torch.abs(sr_flux - hr_flux)) / n_pixels

        # Back-projection
        sr_down = F.interpolate(
            sr, size=lr.shape[-2:], mode="bicubic", align_corners=False
        )
        bp_loss = self.l1(sr_down, lr)

        total = (
            l1_loss
            + self.lambda_flux * flux_loss
            + self.lambda_bp * bp_loss
        )

        return total, {
            "l1": l1_loss.item(),
            "flux": flux_loss.item(),
            "bp": bp_loss.item(),
            "total": total.item(),
        }


class L2SPRegularizer:
    """L2-SP regularization (Li et al., 2018).

    Penalises deviation from pretrained weights rather than raw magnitude:

        L_L2SP = α · Σ_i (θ_i − θ_i^pretrained)²

    This prevents catastrophic forgetting during fine-tuning by anchoring
    parameters near their pretrained values instead of near zero (as standard
    L2 / weight-decay would).
    """

    def __init__(self, model: nn.Module, alpha: float = 0.01) -> None:
        self.pretrained: dict[str, torch.Tensor] = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        self.alpha = alpha

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the L2-SP penalty for the current model state.

        Returns
        -------
        torch.Tensor
            Scalar: α · Σ_i ‖θ_i − θ_i⁰‖²
        """
        l2sp = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.pretrained:
                l2sp = l2sp + (param - self.pretrained[name]).pow(2).sum()
        return self.alpha * l2sp


if __name__ == "__main__":
    # --- CompositeSRLoss ---
    lr = torch.randn(2, 1, 32, 32)
    hr = torch.randn(2, 1, 64, 64)
    sr = torch.randn(2, 1, 64, 64, requires_grad=True)

    criterion = CompositeSRLoss()
    total, components = criterion(sr, hr, lr)
    print("CompositeSRLoss")
    for k, v in components.items():
        print(f"  {k}: {v:.6f}")
    total.backward()
    print(f"  sr.grad norm: {sr.grad.norm().item():.6f}\n")

    # --- L2SPRegularizer ---
    model = nn.Linear(8, 4)
    reg = L2SPRegularizer(model, alpha=0.01)

    # Before any update the penalty should be zero
    p0 = reg.penalty(model)
    print(f"L2SPRegularizer")
    print(f"  penalty before update: {p0.item():.6f}")

    # Perturb weights and check that penalty increases
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param))
    p1 = reg.penalty(model)
    print(f"  penalty after update:  {p1.item():.6f}")
