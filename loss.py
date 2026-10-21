import torch
from torch.nn import functional as F


def supervised_contrastive(features, labels, temperature=.1):
    features = F.normalize(features.float(), dim=1)
    logits = features @ features.t() / temperature
    diagonal = torch.eye(len(features), device=features.device, dtype=torch.bool)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    counts = positives.sum(1)
    valid = counts > 0
    if not valid.any():
        return features.sum() * 0.
    denominator = torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1)
    log_probability = logits - denominator[:, None]
    return -((log_probability * positives).sum(1)[valid] / counts[valid]).mean()


def objective(output, images, labels, model_name):
    ce = F.cross_entropy(output['logits'], labels)
    zero = ce * 0.
    recon, contrast = zero, zero
    if model_name not in ['baseline', 'no_physloss']:
        recon = F.l1_loss(output['reconstruction'], images)
        contrast = supervised_contrastive(output['features'], labels)
    total = ce + .1 * recon + .1 * contrast
    return total, dict(ce=ce, reconstruction=recon, contrastive=contrast)
