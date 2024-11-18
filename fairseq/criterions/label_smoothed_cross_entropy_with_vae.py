# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
from dataclasses import dataclass, field

from fairseq import utils
from fairseq.logging import metrics
from . import FairseqCriterion, register_criterion
from .label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, LabelSmoothedCrossEntropyCriterionConfig

def vae_kld_loss(mean, log_std, padding_mask=None):
    if mean is None or log_std is None:
        return torch.tensor(0)
    if padding_mask is not None:
        # Ensure the padding mask is a boolean tensor
        padding_mask = ~padding_mask.transpose(0, 1).unsqueeze(-1).bool()
        
        # Apply the mask to the tensor
        mean = mean * padding_mask
        log_std = log_std * padding_mask
    
    kld = -0.5 * torch.sum(
        1 + log_std - mean.pow(2) - torch.exp(log_std), dim=-1
    )

    kld = kld.mean()
    return kld

@dataclass
class LabelSmoothedCrossEntropyWithVAECriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    vae_beta: float = field(
        default=1.0,
        metadata={"help": "weight of kld loss"},
    )


@register_criterion('label_smoothed_cross_entropy_with_vae', dataclass=LabelSmoothedCrossEntropyWithVAECriterionConfig)
class LabelSmoothedCrossEntropyCriterionWithVAE(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False, vae_beta=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size=ignore_prefix_size, report_accuracy=report_accuracy)
        self.beta = vae_beta

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'], )
        loss, nll_loss, kld_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'kld_loss': utils.item(kld_loss.data) if reduce else kld_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce=reduce)
        kld_loss = vae_kld_loss(net_output[-3], net_output[-2], net_output[-1])
        loss = loss + self.beta * kld_loss
        return loss, nll_loss, kld_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        kld_loss_sum = sum(log.get("kld_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "kld_loss", kld_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
    
        