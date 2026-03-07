import torch
import torch.nn as nn

def trimmed_mae_loss(prediction, target, mask, trim=0.2):
    M = torch.sum(mask, (1, 2))
    res = prediction - target

    res = res[mask.bool()].abs()

    trimmed, _ = torch.sort(res.view(-1), descending=False)[
        : int(len(res) * (1.0 - trim))
    ]

    return trimmed.sum() / (2 * M.sum())

def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor

def normalize_prediction_robust(target, mask):
    ssum = torch.sum(mask, (1, 2))
    valid = ssum > 0

    m = torch.zeros_like(ssum)
    s = torch.ones_like(ssum)

    m[valid] = torch.median(
        (mask[valid] * target[valid]).view(valid.sum(), -1), dim=1
    ).values
    target = target - m.view(-1, 1, 1)

    sq = torch.sum(mask * target.abs(), (1, 2))
    s[valid] = torch.clamp((sq[valid] / ssum[valid]), min=1e-6)

    return target / (s.view(-1, 1, 1))

class TrimmedProcrustesLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction="batch-based"):
        super(TrimmedProcrustesLoss, self).__init__()

        self.__data_loss = trimmed_mae_loss
        self.__regularization_loss = GradientLoss(scales=scales)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target: torch.Tensor, mask=None):
        if target.dim() == 4:
            assert prediction.shape[1]==1 and target.shape[1]==1
            target = target.squeeze(1)
            prediction = prediction.squeeze(1)
        if mask is None:
            mask = torch.ones_like(target)
        self.__prediction_ssi = normalize_prediction_robust(prediction, mask)
        target_ = normalize_prediction_robust(target, mask)

        total = self.__data_loss(self.__prediction_ssi, target_, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(
                self.__prediction_ssi, target_, mask
            )

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


class GradientLoss(nn.Module):
    def __init__(self, scales=4):
        super().__init__()

        self.__reduction = reduction_batch_based

        self.__scales = scales

    def forward(self, prediction, target, mask=None):
        # if target.dim() == 4:
        #     assert prediction.shape[1]==1 and target.shape[1]==1
        #     target = target.squeeze(1)
        #     prediction = prediction.squeeze(1)
        # if mask is None:
        #     mask = torch.ones_like(target)
        total = 0
        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)