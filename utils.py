
###########

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        print('{}: {}', name, param)
        total_params += param
    print(total_params)
    print(f"Total Trainable Params: {total_params}")
    return total_params

############

from shutil import copyfile
import os
def save_code(tensorboard_folder):
    dst = os.path.join(tensorboard_folder, 'code')
    os.mkdir(dst)
    src_files = [
            'dataset.py',
           'utils_dataset.py',
            'utils.py',
          'train_hcp_timecuts_GGRU.py',
          'train_hcp_timecuts_GGRU_kfold.py',
          'train_ABIDE_timecuts_GGRU_kfold.py',
          'train_ABIDE_timecuts_GGRU_kfold_regression.py',
          'train_hcp_timecuts.py',
          'model_hcp.py']
    for src in src_files:
        copyfile(src, os.path.join(dst, src))

####################

import matplotlib.pyplot as plt

def draw_curves(lst_iter, lst_loss, lst_acc, title):
    plt.plot(lst_iter, lst_loss[0], '-b', label='train loss')
    plt.plot(lst_iter, lst_loss[1], '-r', label='validation loss')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)

    plt.show()

    if lst_acc:
        plt.plot(lst_iter, lst_acc[0], '-b', label='train accuracy')
        plt.plot(lst_iter, lst_acc[1], '-r', label='validation accuracy')

        plt.xlabel("n iteration")
        plt.legend(loc='upper left')
        plt.title(title)

        plt.show()

####################

class style():
    CEND = '\33[0m'
    CBOLD = '\33[1m'
    CITALIC = '\33[3m'
    CURL = '\33[4m'
    CBLINK = '\33[5m'
    CBLINK2 = '\33[6m'
    CSELECTED = '\33[7m'

    CBLACK = '\33[30m'
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CYELLOW = '\33[33m'
    CBLUE = '\33[34m'
    CVIOLET = '\33[35m'
    CBEIGE = '\33[36m'
    CWHITE = '\33[37m'

    CBLACKBG = '\33[40m'
    CREDBG = '\33[41m'
    CGREENBG = '\33[42m'
    CYELLOWBG = '\33[43m'
    CBLUEBG = '\33[44m'
    CVIOLETBG = '\33[45m'
    CBEIGEBG = '\33[46m'
    CWHITEBG = '\33[47m'

    CGREY = '\33[90m'
    CRED2 = '\33[91m'
    CGREEN2 = '\33[92m'
    CYELLOW2 = '\33[93m'
    CBLUE2 = '\33[94m'
    CVIOLET2 = '\33[95m'
    CBEIGE2 = '\33[96m'
    CWHITE2 = '\33[97m'

    CGREYBG = '\33[100m'
    CREDBG2 = '\33[101m'
    CGREENBG2 = '\33[102m'
    CYELLOWBG2 = '\33[103m'
    CBLUEBG2 = '\33[104m'
    CVIOLETBG2 = '\33[105m'
    CBEIGEBG2 = '\33[106m'
    CWHITEBG2 = '\33[107m'

    GREEN_BOLD = '\033[1;32m'


    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

#################

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/focal.html

class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to [1], the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.


    Arguments:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (Optional[str]): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = FocalLoss(alpha=0.5, gamma=2.0, reduction='mean')
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()

    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float, gamma: Optional[float] = 2.0,
                 reduction: Optional[str] = 'none') -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: Optional[float] = gamma
        self.reduction: Optional[str] = reduction
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 2:
            #raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
            raise ValueError("Invalid input shape, we expect BxN. Got: {}"
                             .format(input.shape))
        #if not input.shape[-2:] == target.shape[-2:]:
        #    raise ValueError("input and target shapes must be the same. Got: {}"
        #                     .format(input.shape, input.shape))
        if not input.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    input.device, target.device))
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1) + self.eps

        # create the labels one hot tensor
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).to(input.device)

        # compute the actual focal loss
        weight = torch.pow(1. - input_soft, self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)

        loss = -1
        if self.reduction == 'none':
            loss = loss_tmp
        elif self.reduction == 'mean':
            loss = torch.mean(loss_tmp)
        elif self.reduction == 'sum':
            loss = torch.sum(loss_tmp)
        else:
            raise NotImplementedError("Invalid reduction mode: {}"
                                      .format(self.reduction))
        return loss



if __name__ == "__main__":
    print(":P")