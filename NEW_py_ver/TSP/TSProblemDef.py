
import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    problems = torch.rand(size=(batch_size, problem_size, 2, 128))
    # problems.shape: (batch, problem, 2, 128)
    return problems


def augment_xy_data_by_2_fold(problems):
    # problems.shape: (batch, problem, 2, 128)

    x = problems[:, :, [0], :]
    y = problems[:, :, [1], :]
    # x,y shape: (batch, problem, 1, 128)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((y, x), dim=2)
    

    aug_problems = torch.cat((dat1, dat2), dim=0)
    # shape: (2*batch, problem, 2, 128)

    return aug_problems