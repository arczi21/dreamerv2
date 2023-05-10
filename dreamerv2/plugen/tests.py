import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import distributions
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plugen.features import get_features


def accuracy_test(replay, flow, agnt, device, num_features=2, batch=32):
  accuracy = np.zeros(num_features)
  for j in range(batch):
    data = next(iter(replay.dataset(1, 1)))
    features = get_features(data).reshape(-1)

    data = agnt.wm.preprocess(data)
    embed = agnt.wm.encoder(data).numpy()
    embed = torch.tensor(embed).to(device)

    z, logdet = flow(embed)
    c = z[:, :, :num_features].cpu().detach().numpy().reshape(-1)

    for i in range(num_features):
      if features[i] == 1 and c[i] >= 0:
        accuracy[i] += 1
      elif features[i] == 0 and c[i] <= 0:
        accuracy[i] += 1

  accuracy = accuracy / batch
  return accuracy