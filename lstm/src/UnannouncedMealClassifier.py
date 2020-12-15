#!/usr/bin/env python

import torch.nn as nn
import torch.nn.functional as F


class UnannouncedMealClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(UnannouncedMealClassifier, self).__init__()

        self.linear1 = nn.Linear(vocab_size, 128)
        self.linear2 = nn.Linear(128, num_labels)

    def forward(self, feature_vec):
        hidden1 = self.linear1(feature_vec).clamp(min=0)
        output = self.linear2(hidden1)
        return F.log_softmax(output, dim=1)
