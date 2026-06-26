import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtuples as tt
from pycox.models import CoxPH

# Moduł atencji dla cech wejściowych
class FeatureAttention(nn.Module):
    def __init__(self, in_features):
        super(FeatureAttention, self).__init__()
        self.attention_weights = nn.Linear(in_features, in_features)

    def forward(self, x):
        scores = self.attention_weights(x)
        
        attention_dist = F.softmax(scores, dim=1)
        
        attended_x = x * attention_dist
        
        return attended_x, attention_dist



# DeepSurv z modułem Atencji
class AttentionDeepSurv(nn.Module):
    def __init__(self, in_features, num_nodes, dropout=0.3):
        super(AttentionDeepSurv, self).__init__()
        
        self.attention = FeatureAttention(in_features)

        layers = []
        current_dim = in_features
        for node in num_nodes:
            layers.append(nn.Linear(current_dim, node))
            layers.append(nn.BatchNorm1d(node))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = node

        layers.append(nn.Linear(current_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        attended_x, attention_weights = self.attention(x)
        
        risk = self.mlp(attended_x)
        
        return risk

    def get_attention_weights(self, x):
        with torch.no_grad():
            _, weights = self.attention(x)
        return weights