import torch
import torch.nn.functional as F
from CustomGCNConv import CustomGCNConv
from torch_geometric.nn import GCNConv

_USE_CUSTOM_GCNCONV: bool = False


class FeaturePropagationModule(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 f_size: int = 16, training: bool = True) -> None:
        super().__init__()

        self.training = training

        print(f'Input feature size: {in_features}')

        if _USE_CUSTOM_GCNCONV:
            self.conv_gcn_1 = CustomGCNConv(in_features=in_features, out_features=f_size)
            self.conv_gcn_2 = CustomGCNConv(in_features=f_size, out_features=2*f_size)
            self.conv_gcn_3 = CustomGCNConv(in_features=2*f_size, out_features=out_features)
        else:
            self.conv_gcn_1 = GCNConv(in_channels=in_features, out_channels=f_size)
            self.conv_gcn_2 = GCNConv(in_channels=f_size, out_channels=2*f_size)
            self.conv_gcn_3 = GCNConv(in_channels=2*f_size, out_channels=out_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = self.conv_gcn_1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, training=self.training, p=0.2)

        x2 = self.conv_gcn_2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, training=self.training, p=0.2)
        #
        x3 = self.conv_gcn_3(x2, edge_index)

        # In FPM, we do not really need softmax, since we do not want to do any predictions, this is for toy task debug.
        return F.log_softmax(x3, dim=1)
