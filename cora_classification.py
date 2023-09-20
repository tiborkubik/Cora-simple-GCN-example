import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from FeaturePropagationModule import FeaturePropagationModule


if __name__ == '__main__':
    dataset = Planetoid(root='/tmp/Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)

    # Check some dataset statistics.
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of features: {data.num_features}')
    print(f'Is graph directed? {data.is_directed()}')
    print(f'Number of classes: {dataset.num_classes}')

    # check training nodes
    print("# of nodes to train on: ", data.train_mask.sum().item())
    # check test nodes
    print("# of nodes to test on: ", data.test_mask.sum().item())
    # check validation nodes
    print("# of nodes to validate on: ", data.val_mask.sum().item())

    model = FeaturePropagationModule(in_features=dataset.num_node_features, out_features=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    loss_val_history = []
    accuracy_history = []

    model.train()
    for epoch in range(600):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        # Append the current loss to the loss history list
        loss_history.append(loss.item())

        loss.backward()
        optimizer.step()

        model.eval()

        loss_val = F.nll_loss(out[data.val_mask], data.y[data.val_mask])
        loss_val_history.append(loss_val.item())

        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        accuracy_history.append(acc)
        print(f'Accuracy: {acc:.4f} [Epoch {epoch}]')
        model.train()

    plt.plot(loss_history, label='Training Loss')
    plt.plot(loss_val_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(accuracy_history, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
