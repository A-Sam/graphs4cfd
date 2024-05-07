import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.transforms import NormalizeFeatures
from torch.utils.tensorboard import SummaryWriter  # Import the SummaryWriter

# Load the CORA dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
data = dataset[0]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Initialize the TensorBoard writer
writer = SummaryWriter()

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # Log the loss to TensorBoard
    writer.add_scalar('Loss/train', loss, epoch)

# Don't forget to close the writer when you're done
writer.close()

# To view the TensorBoard:
# Run the command 'tensorboard --logdir=runs' in your terminal and open the provided URL in your web browser.
