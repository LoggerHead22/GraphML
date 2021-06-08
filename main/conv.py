import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import SumPooling
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

### GIN convolution along the graph structure
class GINConv(nn.Module):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__()

        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                 nn.BatchNorm1d(emb_dim),
                                 nn.ReLU(),
                                 nn.Linear(emb_dim, emb_dim))
        self.eps = nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, g, x, edge_attr):# u --(e)--> v
        with g.local_scope():
            edge_embedding = self.bond_encoder(edge_attr) 
            g.ndata['x'] = x # h_{t-1}
            g.apply_edges(fn.copy_u('x', 'm')) # m_{t-1}e = h_{t-1}e
            g.edata['m'] = F.relu(g.edata['m'] + edge_embedding) # h_{t}e = relu(m_{t-1} + edge_emb)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'new_x')) 
            out = self.mlp((1 + self.eps) * x + g.ndata['new_x'])
            #h_t = mlp((1 + eps) * h_t-1 + sum_v (relu(h_t-1 + edge_emb)))
            return out

### GCN convolution along the graph structure
class GCNConv(nn.Module):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GCNConv, self).__init__()

        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, g, x, edge_attr):
        with g.local_scope():
            x = self.linear(x)
            edge_embedding = self.bond_encoder(edge_attr)

            # Molecular graphs are undirected
            # g.out_degrees() is the same as g.in_degrees()
            degs = (g.out_degrees().float() + 1).to(x.device) # D + I 
            norm = torch.pow(degs, -0.5).unsqueeze(-1)                # (N, 1)
            g.ndata['norm'] = norm
            g.apply_edges(fn.u_mul_v('norm', 'norm', 'norm'))

            g.ndata['x'] = x
            g.apply_edges(fn.copy_u('x', 'm'))
            g.edata['m'] = g.edata['norm'] * F.relu(g.edata['m'] + edge_embedding)
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'new_x'))
            out = g.ndata['new_x'] + F.relu(x + self.root_emb.weight) * 1. / degs.view(-1, 1)

            return out

### GNN to generate node embedding
class GNN_node(nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', num_iter=5):
        '''
            num_layers (int): number of GNN message passing layers
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node, self).__init__()
        self.num_layers = num_layers #число слоев
        self.drop_ratio = drop_ratio #p для dropout
        self.JK = JK #тип финального embedding'a (last, sum)
        ### add residual connection or not
        self.residual = residual 
        self.num_iter = num_iter

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of GNNs
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, g, x, edge_attr):
        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        for layer in range(self.num_layers):
            for iteration in range(self.num_iter):

                h = self.convs[layer](g, h_list[-1], edge_attr) # h - hidden_state for all nodes
                if iteration == self.num_iter - 1:
                    h = self.batch_norms[layer](h)

                if layer == self.num_layers - 1 and iteration == self.num_iter - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

                if self.residual:
                    h += h_list[-1]

                h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[(layer + 1) * self.num_iter - 1]

        return node_representation


### Virtual GNN to generate node embedding
class GNN_node_Virtualnode(nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_layers, emb_dim, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin', num_iter=5):
        '''
            num_layers (int): number of GNN message passing layers
            emb_dim (int): node embedding dimensionality
        '''

        super(GNN_node_Virtualnode, self).__init__()
        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual
        self.num_iter = num_iter

        if self.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        ### set the initial virtual node embedding to 0.
        self.virtualnode_embedding = nn.Embedding(1, emb_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        ### List of GNNs
        self.convs = nn.ModuleList()
        ### batch norms applied to node embeddings
        self.batch_norms = nn.ModuleList()

        ### List of MLPs to transform virtual node at every layer
        self.mlp_virtualnode_list = nn.ModuleList()

        for layer in range(num_layers):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        for layer in range(num_layers - 1):
            self.mlp_virtualnode_list.append(nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                                           nn.BatchNorm1d(emb_dim),
                                                           nn.ReLU(),
                                                           nn.Linear(emb_dim, emb_dim),
                                                           nn.BatchNorm1d(emb_dim),
                                                           nn.ReLU()))
        self.pool = SumPooling()

    def forward(self, g, x, edge_attr):
        ### virtual node embeddings for graphs
        virtualnode_embedding = self.virtualnode_embedding( # master node hidden state Embedding(0)
            torch.zeros(g.batch_size).to(x.dtype).to(x.device))

        h_list = [self.atom_encoder(x)]
        batch_id = dgl.broadcast_nodes(g, torch.arange(g.batch_size).to(x.device))
        for layer in range(self.num_layers):
            for iteration in range(self.num_iter):
                ### add message from virtual nodes to graph nodes
                h_list[-1] = h_list[-1] + virtualnode_embedding[batch_id]

                ### Message passing among graph nodes
                h = self.convs[layer](g, h_list[-1], edge_attr)
                if iteration == self.num_iter - 1:
                    h = self.batch_norms[layer](h)

                if layer == self.num_layers - 1 and iteration == self.num_iter - 1:
                    #remove relu for the last layer
                    h = F.dropout(h, self.drop_ratio, training = self.training)
                else:
                    h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

                if self.residual:
                    h = h + h_list[-1]

                h_list.append(h)

            ### update the virtual nodes
            if layer < self.num_layers - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = self.pool(g, h_list[-1]) + virtualnode_embedding
                ### transform virtual nodes using MLP
                virtualnode_embedding_temp = self.mlp_virtualnode_list[layer](
                    virtualnode_embedding_temp)

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        virtualnode_embedding_temp, self.drop_ratio, training = self.training)
                else:
                    virtualnode_embedding = F.dropout(
                        virtualnode_embedding_temp, self.drop_ratio, training = self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layers):
                node_representation += h_list[(layer + 1) * self.num_iter - 1]

        return node_representation
