# coding: utf-8

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_non_zero_graph, build_knn_normalized_graph, \
    build_graph_from_adj


class SOIL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SOIL, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.cl_loss2 = config['cl_loss2']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.knn_k = config['knn_k']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.knn_i = config['knn_i']
        self.knn_a = config['knn_a']

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.norm_adj = self.get_adj_mat()
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)

        # interest graph
        image_adj = build_sim(self.image_embedding.weight.detach())
        text_adj = build_sim(self.text_embedding.weight.detach())
        image_interest = torch.zeros_like(image_adj)
        text_interest = torch.zeros_like(text_adj)
        for user, items in dataset.history_items_per_u.items():
            items = torch.tensor([i for i in items])
            _, cols1 = torch.topk(image_adj[items].sum(dim=0), self.knn_i)
            _, cols2 = torch.topk(text_adj[items].sum(dim=0), self.knn_i)
            cols = torch.cat([cols1, cols2]).unique()
            image_interest[items[:, None], cols] += image_adj[items[:, None], cols]
            text_interest[items[:, None], cols] += text_adj[items[:, None], cols]

        image_interest_adj = build_non_zero_graph(image_interest, is_sparse=self.sparse, norm_type='sym')
        text_interest_adj = build_non_zero_graph(text_interest, is_sparse=self.sparse, norm_type='sym')

        # similarity graph
        image_adj = build_knn_normalized_graph(image_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')
        text_adj = build_knn_normalized_graph(text_adj, topk=self.knn_k, is_sparse=self.sparse, norm_type='sym')

        # mixed graph
        image_interest_adj = torch.add(image_interest_adj, image_adj)
        text_interest_adj = torch.add(text_interest_adj, text_adj)

        torch.cuda.empty_cache()

        self.image_interest_adj = image_interest_adj.to(self.device)
        self.text_interest_adj = text_interest_adj.to(self.device)

        image_adj = build_sim(self.image_embedding.weight.detach())
        text_adj = build_sim(self.text_embedding.weight.detach())
        mm_attractive = torch.zeros_like(self.norm_adj)
        mm_attractive_R = torch.zeros_like(self.R)

        for user, items in dataset.history_items_per_u.items():
            items = torch.tensor([i for i in items])
            k_num = self.knn_a + items.size(0)
            mm_sim = torch.multiply(image_adj[items], text_adj[items])
            mm_value, mm_indices = torch.topk(mm_sim, k_num, dim=-1)

            k_mm_value, k_mm_indices = torch.topk(mm_value.flatten(), k_num)

            mm_indices = mm_indices.flatten()[k_mm_indices]

            uid = torch.zeros_like(mm_indices).fill_(user)
            mm_sparse = torch.sparse_coo_tensor(torch.stack([uid, self.n_users + mm_indices]), k_mm_value,
                                                size=self.norm_adj.size())
            mm_sparse_t = torch.sparse_coo_tensor(torch.stack([self.n_users + mm_indices, uid]), k_mm_value,
                                                  size=self.norm_adj.size())

            mm_R_sparse = torch.sparse_coo_tensor(torch.stack([uid, mm_indices]), k_mm_value, size=self.R.size())

            mm_attractive += mm_sparse
            mm_attractive += mm_sparse_t

            mm_attractive_R += mm_R_sparse

        mm_attractive_adj = build_graph_from_adj(mm_attractive, is_sparse=self.sparse, norm_type='sym', mask=False)
        mm_attractive_adj_R = build_graph_from_adj(mm_attractive_R, is_sparse=self.sparse, norm_type='sym',
                                                   mask=False)
        torch.cuda.empty_cache()

        self.mm_attractive_adj = mm_attractive_adj.to(self.device)
        self.mm_attractive_adj_R = mm_attractive_adj_R.to(self.device)
        # argumented graph
        self.norm_adj = torch.add(self.norm_adj, self.mm_attractive_adj / 2)
        self.R = torch.add(self.R, self.mm_attractive_adj_R / 2)

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.map_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.map_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.tau = 0.5

    def pre_epoch_processing(self):
        pass

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # Feature ID Embedding
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.map_v(image_feats))
        text_item_embeds = torch.multiply(self.item_id_embedding.weight, self.map_t(text_feats))

        # Second-Order Graph Convolution
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        # Interest-Aware Item Graph Convolution
        for i in range(self.n_layers):
            image_item_embeds = torch.sparse.mm(self.image_interest_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        for i in range(self.n_layers):
            text_item_embeds = torch.sparse.mm(self.text_interest_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Attention Fuser
        att_common = torch.cat([self.attention(image_embeds), self.attention(text_embeds)], dim=-1)
        weight_common = self.softmax(att_common)
        common_embeds = weight_common[:, 0].unsqueeze(dim=1) * image_embeds + weight_common[:, 1].unsqueeze(
            dim=1) * text_embeds
        side_embeds = (image_embeds + text_embeds - common_embeds) / 3

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds

        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        bpr_loss = -torch.mean(maxi)

        return bpr_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(
            self.norm_adj, train=True)

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        bpr_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        # item-item constractive loss
        cl_loss = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            side_embeds_users[users], content_embeds_user[users], 0.2)
        # user-item constractive loss
        cl_loss2 = self.InfoNCE(u_g_embeddings, content_embeds_items[pos_items], 0.2) + self.InfoNCE(
            u_g_embeddings, side_embeds_items[pos_items], 0.2)

        return bpr_loss + self.cl_loss * cl_loss + self.cl_loss2 * cl_loss2

    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores
