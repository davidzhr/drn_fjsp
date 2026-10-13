import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F

class GATedge(nn.Module):
    '''
    Machine node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_head, bias=False)
            self.fc_edge = nn.Linear(
                1, out_feats * num_head, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, ope_ma_adj_batch, batch_idxes, feat):
        # 机器节点嵌入: 在工序-机器二部图上做带边特征的图注意力
        # Two linear transformations are used for the machine nodes and operation nodes, respective
        # In linear transformation, an W^O (\in R^{d \times 7}) for \mu_{ijk} is equivalent to
        #   W^{O'} (\in R^{d \times 6}) and W^E (\in R^{d \times 1}) for the nodes and edges respectively
        # 对源(工序)、目标(机器)节点分别做线性变换
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])  # 工序特征(源)
            h_dst = self.feat_drop(feat[1])  # 机器特征(目标)
            if not hasattr(self, 'fc_src'):
                self.fc_src, self.fc_dst = self.fc, self.fc
            feat_src = self.fc_src(h_src)  # 工序线性变换
            feat_dst = self.fc_dst(h_dst)  # 机器线性变换
        else:
            # Deprecated in final experiment
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        feat_edge = self.fc_edge(feat[2].unsqueeze(-1))  # 边(加工时间)线性变换

        # Calculate attention coefficients
        # 计算注意力系数: 源项 el / 目标项 er / 边项 ee
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)  # 工序侧注意力
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)  # 机器侧注意力
        ee = (feat_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)  # 边侧注意力
        # 只在存在边(ope_ma_adj=1)处累加源+边, 再加目标项
        el_add_ee = ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * el.unsqueeze(-2) + ee
        a = el_add_ee + ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * er.unsqueeze(-3)
        eijk = self.leaky_relu(a)  # 工序->机器 边的未归一化注意力
        ekk = self.leaky_relu(er + er)  # 机器自环项

        # Normalize attention coefficients
        # 归一化: 把无边位置屏蔽为 -inf, 再对邻居维 softmax
        mask = torch.cat((ope_ma_adj_batch[batch_idxes].unsqueeze(-1)==1,
                          torch.full(size=(ope_ma_adj_batch[batch_idxes].size(0), 1,
                                           ope_ma_adj_batch[batch_idxes].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)  # 拼上恒为真的自环位
        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)  # 拼接邻居项与自环项
        e[~mask] = float('-inf')  # 无边位置置 -inf
        alpha = F.softmax(e.squeeze(-1), dim=-2)  # 对邻居维归一化得注意力权重
        alpha_ijk = alpha[..., :-1, :]  # 工序邻居的权重
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)  # 自环权重

        # Calculate an return machine embedding
        # 按注意力加权聚合邻居(工序+边)与自身, 得到机器嵌入
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)  # 边特征+工序特征
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)  # 乘以邻居权重
        b = torch.sum(a, dim=-3)  # 对邻居求和(消息聚合)
        c = feat_dst * alpha_kk.squeeze().unsqueeze(-1)  # 自身项加权
        nu_k_prime = torch.sigmoid(b+c)  # sigmoid 激活得到新机器嵌入
        return nu_k_prime

class MLPsim(nn.Module):
    '''
    Part of operation node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, self._num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, feat, adj):
        # 工序嵌入的单类邻居消息传递(机器/紧前/紧后/自身 之一)
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)  # 按邻接矩阵沿边广播邻居特征
        b = torch.sum(a, dim=-2)  # 对邻居求和(聚合)
        c = self.project(b)  # MLP 投影到工序嵌入维度
        return c
