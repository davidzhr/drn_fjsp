import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical  # 分类分布, 用于按动作概率采样/求对数概率
from graph.hgnn import GATedge, MLPsim  # HGNN 的两类基础层: 机器嵌入(GATedge)、工序嵌入子层(MLPsim)
from mlp import MLPCritic, MLPActor  # actor(打分) 与 critic(估值) 的多层感知机


class Memory:
    # PPO 的 rollout 缓冲区: 存一整段采集过程中每个决策点的数据
    def __init__(self):
        self.states = []  # 状态(实际未使用, 下方 act 里被注释掉)
        self.logprobs = []  # 每个动作的对数概率 log pi(a|s)
        self.rewards = []  # 每步奖励
        self.is_terminals = []  # 每步是否为终止(该算例是否完成)
        self.action_indexes = []  # 被选中的展平动作索引

        self.ope_ma_adj = []  # 工序-机器邻接矩阵(每个决策点一份)
        self.ope_pre_adj = []  # 工序-紧前工序邻接矩阵
        self.ope_sub_adj = []  # 工序-紧后工序邻接矩阵
        self.batch_idxes = []  # 未完成算例的下标
        self.raw_opes = []  # 归一化后的工序特征
        self.raw_mas = []  # 归一化后的机器特征
        self.proc_time = []  # 归一化后的加工时间(边特征)
        self.jobs_gather = []  # 各作业当前工序的 gather 索引
        self.eligible = []  # 可行 O-M 动作掩码
        self.nums_opes = []  # 每个算例的工序总数


    def clear_memory(self):
        # 每次 PPO 更新后清空缓冲区(用 del [:] 原地清空, 保持列表对象不变)
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]

        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.nums_opes[:]


class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''
    # 工序节点嵌入模块: 聚合 4 类邻居(机器/紧前/紧后/自身)的信息并融合
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope  # 四类邻居各自的输入维度列表
        self.hidden_size_ope = hidden_size_ope  # MLP 隐层维度
        self.out_size_ope = out_size_ope  # 工序嵌入输出维度
        self.num_head = num_head  # 多头数(最终实验恒为1)
        self.dropout = dropout  # dropout 比例(最终实验恒为0)
        self.gnn_layers = nn.ModuleList()  # 存放对 4 类邻居各一个的消息传递子层

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        # 为每一类邻居建立一个 MLPsim 消息传递层
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        # 融合层: 把 4 类邻居的输出拼接后投影成最终工序嵌入
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),  # 拼接维度 -> 隐层
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),  # 隐层 -> 工序嵌入维度
        )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        # 四类邻居对应的源特征: [机器特征, 工序特征(紧前), 工序特征(紧后), 工序特征(自身)]
        h = (feats[1], feats[0], feats[0], feats[0])
        # Identity matrix for self-loop of nodes
        # 自环用的单位矩阵(第4类"自身"邻居), shape 对齐到 [batch, O, O]
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_idxes])

        # Calculate an return operation embedding
        # 四类邻接矩阵打包: 机器 / 紧前工序 / 紧后工序 / 自身
        adj = (ope_ma_adj_batch[batch_idxes], ope_pre_adj_batch[batch_idxes],
               ope_sub_adj_batch[batch_idxes], self_adj)
        MLP_embeddings = []  # 存放 4 类邻居各自聚合后的嵌入
        # 对每一类邻居做消息传递
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        # 沿最后一维拼接 4 类嵌入 -> [batch, O, out_size_ope*4]
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        # 融合投影得到最终工序嵌入 mu'_{ij}
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime

class HGNNScheduler(nn.Module):
    # 核心策略网络: HGNN 做图嵌入 + actor/critic, 既是 policy 也是 value
    def __init__(self, model_paras):
        super(HGNNScheduler, self).__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = model_paras["num_heads"]  # 列表长度=HGNN迭代次数, 元素=每次的头数
        self.dropout = model_paras["dropout"]

        # Machine node embedding
        # 机器嵌入: 每次 HGNN 迭代一层 GATedge
        self.get_machines = nn.ModuleList()
        # 第一层输入维度用原始特征维(工序 in_size_ope, 机器 in_size_ma)
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                    self.dropout, self.dropout, activation=F.elu))
        # 后续层输入维度改用上一轮的嵌入维度
        for i in range(1,len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                    self.dropout, self.dropout, activation=F.elu))

        # Operation node embedding
        # 工序嵌入: 每次 HGNN 迭代一层 MLPs
        self.get_operations = nn.ModuleList()
        # 第一层: 4 类邻居输入维度 = [机器嵌入维, 工序原始维 x3]
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        # 后续层: 工序侧改用上一轮工序嵌入维度
        for i in range(len(self.num_heads)-1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

        # actor: 对每个候选 O-M 对打分(输出维=action_dim=1)
        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        # critic: 对池化后的全局表示估计状态价值(输出维=1)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        # 不走标准 forward, 改用 act(推理决策) 与 evaluate(更新时重算) 两个方法
        raise NotImplementedError

    def feature_normalize(self, data):
        # 标准化: 减均值除标准差(1e-5 防止除零)
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''
    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        # 验证/DRL-G: 各算例工序数可能不同, 不能整体矩阵归一化, 需逐算例处理
        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                # 只对该算例真实工序(前 nums_opes[i] 个)求均值/标准差
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])  # 非零(可加工)位置的下标
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]  # 取出这些加工时间值
                proc_norm = self.feature_normalize(proc_values)  # 仅对有效值归一化
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm  # 写回原位置
            mean_opes = torch.stack(mean_opes, dim=0)  # 逐算例结果堆叠回批次维
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # 机器特征可整体归一化
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        # DRL-S and scheduling during training have a consistent number of operations
        # 训练/DRL-S: 批内工序数一致, 可直接整矩阵归一化(更快)
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        # 返回归一化后的 (工序特征, 机器特征, 加工时间)
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # 计算当前状态下每个可行 O-M 动作的概率
        # Uncompleted instances
        batch_idxes = state.batch_idxes  # 当前仍未完成的算例下标
        # Raw feats
        # 取原始特征并转成 [batch, 节点数, 特征维]; 只保留未完成算例
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]  # 工序特征 [B, O, in_size_ope]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]  # 机器特征 [B, M, in_size_ma]
        proc_time = state.proc_times_batch[batch_idxes]  # 加工时间(边特征) [B, O, M]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]  # 各算例工序总数
        # 归一化三类特征
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = (copy.deepcopy(features[0]))  # 深拷贝留存, 训练时写入 memory
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))

        # L iterations of the HGNN
        # 进行 L 次(=len(num_heads)) HGNN 迭代, 交替更新机器/工序嵌入
        for i in range(len(self.num_heads)):
            # First Stage, machine node embedding
            # 第一阶段: 机器嵌入 shape: [len(batch_idxes), num_mas, out_size_ma]
            h_mas = self.get_machines[i](state.ope_ma_adj_batch, state.batch_idxes, features)
            features = (features[0], h_mas, features[2])  # 用新机器嵌入替换 features 中的机器项
            # Second Stage, operation node embedding
            # 第二阶段: 工序嵌入 shape: [len(batch_idxes), max(num_opes), out_size_ope]
            h_opes = self.get_operations[i](state.ope_ma_adj_batch, state.ope_pre_adj_batch, state.ope_sub_adj_batch,
                                            state.batch_idxes, features)
            features = (h_opes, features[1], features[2])  # 用新工序嵌入替换 features 中的工序项

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # 机器嵌入均值池化 -> [len(batch_idxes), out_size_ma]
        # There may be different operations for each instance, which cannot be pooled directly by the matrix
        # 验证/DRL-G: 工序数不同, 需逐算例池化(只对真实工序求均值)
        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)  # shape: [len(batch_idxes), d]
        else:
            h_opes_pooled = h_opes.mean(dim=-2)  # 训练/DRL-S: 直接整体均值池化 [len(batch_idxes), out_size_ope]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        # 每个作业"当前待加工工序"的全局id(超过末工序则夹在末工序)
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        # 构造 gather 索引, 从 h_opes 中取出各作业当前工序的嵌入
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]
        h_jobs = h_opes.gather(1, jobs_gather)  # 各作业当前工序嵌入 [B, num_jobs, out_size_ope]
        # Matrix indicating whether processing is possible
        # 该(作业当前工序, 机器)能否加工 shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
        # 为拼接 actor 输入, 把各张量广播到 [B, num_jobs, num_mas, dim]
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)  # 作业工序嵌入
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)  # 机器嵌入
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)  # 机器全局池化
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)  # 工序全局池化
        # Matrix indicating whether machine is eligible
        # 机器是否空闲可用 shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        # 作业是否可调度(未在加工且未完成) shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # 三条件相与得到最终可行动作掩码 shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        # 没有任何可行 O-M 对时报警并返回(异常保护)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        # Input of actor MLP
        # 拼接 4 部分作为 actor 输入, 并转置到 [B, num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated(此处 act 不用, critic 用)
        mask = eligible.transpose(1, 2).flatten(1)  # 掩码展平成 [B, num_mas*num_jobs]

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)  # actor 对每个动作打分并展平 [B, num_mas*num_jobs]
        scores[~mask] = float('-inf')  # 不可行动作置 -inf, softmax 后概率为0
        action_probs = F.softmax(scores, dim=1)  # 归一化为动作概率分布

        # Store data in memory during training
        # 训练时把本决策点所需张量深拷贝存入 memory, 供后续 PPO 更新重算
        if flag_train == True:
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj_batch))
            memories.ope_pre_adj.append(copy.deepcopy(state.ope_pre_adj_batch))
            memories.ope_sub_adj.append(copy.deepcopy(state.ope_sub_adj_batch))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.nums_opes.append(copy.deepcopy(nums_opes))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))

        # 返回: 动作概率, 各作业当前工序id, 池化全局表示
        return action_probs, ope_step_batch, h_pooled

    def act(self, state, memories, dones, flag_sample=True, flag_train=True):
        # 决策一步: 先算动作概率
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, ope_step_batch, _ = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)

        # DRL-S, sampling actions following \pi
        # DRL-S: 按概率分布采样动作
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        # DRL-G: 贪心取最大概率动作
        else:
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        # 把展平动作索引反解回 (机器, 作业, 工序): 索引空间是 num_mas*num_jobs
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()  # 行=机器 = idx // num_jobs
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()  # 列=作业 = idx % num_jobs
        opes = ope_step_batch[state.batch_idxes, jobs]  # 该作业当前待加工工序的全局id

        # Store data in memory during training
        # 训练时记录对数概率与动作索引(采样分支才有 dist)
        if flag_train == True:
            # memories.states.append(copy.deepcopy(state))
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        # 返回 [3, B] 动作张量 (opes, mas, jobs), 供 env.step 消费
        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 jobs_gather, eligible, action_envs, flag_sample=False):
        # PPO 更新时调用: 用存储的张量重算 logprob/value/entropy
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()  # 这里批内全部参与, 直接按序号
        features = (raw_opes, raw_mas, proc_time)  # 直接用已归一化的存储特征

        # L iterations of the HGNN
        # 与推理时相同的 L 次 HGNN 迭代
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, batch_idxes, features)  # 机器嵌入
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, features)  # 工序嵌入
            features = (h_opes, features[1], features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # 机器嵌入池化(更新时批内一致, 直接整体池化)
        h_opes_pooled = h_opes.mean(dim=-2)  # 工序嵌入池化

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_jobs = h_opes.gather(1, jobs_gather)  # 各作业当前工序嵌入
        # 广播拼接, 与 get_action_prob 中构造 actor 输入的方式一致
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        # actor 输入拼接并转置 [B, num_mas, num_jobs, dim]
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # critic 输入: 工序+机器全局池化
        scores = self.actor(h_actions).flatten(1)  # 重新打分
        mask = eligible.transpose(1, 2).flatten(1)  # 用存储的可行掩码

        scores[~mask] = float('-inf')  # 屏蔽不可行动作
        action_probs = F.softmax(scores, dim=1)  # 动作概率
        state_values = self.critic(h_pooled)  # critic 估计状态价值
        dist = Categorical(action_probs.squeeze())  # 构造分布
        action_logprobs = dist.log_prob(action_envs)  # 对历史所选动作求对数概率
        dist_entropys = dist.entropy()  # 策略熵(鼓励探索)
        # 返回: 对数概率, 状态价值, 熵
        return action_logprobs, state_values.squeeze().double(), dist_entropys

class PPO:
    # PPO 算法封装: 维护 policy/policy_old 与优化器, 实现策略更新
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras["lr"]  # learning rate
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras["gamma"]  # discount factor
        self.eps_clip = train_paras["eps_clip"]  # clip ratio for PPO
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff = train_paras["entropy_coeff"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.device = model_paras["device"]  # PyTorch device

        self.policy = HGNNScheduler(model_paras).to(self.device)  # 当前策略(被优化)
        self.policy_old = copy.deepcopy(self.policy)  # 旧策略(采集 rollout 用)
        self.policy_old.load_state_dict(self.policy.state_dict())  # 同步初始权重
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)  # Adam 优化器
        self.MseLoss = nn.MSELoss()  # 价值函数回归损失

    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]  # batch size for updating

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        # 把 memory 各列堆叠, 转置成 (num_envs, 决策步) 再展平成 (样本数, ...), 每个决策点=一条样本
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=0).transpose(0,1).flatten(0,1)
        old_ope_pre_adj = torch.stack(memory.ope_pre_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_ope_sub_adj = torch.stack(memory.ope_sub_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1)
        old_proc_time = torch.stack(memory.proc_time, dim=0).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0,1)  # 奖励保留 (num_envs, 步) 形状供逆序累加
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0,1)  # 终止标记同上
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0,1).flatten(0,1)  # 旧策略对数概率
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0,1).flatten(0, 1)  # 历史动作索引

        # Estimate and normalize the rewards
        # 估计并归一化回报(reward-to-go), 逐个环境处理
        rewards_envs = []
        discounted_rewards = 0  # 累计总回报(仅用于日志输出平均奖励)
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            # 从后往前累加折扣回报, 遇终止则重置
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)  # G_t = r + gamma*G_{t+1}
                rewards.insert(0, discounted_reward)  # 插到最前, 恢复时间正序
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # 逐环境标准化回报(当优势基线)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)  # 拼成一维 [样本数]

        loss_epochs = 0  # 累计各 epoch 的损失(用于日志)
        full_batch_size = old_ope_ma_adj.size(0)  # 样本总数
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)  # 完整 minibatch 个数
        # Optimize policy for K epochs:
        # 对同一批数据训练 K 个 epoch
        for _ in range(self.K_epochs):
            # 遍历所有 minibatch(含最后一个不完整批)
            for i in range(num_complete_minibatches+1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size  # 最后一批取到末尾
                # 用当前策略重新评估这批样本, 得到新 logprob/value/entropy
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_ope_ma_adj[start_idx: end_idx, :, :],
                                         old_ope_pre_adj[start_idx: end_idx, :, :],
                                         old_ope_sub_adj[start_idx: end_idx, :, :],
                                         old_raw_opes[start_idx: end_idx, :, :],
                                         old_raw_mas[start_idx: end_idx, :, :],
                                         old_proc_time[start_idx: end_idx, :, :],
                                         old_jobs_gather[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_action_envs[start_idx: end_idx])

                # 重要性采样比率 r = exp(新logprob - 旧logprob)
                ratios = torch.exp(logprobs - old_logprobs[i*minibatch_size:(i+1)*minibatch_size].detach())
                # 优势 = 归一化回报 - 价值估计(detach, 不回传到 critic)
                advantages = rewards_envs[i*minibatch_size:(i+1)*minibatch_size] - state_values.detach()
                surr1 = ratios * advantages  # 未裁剪项
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # 裁剪项
                # 总损失 = -策略损失(取min) + 价值损失(MSE) - 熵奖励
                loss = - self.A_coeff * torch.min(surr1, surr2)\
                       + self.vf_coeff * self.MseLoss(state_values, rewards_envs[i*minibatch_size:(i+1)*minibatch_size])\
                       - self.entropy_coeff * dist_entropy
                loss_epochs += loss.mean().detach()  # 记录损失(不参与反传)

                self.optimizer.zero_grad()  # 清梯度
                loss.mean().backward()  # 反向传播
                self.optimizer.step()  # 更新 policy 参数

        # Copy new weights into old policy:
        # K 个 epoch 后把新权重同步给旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 返回: 平均损失, 平均回报(供日志打印)
        return loss_epochs.item() / self.K_epochs, \
               discounted_rewards.item() / (self.num_envs * train_paras["update_timestep"])
