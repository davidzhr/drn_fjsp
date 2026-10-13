import sys
import gym
import torch

from dataclasses import dataclass
from env.load_data import load_fjs, nums_detec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy
from utils.my_utils import read_json, write_json


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # 环境状态容器: 打包所有传给模型的张量(静态图 + 动态特征 + 掩码)
    # static  静态部分(整局不变)
    opes_appertain_batch: torch.Tensor = None  # 工序->作业归属
    ope_pre_adj_batch: torch.Tensor = None  # 工序紧前邻接
    ope_sub_adj_batch: torch.Tensor = None  # 工序紧后邻接
    end_ope_biases_batch: torch.Tensor = None  # 每作业末工序全局id
    nums_opes_batch: torch.Tensor = None  # 每算例工序总数

    # dynamic  动态部分(每步更新)
    batch_idxes: torch.Tensor = None  # 未完成算例下标
    feat_opes_batch: torch.Tensor = None  # 工序特征 [B, 6, O]
    feat_mas_batch: torch.Tensor = None  # 机器特征 [B, 3, M]
    proc_times_batch: torch.Tensor = None  # 加工时间(边特征) [B, O, M]
    ope_ma_adj_batch: torch.Tensor = None  # 工序-机器邻接
    time_batch:  torch.Tensor = None  # 各算例当前仿真时间

    mask_job_procing_batch: torch.Tensor = None  # 作业是否在加工中
    mask_job_finish_batch: torch.Tensor = None  # 作业是否已完成
    mask_ma_procing_batch: torch.Tensor = None  # 机器是否在加工中
    ope_step_batch: torch.Tensor = None  # 各作业当前待加工工序指针

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        # 每步 env.step 结束时, 用最新张量刷新 state
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time

def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    '''
    # 把"按作业"的特征按归属关系展开成"按工序"(同作业的工序取同一个作业值)
    return feat_job_batch.gather(1, opes_appertain_batch)

class FJSPEnv(gym.Env):
    '''
    FJSP environment
    '''
    def __init__(self, case, env_paras, data_source='case'):
        '''
        :param case: The instance generator or the addresses of the instances
        :param env_paras: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files
        '''

        # load paras
        # static
        self.show_mode = env_paras["show_mode"]  # Result display mode (deprecated in the final experiment)
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch
        # load instance
        # 加载算例: load_fjs 返回 8 个张量, 对应 tensors[0..7]
        num_data = 8  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0  # 批内最大工序数(用于对齐 padding)
        lines = []
        if data_source=='case':  # Generate instances through generators
            # 训练: 用生成器实时合成算例
            for i in range(self.batch_size):
                lines.append(case.get_case(i)[0])  # Generate an instance and save it
                num_jobs, num_mas, num_opes = nums_detec(lines[i])  # 解析规模
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)  # 取批内最大工序数
        else:  # Load instances from files
            # 验证/测试: 从 .fjs 文件读入算例
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        # load feats
        # 逐算例解析成张量, 按 8 个返回位分别收集(后面再 stack 成 batch)
        for i in range(self.batch_size):
            load_data = load_fjs(lines[i], num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
            ma:
                Number of neighboring operations
                Available time
                Utilization
        '''
        # Generate raw feature vectors
        # 初始化原始特征向量: 工序特征 [B, 6, O], 机器特征 [B, 3, M]
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], num_mas))

        # 工序特征按通道填充(通道0=状态, 初始为0未调度):
        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)  # 通道1: 可加工机器数
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)  # 通道2: 平均加工时间
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)  # 通道3: 所在作业未排工序数
        # 通道5: 工序预计开始时间 = 沿作业链累积前驱平均加工时间(用 cal_cumul 累积矩阵做 bmm)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        # 作业完工时间 = 末工序(开始时间 + 加工时间)
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)  # 通道4: 作业完工时间
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)  # 机器通道0: 可加工工序数
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        # 工序级部分调度表 [B, O, 4]: [状态, 分配机器, 开始时间, 结束时间]
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]  # 开始时间初值
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]  # 结束时间初值
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        # 机器级部分调度表 [B, M, 4]: [是否空闲, 可用时间, 已用加工时长, 当前作业id]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))  # 初始全部空闲

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # 初始 makespan 估计 shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # 是否全部作业完成 shape: (batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        # 保存初始快照, 供 reset() 还原(同批算例可重复调度)
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

    def step(self, actions):
        '''
        Environment transition function
        '''
        # 环境转移: 执行一个 O-M 分配动作, 更新状态并返回 (state, reward, done)
        opes = actions[0, :]  # 被调度的工序(全局id)
        mas = actions[1, :]  # 分配到的机器
        jobs = actions[2, :]  # 对应的作业
        self.N += 1  # 已调度工序计数+1

        # Removed unselected O-M arcs of the scheduled operations
        # 该工序确定机器后, 删除它与其他机器的候选弧(只保留选中的机器)
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        # 更新被调度工序的通道0/1/2: 状态置1, 邻接机器数置1, 加工时间=选中机器的实际时间
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]  # 本次加工时长
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack((torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                                                                        torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                                                                        proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1, opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0  # 已调度工序不再作为后续累积前驱

        # Update 'Number of unscheduled operations in the job'
        # 该作业所有工序的"未排工序数"(通道3)减1
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i]+1] -= 1

        # Update 'Start time' and 'Job completion time'
        # 更新开始时间(通道5)与作业完工时间(通道4)
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]  # 已调度工序: 开始时间=当前时刻
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]  # 已调度标记
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        # 未调度工序: 用累积矩阵估计开始时间(前驱 开始+加工 的累积)
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                            self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze()\
                         * un_scheduled  # estimate start time of unscheduled opes
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[self.batch_idxes,:])

        # Update partial schedule (state)
        # 更新部分调度表: 工序级记录(状态/机器/起止), 机器级记录(占用/可用时间/累计用时/当前作业)
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas), dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))  # 机器置为忙
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times  # 下次可用时间
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times  # 累计加工时长
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()  # 当前加工的作业

        # Update feature vectors of machines
        # 更新机器特征: 可加工工序数 / 可用时间 / 利用率
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]  # 累计用时
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)  # 利用率 = 用时/当前时刻
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # Update other variable according to actions
        # 更新指针与掩码: 作业工序指针+1, 标记作业/机器为忙, 判断作业是否完成
        self.ope_step_batch[self.batch_idxes, jobs] += 1  # 该作业前进到下一道工序
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True  # 作业进入加工中
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True  # 机器进入加工中
        self.mask_job_finish_batch = torch.where(self.ope_step_batch==self.end_ope_biases_batch+1,
                                                 True, self.mask_job_finish_batch)  # 指针越过末工序=作业完成
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # 该算例是否全部作业完成
        self.done = self.done_batch.all()  # 整批是否完成

        # 奖励 = 旧makespan - 新makespan(makespan下界的下降量, 累加后≈-最终makespan)
        max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # 新的 makespan 估计
        self.reward_batch = self.makespan_batch - max
        self.makespan_batch = max

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        # 若当前时刻没有可行 O-M 对, 推进仿真时间直到出现可行对或该算例完成
        flag_trans_2_next_time = self.if_no_eligible()
        # 只要还有算例无可行对且未完成, 就反复推进时间
        while ~((~((flag_trans_2_next_time==0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        # Update the vector for uncompleted instances
        # 收缩 batch_idxes: 已完成的算例不再参与后续计算(半同步)
        mask_finish = (self.N+1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        # 把最新张量写回 state, 返回给调用方
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
            self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch, self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        # 判断每个算例当前是否还有可行 O-M 对; 返回值=0 表示该算例需推进时间
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)  # 各作业当前工序指针(夹到末工序)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))  # 这些工序的加工时间
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)  # 机器空闲
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)  # 作业可调度
        # 累加可行对的加工时间: 结果为0表示该算例当前无任何可行 O-M 对
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # 推进仿真时间到下一个机器完工时刻, 释放完工的机器/作业
        # need to transit
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)  # 需要推进的算例
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        # 复位: 用 __init__ 时保存的 old_* 快照还原所有动态状态, 以便同批算例重复调度
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        return self.state

    def render(self, mode='human'):
        '''
        Deprecated in the final experiment
        '''
        if self.show_mode == 'draw':
            num_jobs = self.num_jobs
            num_mas = self.num_mas
            print(sys.argv[0])
            color = read_json("./utils/color_config")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
            write_json({"gantt_color": color}, "./utils/color_config")
            for batch_id in range(self.batch_size):
                schedules = self.schedules_batch[batch_id].to('cpu')
                fig = plt.figure(figsize=(10, 6))
                fig.canvas.set_window_title('Visual_gantt')
                axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
                y_ticks = []
                y_ticks_loc = []
                for i in range(num_mas):
                    y_ticks.append('Machine {0}'.format(i))
                    y_ticks_loc.insert(0, i + 1)
                labels = [''] * num_jobs
                for j in range(num_jobs):
                    labels[j] = "job {0}".format(j + 1)
                patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
                axes.cla()
                axes.set_title(u'FJSP Schedule')
                axes.grid(linestyle='-.', color='gray', alpha=0.2)
                axes.set_xlabel('Time')
                axes.set_ylabel('Machine')
                axes.set_yticks(y_ticks_loc, y_ticks)
                axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                for i in range(int(self.nums_opes[batch_id])):
                    id_ope = i
                    idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                    id_machine = schedules[id_ope][1]
                    axes.barh(id_machine,
                             0.2,
                             left=schedules[id_ope][2],
                             color='#b2b2b2',
                             height=0.5)
                    axes.barh(id_machine,
                             schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
                             left=schedules[id_ope][2]+0.2,
                             color=color[idx_job],
                             height=0.5)
                plt.show()
        return

    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        # 校验调度方案是否可行(训练/测试每局结束都会调用)
        # 按机器整理甘特图: ma_gantt_batch[算例][机器] = [[工序id, 开始, 结束], ...]
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        # 检查1: 同机器上工序是否重叠, 以及加工时长是否正确
        flag_proc_time = 0  # 加工时长错误计数
        flag_ma_overlap = 0  # 机器上时间重叠计数
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        # 检查2: 同作业内相邻工序是否满足先后顺序(前一道结束不晚于后一道开始)
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        # 检查3: 是否存在未被调度的工序(status != 1)
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        # 任一检查有问题则判为不可行
        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass
