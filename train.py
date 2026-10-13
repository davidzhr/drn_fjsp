import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env


def setup_seed(seed):
    # 固定随机种子, 保证实验可复现(torch/numpy/random 全部设置)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    # 读取配置文件(四块: 环境/模型/训练/测试参数)
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]  # 环境参数
    model_paras = load_dict["model_paras"]  # 模型(HGNN/actor/critic)参数
    train_paras = load_dict["train_paras"]  # 训练(PPO)参数
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)  # 验证环境参数(复制一份改 batch_size)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]  # 验证集并行算例数(=100)
    # 推导 actor/critic 输入维度: actor 拼4部分, critic 拼2部分
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    num_jobs = env_paras["num_jobs"]  # 作业数
    num_mas = env_paras["num_mas"]  # 机器数
    opes_per_job_min = int(num_mas * 0.8)  # 每作业工序数下界(随机算例用)
    opes_per_job_max = int(num_mas * 1.2)  # 每作业工序数上界

    memories = PPO_model.Memory()  # rollout 缓冲区
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])  # PPO(含策略网络+优化器)
    env_valid = get_validate_env(env_valid_paras)  # Create an environment for validation
    maxlen = 1  # Save the best model
    best_models = deque()  # 保存最优模型文件名(只留最近1个)
    makespan_best = float('inf')  # 历史最优(最小)验证 makespan

    # Use visdom to visualize the training process
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    os.makedirs(save_path)
    # Training curve storage path (average of validation set)
    writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    # Training curve storage path (value of each validating instance)
    writer_100 = pd.ExcelWriter('{0}/training_100_{1}.xlsx'.format(save_path, str_time))
    valid_results = []
    valid_results_100 = []
    # 表头列(iterations), 结果列在训练结束后写入, 最后统一 close
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_ave, sheet_name='Sheet1', index=False)
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel(writer_100, sheet_name='Sheet1', index=False)

    # Start training iteration
    # 开始训练迭代
    start_time = time.time()
    env = None
    for i in range(1, train_paras["max_iterations"]+1):
        # Replace training instances every x iteration (x = 20 in paper)
        # 每 parallel_iter 轮(论文=20)更换一批训练算例
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # \mathcal{B} instances use consistent operations to speed up training
            # 随机每作业工序数, 用生成器合成一批算例并建环境(.unwrapped 绕过 gym wrapper)
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras, disable_env_checker=True).unwrapped
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal
        state = env.state  # 初始状态
        done = False  # 整批是否全部完成
        dones = env.done_batch  # 每个算例是否完成
        last_time = time.time()

        # Schedule in parallel
        # 并行调度: 一步步决策直到整批算例都排完
        while ~done:
            with torch.no_grad():  # 采集阶段不建计算图
                actions = model.policy_old.act(state, memories, dones)  # 旧策略出动作(采样), 同时写 memory
            state, rewards, dones = env.step(actions)  # 环境执行一步
            done = dones.all()  # 全部完成才停
            memories.rewards.append(rewards)  # 记录奖励
            memories.is_terminals.append(dones)  # 记录终止标记
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time()-last_time)

        # Verify the solution
        # 校验调度方案(甘特图)是否可行
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()  # 复位环境, 供同批算例下一轮复用

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        # 每 update_timestep 轮(论文=1)做一次 PPO 更新
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)  # 用缓冲数据更新策略
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()  # 更新后清空缓冲区
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        # 每 save_timestep 轮(论文=10)在验证集上评估一次并保存最优
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            # 验证: 返回平均 makespan 和每个算例的 makespan
            vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.policy_old)
            valid_results.append(vali_result.item())
            valid_results_100.append(vali_result_100)

            # Save the best model
            # 若本次验证优于历史最优, 则删旧存新(只保留最优1个)
            if vali_result < makespan_best:
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()  # 取出旧最优文件名
                    os.remove(delete_file)  # 删除磁盘上的旧模型
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)  # 保存当前最优权重

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result.item()]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))

    # Save the data of training curve to files
    data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
    data.to_excel(writer_ave, sheet_name='Sheet1', index=False, startcol=1)
    writer_ave.close()  # 新版 pandas: close() 内部会保存
    column = [i_col for i_col in range(100)]
    data = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')), columns=column)
    data.to_excel(writer_100, sheet_name='Sheet1', index=False, startcol=1)
    writer_100.close()  # 新版 pandas: close() 内部会保存

    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()