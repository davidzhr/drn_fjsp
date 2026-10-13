import gym
import env
import PPO_model
import torch
import time
import os
import copy

def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    # 构建验证环境: 读取 ./data_dev/{作业数}{机器数} 目录下全部 .fjs 文件作为验证集
    file_path = "./data_dev/{0}{1}/".format(env_paras["num_jobs"], str.zfill(str(env_paras["num_mas"]),2))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]  # 拼成完整路径
    env = gym.make('fjsp-v0', case=valid_data_files, env_paras=env_paras, data_source='file',
                   disable_env_checker=True).unwrapped
    return env

def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    # 在验证集上贪心跑完整调度, 返回平均 makespan 与每个算例的 makespan
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            # 验证固定贪心(flag_sample=False)且不写 memory(flag_train=False)
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    gantt_result = env.validate_gantt()[0]  # 校验方案可行性
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    makespan = copy.deepcopy(env.makespan_batch.mean())  # 平均 makespan
    makespan_batch = copy.deepcopy(env.makespan_batch)  # 每个算例的 makespan
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch
