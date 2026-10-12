"""
独立调试 load_fjs 的脚本。
复刻 env/fjsp_env.py 中 FJSPEnv.__init__ 对 load_fjs 的调用方式：
    先用 nums_detec 解析规模，再调用 load_fjs。
直接运行：  python debug_load_fjs.py
可在 load_data.py 的 load_fjs / edge_detec 里打断点，或在本文件里打断点单步跟踪。
"""
import torch

from env.load_data import load_fjs, nums_detec

# 任选一个算例文件
CASE_PATH = "data_dev/1005/10j_5m_001.fjs"


def main():
    torch.set_printoptions(linewidth=200, threshold=10000)

    with open(CASE_PATH) as f:
        lines = f.readlines()

    # 第一步：解析规模（和 fjsp_env.py 一致）
    num_jobs, num_mas, num_opes = nums_detec(lines)
    print(f"num_jobs={num_jobs}, num_mas={num_mas}, num_opes={num_opes}\n")

    # 第二步：调用 load_fjs（断点可打在这一行）
    (matrix_proc_time,
     matrix_ope_ma_adj,
     matrix_pre_proc,
     matrix_sub_proc,          # = matrix_pre_proc.t()
     opes_appertain,
     num_ope_biases,
     nums_ope,
     matrix_cal_cumul) = load_fjs(lines, num_mas, num_opes)

    print("=== matrix_proc_time  [num_opes x num_mas] 加工时间 ===")
    print(matrix_proc_time)
    print("\n=== matrix_ope_ma_adj  工序-机器可加工邻接(0/1) ===")
    print(matrix_ope_ma_adj)
    print("\n=== matrix_pre_proc  紧前约束(True=i在i+1前) ===")
    print(matrix_pre_proc.int())
    print("\n=== opes_appertain  每个工序归属的作业id ===")
    print(opes_appertain)
    print("\n=== num_ope_biases  每个作业首工序的全局id ===")
    print(num_ope_biases)
    print("\n=== nums_ope  每个作业的工序数 ===")
    print(nums_ope)
    print("\n=== matrix_cal_cumul  前驱累积矩阵 ===")
    print(matrix_cal_cumul)


if __name__ == "__main__":
    main()
