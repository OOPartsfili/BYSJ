"""
打分数据合并脚本：将 score_data 中的打分合并到 merge1_data 中。

逻辑：
  - score_data 是稀疏事件（INIT → D → FINAL），记录了回放过程中的打分时刻
  - merge1_data 是连续高频数据（~65Hz），包含仿真+眼动+手脚
  - score_data 的 sim_time 是相对时间（从 0 开始），与 merge1 的场景时长一致
  - 将 merge1 的 time 列归一化为相对时间后，按 sim_time 对齐打分
  - 打分为阶梯函数：两次打分事件之间，分数保持不变

匹配规则：
  score 文件名包含场景编号和时间戳（如 score_Scene_1_01_1774511173_...），
  merge1 文件名为 Scene_1_01_1774511173_merged.csv（排除含 crush 的文件）

输出: processed_data/{experiment}/{subject}/merge2_data/

用法: python score_merge_data.py
"""

import os
import sys
import re
import glob
import numpy as np
import pandas as pd

BYSJ_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BYSJ_DIR, "data")
PROCESSED_DIR = os.path.join(BYSJ_DIR, "processed_data")


def extract_scene_key(filename):
    """从文件名中提取场景标识，如 'Scene_1_01_1774511173'"""
    # 匹配 Scene_X_XX_TIMESTAMP 模式
    m = re.search(r'(Scene_\d+_\d+_\d+)', filename)
    return m.group(1) if m else None


def merge_score_to_scene(merge1_file, score_file, output_file):
    """将打分数据合并到单个场景的 merge1 数据中"""

    # 读取 merge1 数据
    df_merge = pd.read_csv(merge1_file)
    sim_times = df_merge["time"].values.astype(float)
    # 归一化为相对时间（从 0 开始）
    t0 = sim_times[0]
    rel_times = sim_times - t0

    # 读取打分数据
    df_score = pd.read_csv(score_file)
    # sim_time 列就是相对时间
    score_times = df_score["sim_time"].values.astype(float)
    score_vals = df_score["score"].values.astype(float)

    # 阶梯插值：对每个 merge1 时刻，找到 <= 该时刻的最后一个打分事件
    # 用 np.searchsorted 找到插入位置，然后向左取
    indices = np.searchsorted(score_times, rel_times, side="right") - 1
    indices = np.clip(indices, 0, len(score_vals) - 1)
    merged_scores = score_vals[indices]

    # 添加列
    df_merge["score"] = merged_scores

    # 保存
    df_merge.to_csv(output_file, index=False, encoding="utf-8-sig")
    return len(df_merge)


def process_subject(subject_name, experiment, data_base, processed_base):
    """处理单个被试者"""

    score_dir = os.path.join(data_base, experiment, subject_name, "score_data")
    merge1_dir = os.path.join(processed_base, experiment, subject_name, "merge1_data")
    merge2_dir = os.path.join(processed_base, experiment, subject_name, "merge2_data")

    if not os.path.isdir(score_dir):
        print(f"  [跳过] 未找到 score_data 目录")
        return
    if not os.path.isdir(merge1_dir):
        print(f"  [跳过] 未找到 merge1_data 目录")
        return

    # 构建 score 文件的场景键映射
    score_files = sorted(glob.glob(os.path.join(score_dir, "score_*.csv")))
    score_map = {}
    for sf in score_files:
        key = extract_scene_key(os.path.basename(sf))
        if key:
            score_map[key] = sf

    # 找到所有非 crush 的 merge1 文件
    merge1_files = sorted(glob.glob(os.path.join(merge1_dir, "*_merged.csv")))
    non_crush = [f for f in merge1_files if "crush" not in os.path.basename(f)]

    if not non_crush:
        print(f"  [跳过] merge1_data 下无非 crush 文件")
        return

    os.makedirs(merge2_dir, exist_ok=True)

    for mf in non_crush:
        basename = os.path.basename(mf)
        scene_key = extract_scene_key(basename)

        if scene_key not in score_map:
            print(f"    [跳过] {scene_key}: 无对应打分文件")
            continue

        output_file = os.path.join(merge2_dir, basename)
        n_rows = merge_score_to_scene(mf, score_map[scene_key], output_file)
        print(f"    {scene_key}: {n_rows} 行 -> {basename}")


def main():
    print("=" * 60)
    print("打分合并脚本")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {PROCESSED_DIR}")
    print("=" * 60)

    # 遍历所有实验批次和被试者
    for experiment in sorted(os.listdir(DATA_DIR)):
        exp_dir = os.path.join(DATA_DIR, experiment)
        if not os.path.isdir(exp_dir):
            continue

        for subject in sorted(os.listdir(exp_dir)):
            subject_dir = os.path.join(exp_dir, subject)
            if not os.path.isdir(subject_dir):
                continue

            # 检查是否有 score_data 目录
            if not os.path.isdir(os.path.join(subject_dir, "score_data")):
                continue

            print(f"\n{'─' * 60}")
            print(f"处理: {experiment}/{subject}")
            print(f"{'─' * 60}")

            process_subject(subject, experiment, DATA_DIR, PROCESSED_DIR)

    print(f"\n{'=' * 60}")
    print("全部打分合并完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
