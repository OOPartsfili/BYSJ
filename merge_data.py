"""
数据合并脚本：以仿真数据的时间轴为基准，
将眼动和手脚数据通过时间插值对齐到仿真的每一行，
合并为一份完整的 CSV 文件。

逻辑：
  - 仿真数据 ~65Hz，眼动 ~60Hz，手脚 ~20Hz
  - 以仿真数据的 time 列为基准时间轴
  - 对眼动数据的数值列进行线性插值对齐
  - 对手脚数据的数值列进行线性插值对齐
  - 非数值列(如字符串/字典列)使用最近邻匹配

输出: processed_data/{experiment}/{subject}/merge1_data/

用法: python merge_data.py
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

BYSJ_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BYSJ_DIR, "processed_data")


def interpolate_to_target(df_source, source_time_col, target_times,
                           prefix="", numeric_only=True):
    """
    将 df_source 按 source_time_col 插值到 target_times 时间轴上。

    对数值列: 线性插值
    对非数值列: 最近邻匹配 (选取时间最接近的行)

    返回: DataFrame，行数 = len(target_times)
    """
    df = df_source.copy()
    src_times = df[source_time_col].values.astype(float)

    # 分离数值列和非数值列
    other_cols = [c for c in df.columns if c != source_time_col]
    numeric_cols = []
    non_numeric_cols = []
    for c in other_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            non_numeric_cols.append(c)

    result = pd.DataFrame()

    # 数值列: 线性插值 (跳过 NaN 值，仅在有效数据点间插值)
    for col in numeric_cols:
        vals = df[col].values.astype(float)
        # 去除 NaN，仅用有效数据点进行插值
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() == 0:
            # 该列全为 NaN，直接填 NaN
            col_name = f"{prefix}{col}" if prefix else col
            result[col_name] = np.nan
            continue
        valid_times = src_times[valid_mask]
        valid_vals = vals[valid_mask]
        interpolated = np.interp(target_times, valid_times, valid_vals)
        col_name = f"{prefix}{col}" if prefix else col
        result[col_name] = interpolated

    # 非数值列: 最近邻匹配
    if non_numeric_cols and not numeric_only:
        # 构建最近邻索引
        nearest_idx = np.searchsorted(src_times, target_times, side="left")
        nearest_idx = np.clip(nearest_idx, 0, len(src_times) - 1)
        # 检查左边是否更近
        left_idx = np.clip(nearest_idx - 1, 0, len(src_times) - 1)
        use_left = np.abs(target_times - src_times[left_idx]) < \
                    np.abs(target_times - src_times[nearest_idx])
        nearest_idx[use_left] = left_idx[use_left]

        for col in non_numeric_cols:
            col_name = f"{prefix}{col}" if prefix else col
            result[col_name] = df[col].iloc[nearest_idx].values

    return result


def process_scene_merge(sim_file, gaze_file, hf_file, output_file):
    """对单个场景执行合并"""

    # 读取仿真数据 (基准)
    df_sim = pd.read_csv(sim_file)
    target_times = df_sim["time"].values.astype(float)
    n_target = len(target_times)

    # 读取眼动数据
    df_gaze = pd.read_csv(gaze_file, low_memory=False)
    # StorageTime 是毫秒，转为秒
    df_gaze["StorageTime_sec"] = df_gaze["StorageTime"] / 1000.0

    # 选取眼动中需要插值的数值列（排除一些非关键的超大字符串列）
    # 眼动核心数值列
    gaze_key_cols = [
        "StorageTime_sec","Screenpoint_x","Screenpoint_y",
        "EstimatedDelay", "TimeStamp", "FrameRate",
        "HeadHeading", "HeadPitch", "HeadRoll",
        "FilteredGazeHeading", "FilteredGazePitch",
        "FilteredLeftGazeHeading", "FilteredLeftGazePitch",
        "FilteredRightGazeHeading", "FilteredRightGazePitch",
        "EyelidOpening", "LeftEyelidOpening", "RightEyelidOpening",
        "PupilDiameter", "LeftPupilDiameter", "RightPupilDiameter",
        "FilteredPupilDiameter", "FrameNumber", "StorageTime",
    ]
    # 只保留存在的列
    gaze_key_cols = [c for c in gaze_key_cols if c in df_gaze.columns]
    df_gaze_subset = df_gaze[gaze_key_cols].copy()

    # 插值眼动
    gaze_interp = interpolate_to_target(
        df_gaze_subset,
        source_time_col="StorageTime_sec",
        target_times=target_times,
        prefix="gaze_",
        numeric_only=True,
    )

    # 读取手脚数据
    df_hf = pd.read_csv(hf_file, header=None,
                          names=["timestamp_ms", "dist1", "dist2", "dist3"])
    df_hf["timestamp_sec"] = df_hf["timestamp_ms"] / 1000.0

    # 插值手脚
    hf_interp = interpolate_to_target(
        df_hf[["timestamp_sec", "dist1", "dist2", "dist3"]],
        source_time_col="timestamp_sec",
        target_times=target_times,
        prefix="hf_",
        numeric_only=True,
    )

    # 合并: 仿真 + 眼动插值 + 手脚插值
    df_merged = pd.concat([df_sim.reset_index(drop=True),
                            gaze_interp.reset_index(drop=True),
                            hf_interp.reset_index(drop=True)], axis=1)

    # 填充缺失值：首行缺失用首次有效值填充，尾部缺失用末次有效值填充
    for col in df_merged.columns:
        if df_merged[col].isna().any():
            # bfill 填充开头的 NaN，ffill 填充结尾的 NaN
            df_merged[col] = df_merged[col].bfill().ffill()

    df_merged.to_csv(output_file, index=False, encoding="utf-8-sig")
    return n_target


def process_subject_merge(seg_dir, merge_output_dir):
    """对一个被试者的所有场景执行合并"""

    # 找到所有仿真文件
    sim_files = sorted(glob.glob(os.path.join(seg_dir, "*_仿真.csv")))
    if not sim_files:
        print(f"  [跳过] segment_data 下无仿真文件")
        return

    os.makedirs(merge_output_dir, exist_ok=True)

    for sim_file in sim_files:
        # 从文件名推断场景名称
        basename = os.path.basename(sim_file)
        scene_name = basename.replace("_仿真.csv", "")

        gaze_file = os.path.join(seg_dir, f"{scene_name}_眼动.csv")
        hf_file = os.path.join(seg_dir, f"{scene_name}_手脚.csv")

        if not os.path.exists(gaze_file):
            print(f"    [跳过] {scene_name}: 缺少眼动文件")
            continue
        if not os.path.exists(hf_file):
            print(f"    [跳过] {scene_name}: 缺少手脚文件")
            continue

        output_file = os.path.join(merge_output_dir, f"{scene_name}_merged.csv")

        n_rows = process_scene_merge(sim_file, gaze_file, hf_file, output_file)
        print(f"    {scene_name}: {n_rows} 行 -> "
              f"{os.path.basename(output_file)}")


def main():
    if not os.path.isdir(PROCESSED_DIR):
        print(f"[错误] 未找到 processed_data 目录: {PROCESSED_DIR}")
        print(f"请先运行 segment_data.py 进行数据分割。")
        sys.exit(1)

    print("=" * 60)
    print("数据合并脚本 (插值对齐)")
    print(f"输入目录: {PROCESSED_DIR}")
    print("=" * 60)

    # 遍历 processed_data/{experiment}/{subject}/segment_data/
    for experiment in sorted(os.listdir(PROCESSED_DIR)):
        exp_dir = os.path.join(PROCESSED_DIR, experiment)
        if not os.path.isdir(exp_dir):
            continue

        for subject in sorted(os.listdir(exp_dir)):
            subject_dir = os.path.join(exp_dir, subject)
            if not os.path.isdir(subject_dir):
                continue

            seg_dir = os.path.join(subject_dir, "segment_data")
            if not os.path.isdir(seg_dir):
                continue

            merge_dir = os.path.join(subject_dir, "merge1_data")

            print(f"\n{'─' * 60}")
            print(f"合并: {experiment}/{subject}")
            print(f"{'─' * 60}")

            process_subject_merge(seg_dir, merge_dir)

    print(f"\n{'=' * 60}")
    print("全部合并完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
