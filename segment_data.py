"""
数据分割脚本：遍历 data/ 下所有被试者目录，
基于 carla_data 中各场景的时间戳，从整段眼动数据和手脚距离数据中
分割出对应片段，输出到 processed_data/ 对应路径下的 segment_data/ 文件夹。

用法: python segment_data.py
"""

import os
import sys
import ast
import glob
import pandas as pd

# ========== 路径配置 ==========
BYSJ_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BYSJ_DIR, "data")
PROCESSED_DIR = os.path.join(BYSJ_DIR, "processed_data")


def parse_gaze_point(raw_str):
    """解析 FilteredClosestWorldIntersection 列，
    提取 ObjectPoint 的 x, y 坐标，并根据 ObjectName 偏移，
    返回 (Screenpoint_x, Screenpoint_y)。"""
    try:
        data = ast.literal_eval(raw_str)
    except (ValueError, SyntaxError):
        return (float('nan'), float('nan'))

    if not isinstance(data, dict):
        return (float('nan'), float('nan'))

    object_name = data.get('ObjectName', 'ScreenMiddle')
    if not object_name:
        object_name = 'ScreenMiddle'

    obj_pt = data.get('ObjectPoint', {})
    x = obj_pt.get('x', float('nan'))
    y = obj_pt.get('y', float('nan'))

    if object_name == 'ScreenMiddle':
        x += 1920
    elif object_name == 'ScreenRight':
        x += 3840
    # ScreenLeft 不偏移

    return (x, y)


def process_subject(subject_dir, output_dir):
    """处理单个被试者的数据分割"""

    subject_name = os.path.basename(subject_dir)
    carla_dir = os.path.join(subject_dir, "carla_data")
    gaze_dir = os.path.join(subject_dir, "Gazing_point_program")

    # 检查必要目录/文件
    if not os.path.isdir(carla_dir):
        print(f"  [跳过] 未找到 carla_data 目录")
        return
    handfoot_files = glob.glob(os.path.join(subject_dir, "毕设_*.csv"))
    if not handfoot_files:
        print(f"  [跳过] 未找到手脚距离数据文件 (毕设_*.csv)")
        return
    gaze_files = glob.glob(os.path.join(gaze_dir, "received_data_*.csv"))
    if not gaze_files:
        print(f"  [跳过] 未找到眼动数据文件")
        return

    seg_output_dir = os.path.join(output_dir, "segment_data")
    os.makedirs(seg_output_dir, exist_ok=True)

    # ----- 1. 读取各场景时间范围 -----
    scene_files = sorted(glob.glob(os.path.join(carla_dir, "Scene_*.csv")))
    if not scene_files:
        print(f"  [跳过] carla_data 下无 Scene_*.csv 文件")
        return

    scene_info = []
    for sf in scene_files:
        basename = os.path.basename(sf).replace(".csv", "")
        df_scene = pd.read_csv(sf)
        t_min = df_scene["time"].min()
        t_max = df_scene["time"].max()
        scene_info.append({
            "name": basename,
            "file": sf,
            "t_start": t_min,
            "t_end": t_max,
        })
        print(f"    {basename}: {t_min:.3f} ~ {t_max:.3f}  "
              f"(时长 {t_max - t_min:.1f}s, {len(df_scene)} 行)")

    # ----- 2. 读取眼动数据 -----
    gaze_file = gaze_files[0]
    print(f"    读取眼动: {os.path.basename(gaze_file)} ...")
    df_gaze = pd.read_csv(gaze_file, low_memory=False)
    df_gaze["StorageTime_sec"] = df_gaze["StorageTime"] / 1000.0

    # 解析 FilteredClosestWorldIntersection -> Screenpoint_x, Screenpoint_y
    if "FilteredClosestWorldIntersection" in df_gaze.columns:
        print(f"    解析注视点坐标 (FilteredClosestWorldIntersection) ...")
        coords = df_gaze["FilteredClosestWorldIntersection"].apply(parse_gaze_point)
        df_gaze["Screenpoint_x"] = coords.apply(lambda c: c[0])
        df_gaze["Screenpoint_y"] = coords.apply(lambda c: c[1])
    else:
        print(f"    [警告] 未找到 FilteredClosestWorldIntersection 列")

    print(f"    眼动总行数: {len(df_gaze)}")

    # ----- 3. 读取手脚距离数据 -----
    handfoot_file = handfoot_files[0]
    print(f"    读取手脚: {os.path.basename(handfoot_file)} ...")
    # 先尝试无表头读取，检测第一行是否为表头
    df_hf = pd.read_csv(handfoot_file, header=None,
                         names=["timestamp_ms", "dist1", "dist2", "dist3"])
    # 如果第一个值不是数字，说明文件有表头行，需要跳过
    first_val = str(df_hf.iloc[0, 0])
    if not first_val.replace('.', '', 1).isdigit():
        print(f"    检测到表头行，重新读取 ...")
        df_hf = pd.read_csv(handfoot_file)
        df_hf.columns = ["timestamp_ms", "dist1", "dist2", "dist3"]
    df_hf["timestamp_ms"] = pd.to_numeric(df_hf["timestamp_ms"], errors="coerce")
    df_hf["timestamp_sec"] = df_hf["timestamp_ms"] / 1000.0
    print(f"    手脚总行数: {len(df_hf)}")

    # ----- 4. 按场景切割并保存 -----
    for info in scene_info:
        name = info["name"]
        t_start = info["t_start"]
        t_end = info["t_end"]

        # 仿真数据
        df_sim = pd.read_csv(info["file"])
        sim_out = os.path.join(seg_output_dir, f"{name}_仿真.csv")
        df_sim.to_csv(sim_out, index=False, encoding="utf-8-sig")

        # 眼动数据
        mask_gaze = (df_gaze["StorageTime_sec"] >= t_start) & \
                     (df_gaze["StorageTime_sec"] <= t_end)
        df_gaze_seg = df_gaze.loc[mask_gaze].drop(columns=["StorageTime_sec"])
        gaze_out = os.path.join(seg_output_dir, f"{name}_眼动.csv")
        df_gaze_seg.to_csv(gaze_out, index=False, encoding="utf-8-sig")

        # 手脚数据
        mask_hf = (df_hf["timestamp_sec"] >= t_start) & \
                   (df_hf["timestamp_sec"] <= t_end)
        df_hf_seg = df_hf.loc[mask_hf].drop(columns=["timestamp_sec"])
        hf_out = os.path.join(seg_output_dir, f"{name}_手脚.csv")
        df_hf_seg.to_csv(hf_out, index=False, header=False, encoding="utf-8-sig")

        print(f"    {name}  仿真:{len(df_sim)}  眼动:{len(df_gaze_seg)}  "
              f"手脚:{len(df_hf_seg)}")

    print(f"  => 输出目录: {seg_output_dir}")


def main():
    if not os.path.isdir(DATA_DIR):
        print(f"[错误] 未找到数据目录: {DATA_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("数据分割脚本")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {PROCESSED_DIR}")
    print("=" * 60)

    # 遍历 data/ 下所有实验批次 -> 被试者
    for experiment in sorted(os.listdir(DATA_DIR)):
        exp_dir = os.path.join(DATA_DIR, experiment)
        if not os.path.isdir(exp_dir):
            continue

        for subject in sorted(os.listdir(exp_dir)):
            subject_dir = os.path.join(exp_dir, subject)
            if not os.path.isdir(subject_dir):
                continue

            print(f"\n{'─' * 60}")
            print(f"处理: {experiment}/{subject}")
            print(f"{'─' * 60}")

            output_dir = os.path.join(PROCESSED_DIR, experiment, subject)
            process_subject(subject_dir, output_dir)

    print(f"\n{'=' * 60}")
    print("全部分割完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
