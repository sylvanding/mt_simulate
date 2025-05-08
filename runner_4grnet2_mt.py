import numpy as np
import pandas as pd
import re
import os
import pathlib
import math
import multiprocessing
import h5py


def normalize_pc_pair(input, gt=None, return_params=False):
    """
    对点云数据进行归一化,支持单个点云或点云对的归一化

    Args:
        input: 输入点云 (b, n, 3)
        gt: 目标点云 (b, m, 3), 可选
        return_params: 是否返回归一化参数

    Returns:
        normalized_input: 归一化后的输入点云
        normalized_gt: 归一化后的目标点云(如果提供)
        params: 归一化参数(如果return_params=True)
    """
    # 计算中心点和最远距离
    centroid = np.mean(input, axis=1, keepdims=True)  # (b, 1, 3)
    input_centered = input - centroid
    furthest_distances = np.amax(
        np.abs(input_centered), axis=(1, 2), keepdims=True
    )  # (b, 1, 1)
    furthest_distances = np.repeat(furthest_distances, 3, axis=2)

    # 归一化input
    normalized_input = input_centered / furthest_distances

    # 如果提供了gt,使用相同参数归一化
    normalized_gt = None
    if gt is not None:
        gt_centered = gt - centroid
        normalized_gt = gt_centered / furthest_distances

    if return_params:
        params = {"centroid": centroid, "scale": furthest_distances}
        return normalized_input, normalized_gt, params

    return normalized_input, normalized_gt


def read_mt_data(filename):
    """读取微管数据文件"""
    mt_data = []
    current_mt = None

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("<TUBE>"):
            if current_mt is not None:
                mt_data.append(current_mt)
            current_mt = {"points": [], "radius": 0, "thickness": 0}
            # 提取微管编号
            match = re.search(r"tube (\d+)", line)
            if match:
                current_mt["id"] = int(match.group(1))
        elif line.startswith("radius:"):
            current_mt["radius"] = float(line.split(":")[1].strip().strip(","))
        elif line.startswith("thickness:"):
            current_mt["thickness"] = float(line.split(":")[1].strip().strip(","))
        elif line.startswith("node:"):
            coords = line.split(":")[1].strip().split(",")
            point = [float(x) for x in coords]
            current_mt["points"].append(point)

    if current_mt is not None:
        mt_data.append(current_mt)

    return mt_data


def generate_membrane_points(mt, n_points=256, thickness_offset=2):
    """在微管膜表面生成随机点"""
    points = np.array(mt["points"])
    membrane_points = []

    # 将点数平均分配到每个节点
    points_per_node = max(1, n_points // (len(points) - 1))
    remaining_points = n_points - points_per_node * (len(points) - 1)

    for i in range(len(points) - 1):
        # 当前节点和下一个节点
        p1 = points[i]
        p2 = points[i + 1]

        # 计算这段需要生成的点数
        current_points = points_per_node + (1 if remaining_points > 0 else 0)
        if remaining_points > 0:
            remaining_points -= 1

        if current_points == 0:
            continue

        # 计算方向向量
        direction = p2 - p1
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10:  # 避免除以零
            continue
        direction = direction / direction_norm

        # 生成随机点
        for _ in range(current_points):
            # 在两点之间随机选择位置
            t = np.random.random()
            center = p1 + t * direction * direction_norm

            # 生成垂直于方向向量的随机单位向量
            while True:
                random_vec = np.random.randn(3)
                random_vec = random_vec - np.dot(random_vec, direction) * direction
                norm = np.linalg.norm(random_vec)
                if norm > 1e-10:  # 确保向量不为零
                    random_vec = random_vec / norm
                    break

            # 在膜表面上生成点（距离中轴线 thickness - thickness_offset 的距离）
            # 使用 thickness_offset 参数
            membrane_radius = mt["thickness"] - thickness_offset
            if membrane_radius <= 0:
                print(
                    f"警告: 微管 {mt.get('id', 'N/A')} 的 thickness ({mt['thickness']}) 小于或等于 offset ({thickness_offset})，生成的点将位于中轴线上或内部。"
                )
                membrane_radius = 0  # 避免负半径

            membrane_point = center + random_vec * membrane_radius
            membrane_points.append(
                [mt["id"], membrane_point[0], membrane_point[1], membrane_point[2]]
            )

    return membrane_points


def process_file(args):
    """处理单个txt文件，并返回点云数据"""
    (
        txt_file,
        min_mt_num,
        max_mt_num,
        n_points_per_pc,
        # base_output_dir 不再需要传入 process_file
        thickness_offset,
    ) = args

    try:
        # 读取微管数据
        mt_data = read_mt_data(str(txt_file))

        # 根据每个微管的点数分配总点数
        original_mt_count = len(mt_data)  # 记录原始微管数量
        if len(mt_data) > 0:
            # 随机选择的微管数量，在min_mt_num和max_mt_num之间
            min_mt_num = max(min_mt_num, 1)
            max_mt_num = min(max_mt_num, len(mt_data))
            select_num = np.random.randint(min_mt_num, max_mt_num + 1)
            # 随机选择微管
            selected_indices = np.random.choice(len(mt_data), select_num, replace=False)
            mt_data = [mt_data[i] for i in selected_indices]
            print(
                f"文件 {txt_file.name}: 从{original_mt_count}条微管中随机选择了{select_num}条"
            )
        else:
            print(f"文件 {txt_file.name}: 没有读取到微管数据，跳过处理。")
            return None  # 返回 None

        total_points = sum(len(mt["points"]) for mt in mt_data)
        # 避免 total_points 为 0 导致除零错误
        if total_points == 0:
            print(f"文件 {txt_file.name}: 所选微管总点数为 0，跳过处理。")
            return None  # 返回 None

        n_points_list = [
            math.ceil(n_points_per_pc * len(mt["points"]) / total_points)
            for mt in mt_data
        ]

        # 为每个微管生成膜表面点
        all_points = []
        for mt, n_points in zip(mt_data, n_points_list):
            # 确保 n_points 大于 0
            if n_points <= 0:
                continue
            # 传递 thickness_offset 参数
            membrane_points = generate_membrane_points(
                mt, n_points=n_points, thickness_offset=thickness_offset
            )
            all_points.extend(membrane_points)

        # 检查是否生成了点
        if not all_points:
            print(f"文件 {txt_file.name}: 未能生成任何点，跳过处理。")
            return None

        # 转换为numpy数组
        points_array = np.array(all_points)
        print(f"文件 {txt_file.name}: 共生成{len(points_array)}个点")

        # 检查点云的点数是否为n_points_per_pc
        current_points = len(points_array)

        # 提取坐标部分（不包含mt_id）
        coords = points_array[:, 1:4]  # 只保留x, y, z坐标

        if current_points < n_points_per_pc:
            # 如果点数少于目标点数，随机复制点
            print(
                f"文件 {txt_file.name}: 点数不足，从{current_points}个点随机复制到{n_points_per_pc}个点"
            )
            # 计算需要复制的点数
            points_to_add = n_points_per_pc - current_points
            # 随机选择索引进行复制
            indices_to_duplicate = np.random.choice(
                current_points, points_to_add, replace=True
            )
            # 复制选定的点
            additional_coords = coords[indices_to_duplicate]
            # 合并原始点和复制的点
            coords = np.vstack((coords, additional_coords))
        elif current_points > n_points_per_pc:
            # 如果点数多于目标点数，随机删除点
            print(
                f"文件 {txt_file.name}: 点数过多，从{current_points}个点随机删减到{n_points_per_pc}个点"
            )
            # 随机选择要保留的点的索引
            indices_to_keep = np.random.choice(
                current_points, n_points_per_pc, replace=False
            )
            # 只保留选定的点
            coords = coords[indices_to_keep]

        # 确认最终点数
        print(f"文件 {txt_file.name}: 最终点数为{len(coords)}个点")

        # --- 不再保存单个 H5 文件，直接返回坐标数据 ---
        return coords  # 返回 NumPy 数组

    except Exception as e:
        print(f"处理文件 {txt_file.name} 时出错: {e}")
        import traceback

        traceback.print_exc()  # 打印详细错误信息
        return None  # 出错时返回 None


def pc_random_sample(pc_data, n_points_per_pc_sparse):
    """
    对点云数据进行随机采样，若点数不足则随机复制补齐

    Args:
        pc_data: 输入点云数据，形状为 (num_files, n_points_per_pc, 3)
        n_points_per_pc_sparse: 目标采样点数

    Returns:
        sampled_data: 采样后的点云数据，形状为 (num_files, n_points_per_pc_sparse, 3)
    """
    num_files = pc_data.shape[0]
    sampled_data = np.zeros((num_files, n_points_per_pc_sparse, 3), dtype=np.float32)

    for i in range(num_files):
        current_points = pc_data[i].shape[0]
        if current_points < n_points_per_pc_sparse:
            # 点数不足，随机复制补齐
            points_to_add = n_points_per_pc_sparse - current_points
            indices_to_duplicate = np.random.choice(
                current_points, points_to_add, replace=True
            )
            additional_points = pc_data[i][indices_to_duplicate]
            sampled_data[i] = np.vstack((pc_data[i], additional_points))
        elif current_points > n_points_per_pc_sparse:
            # 点数过多，随机采样
            indices_to_keep = np.random.choice(
                current_points, n_points_per_pc_sparse, replace=False
            )
            sampled_data[i] = pc_data[i][indices_to_keep]
        else:
            # 点数刚好，无需处理
            sampled_data[i] = pc_data[i]

    return sampled_data


def filter_and_pad_point_cloud(pc_data, z_threshold=(-25, 100)):
    """
    过滤掉点云中 Z 坐标大于阈值的点，并复制点以恢复到原始点数。

    Args:
        pc_data (np.ndarray): 输入的点云数据，形状为 (num_point_clouds, N, 3)。
        z_threshold (float): Z 坐标的过滤阈值。默认为 0.25。

    Returns:
        np.ndarray: 处理后的点云数据，形状与输入相同 (num_point_clouds, N, 3)。
        对于每个点云，Z > z_threshold 的点被移除，
        然后通过随机复制剩余点的方式将点数恢复到 N。
        如果过滤后没有剩余点，则返回一个由零填充的点云。
    """
    num_point_clouds, original_num_points, _ = pc_data.shape
    processed_pcs = []

    for i in range(num_point_clouds):
        single_pc = pc_data[i]

        # 1. 过滤点
        mask = (single_pc[:, 2] >= z_threshold[0]) & (single_pc[:, 2] <= z_threshold[1])
        filtered_pc = single_pc[mask]

        num_filtered_points = filtered_pc.shape[0]

        # 2. 复制点以恢复到 original_num_points
        if num_filtered_points == original_num_points:
            # 没有点被过滤
            processed_pcs.append(filtered_pc)
        elif num_filtered_points == 0:
            # 所有点都被过滤掉了，用零填充
            processed_pcs.append(
                np.zeros((original_num_points, 3), dtype=pc_data.dtype)
            )
        else:
            # 点数不足，需要复制
            points_to_add = original_num_points - num_filtered_points
            # 从过滤后的点中随机选择索引（允许重复）
            indices_to_duplicate = np.random.choice(
                num_filtered_points, points_to_add, replace=True
            )
            duplicated_points = filtered_pc[indices_to_duplicate]
            # 合并过滤后的点和复制的点
            final_pc = np.vstack((filtered_pc, duplicated_points))
            processed_pcs.append(final_pc)

    return np.stack(processed_pcs, axis=0)


def main():
    # 参数设置
    n_points_per_pc = 16384  # 每个点云的点数
    min_mt_num = 30  # 每个点云中的最小微管数量
    max_mt_num = 40  # 每个点云中的最大微管数量
    thickness_offset_val = 5.5  # 在中轴线附近生成的膜表面点距离中轴线的距离为 thickness - thickness_offset_val，thickness = 6 px

    # 下采样参数
    downsample_rate = 8
    n_points_per_pc_sparse = n_points_per_pc // downsample_rate  # 2048

    # 输出目录和 H5 文件名
    output_file_name = "mt_pc_%s_%s_%s_%s_%s" % (
        n_points_per_pc,
        n_points_per_pc_sparse,
        min_mt_num,
        max_mt_num,
        thickness_offset_val,
    )
    base_output_dir = pathlib.Path("./outputs")
    base_output_dir.mkdir(parents=True, exist_ok=True)
    # 定义聚合 H5 文件的完整路径
    aggregate_h5_filepath = base_output_dir / f"{output_file_name}.h5"

    # 获取所有微管中轴线的 txt 文件
    input_dir = pathlib.Path("./mid_axial")  # 微管中轴线文件夹
    txt_files = list(input_dir.glob("*.txt"))

    # 设置进程数
    num_processes = multiprocessing.cpu_count()
    print(f"使用 {num_processes} 个进程进行并行处理...")

    # 准备传递给 process_file 的参数列表 (移除 base_output_dir)
    args_list = [
        (
            txt_file,
            min_mt_num,
            max_mt_num,
            n_points_per_pc,
            # base_output_dir, # 不再需要传递给 process_file
            thickness_offset_val,
        )
        for txt_file in txt_files
    ]

    # 创建进程池并执行任务，收集结果（现在是点云数组或 None）
    gt_data = []
    processed_file_names = []  # 存储成功处理的文件名，以便追踪
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_file, args_list)
        # 过滤掉 None 的结果 (处理失败或未生成点的文件)
        for i, result in enumerate(results):
            if result is not None:
                gt_data.append(result)
                processed_file_names.append(txt_files[i].name)  # 记录成功处理的文件名

    # --- 将所有收集到的点云保存到一个 H5 文件中 ---
    if gt_data:
        print(f"\n成功处理了 {len(gt_data)} 个文件，准备聚合保存...")
        # 将列表中的 NumPy 数组堆叠成一个大的 NumPy 数组
        # 形状将是 (num_files, n_points_per_pc, 3)
        gt_data = np.stack(gt_data, axis=0)
        
        # filter
        gt_data = filter_and_pad_point_cloud(gt_data, z_threshold=(-25, 100))

        # sample and normalize
        input_data = pc_random_sample(gt_data, n_points_per_pc_sparse)
        gt_data, input_data, norm_params = normalize_pc_pair(gt_data, input_data, True)

        print(f"np.max(input_data): {np.max(input_data, axis=(0, 1))}")
        print(f"np.min(input_data): {np.min(input_data, axis=(0, 1))}")
        print(f"np.max(gt_data): {np.max(gt_data, axis=(0, 1))}")
        print(f"np.min(gt_data): {np.min(gt_data, axis=(0, 1))}")

        try:
            with h5py.File(aggregate_h5_filepath, "w") as f:
                # 保存聚合的点云数据
                f.create_dataset("input_data", data=input_data.astype(np.float32))
                f.create_dataset("gt_data", data=gt_data.astype(np.float32))
                # Store normalization parameters
                f.create_dataset(
                    "norm_params/centroid",
                    data=norm_params["centroid"].astype(np.float32),
                )
                f.create_dataset(
                    "norm_params/scale", data=norm_params["scale"].astype(np.float32)
                )

            print(f"\n所有点云已成功聚合保存到: {aggregate_h5_filepath}")
            print(f"H5 文件包含 'gt_data' 数据集，形状为: {gt_data.shape}")
            print(f"'source_files' 数据集，包含 {len(processed_file_names)} 个源文件。")

        except Exception as save_e:
            print(f"\n聚合保存到 H5 时出错: {save_e}")
    else:
        print("\n未能成功处理任何文件，未生成聚合 H5 文件。")

    print("所有文件处理完成。")


if __name__ == "__main__":
    main()
