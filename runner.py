import numpy as np
import pandas as pd
import re
import os
import pathlib
import math
import multiprocessing


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
    """处理单个txt文件，并将点云保存为CSV文件"""
    (
        txt_file,
        min_mt_num,
        max_mt_num,
        n_points_per_pc,
        base_output_dir,
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
            return None  # 返回 None

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

        # --- 保存到 CSV 文件 ---
        csv_filename = txt_file.with_suffix(".csv").name
        csv_filepath = base_output_dir / csv_filename
        try:
            np.savetxt(csv_filepath, coords, delimiter=",", fmt="%f")
            print(f"文件 {txt_file.name}: 已成功保存点云到 {csv_filepath}")
            return str(csv_filepath)  # 返回保存的文件路径
        except Exception as save_e:
            print(f"文件 {txt_file.name}: 保存到 CSV 时出错: {save_e}")
            return None

    except Exception as e:
        print(f"处理文件 {txt_file.name} 时出错: {e}")
        import traceback

        traceback.print_exc()  # 打印详细错误信息
        return None  # 出错时返回 None


def main():
    # 参数设置
    n_points_per_pc = 16384  # 每个点云的点数
    min_mt_num = 4  # 每个点云中的最小微管数量
    max_mt_num = 16  # 每个点云中的最大微管数量
    thickness_offset_val = 4  # 在中轴线附近生成的膜表面点距离中轴线的距离为 thickness - thickness_offset_val，thickness = 6 px

    # csv 输出目录
    base_output_dir = pathlib.Path("./outputs/pc_%s" % n_points_per_pc)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有微管中轴线的 txt 文件
    input_dir = pathlib.Path("./mid_axial")  # 微管中轴线文件夹
    txt_files = list(input_dir.glob("*.txt"))

    # 设置进程数
    num_processes = multiprocessing.cpu_count()
    print(f"使用 {num_processes} 个进程进行并行处理...")

    # 准备传递给 process_file 的参数列表
    args_list = [
        (
            txt_file,
            min_mt_num,
            max_mt_num,
            n_points_per_pc,
            base_output_dir,
            thickness_offset_val,
        )
        for txt_file in txt_files
    ]

    # 创建进程池并执行任务，收集结果（现在是文件路径或 None）
    saved_files = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_file, args_list)
        # 过滤掉 None 的结果 (处理失败或未生成点的文件)
        saved_files = [r for r in results if r is not None]

    if saved_files:
        print(f"\n成功处理并保存了 {len(saved_files)} 个文件到目录: {base_output_dir}")
    else:
        print("\n未能成功处理或保存任何文件。")

    print("所有文件处理完成。")


if __name__ == "__main__":
    main()
