import h5py
import matplotlib.pyplot as plt
import numpy as np

# 读取H5文件
file_path = "outputs/mt_pc_16384_2048_30_40_5.5.h5"
with h5py.File(file_path, 'r') as f:
    gt_data = f['gt_data'][:]

# 提取z轴数据
z_values = gt_data[:, :, 2].flatten()

# 绘制z轴分布直方图
plt.figure(figsize=(10, 6))
plt.hist(z_values, bins=100, density=True, alpha=0.7, color='blue')
plt.title('Z-axis Distribution of gt_data')
plt.xlabel('Z Value')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)

# 保存分布图片
output_image_path = "outputs/z_distribution_gt_data.png"
plt.savefig(output_image_path)
plt.close()

print(f"Z轴分布图已保存至: {output_image_path}")
