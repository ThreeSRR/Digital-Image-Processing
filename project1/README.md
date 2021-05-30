# Project1

编程复现论文[`Lisheng Wang, Jing Bai, Threshold selection by clustering gray levels of boundary, Pattern Recognition Letters 24 (2003) 1983–1999`](./Threshold_selection_by_clustering_gray_levels_of_boundary.pdf)中的阈值分割算法，用于附件中几个图像的单阈值或多阈值分割测试。

1. 与论文中对应的实验结果进行比较

2. 观察选择不同梯度阈值时，对边缘检测结果的影响，以及对阈值计算结果的影响

3. 观察采用不同梯度计算模型（robert、sobel、prewitt）时，对结果是否有影响

4. 观察直接用Laplacian算在在图像中检测零交叉与采用本文方法计算（梯度约束的Laplacian检测）的零交叉在分布上存在哪些不同

   

## Code Structure

- [`segmentation_bilevel.py`](./segmentation_bilevel.py)为单阈值分割的代码
- [`segmentation_multi.py`](./segmentation_multi.py)为多阈值分割的代码
- [`bilevel-thres/`](./bilevel-thres/)为单阈值分割使用的图像
- [`multi-thres/`](./multi-thres/)为多阈值分割使用的图像
- [`result/`](./result/)为部分实验结果

