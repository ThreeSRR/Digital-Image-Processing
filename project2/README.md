# Project2

### Task

对彩色眼底视网膜图像进行预处理，实现尺寸、位置、颜色的归一化，用于以后的病灶识别任务

1. 空域对齐：检测视盘中心和黄斑中心，将所有图像的视盘中心和黄斑中心对齐
2. 颜色归一化：根据参考图像，使图像的颜色直方图与参考图像相近
3. 血管检测与填充



### Structure

- [`unet-code/`](./unet-code/)为unet进行血管分割部分代码
- [`unet-results/`](./unet-results/)为DRIVE数据集上血管分割、空域对齐与血管填充结果
- [`color_normalization.py`](./color_normalization.py)为颜色归一化代码
- [`dot_detection_and_align.py`](./dot_detection_and_align.py)为视盘中心检测、黄斑检测与空域对齐代码
- [`vessel-removal.py`](./vessel_removal.py)为血管区域填充代码

