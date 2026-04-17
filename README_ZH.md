# 3D Gaussian Splatting for Real-Time Radiance Field Rendering

| [网页](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [完整论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) | [视频](https://youtu.be/T_kXY43VZnk) | [GRAPHDECO 其他出版物](http://www-sop.inria.fr/reves/publis/gdindex.php) | [FUNGRAPH 项目页面](https://fungraph.inria.fr) |<br>
| [T&T+DB COLMAP (650MB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) | [预训练模型 (14 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip) | [Windows 查看器 (60MB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip) | [评估图像 (7 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip) |<br>

本仓库包含论文“3D Gaussian Splatting for Real-Time Radiance Field Rendering”的官方作者实现，论文可在[此处](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)找到。我们还提供了用于生成论文中误差指标的参考图像，以及最近创建的预训练模型。

摘要：*辐射场方法最近在由多张照片或视频捕捉的场景的新视角合成方面取得了革命性进展。然而，要实现高视觉质量仍然需要训练和渲染成本高昂的神经网络，而最近的快速方法不可避免地需要在速度和质量之间进行权衡。对于无边界且完整的场景（而非孤立物体）以及1080p分辨率的渲染，目前没有任何方法能够实现实时显示速率。我们引入了三个关键要素，使得我们在保持有竞争力的训练时间的同时，能够实现最先进的视觉质量，并且重要的是，在1080p分辨率下实现高质量实时（≥30 fps）的新视角合成。首先，从相机标定过程中产生的稀疏点开始，我们使用3D高斯函数来表示场景，这些高斯函数保留了连续体积辐射场在场景优化方面的理想特性，同时避免了空区域中不必要的计算；其次，我们执行3D高斯函数的交错优化/密度控制，特别是优化各向异性协方差以实现场景的精确表示；第三，我们开发了一种快速的可见性感知渲染算法，该算法支持各向异性抛雪球，既加速了训练又实现了实时渲染。我们在几个已有数据集上展示了最先进的视觉质量和实时渲染效果。*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>

## 新特性！

我们维护和更新代码的资源有限。不过，自原始发布以来，我们添加了一些新特性，这些特性受到其他研究人员在 3DGS 上所做的许多优秀工作的启发。我们将在资源允许的范围内添加其他特性。

**2024年10月更新**：我们集成了[训练速度加速](#训练速度加速)，并使其兼容[深度正则化](#深度正则化)、[抗锯齿](#抗锯齿)和[曝光补偿](#曝光补偿)。我们通过修复错误和增加[顶视图](#sibr-顶视图)中的功能增强了 SIBR 实时查看器，顶视图允许可视化输入相机和用户相机。

**2024年春季更新**：
Orange Labs 慷慨地增加了用于 VR 观看的 [OpenXR 支持](#openxr-支持)。

## 分步教程

Jonathan Stephens 制作了一个很棒的分步教程，用于在您的机器上设置 Gaussian Splatting，以及从视频创建可用数据集的说明。如果下面的说明对您来说太枯燥，请点击[此处](https://www.youtube.com/watch?v=UXtuigy_wYc)查看。

## Colab

用户 [camenduru](https://github.com/camenduru) 慷慨地提供了一个 Colab 模板，该模板使用本仓库的源代码（状态：2023年8月！），以便快速简便地使用该方法。请点击[此处](https://github.com/camenduru/gaussian-splatting-colab)查看。

## 克隆仓库

仓库包含子模块，因此请使用以下命令检出：

```shell
# HTTPS
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

## 概述

代码库有 4 个主要组件：
- 基于 PyTorch 的优化器，用于从 SfM 输入生成 3D 高斯模型
- 网络查看器，允许连接并可视化优化过程
- 基于 OpenGL 的实时查看器，用于实时渲染训练好的模型
- 一个脚本，帮助您将自己的图像转换为可用于优化的 SfM 数据集

这些组件在硬件和软件方面有不同的要求。它们已在 Windows 10 和 Ubuntu Linux 22.04 上测试。设置和运行每个组件的说明见以下章节。

## 优化器

优化器在 Python 环境中使用 PyTorch 和 CUDA 扩展来生成训练好的模型。

### 硬件要求

- 支持 CUDA 且计算能力为 7.0+ 的 GPU
- 24 GB 显存（以达到论文评估质量）
- 对于较小的显存配置，请参阅常见问题解答

### 软件要求
- Conda（推荐用于简化设置）
- 用于 PyTorch 扩展的 C++ 编译器（Windows 上我们使用 Visual Studio 2019）
- 用于 PyTorch 扩展的 CUDA SDK 11，*在 Visual Studio 之后*安装（我们使用 11.8，**11.6 存在已知问题**）
- C++ 编译器和 CUDA SDK 必须兼容

### 设置

#### 本地设置

我们默认提供的安装方法基于 Conda 包和环境管理：
```shell
SET DISTUTILS_USE_SDK=1 # 仅限 Windows
conda env create --file environment.yml
conda activate gaussian_splatting
```
请注意，此过程假设您已安装 CUDA SDK **11**，而不是 **12**。有关修改，请参见下文。

提示：使用 Conda 下载包并创建新环境可能需要大量磁盘空间。默认情况下，Conda 将使用系统主硬盘。您可以通过指定不同的包下载位置和不同驱动器上的环境来避免这种情况：

```shell
conda config --add pkgs_dirs <Drive>/<pkg_path>
conda env create --file environment.yml --prefix <Drive>/<env_path>/gaussian_splatting
conda activate <Drive>/<env_path>/gaussian_splatting
```

### 运行

要运行优化器，只需使用：
```shell
python train.py -s <COLMAP 或 NeRF 合成数据集的路径>
```

<details>
<summary><span style="font-weight: bold;">train.py 的命令行参数</span></summary>

  #### --source_path / -s
  包含 COLMAP 或合成 NeRF 数据集的源目录路径。
  #### --model_path / -m 
  训练模型应存储的路径（默认为 ```output/<随机>```）。
  #### --images / -i
  COLMAP 图像的替代子目录（默认为 ```images```）。
  #### --eval
  添加此标志以使用 MipNeRF360 风格的训练/测试拆分进行评估。
  #### --resolution / -r
  指定训练前加载图像的分辨率。如果提供 ```1, 2, 4``` 或 ```8```，则分别使用原始分辨率、1/2、1/4 或 1/8 分辨率。对于所有其他值，将宽度缩放到给定数字，同时保持图像宽高比。**如果未设置且输入图像宽度超过 1.6K 像素，输入会自动缩放到此目标。**
  #### --data_device
  指定将源图像数据放在何处，默认为 ```cuda```，如果在大规模/高分辨率数据集上训练，建议使用 ```cpu```，这将减少显存消耗，但会略微减慢训练速度。感谢 [HrsPythonix](https://github.com/HrsPythonix)。
  #### --white_background / -w
  添加此标志以使用白色背景而不是黑色（默认），例如，用于 NeRF 合成数据集的评估。
  #### --sh_degree
  要使用的球谐函数阶数（不大于 3）。默认为 ```3```。
  #### --convert_SHs_python
  标志，使管道使用 PyTorch 而非我们自己的代码计算 SH 的前向和反向。
  #### --convert_cov3D_python
  标志，使管道使用 PyTorch 而非我们自己的代码计算 3D 协方差的前向和反向。
  #### --debug
  如果遇到错误，启用调试模式。如果光栅化器失败，会创建一个 ```dump``` 文件，您可以在 issue 中转发给我们，以便我们查看。
  #### --debug_from
  调试很**慢**。您可以指定一个迭代次数（从 0 开始），在此之后上述调试才会激活。
  #### --iterations
  训练的总迭代次数，默认为 ```30_000```。
  #### --ip
  启动 GUI 服务器的 IP，默认为 ```127.0.0.1```。
  #### --port 
  GUI 服务器使用的端口，默认为 ```6009```。
  #### --test_iterations
  空格分隔的迭代次数，训练脚本在这些迭代时计算测试集上的 L1 和 PSNR，默认为 ```7000 30000```。
  #### --save_iterations
  空格分隔的迭代次数，训练脚本在这些迭代时保存高斯模型，默认为 ```7000 30000 <iterations>```。
  #### --checkpoint_iterations
  空格分隔的迭代次数，用于存储检查点以便稍后继续训练，保存在模型目录中。
  #### --start_checkpoint
  保存的检查点路径，用于继续训练。
  #### --quiet 
  标志，忽略写入标准输出管道的任何文本。
  #### --feature_lr
  球谐函数特征学习率，默认为 ```0.0025```。
  #### --opacity_lr
  不透明度学习率，默认为 ```0.05```。
  #### --scaling_lr
  缩放学习率，默认为 ```0.005```。
  #### --rotation_lr
  旋转学习率，默认为 ```0.001```。
  #### --position_lr_max_steps
  位置学习率从 ```initial``` 到 ```final``` 的步数（从 0 开始）。默认为 ```30_000```。
  #### --position_lr_init
  初始 3D 位置学习率，默认为 ```0.00016```。
  #### --position_lr_final
  最终 3D 位置学习率，默认为 ```0.0000016```。
  #### --position_lr_delay_mult
  位置学习率乘数（参见 Plenoxels），默认为 ```0.01```。
  #### --densify_from_iter
  开始密化（densification）的迭代次数，默认为 ```500```。
  #### --densify_until_iter
  停止密化的迭代次数，默认为 ```15_000```。
  #### --densify_grad_threshold
  决定是否应基于 2D 位置梯度对点进行密化的阈值，默认为 ```0.0002```。
  #### --densification_interval
  密化频率，默认为 ```100```（每 100 次迭代）。
  #### --opacity_reset_interval
  重置不透明度的频率，默认为 ```3_000```。
  #### --lambda_dssim
  SSIM 对总损失的影响（0 到 1），默认为 ```0.2```。
  #### --percent_dense
  场景范围（0-1）的百分比，点必须超过该值才能被强制密化，默认为 ```0.01```。

</details>
<br>

请注意，与 MipNeRF360 类似，我们的目标图像分辨率在 1-1.6K 像素范围内。为方便起见，可以传入任意大小的输入，如果其宽度超过 1600 像素，将自动调整大小。我们建议保留此行为，但您可以通过设置 ```-r 1``` 强制训练使用更高分辨率的图像。

MipNeRF360 场景由论文作者托管在[此处](https://jonbarron.info/mipnerf360/)。您可以在此处找到我们用于 Tanks&Temples 和 Deep Blending 的 SfM 数据集[此处](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip)。如果您不提供输出模型目录（```-m```），训练好的模型将写入 ```output``` 目录内具有随机唯一名称的文件夹。此时，可以使用实时查看器查看训练好的模型（参见下文）。

### 评估
默认情况下，训练好的模型使用数据集中的所有可用图像。要使用保留的测试集进行训练和评估，请使用 ```--eval``` 标志。这样，您可以渲染训练/测试集并产生误差指标，如下所示：
```shell
python train.py -s <COLMAP 或 NeRF 合成数据集的路径> --eval # 使用训练/测试拆分进行训练
python render.py -m <训练模型的路径> # 生成渲染图像
python metrics.py -m <训练模型的路径> # 计算渲染图像的误差指标
```

如果您想评估我们的[预训练模型](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip)，您需要下载相应的源数据集，并使用额外的 ```--source_path/-s``` 标志向 ```render.py``` 指示其位置。注意：预训练模型是使用发布时的代码库创建的。此代码库已经过清理并包含错误修复，因此评估它们得到的指标将与论文中的不同。
```shell
python render.py -m <预训练模型的路径> -s <COLMAP 数据集的路径>
python metrics.py -m <预训练模型的路径>
```

<details>
<summary><span style="font-weight: bold;">render.py 的命令行参数</span></summary>

  #### --model_path / -m 
  要为其创建渲染图像的训练模型目录的路径。
  #### --skip_train
  标志，跳过渲染训练集。
  #### --skip_test
  标志，跳过渲染测试集。
  #### --quiet 
  标志，忽略写入标准输出管道的任何文本。

  **以下参数将根据训练时使用的参数自动从模型路径中读取。但是，您可以通过在命令行上显式提供它们来覆盖。**

  #### --source_path / -s
  包含 COLMAP 或合成 NeRF 数据集的源目录路径。
  #### --images / -i
  COLMAP 图像的替代子目录（默认为 ```images```）。
  #### --eval
  添加此标志以使用 MipNeRF360 风格的训练/测试拆分进行评估。
  #### --resolution / -r
  更改训练前加载图像的分辨率。如果提供 ```1, 2, 4``` 或 ```8```，则分别使用原始分辨率、1/2、1/4 或 1/8 分辨率。对于所有其他值，将宽度缩放到给定数字，同时保持图像宽高比。默认为 ```1```。
  #### --white_background / -w
  添加此标志以使用白色背景而不是黑色（默认），例如，用于 NeRF 合成数据集的评估。
  #### --convert_SHs_python
  标志，使管道使用 PyTorch 而非我们自己的代码计算 SH 进行渲染。
  #### --convert_cov3D_python
  标志，使管道使用 PyTorch 而非我们自己的代码计算 3D 协方差进行渲染。

</details>

<details>
<summary><span style="font-weight: bold;">metrics.py 的命令行参数</span></summary>

  #### --model_paths / -m 
  空格分隔的模型路径列表，用于计算这些模型的指标。
</details>
<br>

我们还提供了 ```full_eval.py``` 脚本。该脚本指定了我们评估中使用的例程，并演示了一些附加参数的使用，例如 ```--images (-i)``` 用于在 COLMAP 数据集中定义替代图像目录。如果您已下载并解压所有训练数据，可以像这样运行它：
```shell
python full_eval.py -m360 <mipnerf360 文件夹> -tat <tanks and temples 文件夹> -db <deep blending 文件夹>
```
在当前版本中，此过程在我们包含 A6000 的参考机器上大约需要 7 小时。如果您想对我们的预训练模型进行全面评估，可以指定其下载位置并跳过训练。
```shell
python full_eval.py -o <预训练模型目录> --skip_training -m360 <mipnerf360 文件夹> -tat <tanks and temples 文件夹> -db <deep blending 文件夹>
```

如果您想计算我们论文中[评估图像](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip)的指标，也可以跳过渲染。在这种情况下，无需提供源数据集。您可以同时计算多个图像集的指标。
```shell
python full_eval.py -m <评估图像目录>/garden ... --skip_training --skip_rendering
```

<details>
<summary><span style="font-weight: bold;">full_eval.py 的命令行参数</span></summary>
  
  #### --skip_training
  标志，跳过训练阶段。
  #### --skip_rendering
  标志，跳过渲染阶段。
  #### --skip_metrics
  标志，跳过指标计算阶段。
  #### --output_path
  放置渲染图像和结果的目录，默认为 ```./eval```，如果评估预训练模型，则设置为预训练模型位置。
  #### --mipnerf360 / -m360
  MipNeRF360 源数据集的路径，如果训练或渲染则需要。
  #### --tanksandtemples / -tat
  Tanks&Temples 源数据集的路径，如果训练或渲染则需要。
  #### --deepblending / -db
  Deep Blending 源数据集的路径，如果训练或渲染则需要。
</details>
<br>

## 交互式查看器
我们为我们的方法提供了两个交互式查看器：远程查看器和实时查看器。我们的查看解决方案基于 GRAPHDECO 小组为多个新视角合成项目开发的 [SIBR](https://sibr.gitlabpages.inria.fr/) 框架。

### 硬件要求
- 支持 OpenGL 4.5 的 GPU 和驱动程序（或最新的 MESA 软件）
- 建议 4 GB 显存
- 支持 CUDA 且计算能力为 7.0+ 的 GPU（仅限实时查看器）

### 软件要求
- Visual Studio 或 g++，**不是 Clang**（Windows 上我们使用 Visual Studio 2019）
- CUDA SDK 11，*在 Visual Studio 之后*安装（我们使用 11.8）
- CMake（较新版本，我们使用 3.24）
- 7zip（仅限 Windows）

### 预构建的 Windows 二进制文件
我们在此处提供预构建的 Windows 二进制文件[此处](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/binaries/viewers.zip)。我们建议在 Windows 上使用它们以实现高效设置，因为构建 SIBR 涉及多个外部依赖项，这些依赖项必须即时下载和编译。

### 在 SIBR 查看器中导航
SIBR 界面提供了几种在场景中导航的方法。默认情况下，您将启动 FPS 导航器，您可以使用 ```W, A, S, D, Q, E``` 进行相机平移，使用 ```I, K, J, L, U, O``` 进行旋转。或者，您可能希望使用轨迹球风格的导航器（从浮动菜单中选择）。您还可以使用 ```Snap to``` 按钮捕捉到数据集中的相机，或使用 ```Snap to closest``` 找到最近的相机。浮动菜单还允许您更改导航速度。您可以使用 ```Scaling Modifier``` 来控制显示的高斯函数的大小，或显示初始点云。

### 运行网络查看器

解压或安装查看器后，您可以在 ```<SIBR install dir>/bin``` 中运行编译好的 ```SIBR_remoteGaussian_app[_config]``` 应用程序，例如：
```shell
./<SIBR install dir>/bin/SIBR_remoteGaussian_app
```
网络查看器允许您连接到正在运行的训练进程（在同一台或不同的机器上）。如果您在同一台机器和操作系统上训练，则不需要命令行参数：优化器会将训练数据的位置通信给网络查看器。默认情况下，优化器和网络查看器将尝试在 **localhost** 的端口 **6009** 上建立连接。您可以通过向优化器和网络查看器都提供匹配的 ```--ip``` 和 ```--port``` 参数来更改此行为。如果由于某种原因，优化器用于查找训练数据的路径对网络查看器不可达（例如，因为它们在不同的（虚拟）机器上运行），您可以通过使用 ```-s <source path>``` 向查看器指定覆盖位置。

<details>
<summary><span style="font-weight: bold;">网络查看器的主要命令行参数</span></summary>

  #### --path / -s
  覆盖模型源数据集路径的参数。
  #### --ip
  用于连接到正在运行的训练脚本的 IP。
  #### --port
  用于连接到正在运行的训练脚本的端口。
  #### --rendering-size 
  接受两个空格分隔的数字，定义网络渲染的分辨率，默认宽度为 ```1200```。
  请注意，要强制使用与输入图像不同的宽高比，您还需要 ```--force-aspect-ratio```。
  #### --load_images
  标志，加载源数据集图像以在每个相机的顶视图中显示。
</details>
<br>

### 运行实时查看器

解压或安装查看器后，您可以在 ```<SIBR install dir>/bin``` 中运行编译好的 ```SIBR_gaussianViewer_app[_config]``` 应用程序，例如：
```shell
./<SIBR install dir>/bin/SIBR_gaussianViewer_app -m <训练模型的路径>
```

只需提供指向训练模型目录的 ```-m``` 参数即可。或者，您可以使用 ```-s``` 指定训练输入数据的覆盖位置。要使用自动选择之外的分辨率，请指定 ```--rendering-size <width> <height>```。如果您想要精确的分辨率并且不介意图像失真，可以结合使用 ```--force-aspect-ratio```。

**要解锁全帧率，请禁用您机器上的垂直同步（V-Sync）以及应用程序中的垂直同步（菜单 → Display）。在多 GPU 系统（例如笔记本电脑）中，您的 OpenGL/显示 GPU 应与您的 CUDA GPU 相同（例如，通过在 Windows 上设置应用程序的 GPU 偏好，见下文）以获得最佳性能。**

除了初始点云和抛雪球点，您还可以从浮动菜单中选择将高斯函数渲染为椭球体来可视化它们。
SIBR 还有许多其他功能，有关查看器、导航选项等的更多详细信息，请参阅[文档](https://sibr.gitlabpages.inria.fr/)。还有一个顶视图（可从菜单中获得），显示输入相机和原始 SfM 点云的位置；请注意，启用顶视图会降低渲染速度。实时查看器还使用了更激进的快速剔除，可以在浮动菜单中切换。如果您遇到可以通过关闭快速剔除来解决的问题，请告知我们。

<details>
<summary><span style="font-weight: bold;">实时查看器的主要命令行参数</span></summary>

  #### --model-path / -m
  训练模型的路径。
  #### --iteration
  如果有多个可用状态，指定加载哪个状态。默认为最新的可用迭代。
  #### --path / -s
  覆盖模型源数据集路径的参数。
  #### --rendering-size 
  接受两个空格分隔的数字，定义实时渲染的分辨率，默认宽度为 ```1200```。请注意，要强制使用与输入图像不同的宽高比，您还需要 ```--force-aspect-ratio```。
  #### --load_images
  标志，加载源数据集图像以在每个相机的顶视图中显示。
  #### --device
  如果有多个 CUDA 设备，指定用于光栅化的设备索引，默认为 ```0```。
  #### --no_interop
  强制禁用 CUDA/GL 互操作。在可能不符合规范的系统上使用（例如，使用 MESA GL 4.5 软件渲染的 WSL2）。
</details>
<br>

## 处理您自己的场景

我们的 COLMAP 加载器期望源路径位置具有以下数据集结构：

```
<location>
|---images
|   |---<image 0>
|   |---<image 1>
|   |---...
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

对于光栅化，相机模型必须是 SIMPLE_PINHOLE 或 PINHOLE。我们提供了一个转换脚本 ```convert.py```，用于从输入图像中提取 undistorted 图像和 SfM 信息。或者，您可以使用 ImageMagick 调整 undistorted 图像的大小。这种缩放类似于 MipNeRF360，即在相应文件夹中创建分辨率为原始分辨率 1/2、1/4 和 1/8 的图像。要使用它们，请首先安装最新版本的 COLMAP（最好是 CUDA 加速版）和 ImageMagick。将您想要使用的图像放在目录 ```<location>/input``` 中。
```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
如果 COLMAP 和 ImageMagick 在您的系统路径中，您只需运行：
```shell
python convert.py -s <location> [--resize] # 如果不调整大小，则不需要 ImageMagick
```
或者，您可以使用可选参数 ```--colmap_executable``` 和 ```--magick_executable``` 指向相应的路径。请注意，在 Windows 上，可执行文件应指向负责设置执行环境的 COLMAP ```.bat``` 文件。完成后，```<location>``` 将包含预期的 COLMAP 数据集结构，其中包含 undistorted、调整大小后的输入图像，以及原始图像和目录 ```distorted``` 中的一些临时（distorted）数据。

如果您有自己的未经过 undistortion 的 COLMAP 数据集（例如，使用 ```OPENCV``` 相机），您可以尝试仅运行脚本的最后部分：将图像放在 ```input``` 中，将 COLMAP 信息放在子目录 ```distorted``` 中：
```
<location>
|---input
|   |---<image 0>
|   |---<image 1>
|   |---...
|---distorted
    |---database.db
    |---sparse
        |---0
            |---...
```
然后运行：
```shell
python convert.py -s <location> --skip_matching [--resize] # 如果不调整大小，则不需要 ImageMagick
```

<details>
<summary><span style="font-weight: bold;">convert.py 的命令行参数</span></summary>

  #### --no_gpu
  标志，避免在 COLMAP 中使用 GPU。
  #### --skip_matching
  标志，指示图像已有 COLMAP 信息。
  #### --source_path / -s
  输入的位置。
  #### --camera 
  用于早期匹配步骤的相机模型，默认为 ```OPENCV```。
  #### --resize
  标志，创建输入图像的调整大小版本。
  #### --colmap_executable
  COLMAP 可执行文件的路径（Windows 上为 ```.bat```）。
  #### --magick_executable
  ImageMagick 可执行文件的路径。
</details>
<br>

### 训练速度加速

我们集成了来自 [Taming-3dgs](https://humansensinglab.github.io/taming-3dgs/)<sup>1</sup> 的即插即用替换，以及 [fused ssim](https://github.com/rahul-goel/fused-ssim/tree/main)，到原始代码库中以加速训练时间。安装后，加速的光栅化器使用 `--optimizer_type default` 可带来 **×1.6 的训练时间加速**，使用 `--optimizer_type sparse_adam` 可带来 **×2.7 的训练时间加速**。

要获得更快的训练时间，您必须首先将加速的光栅化器安装到您的环境中：

```bash
pip uninstall diff-gaussian-rasterization -y
cd submodules/diff-gaussian-rasterization
rm -r build
git checkout 3dgs_accel
pip install .
```

然后，在运行 `train.py` 时可以添加以下参数以使用稀疏 Adam 优化器：

```bash
--optimizer_type sparse_adam
```

*请注意，此自定义光栅化器的行为与原始版本不同，有关训练时间的更多详细信息，请参阅[训练时间比较统计](results.md/#training-times-comparisons)。*

*1. Mallick and Goel, et al. ‘Taming 3DGS: High-Quality Radiance Fields with Limited Resources’. SIGGRAPH Asia 2024 Conference Papers, 2024, https://doi.org/10.1145/3680528.3687694, [github](https://github.com/humansensinglab/taming-3dgs)*

### 深度正则化

为了获得更好的重建场景，我们在优化过程中使用深度图作为每个输入图像的先验信息。它在无纹理区域（例如道路）上效果最好，并且可以去除漂浮物。有几篇论文使用类似的想法来改进 3DGS 的各个方面（例如 [DepthRegularizedGS](https://robot0321.github.io/DepthRegGS/index.html)、[SparseGS](https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/)、[DNGaussian](https://fictionarry.github.io/DNGaussian/)）。我们集成的深度正则化是我们的 [Hierarchical 3DGS](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/) 论文中使用的，但应用于原始 3DGS；对于某些场景（例如 DeepBlending 场景），它显著提高了质量；对于其他场景，它要么影响很小，要么甚至可能更差。有关显示潜在收益和质量统计的示例结果，请参见[此处](results.md)。

在合成数据集上训练时，可以生成深度图，并且它们无需进一步处理即可用于我们的方法。

对于真实世界数据集，应为每个输入图像生成深度图，请按以下步骤生成：
1. 克隆 [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#usage)：
    ```
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    ```
2. 从 [Depth-Anything-V2-Large](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) 下载权重并将其放在 `Depth-Anything-V2/checkpoints/` 下。
3. 生成深度图：
   ```
   python Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path <输入图像的路径> --outdir <输出路径>
   ```
5. 使用以下命令生成 `depth_params.json` 文件：
    ```
    python utils/make_depth_scale.py --base_dir <colmap 路径> --depths_dir <生成的深度图路径>
    ```

如果您想在训练时使用深度正则化，应设置一个新参数 `-d <深度图路径>`。

### 曝光补偿

为了补偿不同输入图像中的曝光变化，我们像 [Hierarchical 3dgs](https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/) 一样为每个图像优化了一个仿射变换。

这可以极大地改善“野外”拍摄的重建结果，例如，当使用相机曝光设置未固定的智能手机拍摄时。有关显示潜在收益和质量统计的示例结果，请参见[此处](results.md)。

添加以下参数以启用它：
```
--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp
```
同样，其他优秀论文也使用了类似的想法，例如 [NeRF-W](https://nerf-w.github.io/)、[URF](https://urban-radiance-fields.github.io/)。

### 抗锯齿

我们添加了来自 [Mip Splatting](https://niujinshuchong.github.io/mip-splatting/) 的 EWA 滤波器到我们的代码库中以消除锯齿。它默认是禁用的，但您可以通过在训练场景时向 `train.py` 添加 `--antialiasing` 或在渲染时向 `render.py` 添加 `--antialiasing` 来启用它。可以在 SIBR 查看器中切换抗锯齿，默认是禁用的，但在查看使用 `--antialiasing` 训练的场景时应该启用它。
*此场景是使用 `--antialiasing` 训练的*。

### SIBR：顶视图
> `Views > Top view`

`顶视图`在另一个视图中渲染 SfM 点云，并显示相应的输入相机和`点视图`中的用户相机。这允许可视化查看器距离输入相机有多远等。

这是一个 3D 视图，因此用户可以像在`点视图`中一样在其中导航（可用模式：FPS、轨迹球、轨道）。

选项可用于自定义此视图，可以启用/禁用网格，并可以修改它们的缩放比例。
一个有用的附加功能是移动到输入图像的位置，并逐渐淡出到该位置的 SfM 点视图（例如，以验证相机对齐）。可以在`顶视图`中显示来自输入相机的视图（*请注意，必须在命令行中设置 `--images-path`*）。可以通过单击 `Top view settings > Cameras > Snap to closest` 将`顶视图`相机捕捉到距离`点视图`中用户相机最近的输入相机。

### OpenXR 支持

OpenXR 在分支 `gaussian_code_release_openxr` 中得到支持。
在该分支中，您可以找到 VR 支持的文档[此处](https://gitlab.inria.fr/sibr/sibr_core/-/tree/gaussian_code_release_openxr?ref_type=heads)。

## 常见问题解答
- *如何将其用于更大的数据集* 这通常发生在例如驾驶数据集中（汽车近景，建筑物远景）。对于此类场景，您可以降低 ```--position_lr_init```、```--position_lr_final``` 和 ```--scaling_lr```（x0.3、x0.1...）。场景越广泛，这些值应该越低。下面，我们使用默认学习率（左）和 ```--position_lr_init 0.000016 --scaling_lr 0.001"```（右）。

- *我没有 24 GB 的显存用于训练，该怎么办？* 显存消耗由正在优化的点数量决定，这些点会随着时间的推移而增加。如果您只想训练到 7k 次迭代，您将需要显着更少的显存。要执行完整的训练程序并避免内存不足，您可以增加 ```--densify_grad_threshold```、```--densification_interval``` 或减少 ```--densify_until_iter``` 的值。但请注意，这将影响结果的质量。也可以尝试将 ```--test_iterations``` 设置为 ```-1``` 以避免测试期间的内存峰值。如果 ```--densify_grad_threshold``` 非常高，则不应发生密化，并且如果场景本身成功加载，训练应该能够完成。

- *等等，但是 ```<插入功能>``` 没有优化，可以做得更好？* 有几个部分我们甚至没有时间考虑改进（到目前为止）。您使用此原型获得的性能可能是一个相当慢的基线。
