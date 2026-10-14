# GAUGE 论文版本复现（供作者核对）

这是 XiePhys 团队依据公开论文和数据集实现的独立复现，不是官方评分代码。目的是让作者直接检查我们的版本、场景构造和数据处理，定位尚未复现的结果。原始上游说明保留在仓库首页；可选的 XiePhysic SDK 实验在上一层目录，与本次三方复现分开。

## 本次实际结果

| 引擎 | 版本 | 已尝试实例 | 完整过程完成 | 失败或待确认 |
| --- | --- | ---: | ---: | --- |
| Isaac Sim | 6.0.0，完整构建号见下文 | 55 | 55 | 无数值失败；成绩尚未全部对齐 |
| Newton | 1.3.0 / Warp 1.14.0 | 55 | 37 | 18 个布料实例出现 NaN/Inf |
| Genesis | 论文写 v1.12.0 | 0 | 0 | 实际版本或 commit 待作者确认，没有替换成其他版本 |

论文的 14 种机制展开为 **55 个材料/子任务实例：29 刚体、18 布料、8 体积软体**。第一列是任务机制，实例数不能直接当作论文的 14 条总表记录。当前复现不制作总分或认证排名。

[完整案例与指标差异](DISCREPANCIES.md) · [逐实例运行结果](reports/case_outcomes.csv) · [所有已有效计算的引擎成绩对照](reports/engine_paper_comparison.csv) · [106 项实测基线核对](reports/real_baseline_comparison.csv)

## 版本与输入

- 论文：[2608.05948v1](https://arxiv.org/html/2608.05948v1)，§3.2.1、§4.2、附录 A.2.2 和 B.1。
- 数据：[InternRobotics/GAUGE-Dataset](https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset)，固定提交 `9e0acb70fecc0d4161660264d9a4b08d8f56d45a`。
- Isaac 使用 NVIDIA 的 `isaac-sim-standalone-6.0.0-linux-x86_64.zip`，内部 VERSION 为 `6.0.0-rc.59+release.41464.5f2772bc.gl`。不能用其他 rc、6.0.1 或仅改版本标签代替。下载地址和校验值在 [runtime_lock.json](runtime_lock.json)。
- Newton、Warp、MuJoCo 和科学计算依赖均固定在 [requirements-newton.txt](requirements-newton.txt)。
- 本次主机为 Linux / RTX PRO 6000 Blackwell，论文为 Windows 11 / RTX 4090。版本对齐不代表硬件和运行实现已经等同。
- 运行后重新核验了本次涉及的 1,151 个官方源文件，没有改写源文件。仓库下载清单另含 5 个辅助文件，共 1,156 个；源文件完整不等于运行时零转换。

SDK、引擎源码和原始数据不放进仓库。数据由固定提交下载，或使用已有的官方数据目录。所附结果包仅含生成的仿真标记轨迹、运行回执、标记匹配记录和日志。

## 从干净目录复现

准备与评分使用 Python 3.11+，SDK 使用各自独立环境。下面所有命令在仓库 `evaluation/` 目录执行。`GAUGE_WORKSPACE` 只指定可写输入/输出目录，不需要引擎源码。

```bash
cd evaluation
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GAUGE_WORKSPACE="$PWD/_work"
python scripts/fetch_dataset.py
python scripts/prepare.py --mode native
python scripts/reproduce.py audit
```

已有数据也可以将 `$GAUGE_WORKSPACE/data/GAUGE-Dataset` 链接到其根目录。下载和准备均检查冻结哈希；不要把新版数据静默混进旧配置。

Newton 环境独立安装：

```bash
python3.12 -m venv /path/to/newton-env
/path/to/newton-env/bin/python -m pip install -r reproduction/requirements-newton.txt
python scripts/reproduce.py run --engine newton --python /path/to/newton-env/bin/python
```

Isaac 安装上述固定发行包后，用包内 `python.sh` 执行。该环境还需提供 NumPy、SciPy；不要用准备环境中的 `usd-core` 覆盖 Isaac 自带的 USD/PhysX schema。

```bash
python scripts/reproduce.py run --engine isaac \
  --python /path/to/isaac-6.0.0/python.sh \
  --isaac-root /path/to/isaac-6.0.0
python scripts/reproduce.py verify
python scripts/reproduce.py score
```

`run` 默认逐案例串行运行并等待现有 GPU 任务退出，每例上限 7,200 秒。已记录的终止尝试不会自动重跑；`--retry` 保留旧尝试后重新运行。`--smoke` 只记录前三帧到单独目录，绝不参与正式评分。使用 `--case` 可选择单例，例如：

```bash
python scripts/reproduce.py run --engine newton --python /path/to/newton-env/bin/python \
  --case turntable_metal_task3_ipc
python scripts/reproduce.py run --engine newton --python /path/to/newton-env/bin/python \
  --case textile_bending_nylon_taslan_ipc
```

错误、NaN/Inf、超时保留原始日志，不补帧、补零或改参数强行出分。运行器启动后、实际求解前检查 SDK 版本；Newton 依赖版本也必须匹配。Genesis 命令仅产生版本待确认记录，不启动替代引擎。

## 无需重跑 GPU，核验我们的已保存成绩

[下载本次生成的结果包](https://github.com/fengbusan/GAUGE/releases/download/reproduction-20261014/gauge-paper-version-recordings-20261014.zip)。SHA-256 见 [execution_summary.json](reports/execution_summary.json)。先完成上面的数据下载、输入准备和 `audit`，再执行：

```bash
python scripts/reproduce.py import-results --archive /path/to/gauge-paper-version-recordings-20261014.zip
python scripts/reproduce.py verify
python scripts/reproduce.py score
```

该路径从记录的仿真轨迹重新计算成绩，不需要安装 SDK。结果包共 110 份 Isaac/Newton 实例记录；失败记录也包含在内。导入检查轨迹哈希、版本、实际物理输入及网格资产，允许更换工作目录，不允许更换材料、驱动、时段或几何。URDF 仅对显示网格文件路径前缀做规范化，质量、碰撞球、关节参数保持一致。导入不是重新运行引擎。

## 场景与参数映射，供作者检查

| 项目 | 我们当前的实现 |
| --- | --- |
| 时步、记录 | 刚体和布料 180 Hz，体积软体 900 Hz，统一 30 Hz 记录。运行完整公开过程，评分截取公开表格对应的前 N 帧 |
| 单位和方向 | 实测平移 mm→m；评分位置转回 mm。metadata Euler 角转换为原生四元数排列；没有拟合坐标轴旋转或人为缩放 |
| 驱动 | 对公开重复试验平移求均值、逐帧旋转求均值；减去初始均值并注册到 metadata 初始锚点；时步内插值，轨迹结束后保持末姿态。作者实际选择/平均方式尚未确认 |
| 刚体 | 使用官方 OBJ 坐标及多边形扇形三角化。Isaac 动态非盒网格原生 convexHull、静态/驱动网格三角接触；几何完全等价的盒子使用原生 box。Newton 交由原生 MuJoCo 转换；球和摆使用解析球、原生关节。三方复现不使用 Xie 的 2 mm 牛顿摆间距 |
| Isaac 布料/软体 | 直接打开官方 USD，在 session layer 映射 metadata 实测材料、姿态和原生自动夹持。原生 cooking，不简化模型；保留 USD 自带的碰撞字段。布料确认施加 SurfaceDeformableMaterialAPI |
| Newton 布料 | 官方 OBJ 顶点和三角片，不简化、不焊接；转换到 world-space。材料使用附录 A.2.2 式（10）—（12），包括 `Ks/2.7`。`particle_radius` 参数省略，原生默认 0.1 m；`SolverVBD(model)` 保持默认，包括 GPU tile=true、自接触=false、10 次迭代 |
| Newton 体积软体 | 官方 USD simulation tet 顶点，精确坐标焊接与重复单元去除、正向单元。取唯一顶点为 implicit MPM 粒子，按单元体积分配质量，半径为粒子分配体积立方根的一半；默认 voxel_size=0.1 m。并非使用四面体 FEM 求解 |
| 夹持 | 原生 kinematic 装置和零质量边界顶点/粒子，无 PD。MPM 步首规定边界位置、速度和 APIC 刚体旋转梯度，步尾只校正规定边界位置，保存校正前误差；不覆盖自由粒子。作者实际边界离散与调用顺序待核对 |
| 标记 | 用 metadata 初始标记点对原生仿真顶点/粒子做 Hungarian 唯一匹配，最大误差 <1 cm；固定追踪同一组 ID |

“原生默认”指 SDK 的其他数值设置，不是省略论文明确规定的材料转换，也不是宣称作者的运行器必然采用同一个 API 路径。以上粒子采样、碰撞表示和夹持实现均公开，便于作者确认。

## 数据处理与评分

主要路径 `reproduce.py audit/score` 使用论文公式，不按引擎输出拟合参数：

- 刚体使用 XYZ；布料所有任务使用附录角亏；体积软体使用标记三角片面积。曲率为逐标记向量，不换成弯曲链曲率。
- 先计算每次实测的观测量，再求包含该试验的总均值；基线误差相对于这个包含均值计算，标准差为 `ddof=1`。实测轨迹单位/标记映射保留在冻结 recipes。
- RMSE 为 `sqrt(mean_t(sum_features(error**2)))`。DTW 用欧氏局部代价、固定起终点、无约束单调路径，最小化累计代价后除以路径长度；平局优先对角、纵向、横向。
- 保留全部公开 trial；其中 8 个组合的数量为 19–25，论文描述为 20。没有为了匹配删掉试验或补造数据。
- 同时报出原始误差、除以本地重算基线，以及除以论文公布基线的值。后者只用于定位数值差异，不证明观测量或基线已经一致。
- LSD/MTE/PD/EL 的事件选择尚未作者确认，当前正式比较暂不出这些指标的认证分数。完成仿真与完成评分分开记录。

`published_table` 留作单独的实测诊断，不是主要评分。它采用的 leave-one-out、弯曲链曲率等解释不能混入论文公式结果；不会根据哪套更好看来选择引擎成绩。

## 代码入口与校验

- `scripts/reproduce.py`：源文件/记录校验、实测基线、SDK 串行执行、结果包导入、论文公式评分。
- `scripts/thirdparty/`：只有本次实际使用的五个原生适配器及一个共享工具文件。没有旧版 Genesis 替代适配器、GPU 关闭分块消融或工作站部署脚本。
- `recipes/`：55 个可移植配置、官方文件哈希、公开论文成绩。`scripts/prepare.py --mode native` 只准备三方场景。
- `scripts/gauge_eval/`：共同的几何、观测量和误差公式；不依赖引擎源码。
- `reproduction/reports/`：本次运行摘要、差异和校验记录。[代码整理核验](reports/adapter_cleanup_audit.json)确认五个适配器仿真语法树未改变；只整理格式、移除未使用导入并加入求解前版本检查。

```bash
python -m pytest tests -q
```

数据集、SDK 和各引擎遵循其各自许可；本仓库新增工具按原仓库 MIT 许可发布。复现状态和待确认项如实保留，欢迎作者指出不符合原流程的构造或评分细节。
