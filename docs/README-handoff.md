# 交接文档 · Windows → Ubuntu

> 本文档记录 2026-11-09 至 2026-11-11 在 Windows 上的全部决策、硬件、待办、坑。
> Ubuntu 上开新 TRAE 会话时，让它先读 `docs/README-handoff.md` 再继续。
> 仓库：https://github.com/monicaffy108-alt/lehome-garment-folding

## 1. 项目是什么

**LeHome Challenge 2026（ICRA 2026）复现**：
双臂 SO-101 机械臂在 Isaac Lab 仿真中折叠四类衣物（长袖、短袖、长裤、短裤）。

这不是竞赛参赛，是个人复现 + 消融。目标是：跑通官方 baseline → 复刻冠军方案核心组件 → 量化每一步带来的提升。

**为什么选这个题目**：
- 手上已有一条借到的 SO-101 臂（真机可做 sim-to-real）
- 手上的 LinkerBot L20 + ACT 部署经验直接相关
- Kaggle Top 7 / 802 的算法经验可复用
- 适合放进 GitHub profile 的 Projects 栏

## 2. 硬件（Windows 这边）

| 项目 | 型号 | 备注 |
|---|---|---|
| GPU | RTX 5070 (12 GB) | Blackwell，Isaac Sim 能跑，ACT/DP 训练够 |
| CPU | Intel i7-14700F (28 线程) | 仿真 CPU 跑（官方写明的），这是优势 |
| RAM | 32 GB | 够 |
| 系统盘 | 可用 ~924 GB | Isaac Sim 15–20 GB + 数据集几十 GB，够 |
| WSL | Ubuntu **26.04** | ❌ **版本太新，Isaac Sim 5.1 装不上**。不要用 WSL 装 Isaac Sim |
| 原生 Ubuntu | **需要 22.04 或 24.04** | 这是后续工作的平台 |

## 3. 真机（SO-101）状态

- 已借到一条 SO-101 臂（leader 还是 follower 未确认）
- **wrist_roll（第 6 关节）失效**：末端绕轴旋转不动
- 折叠任务里这是最关键的关节之一，但不影响：遥操作链路验证 / 相机管线 / 策略推理速度 / 仿真里的全部工作
- 可以把它锁死，当作 **5-DoF** 机械臂使用，Isaac Lab 里把对应关节设 fixed 即可 → 这本身就是一个有价值的消融实验
- 舵机是 Feetech STS3215（6 个，串在一条串行总线上）
- 故障诊断步骤见本文件末尾 "SO-101 舵机诊断" 章节

## 4. 路线图（5 个阶段，2026-11-11 修订）

**修订原因**：原路线是先训自己的 ACT baseline。改为「**先用 Ilia 公开的权重跑通官方评测，再自己重训**」——
即使自己重训效果差一点，也有「复现了原始结果（~79.63%）」这个保底产出，不会被一次失败的训练卡住整个项目。
**复现目标定为 Ilia 的方案，不做 YifanLiu 的**（Yifan 只公开了主页描述，没有代码/权重，没法验证对错）。

| 阶段 | 内容 | 交付物 | 算力评估 |
|---|---|---|---|
| 0 · 环境 | Ubuntu 22.04 + Isaac Sim 5.1 + Isaac Lab 2.3.1 + uv + Python 3.11 | 能跑通官方 smoke test（空场景 + 加载 SO-101） | 不花 GPU，主要花时间下包 |
| 1 · 跑通 Ilia 权重 | 从 https://huggingface.co/IliaLarchenko/lehome_sim 下载 checkpoint，用他的代码跑官方评测脚本 | **成功率截图/录屏，若复现到 ~79% 就是最强证据** | 推理为主，GPU 够 |
| 2 · 重训 baseline | 从 Ilia 的 baseline config 出发自己重训（ACT/DP 起步） | 自己的 baseline 成功率，和 Ilia 报告对照 | GPU 够，1 天内出数字 |
| 3 · 自价值预测头 | 同一网络加成功概率 + 进度预测 | 消融表第一行：加头的成功率变化 | 显存略增，还是够 |
| 4 · 失败检测 + 候选选择 | 用自价值预测触发切换候选动作 | 消融表第二行 | 推理成本增，CPU rollout 变多 |
| 5 · 数据飞轮（有剩余时间才做） | 小规模 rollout 回灌（~10² 条，RFT 思路参考 Yifan） | 成功率 vs 新增轨迹数曲线 | 仿真 CPU bound，慢但能扛 |
| 6 · 写作 | 失败分析 + 消融 + 结论 | 仓库 README 的 Results 表填满 | 无 |

**注意**：官方 README 写明「**仿真当前只支持 CPU 设备**」——GPU 只用于训练，rollout 是 CPU 跑的。这点直接影响时间预期。

## 5. 关键技术栈版本

| 层 | 版本 | 备注 |
|---|---|---|
| 仿真 | Isaac Sim 5.1.0 | Ubuntu 22.04 主支持，WSL 实验性 |
| 机器人学习框架 | Isaac Lab 2.3.1 | 要指定 Isaac Sim 路径 |
| 策略库 | LeRobot 0.4.2 | 含 ACT / DP / SmolVLA 三个 config |
| Python | 3.11 | Isaac Sim 要求 |
| 包管理 | uv | 官方推荐，快 |
| 数据 | `lehome/asset_challenge` + `lehome/dataset_challenge_merged`（HuggingFace, parquet） | 每类衣服 25 条 demo（10 已见 + 10 未见）|
| 许可 | Apache 2.0 | |

## 6. 参考材料

| 资源 | URL |
|---|---|
| 本项目仓库 | https://github.com/monicaffy108-alt/lehome-garment-folding |
| 官方 benchmark 仓库 | https://github.com/lehome-official/lehome-challenge |
| 挑战赛官网 | https://lehome-challenge.com/ |
| **Ilia Larchenko 方案页（线上仿真赛 #1 of 62 / 真机 #2，单人队）** | https://ilialarchenko.com/projects/lehome2026/ |
| **Ilia 技术报告（arXiv 2606.27163，消融表 + sim2real 工具链在这）** | https://arxiv.org/abs/2606.27163 |
| **Ilia 代码仓库（lehome_solution，架构/训练命令/checkpoint 在这）** | https://github.com/IliaLarchenko/lehome_solution |
| **Ilia 模型权重（HuggingFace）** | https://huggingface.co/IliaLarchenko/lehome_sim |
| Ilia YouTube（折叠过程/失败案例/真机部署完整视频） | https://www.youtube.com/@ilialarchenko |
| Isaac Lab 仿真环境论文 | https://arxiv.org/abs/2604.22363 |
| YifanLiu 主页（Ebtech-MIAA 17th worldwide，初赛 #2） | https://yifanliu-ai.github.io/ |
| YifanLiu 的 workspace（**只是官方仓库副本**，不是他的方案代码） | https://github.com/YifanLiu-AI/lehome-challenge |
| 真机冠军 sZs 队写-up（港中深，技术细节未公开，先别啃） | https://sds.cuhk.edu.cn/article/2439 |

## 6.1 方案对比：Ilia vs YifanLiu（2026-11-11 确认）

**核心共通点**：两边都认识到「demo 太少，rollout 数据才是扩展泛化的关键」——Yifan 走离线 RFT 飞轮，Ilia 走 AWR + 自价值驱动的在线 RL。

| 维度 | Ilia（#1 sim / #2 real） | YifanLiu（17th） |
|---|---|---|
| 策略类型 | **VLA + RL**（自价值预测 + 失败检测 + 候选选择） | **X-VLA + RFT 数据飞轮** |
| 失败处理 | 同一网络实时判断"我在失败"，切换候选动作 | 靠离线 RFT 回灌，不做实时检测 |
| 训练架构 | 异步分布式，HF Hub 协调 rollout 和训练 | 单机，算力受限只做 ~100 条 rollout |
| 推理层 | Thompson sampling 自动调超参 | 无 |
| 真机 | sim2real 一周搞定：DAgger HIL + 相机对齐工具 + 强增强 | 没上真机 |
| 开源 | **代码 + 模型权重 + 报告 + 视频全开源** | 只有主页描述 |

**Ilia 方案核心架构**（我们的复现目标）：

```
同一个 VLA 网络 → 同时输出：动作 / 成功概率 / 任务进度 / 未来相关量
  ↓
  ├── Advantage estimation（算 advantage 做 RL）
  ├── Live failure detection（实时失败检测）
  └── Candidate selection（候选动作选择）
工程优化：AWR + RECAP（flow-matching VLA）、异步分布式训练、
         Thompson sampling 推理调参、sim2real（相机对齐 + 增强 + DAgger）
```

**对本项目的影响**：Stage 3 的「自价值预测 + 失败检测 + 候选选择」就是 Ilia 的核心组件，照 arXiv 2606.27163 的消融表复现。真机阶段直接抄他 sim2real 工具链。

## 7. 为什么不直接上真机

**真机验证的工作量分布**：
标定 > 采数据 > 评测等待 > 训练。花在「让硬件听话」的时间会远超花在模型上的时间。
所以：**先在仿真里把 1–3 步跑通，真机作为第二阶段**。

真机验证需要：
- 原生 Ubuntu 22.04（**不能在 WSL 跑真机**，USB 转串口在 WSL2 里断连是常态）
- SO-101 双臂 × 2（遥操作要 leader + follower）
- 3 路 RGB 相机（头俯视 + 两腕）
- 7.4V 电源（电流要足，不要只靠 USB 供电）
- 桌面、支架、四类衣物各若干件、深色纯色背景、稳定光源
- 评测协议：每类衣服 20 实例 × 至少 20 rollout、固定随机种子、记录失败类型

## 8. 已做的坑清单

| 坑 | 后果 | 怎么避 |
|---|---|---|
| WSL Ubuntu 26.04 太新 | Isaac Sim 5.1 装不上（Python 3.13+、glibc 太新） | 用原生 Ubuntu 22.04 或 24.04 |
| 仿真不是 GPU 跑的 | 训练是 GPU，但 rollout 是 CPU，算力瓶颈不在 GPU 在 CPU 线程数 | 这个 28 线程 CPU 是优势，让仿真并行跑 |
| LeHome 官方仓库在 YifanLiu 的账户里是副本 | 他的 workspace 只存官方 baseline config，没有他自己的训练产物 | 看他的主页（GitHub Pages）而不是那个仓库 |
| SO-101 串行总线 | 中间某个舵机坏了，它后面的全都失联 | 断电重插；怀疑坏的舵机跟已知好的对调位置；用 scservo_sdk 逐个 ping ID |
| 不应该在关节卡死时给扭矩指令 | 电流堵转会烧驱动、烤变形舵机壳 | 先断电再锁死；或者用 `set_torque_enable(False)` |

## 9. 当前仓库状态（2026-11-11）

```
lehome-garment-folding/
├── .gitignore               # 标准 Python + Isaac Sim 忽略项
├── README.md                # 项目定位 + 路线图 + 算力评估 + Results 占位表
└── docs/
    └── README-handoff.md    # 本文档
```

**还没有任何训练产物、没有任何数据、没有 checkpoint**。
Stage 0 还没开始——Isaac Sim 还没装、环境还没搭。

## 10. Ubuntu 那边第一步怎么做

打开 Ubuntu 上的 TRAE，让它先读本文档，然后执行：

```bash
# 确认系统版本
lsb_release -a          # 应该是 22.04 或 24.04

# 确认 GPU 驱动
nvidia-smi              # 应该能看到 RTX 5070

# 确认磁盘
df -h /

# 克隆两个仓库
git clone https://github.com/monicaffy108-alt/lehome-garment-folding.git
git clone https://github.com/lehome-official/lehome-challenge.git

# 装 uv 和 Python 3.11
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11

# 然后去看 Isaac Sim 5.1 官方安装文档
# https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_apt.html
```

**⚠️ 先跑官方 smoke test 再下数据集**（Isaac Sim 装好先开个空场景、加载 SO-101 机械臂确认渲染和物理都正常，再去下几十 GB 的数据）。

## 11. 概念速查（面试时会被问到）

| 术语 | 一句话 |
|---|---|
| **泛化** | 模型在训练没见过的新数据上还能不能干好；25 条 demo 容易记住"这件衣服长这样"换个花色就崩 |
| **RL（强化学习）** | 让模型自己试错，靠「结果好坏」的反馈学策略，而不是靠人给的示范（示范是模仿学习）；Ilia 的 AWR、advantage estimation 都属于 RL |
| **数据飞轮** | 自己训练 → 自己跑 → 筛好数据 → 回灌训练 → 模型变强 → 再跑 的循环 |
| **遥操作补仿真数据** | 人拖着 leader 臂，follower 在仿真里跟随，录下来当 demo；这队试了反而掉点，因为人操作节奏和模型分布不匹配 |
| **RFT（拒绝采样微调）** | 让模型自己跑很多次，失败轨迹扔掉，只拿成功的来监督微调；不需要写 reward 函数，便宜 |
| **rollout** | 模型在环境里完整跑一次任务产生的整段轨迹 |
| **自价值预测** | 同一个网络同时输出动作 + 成功概率 + 任务进度；成功概率低就触发失败检测、切换候选动作；冠军方案的核心组件 |
| **sim-to-real gap** | 仿真里训好的策略上真机就崩，原因是仿真和真机的感知、动力学、摩擦都有差异 |

---

## 附：SO-101 舵机诊断

断电，拔线，用手拧那个关节感受阻力：

| 手感 | 判断 | 处理 |
|---|---|---|
| 完全松，能自由转 | 齿轮或电机断 | 换 STS3215（一百多块），用 Feetech 上位机把 ID 和波特率改成原来一致 |
| 有一点点阻力但转不到目标角 | 齿轮卡住或有异物 | 拆开看 |
| 上电后能顶住力矩，但不动 | 编码器故障或驱动异常 | 换舵机 |
| 完全没反应，连 ping 都不回 | 通信或供电断 | 先查线，再查舵机 |

```bash
# Windows/PowerShell 里扫总线（scservo_sdk）
python -c "
from scservo_sdk import PortHandler, PacketHandler, COMM_SUCCESS
p = PortHandler('COM5'); pk = PacketHandler(1.0)
p.openPort(); p.setBaudRate(1000000)
for i in range(1,7):
    m, r, e = pk.ping(p, i)
    print(i, 'OK' if m == COMM_SUCCESS else 'FAIL', e)
"
```

**串行总线注意**：中间某个舵机坏了，它后面的所有舵机也会失联。所以扫出来 5 和 6 同时不通，问题可能在 5 前面的线，不一定是两个舵机都坏。
