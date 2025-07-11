# 强化学习加密货币交易机器人 - 高级版

## 项目简介

本项目是一个**先进的深度强化学习交易系统**，专门针对 BTC/USDT 高频交易场景。采用了最新的强化学习技术，包括 **Double Dueling DQN**、**Prioritized Experience Replay (PER)**、**LSTM 记忆网络** 和 **自适应噪声网络**，旨在从历史 K 线数据中学习最优交易策略。

## 🚀 核心特性

### 深度学习架构
- **Double Dueling DQN**：结合双网络和对决架构，有效减少 Q 值高估问题
- **LSTM 时序记忆**：持续化隐藏状态，更好地捕获市场时间序列特征
- **Prioritized Experience Replay**：基于 TD-error 的优先级采样，提升学习效率
- **Noisy Networks**：可选的参数化噪声层，实现自动探索策略

### 训练优化技术
- **自适应 ε-greedy 衰减**：智能探索率调度，平衡探索与利用
- **梯度累积 & AMP**：支持混合精度训练，提升 GPU 利用率
- **动态目标网络更新**：基于步数而非回合数的更新策略
- **检查点与最佳模型保存**：自动保存训练过程中的最优权重

### 环境与奖励设计
- **归一化对数收益率**：解决不同价格尺度下的奖励稳定性问题
- **基准比较奖励**：与买入持有策略的相对性能评估
- **风险管理**：内置破产保护和手续费惩罚机制
- **改进的动作空间**：21 个离散动作，支持分级买卖策略

### 数据处理与性能
- **离线特征缓存**：技术指标预计算，显著提升 I/O 性能
- **Parquet 格式存储**：高效的数据压缩与快速加载
- **并行计算优化**：支持多环境并行训练（可扩展）
- **内存优化**：直接 tensor 化批次数据，减少 CPU-GPU 传输

### 实时 GUI 监控
- **多维度可视化**：价格走势、资产净值、奖励曲线等
- **实时性能指标**：ε 衰减、损失函数、TD 误差等
- **动作分布分析**：买卖持有行为的统计可视化
- **训练控制面板**：一键开始/暂停/保存/加载功能

## 🛠 技术栈

### 核心依赖
- **PyTorch ≥2.0.0**：深度学习框架，支持 CUDA/MPS
- **PyQt6**：现代化 GUI 框架
- **pandas-ta**：技术指标计算库
- **pyarrow**：高性能数据序列化

### 性能优化
- **混合精度训练**：torch.cuda.amp 支持
- **JIT 编译**：numba 加速计算密集型操作
- **向量化操作**：NumPy/PyTorch 高效矩阵运算

### 可视化与监控
- **pyqtgraph**：高性能实时图表库
- **matplotlib/plotly**：静态与交互式图表
- **TensorBoard**：训练日志可视化（可选）

## 📊 GUI 仪表盘功能

### 主要图表
- **BTC 价格走势图**：带交易信号标记的实时价格
- **投资组合对比图**：智能体 vs 买入持有基准
- **回合奖励图**：每回合累计奖励与收益率

### 监控指标
- **实时状态**：当前回合、步数、利润率、最大回撤
- **交易统计**：总交易次数、手续费、持仓情况
- **学习进度**：探索率、训练损失、经验池大小
- **动作分析**：买卖持有行为分布统计

### 高级功能
- **动态配置**：支持运行时切换 PER、噪声网络等特性
- **模型管理**：便捷的权重加载、保存和检查点恢复
- **数据切换**：支持加载不同的历史数据集
- **性能分析**：TD 误差、收益率对比等高级指标

## 🎯 项目结构

```
qianghualearning1/
├─ binance_kline_downloader.py     # Binance API 数据下载工具
├─ preprocess_data.py              # 数据预处理脚本
├─ data/
│  ├─ BTCUSDT_5m_2years.csv        # 示例：BTC 5分钟 K线数据
│  └─ *.parquet                    # 缓存的预处理特征文件
├─ saved_models/                   # 模型权重存储目录
│  ├─ best_model.pth               # 最佳性能模型
│  └─ checkpoint_ep_*.pth          # 定期检查点
├─ logs/                           # 训练日志目录
├─ src/
│  ├─ __init__.py
│  ├─ agent.py                     # Double Dueling DQN 智能体
│  ├─ config.py                    # 全局配置与超参数
│  ├─ gui.py                       # PyQt6 实时 GUI
│  ├─ main.py                      # 程序入口
│  ├─ model.py                     # 神经网络架构定义
│  ├─ replay_buffer.py             # PER 经验回放缓冲区
│  ├─ trading_env.py               # 交易环境与奖励函数
│  └─ trainer.py                   # 训练循环与优化逻辑
├─ README.md
└─ requirements.txt
```

## 🚦 快速开始

### 环境安装

```bash
# 克隆项目
git clone <repository-url>
cd qianghualearning1

# 安装依赖（推荐使用虚拟环境）
pip install -r requirements.txt
```

### 数据预处理（可选但推荐）

```bash
# 预处理单个文件
python preprocess_data.py --input data/BTCUSDT_5m_2years.csv

# 批量处理数据目录
python preprocess_data.py --input-dir data/

# 查看缓存状态
python preprocess_data.py --input-dir data/ --status
```

### 启动训练

```bash
# 启动 GUI 应用
python src/main.py
```

### GUI 操作流程

1. **加载数据**：点击 "Load Dataset" 选择数据文件
2. **配置选项**：勾选 Prioritized Replay、Noisy Networks 等高级特性
3. **加载模型**：（可选）点击 "Load Weights" 从检查点继续训练
4. **开始训练**：点击 "Start Training" 开始学习过程
5. **实时监控**：观察各项指标和图表变化
6. **保存模型**：训练过程中会自动保存最佳模型

## ⚙️ 核心算法改进

### 1. Double Dueling DQN
- **Value Stream**：估计状态价值 V(s)
- **Advantage Stream**：估计动作优势 A(s,a)  
- **组合公式**：Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
- **Double DQN**：策略网络选择动作，目标网络评估价值

### 2. Prioritized Experience Replay
- **TD-error 优先级**：|δ| = |Q(s,a) - (r + γ max Q(s',a'))|
- **重要性采样**：避免优先级采样引入的偏差
- **Sum Tree 数据结构**：O(log n) 的高效采样与更新

### 3. 奖励函数优化
```python
# 归一化对数收益率
agent_return = log(net_worth_t / net_worth_{t-1})
baseline_return = log(baseline_t / baseline_{t-1})

# 综合奖励
reward = agent_return * scale + 
         (agent_return - baseline_return) * benchmark_weight - 
         (fee / net_worth) * fee_penalty_weight
```

### 4. LSTM 状态持续化
- **Episode 内持续**：hidden state 在单个回合内保持连续性
- **Episode 间重置**：每个新回合开始时清零隐藏状态
- **梯度流优化**：更好的时序依赖建模

## 📈 性能优化特性

### GPU 加速训练
- **自动设备检测**：CUDA > MPS > CPU 的智能选择
- **混合精度训练**：FP16/FP32 混合，提升训练速度
- **梯度累积**：模拟更大批次训练效果

### 内存与 I/O 优化
- **直接 Tensor 化**：减少 numpy ↔ tensor 转换开销
- **特征缓存系统**：技术指标离线计算，5-10x I/O 提升
- **批次预获取**：减少训练过程中的内存分配

### 训练稳定性
- **梯度裁剪**：防止梯度爆炸
- **目标网络软更新**：渐进式参数更新
- **学习率调度**：自适应学习率衰减

## 🔬 实验与调参指南

### 关键超参数
```python
# 网络架构
HIDDEN_SIZE = 128           # LSTM 隐藏层大小
WINDOW_SIZE = 60            # 历史窗口长度

# 训练参数  
LEARNING_RATE = 1e-4        # 学习率
BATCH_SIZE = 64             # 批次大小
GAMMA = 0.99                # 折扣因子

# 探索策略
EPSILON_START = 0.9         # 初始探索率
EPSILON_MIN = 0.01          # 最小探索率
EPSILON_DECAY = 0.9995      # 衰减率

# PER 参数
PER_ALPHA = 0.6             # 优先级指数
PER_BETA_START = 0.4        # 重要性采样初始权重
```

### 调参建议
1. **学习率**：过高导致不稳定，过低收敛慢
2. **批次大小**：增大提升稳定性，但需要更多显存
3. **探索率衰减**：应根据训练轮数调整衰减速度
4. **奖励权重**：平衡基准比较和手续费惩罚的重要性

## 📋 系统要求

### 最低配置
- **CPU**：4 核心，支持 AVX2 指令集
- **内存**：8GB RAM
- **显卡**：集成显卡（CPU 训练）
- **存储**：2GB 可用空间

### 推荐配置  
- **CPU**：8+ 核心 (Intel i7/AMD Ryzen 7)
- **内存**：16GB+ RAM
- **显卡**：NVIDIA RTX 3060+ / Apple M1 Pro+
- **存储**：SSD，10GB+ 可用空间

### 开发环境
- **Python**：3.8+
- **CUDA**：11.6+ (NVIDIA GPU)
- **操作系统**：Windows 10+, macOS 12+, Ubuntu 20.04+

## 🔧 故障排除

### 常见问题
1. **CUDA 不可用**：确认 PyTorch CUDA 版本与显卡驱动匹配
2. **内存不足**：减小 `BATCH_SIZE` 或 `WINDOW_SIZE`
3. **GUI 显示异常**：更新 PyQt6 版本或检查显示驱动
4. **训练不收敛**：调整学习率或检查奖励函数设计

### 性能调优
- 启用数据预处理缓存以提升加载速度
- 使用混合精度训练减少显存占用
- 根据硬件性能调整并行线程数

## 📚 扩展功能

### 计划中的特性
- **多币种支持**：ETH、BNB 等其他加密货币
- **连续动作空间**：TD3/SAC 算法支持
- **多环境并行**：异步训练加速
- **实盘交易接口**：币安 API 集成

### 研究方向
- **多智能体协作**：策略组合与对抗训练
- **元学习适应**：快速适应新市场条件
- **风险约束优化**：VaR/CVaR 风险控制

## ⚠️ 免责声明

**本项目仅供学术研究与技术学习使用**。加密货币交易存在极高风险，可能导致重大财务损失。任何基于本代码进行的实际交易操作，风险由使用者自行承担。

**风险提示**：
- 历史业绩不代表未来表现
- 强化学习模型存在过拟合风险  
- 市场环境变化可能影响策略有效性
- 交易手续费和滑点会影响实际收益

## 📄 许可证

本项目采用 MIT 许可证。详情请参见 [LICENSE](LICENSE) 文件。

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！请确保：
- 代码遵循项目风格规范
- 添加必要的单元测试
- 更新相关文档说明

---

**祝您研究愉快！** 🚀📈 