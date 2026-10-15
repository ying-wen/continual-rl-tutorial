# 流式强化学习：交互协议与更新稳定性

一条经验处理完就会离开，延迟结果怎样改变过去的预测，又怎样影响下一次行动？从样本 TD 推到资格迹和 actor–critic，再检查深度更新、学习器状态，以及 GVF、技能与平均奖励在同一条流中的时间和资源接口。

## 本章内容

- 明确流式数据权限、单步计算与持久内存预算，区分在线、离策略、流式协议和不可重置生命期。
- 从样本半梯度与有限前向目标推导资格迹，区分冻结参数恒等式和变参非线性执行。
- 推导 critic 误差用于行动学习的条件，并沿实际 Stream-AC 主循环检查旧参数、更新和下一动作的时序。
- 比较更新尺度控制与元信用，逐项恢复学习器状态，并说明 GVF、option 与平均奖励怎样共享经验而保留各自问题。

## 阅读这条论证

流式学习规定经验怎样被使用，并不另设一种收益目标。先把给定预测问题写成逐步更新，再让 critic 的误差改变策略。接着检查尺度、终止、学习器记忆和计算预算，最后把预测、技能与平均奖励接入同一条经验流。

[规定数据与计算条件](streaming.md#lesson-setting) → [从样本到资格更新](streaming.md#lesson-derive) → [分析更新尺度](streaming.md#lesson-scale) → [从误差到下一次行动](streaming.md#lesson-streamx) → [保存完整学习过程](streaming.md#lesson-learner-state) → [连接预测、技能与长期目标](streaming.md#lesson-components) → [对应实际实现](streaming.md#lesson-code)

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [时间信用分配：从资格迹到深度梯度学习](credit.md)：理解保留的资格与敏感度，再规定单次处理经验时能保存什么。


### 价值函数

$v(s)$ 预测从状态 $s$ 出发按固定策略累积的折扣奖励。学习器根据不断到来的经验修正这个预测。

### 半梯度 TD

误差 $\delta=r+\gamma v(s')-v(s)$ 中的两个预测使用同一份旧参数；更新方向只对当前预测求导。

### 资格迹

与参数同维的向量，积累过去预测对参数的敏感性。它保存信用分配所需的统计量，而不保存原始样本。

<a id="problem-definition"></a>

## 本章的问题定义

每次真实经验到来后及时学习，不重放历史转移，持久内存和每步计算有明确预算；该协议独立于是否允许环境重置。

### 给定条件与符号

- 基础预测或控制目标、新经验因果流、持久字节预算和单步时间预算。
- 网络、资格迹、尺度统计与更新控制器；真实终止清理规则。

### 需要求解的对象

满足协议且可持续执行的学习更新；更新尺度局部可检查，长期预测或控制质量另行评价。

### 信息与数据权限

允许保存参数、优化器状态、资格迹和在线统计，不保存供再次训练的历史转移。当前奖励统计、目标计算与更新顺序都需明确。

$$
M_t=U(M_{t-1},\xi_t),\qquad \operatorname{bytes}(M_t)\le B,\qquad C_t\le C_{\max}
$$

$\xi_t$ 是刚收到的经验，$M_t$ 是全部持久学习状态，$U$ 是更新映射，$B$ 为内存预算；$C_t$ 是包括动作选择、学习与规划的本步总计算，$C_{\max}$ 为上限。历史转移不得被重取训练。基础任务目标沿用预测或控制规格，这些资源约束不是新的奖励函数。

### 成立条件与解的含义

- 预算包含网络、优化器、迹和统计；batch size、无GPU或常数缓存大小不单独证明协议。
- 局部输出线性化、范数代理和逐坐标位移界的前提分别声明；终止不自动清除长期尺度统计。

判断准则：日志证明每条转移使用权限与延迟，持久内存不随寿命增长；手算更新/尺度边界正确；长期实验报告非有限更新、恢复及外部收益，参数位移界不替代收益。

### 适用边界

- 小参数更新不保证TD误差下降、策略改善或单生命期安全。
- 不把观测归一化和自适应统计当作无成本且不影响函数的预处理。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变信息或数据协议 · [时间信用分配与资格迹](credit.md)：资格迹用持久统计承接延迟反馈；前向目标、在线参数变化和实际终止仍决定这条统计怎样更新。

- 组合不同学习问题 · [元学习与学习规则的适应](meta.md)：预设尺度控制限制本次更新，元学习根据后续表现改规则；后者还需保存规则敏感度并核算每步成本。

- 改变信息或数据协议 · [通用价值函数与预测知识](gvf.md)：一条新经验可即时服务多个预测，但每个问题保留自己的cumulant、continuation、目标策略和学习状态。

- 改变信息或数据协议 · [Options 与技能发现](options.md)：技能可用有限累加器保存段内后果；高层边界更新与原始步更新是不同调度，停止技能不终止外部控制。

- 组合不同学习问题 · [平均奖励与差分价值](average.md)：平均奖励另定义奖励率和差分价值；逐步中心化使用哪个率、技能耗时按哪个时钟计量，都需与更新时序一致。

- 改变信息或数据协议 · [持续学习的智能体架构](architectures.md)：完整智能体要合计行动、预测、模型和规划成本，保存全部会改变下一次更新的状态；无重置与无重放分别规定。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

没有回放可反复纠正当前过冲；奖励、特征和迹尺度变化会放大单步更新。

### 本章的核心思路

用可递推统计保留信用并按声明的局部尺度限制更新，分别检验协议和任务表现。

1. [从样本目标推到资格更新](streaming.md#lesson-derive)：先由固定策略的Bellman关系构造停止梯度标签，再把有限多步目标变成可递推资格；在线变参时另查等价条件。

2. [定位尺度与过冲](streaming.md#lesson-scale)：由特征、误差和资格计算一次更新造成的输出变化，说明范数缩放控制了什么以及其局部前提。

3. [让critic误差影响下一次行动](streaming.md#lesson-streamx)：在声明的策略梯度条件下构造actor方向，按旧参数计算critic与actor信号，再共同写入并选择下一动作。

4. [保存能重现下一步的学习状态](streaming.md#lesson-learner-state)：权重之外，资格、优化器和运行统计也决定更新；逐项规定终止、恢复和单独重置时哪些状态保留。

5. [在同一条流上组合多个问题](streaming.md#lesson-components)：GVF各守预测规格，技能各守执行边界，平均奖励按原始时间中心化；模块共享经验而分别计入状态和计算预算。

结论与条件：有限增量和正稳定项下可检验逐坐标位移界；线性固定标签的输出关系只约束本次更新。Stream-X/Intentional深度组合不提供无条件长期收敛。

### 相关方法改变了什么

- ObGD：以TD误差与迹范数做廉价整体缩放，依赖代理尺度解释。

- 逐坐标StreamingOptimizer：用衰减最大增量提供坐标位移边界，不保证输出收缩。

- Intentional Updates：按局部输出变化选尺度，含预条件、迹和在线平均近似。


<a id="lesson-setting"></a>

## 1 · 流式学习的数据与计算约束

设机器人每 20 毫秒接收一次传感器信息。环境继续向前走，学习器则可以立刻更新、先收集三条再更新，或反复处理刚收到的同一条经验。这些选择会改变下一次行动能用上哪些知识。先把两个计数分开：环境时钟数真实转移，学习时钟数参数更新。

下图假定一次更新耗时 12 ms，学习任务应在下一条经验到来前完成。环境转移编号是 $t$，更新编号是 $k$。每条经验更新一次时，两者按固定关系增长；收集三条经验做一次批更新，或每条经验重复更新两次，就会改变这个关系。图中的时间是假定成本，用来检查调度。真实控制还要计入感知、通信和动作计算。

![经验以小帧沿时间流入。逐条更新及时完成，三条经验汇成一次批更新，重复更新使等待逐渐拉长。](https://yingwen.io/crl-figures/concept-streaming-clocks.svg)

小帧是新经验，色块是更新，粉色带是尚未开始处理的等待时间。环境每 20 ms 到来一条经验，每次更新假定耗时 12 ms；重复两次后，等待依次为 0、4、8、12 ms。块内数字是学习更新计数。原创调度算例。

逐条处理在 32、52、72、92 ms 完成。重复两次需要每周期 24 ms，按到来顺序处理的完成时刻变成 44、68、92、116 ms；待处理的经验开始积压。若只保存最新样本，就必须丢弃来不及处理的数据。若系统遇到迟到便维持旧动作，环境会在旧控制下多运行一段；若暂停模拟器等待计算，得到的又是另一种交互条件。因此，按相同环境步数画出的学习曲线，还需要延迟协议才能解释。

| 名称回答的问题 | 允许的选择 | 一个具体组合 |
| --- | --- | --- |
| Online：训练时还有新交互吗？ | 策略与环境继续产生数据，并在交互过程中学习 | PPO 可以在线收集 rollout，再做批更新 |
| Off-policy：学习谁的后果？ | 目标策略 π 可不同于实际行为 b | 当前转移可以立即用于学习另一策略的 GVF，无需 replay |
| Batch：一次更新用多少条样本？ | 一条，或将多条梯度求和、平均 | 同一批可只使用一次，也可优化多轮 |
| Replay：旧样本会被重新读取吗？ | 保留转移，之后抽取并重算学习目标 | 每次只抽一条的 replay 仍然是 replay |
| 严格一步 streaming：何时用、用几次、保留什么？ | 本章规定即到即用一次，不积攒训练批次，不留原始转移供重放 | 固定维参数、资格迹与运行统计可以跨步保留 |

这里的 batch 指更新分组方式；有些文献把仅从固定数据集学习也称为 batch RL，此时还需说明不再采集新交互。环境是否允许 episode 重置，则由另一个协议规定：每回合结束后重置环境、保留已学参数，仍然可以逐条流式学习。折扣还是平均奖励、固定策略还是控制、网络是否深，也各自需要选择。

内存限制关心保存了什么及其大小。权重保存已经学到的预测，RNN 隐状态概括当前历史，资格迹保存过去预测的参数敏感性，优化器统计保存更新尺度。这些状态都要计入持久内存预算。以十万个 float32 参数为例，权重和一条同维资格迹各占 0.4 MB；若再有两条优化器统计，总计 1.6 MB。相比之下，十万张单通道 84×84 字节图像仅像素就占 705.6 MB，还没有算动作、奖励与索引。这里使用十进制 MB，比较的是明确给定的存储内容。

保存摘要也决定以后还能提出什么问题。保留原始图像，可以用更新后的编码器重新计算特征；只保留旧特征或梯度统计，则承诺以后通过这些摘要使用经验。流式限制因此同时是计算限制和信息限制。资格迹选择保留信用分配需要的敏感性，在线元学习还可以保留学习规则的敏感性；它们占用同一份资源预算，却回答不同的问题。

更完整的约定是：在固定架构下，持久学习状态的大小 $M_t\le B$，每条转移的处理成本 $C_t\le C_{\max}$，两项上界都不随已活过的步数 $t$ 增长。环境还可以给出墙钟截止时间 $\Delta$。$O(d)$ 的资格向量说明内存随参数数目 $d$ 增长，尚未说明这台设备能否在 $\Delta$ 内完成。固定网络的精确 RTRL 也可与生命期长度无关，却可能因状态—参数敏感度矩阵太大而超出给定预算。[状态构建章](state.md)讨论该矩阵的计算；本章的限制须同时报出 $d,B,C_{\max}$ 与实际延迟。

| 约束或对象 | 保存内容或度量 | 边界 |
| --- | --- | --- |
| 严格 streaming 数据 | 当前样本；固定维参数和统计 | 不重放旧转移，不等待未来样本组成训练 batch |
| 资格迹 | 每个参数一个历史敏感性统计 | 不能无损恢复历史样本 |
| 实时计算预算 | 一次或固定少数次前后向 | 限制每步计算与历史回溯 |
| 学习评价 | 实际每步延迟、内存、累计奖励 | GPU 吞吐量不能单独代表实时控制能力 |

把 DQN 的 replay capacity 改成 1 是可用的诊断基线，却未必保留 DQN 的稳定性。它移除了跨经验的样本混合，也使从这个 buffer 抽出的批次只包含同一条经验；但不会自动移除 target network。若同一转移仍被重复更新，也不符合本章严格的一次使用协议。比较时应分别说明回放、更新次数、目标副本和批大小怎样改变。研究问题应是“哪些机制能在这个协议下工作”，而不是只换一个 buffer 参数。

<a id="course-streaming-tracking-floor"></a>

## 持续更新不等于持续学会：噪声、遗忘与追踪的最小计算

流式协议允许每步更新，但不保证更新还来得及追踪目标，也不保证网络保有学习能力。先去掉 TD、自举和神经网络，只研究一个标量预测。这个小问题已包含持续学习中的两个相反要求：平均更多数据能减少噪声，给旧数据较小权重才能及时适应。

$$
y_t=m_t+\epsilon_t,\qquad w_t=(1-\alpha)w_{t-1}+\alpha y_t,\qquad 0<\alpha\leq1.
$$

$m_t$ 是此刻要预测的均值；噪声独立、零均值且方差为 $\sigma^2$。$w_t$ 是读取当前样本后的预测。这个固定步长更新只需要一个数的持久存储。

$$
w_t=(1-\alpha)^t w_0+\alpha\sum_{k=1}^{t}(1-\alpha)^{t-k}y_k.
$$

旧样本的权重随年龄几何衰减。没有保存原始数据，不等于不受历史影响；这里的历史保存在统计量中。固定步长也不是完全遗忘：过去信号仍以递减权重参与。

$$
m_t=m:\quad \lim_{t\to\infty}\operatorname{Var}(w_t-m)=\frac{\alpha\sigma^2}{2-\alpha};\qquad m_t=m_0+vt:\quad \lim_{t\to\infty}\mathbb E[w_t-m_t]=-\frac{1-\alpha}{\alpha}v.
$$

左式由误差方差递推求固定点；右式由均值误差递推求固定点。前者说明小步长降低稳态噪声，后者说明小步长增加持续线性漂移下的滞后。漂移例是局部追踪诊断，不是有界奖励的无限时域控制模型。

取 σ²=1、每步均值漂移 v=0.01。α=0.1 时，稳态方差约 0.0526、均值滞后 −0.09；α=0.01 时，方差降至约 0.00503，滞后却变为 −0.99。把目标漂移阶段的误差全部称为随机噪声，并继续减小步长，会使问题更严重。

这个更新始终有非零灵敏度：对新样本的导数为 α。因此它的追踪失败可以来自步长与时间尺度失配，而不是可塑性丧失。神经网络则多出梯度通路、特征退化和优化器统计等困难。必须先分开“仍能改变但改变太慢”与“相同训练协议下越来越难学到新目标”。

| 要区分的困难 | 先固定什么 | 比较什么 |
| --- | --- | --- |
| 采样噪声 | 固定均值与数据生成过程 | 不同步长的估计方差 |
| 目标漂移 | 相同漂移轨迹与输入尺度 | 滞后、变化后的累计误差 |
| 表示或优化器老化 | 同一 probe 数据、容量和训练预算 | 已有网络与匹配的新网络学习能力 |
| 控制造成的数据变化 | 相同控制权限，另设固定数据诊断 | 新状态覆盖与同数据学习表现 |

接回 CRL，同一个 GVF 会因目标策略或问题函数改变而需要追踪；option 模型会因技能改善而改变；价值又受模型规划与 actor 更新影响。即使外部世界固定，这些内部目标仍可能移动。多个学习器都“使用常数步长”不是完整答案：各自的噪声、变化速度和资源成本不同，且它们的更新会影响彼此。

思考：若减小 critic 步长使 TD loss 更平稳，却降低变化后的收益，怎样检验是目标追踪太慢、探索减少还是表示退化？先在共同固定的数据流上定位学习问题，再回到各自产生经验的闭环实验。两类实验回答不同问题，不能互相替代。

<a id="research-openmind-time-discretization"></a>

## 学得及时，还要说明奖励发生在何时

逐转移更新并不自动统一物理时间。若一个机器人每 10 毫秒控制一次，另一个每 100 毫秒一次，相同的每步折扣会定义不同的未来尺度。De Asis 与 Sutton（RLC 2024）进一步指出：奖励在区间末端到达，却把第一条奖励视为无需折扣，也会改变连续时间回报的离散近似。

$$
\widetilde G(t)=\int_t^T e^{-\kappa(u-t)}r(u)\,\mathrm du,\qquad G_t^{\rm RP}=\sum_{k=t}^{T-1}e^{-\kappa\sum_{i=t}^{k}\Delta_{i+1}}r_{k+1}\Delta_{k+1}
$$

这是原文式 (7) 的右端点形式，将每单位时间折扣写成 $e^{-\kappa}$。$\Delta_{i+1}>0$ 是真实区间时长，$r_{k+1}$ 是区间末采样的奖励率；若环境已经返回区间积分奖励，不能未经检查再乘时长。

传统离散回报的第一条奖励权重为一；右端点近似则同时将奖励和折扣放在区间末端。固定 Δ 时，两种回报只差一个共同正比例，策略排序不因此改变。若 Δ 随时间、动作或计算耗时变化，该比例就不再能从整个和式中提出。

$$
G_t^{\rm RP}=e^{-\kappa\Delta_{t+1}}\left(r_{t+1}\Delta_{t+1}+G_{t+1}^{\rm RP}\right)
$$

这是上述目标直接给出的递推。相应 TD target 也应同时折扣本区间奖励率近似与后继价值；先确定传感器奖励的时间语义，才能选择这个公式。

手算积分检查：奖励率恒为 1，每单位时间折扣 0.9，取两段时长 1 和 2。右端点近似是 $0.9+2\times0.9^3=2.358$；若折扣取左端点、奖励取右端点，则是 $1+2\times0.9=2.8$。二者都只是积分近似；在这个例子中准确积分为 $(1-0.9^3)/(-\log0.9)\approx2.572$。

论文研究的是给定奖励采样语义下的目标离散化，不是关于所有控制问题的万能修正。它既不消除低采样频率漏失事件，也不解决动作延迟、探索风险或表示学习。奖励已在高频内部正确积分的 SMDP／option 接口，需要按其实际定义处理。

与本章更新尺度控制的关联在于：Intentional Updates 约束每次学习产生的局部变化；真实时间目标决定这些次数及奖励延迟意味着什么。Kris 的 Design for Learning 立场文章进一步讨论硬件可恢复性与学习算法共同设计，但该文章本身不是持续机器人性能实验。

- 思考：固定 Δ 的共同比例虽不改最优排序，为什么仍会改变学习率、奖励尺度与数值难度？
- 实验设计：固定每秒目标，记录真实时间戳，在相同物理时长下改变控制周期及其抖动；分别报告回报积分误差、每秒更新数、决策延迟和实际控制收益。

<a id="lesson-derive"></a>

## 2 · 流式协议中的资格迹

先固定一个 Markov 状态表示、策略 $\pi$ 和折扣 $0\le\gamma<1$，预测其回报。Bellman 方程需要对可能的奖励和后继状态求期望，流式学习器实际只收到一条 $(S_t,A_t,R_{t+1},S_{t+1})$。用旧网络构造标签，再固定这个标签求导，得到可立即执行的一步更新：

$$
\begin{aligned}y_t&=R_{t+1}+\gamma_{t+1}\hat v(S_{t+1};w_t),\\\ell_t(w;y_t)&=\tfrac12[y_t-\hat v(S_t;w)]^2,\\-\nabla_w\ell_t(w_t;y_t)&=\delta_t\nabla_w\hat v(S_t;w_t),\qquad\delta_t=y_t-\hat v(S_t;w_t).\end{aligned}
$$

只在本次求导时固定 y；下一条经验用新参数产生新标签。若沿后继预测也求导，就改变了更新方向。

它为什么与所要预测的价值有关？把 $w$ 作为固定的分析参数，从 $s$ 按 $\pi$ 采样一步，则 $\mathbb E_\pi[\delta(w)\mid S=s]=(\mathcal T_\pi\hat v_w)(s)-\hat v_w(s)$。表格表示时，各状态的平均更新在寻找 Bellman 方程的零点。共享参数时，实际方向是残差乘特征的加权和，不同状态可能彼此抵消；这连接到 [GVF 章的投影固定点](gvf.md)。该等式解释固定预测器上的一步采样，尚未给变化参数、相关数据下的整条学习轨迹一个收敛证明。

一步目标可以立即计算，但终点奖励传回很早的状态可能需要多次再访问。限制原始样本保留后，资格迹选择保存过去输出的梯度摘要。为看清摘要为何是指数衰减，先冻结整条轨迹上的价值参数，比较多步目标。

$$
G_t^{(n)}=\sum_{k=1}^{n}\gamma^{k-1}R_{t+k}+\gamma^n v(S_{t+n})
$$

n-step target 比一步 TD 看得更远，却要等 n 步才能构造。有限回合末尾的 bootstrap 为零。

先固定价值参数。将 $\delta_t,\gamma\delta_{t+1},\ldots$ 相加时，相邻两项中的状态价值相消，只留下首项 $-v(S_t)$ 和末尾的 bootstrap。这一望远镜求和说明了多步回报与 TD 误差序列的关系。

$$
G_t^{(n)}-v(S_t)=\sum_{k=0}^{n-1}\gamma^k\delta_{t+k}
$$

等式要求右侧所有 δ 使用与左侧同一套固定价值。在线每步变参数时，普通 accumulating trace 与这个冻结参数前向视角只有近似对应；不能把有限步长下的等价说成精确。

$$
\begin{aligned}G_t^\lambda&=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)},\\G_t^\lambda-v(S_t)&=\sum_{k=0}^{\infty}(\gamma\lambda)^k\delta_{t+k}.\end{aligned}
$$

无限混合式先取 $0\le\lambda<1$，并要求回报和右侧级数收敛；$\lambda=1$ 用相应极限或完整回报定义，不直接代入第一行的零乘无限和。有限回合让最后一个 Monte Carlo return 承接剩余权重。$\lambda=0$ 退化为一步 TD，接近 1 则使较远的 TD error 参与信用分配。

前向视角考虑未来哪些 $\delta$ 影响当前预测；后向视角在误差到来时，给过去相关的预测分配信用。为避免无限求和的额外条件，取终止于 $T$ 的有限轨迹，令 $g_t=\nabla_w\hat v(S_t;w)$，参数 $w$ 始终冻结。将每个预测的前向增量相加，并按误差到达时刻重新分组：

$$
\begin{aligned}\sum_{t=0}^{T-1}(G_t^\lambda-\hat v_w(S_t))g_t&=\sum_{t=0}^{T-1}\sum_{k=t}^{T-1}(\gamma\lambda)^{k-t}\delta_k g_t\\&=\sum_{k=0}^{T-1}\delta_k\underbrace{\sum_{t=0}^{k}(\gamma\lambda)^{k-t}g_t}_{z_k},\\z_k&=\gamma\lambda z_{k-1}+g_k.\end{aligned}
$$

同一个误差只需乘一个递推向量。这个冻结参数的总方向恒等式也适用于可微非线性预测器；它没有在求和中途改变网络。

现在把该递推用作在线算法：每收到一条经验就计算当前梯度并更新参数。线性固定特征下 $g_t=x_t$；神经网络下 $g_t=\nabla_w\hat v(S_t;w_t)$ 随参数版本改变。可执行的普通 accumulating TD(λ) 为：

$$
\begin{aligned}z_t&=\gamma_t\lambda z_{t-1}+\nabla_w v(S_t;w_t),\\\delta_t&=R_{t+1}+\gamma_{t+1}v(S_{t+1};w_t)-v(S_t;w_t),\\w_{t+1}&=w_t+\alpha_t\delta_t z_t.\end{aligned}
$$

γt 属于到达 St 的转移，用来衰减旧痕迹；γt+1 属于当前刚完成的转移，用来构造 target。常数 γ 时区别被隐藏，终止和 GVF 的状态依赖 discount 会把错误暴露出来。

若 $S_{t+1}$ 真正终止，$\gamma_{t+1}=0$，不再 bootstrap；但本次奖励仍通过 $z_t$ 分给本回合的过去状态。痕迹在完成终点更新后、新回合开始前清空。若只是时间限制截断而真实过程仍会继续，价值目标保留 bootstrap；若随后环境另行重置，资格迹则不能跨越这次非连续转移。

**算法：线性 accumulating TD(λ)；终点奖励在清迹之前更新**

1. 初始化参数 $w$、$z=0$，选择 $\lambda$ 与 $\alpha$
1. 每步接收 $(x,r,x')$ 和 $\gamma_{\rm current},\gamma_{\rm next}$：
  1. 保存旧参数 $w_{\rm old}$
  1. 计算 $v=x^\top w_{\rm old}$、$v'=x'^\top w_{\rm old}$
  1. 计算 $\delta=r+\gamma_{\rm next}v'-v$
  1. 更新 $z\leftarrow\gamma_{\rm current}\lambda z+x$
  1. 更新 $w\leftarrow w_{\rm old}+\alpha\delta z$
1. 新回合开始时令 $z=0$

因此网络里的 $z_t=\sum_{k=0}^{t}(\gamma\lambda)^{t-k}\nabla_w\hat v(S_k;w_k)$，通常不等于把旧输入全部放进当前网络所得的 $\sum_{k=0}^{t}(\gamma\lambda)^{t-k}\nabla_w\hat v(S_k;w_t)$。例如 $\hat v(x;a,b)=b\tanh(ax)$，对 $a$ 的梯度为 $b[1-\tanh^2(ax)]x$；即使只更新 $b$，同一旧输入对 $a$ 的梯度也变了。保存数值资格保留了当时的信用方向，重算旧输入则需要额外信息与计算。[信用分配章](credit.md#lesson-online)进一步展开变参导致的望远镜余项与 true-online 修正。

若行为是 $b$、预测的是 $\pi$，当前样本也可立即构造 $\rho_t=\pi(A_t\mid S_t)/b(A_t\mid S_t)$。普通状态价值 IS 迹改为 $z_t=\rho_t[\gamma_t\lambda z_{t-1}+g_t]$，要求行为覆盖目标动作。这个比率修正动作来源，行为的状态访问权重仍然存在；共享函数逼近下的离策略稳定性需要另行处理。GTD 与 emphatic 方法改变的正是目标几何或状态权重，它们也可以增量执行。

| 方法 | 保存的痕迹 | 关键限制 |
| --- | --- | --- |
| TD(λ) | 过去 ∇v 的衰减和 | 普通在线 accumulating 版本不是任意步长下精确前向等价 |
| Momentum SGD | 过去 loss gradient 的衰减和 | 每步历史误差已乘在各自梯度上；不同于当前 δ 乘全部痕迹 |
| True online TD(λ) | Dutch trace 和预测差修正 | 对指定线性在线 λ-return 有精确等价；不能直接宣称任意深网同样成立 |

<a id="experiment-neural_td_trace"></a>

### 实验：实验 · 终点奖励怎样沿历史梯度传回去？

逐条转移更新且不保存经验时，非线性资格迹如何加快早期传播，又为何不保证始终优于 TD(0)？

**环境与可用信息。** 固定策略依次经过六个状态。前五次转移奖励为零，最后一次奖励服从 Bernoulli(0.7)，随后真实终止。折扣为 0.9。状态编码为位置的线性坐标及其平方，价值网络为 2–6–1 tanh。

**设置。** 种子 0–4，各运行 1200 个转移，即 200 个完整回合。所有参数独立均匀初始化于 [−0.4,0.4]。每步半梯度更新，步长 0.03；实验 λ=0.8，对照 λ=0。网络、初值与奖励随机流匹配，不用回放或目标网络。

**检验的机制。** 先构造当前迹，再用到达的 TD 误差更新参数；终点奖励分配后才清迹。非线性网络的历史梯度是在历史参数上计算的，并不会因当前参数变化而被重新计算。长迹既传播早期信用，也传播奖励噪声。

**测量。** 主图是六状态真实价值的 RMSE。原始日志还保存 trace_norm、gradient_drift、完成回合数与保留转移数。gradient_drift 只测同一上一输入的梯度变化，不是整条历史迹的误差。

```bash
python3 implementations/streaming_composition/neural_td_trace.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](../assets/crl-figures/result-neural_td_trace.svg)

横轴：environment_steps。纵轴：全状态预测 RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 六个状态的神经预测与0.7×0.9^(5−s)逐项比较，取均匀 RMSE。0.7来自末端奖励成功概率，参照不是本回合抽到的0或1。

**step：怎样计时。** step 是真实原始转移，包含末端奖励转移；完成回合数、梯度次数和迹范数另记。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算各记录时刻的跨种子均值、样本标准差和末点误差；没有保存全部预测向量，不能仅凭 value 重新计算状态权重或逐状态误差。

计算位置：[streaming_composition/neural_td_trace.py](../implementations/streaming_composition/neural_td_trace.py) · [streaming_composition/neural_td0.py](../implementations/streaming_composition/neural_td0.py) · [streaming_composition/_common.py](../implementations/streaming_composition/_common.py)

</details>

**结果分析。** 第 60 步，λ=0.8 的平均 RMSE 为 0.142，对照为 0.286，早期传播更快；第 600 步则为 0.0937 对 0.0607。末尾为 0.0478 对 0.0509，差异相对种子波动很小。不能只挑某一个时刻宣布迹总是更好。

**结论边界。** 这是平稳、同策略、Markov 输入的预测实验。普通非线性累积迹没有获得 true-online 线性等价保证。两者都使用固定步长，末端随机奖励会持续带来波动。

**继续实验。** 保留同一奖励流，把 λ 改为 0、0.4、0.8、1；分别观察前 60 步与后 600 步误差。再故意在终点更新之前清迹，定位哪些早期状态不再获得这次奖励的信用。

[源码](../implementations/streaming_composition/neural_td_trace.py) · [逐种子记录](https://yingwen.io/crl-code/results/neural_td_trace/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/neural_td_trace/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/neural_td_trace/curves.json)

<a id="lesson-scale"></a>

## 3 · 数值尺度与更新过冲

考虑一维固定标签回归：预测为 $wx$，误差为 $e=y-wx$，更新为 $w_+=w+\alpha ex$。代入后得到 $e_+=e(1-\alpha x^2)$。即使 $\alpha=0.1$，只要 $x=10$，有效步长也达到 10，误差变号且放大 9 倍。步长的作用取决于特征尺度。

$$
\xi=\frac{e-e_+}{e}=\alpha x^2,\qquad 0<\xi\leq1\ \Longrightarrow\ \text{本次固定标签误差不越过零点}
$$

这是当前样本的代数性质，不等于新样本泛化、长期控制稳定性或单生命期安全。e=0 时用更新为零处理，不做 0/0。

TD 更新会同时改变当前预测和下一状态 bootstrap。线性情况下直接展开，得到 $\xi=\alpha z^\top(x-\gamma x')$，而非 $\alpha x^\top x$。它可以为负，即同一样本的 TD 误差变大；这体现半梯度 TD 与固定标签回归的区别。

$$
\delta_+=\delta-\alpha\delta\,z^\top(x-\gamma x'),\qquad \xi=\alpha z^\top(x-\gamma x')
$$

式中 z 在这一步更新中视为固定。深网使用局部线性化后只能得到近似；若再用更便宜的范数代理，还需要额外梯度尺度假设。

$$
\begin{aligned}M&=\alpha\kappa\max(|\delta|,1)\|z\|_1,\\\alpha_{\rm used}&=\frac{\alpha}{\max(1,M)},\\w_+&=w+\alpha_{\rm used}\delta z.\end{aligned}
$$

这是 ObGD 的代数更新形式，κ 是安全缩放系数。它用廉价统计抑制大更新；对任意网络、任意输入并不存在无条件不越界保证。零痕迹时分母至少 1，不会除零。

一个尺度反例是 $y=1$、$x=10$、$w=0$、$\alpha=0.1$、$\kappa=2$。此时 $z=10$、$M=2$，所以 $\alpha_{\rm used}=0.05$，更新后 $w_+=0.5$，预测为 5，仍然超过目标 1。这里梯度尺度不满足范数代理所需条件，说明该更新的实际性质依赖输入与网络尺度。

在线输入统计采用 Welford 递推：先用旧均值计算差，再更新新均值，最后用新旧两侧差的乘积更新 M2。所有统计只能使用已经到来的观测。全数据预先归一化会泄露未来；累积统计在漂移中会越来越迟钝，指数滑动统计则是另一种估计器，需要独立比较。

$$
\begin{aligned}n&\leftarrow n+1,\quad d=x-\mu_{\rm old},\\\mu&\leftarrow\mu_{\rm old}+d/n,\\M_2&\leftarrow M_2+d(x-\mu),\quad \sigma^2=M_2/(n-1).\end{aligned}
$$

n=1 时必须定义方差初始化和 ε，避免第一步除零。归一化统计的改变也在改变价值函数输入坐标，因此不是与学习完全无关的预处理。

<a id="experiment-extended-obgd2024"></a>

### 实验：实验 · 限制更新幅度，何时只是把学习变慢？

如果固定步长已经稳定，ObGD 的保守缩放是否仍应提高学习速度？

**环境与可用信息。** 两个状态确定性交替、永不终止。到达状态 1 获得奖励幅度，另一转移奖励为零；第 601 步幅度从 1 降为 0.5。输入为二状态 one-hot，网络为 2–16–1 tanh，折扣 0.8。

**设置。** 五种子各运行 1200 个真实转移。网络使用相同 seed 的 PyTorch Linear 默认初始化。两者都用累积梯度迹，λ=0.8，基础步长 0.03；ObGD 额外采用 κ=2 的全迹 L1 缩放。没有回放、目标副本、LayerNorm 或稀疏初始化。

**检验的机制。** 对照直接使用 0.03；ObGD 依据当前误差和迹范数缩小有效步长。它没有修改当前预测问题，也没有修正离策略分布。这个温和任务隔离了“缩小增量”本身的效果。

**测量。** 纵轴是两个状态相对于解析真值的冻结 MSE，越低越好。第 600 步仍属于旧奖励阶段；第 601 步才开始新阶段。日志中的 step_size 与 trace_l1 可解释误差下降的速度。

```bash
python3 implementations/extended_adaptation/obgd2024.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/extended-obgd2024/curves.svg)

横轴：environment_steps。纵轴：frozen_value_mse。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 60 步，ObGD 的平均 MSE 为 0.0653，固定步长为 0.00269。改变奖励后第 660 步，分别为 0.0157 与 0.000750。最后两者都近于零。这个设置没有显示 ObGD 优势，而是显示限制增量的适应速度代价。

**结论边界。** 这是 2024 优化器加非线性 TD 的组件，不是完整 Stream-X，也不是 2026 StreamingOptimizer。未扫描尺度、基础步长或任务难度，不能由此判定作者系统无效。

**继续实验。** 在不改变任务奖励目标的前提下，系统改变输入尺度和基础步长，预先记录稳定区域与恢复时间。保留所有发散配置，不通过事后裁剪参数把失败曲线变成成功。

[源码](../implementations/extended_adaptation/obgd2024.py) · [逐种子记录](https://yingwen.io/crl-code/results/extended-obgd2024/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/extended-obgd2024/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/extended-obgd2024/curves.json)

<a id="lesson-streamx"></a>

## 4 · 从 critic 误差到 Stream-AC 的下一次行动

机器人不仅预测固定策略，还要改变行为。令 actor 的参数为 $\theta$，critic 的参数为 $w$。先冻结 $\pi_\theta$ 并假设 critic 恰好是它的真实价值，则一次转移的条件平均误差为优势：

$$
\begin{aligned}\mathbb E[\delta\mid S=s,A=a]&=q_{\pi_\theta}(s,a)-v_{\pi_\theta}(s),\\\mathbb E[\delta\nabla_\theta\log\pi_\theta(A\mid s)\mid S=s]&=\sum_a\nabla_\theta\pi_\theta(a\mid s)q_{\pi_\theta}(s,a).\end{aligned}
$$

第二行使用 π∇logπ=∇π，以及状态基线乘 ∇Σaπ(a|s)=0。动作之外的奖励与后继仍是随机样本。

这说明当前 $\delta$ 为什么可用于加强或减弱刚执行的动作。若 $\hat v_w=v_\pi+e$，第一行还多出 $\gamma\mathbb E[e(S')\mid s,a]-e(s)$；当前状态的基线误差在动作平均中抵消，后继误差一般不会。即使动作刚由当前 actor 产生，critic 的逼近误差仍会偏移 actor 的平均方向。

在整个任务上，仍须说明状态怎样加权。指定初始分布的折扣目标 $J(\theta)=\mathbb E[\sum_{t\ge0}\gamma^tR_{t+1}]$ 需要相应的折扣访问权重；逐时刻写成轨迹和时含 $\gamma^t$。Stream-AC 的实际增量循环没有这个从回合起点累计的外层因子，并加入资格迹、熵和更新缩放。因此上式用于解释局部信号，不将完整实现称为该固定起点目标的无偏梯度。平均奖励的状态权重与差分 critic 在后面的接口节另行规定。

先忽略熵和缩放，actor 与 critic 各保存一条迹：$z_t^v=\gamma\lambda_vz_{t-1}^v+\nabla_w\hat v_w(S_t)$、$z_t^\pi=\gamma\lambda_\pi z_{t-1}^\pi+\nabla_\theta\log\pi_\theta(A_t\mid S_t)$，分别以 $\alpha_v\delta_tz_t^v$ 和 $\alpha_\pi\delta_tz_t^\pi$ 更新。它们利用同一反馈，却对不同输出分配信用。actor 改变以后，下一次采样的行动概率变化，critic 追踪的策略也随之变化；固定轨迹预测算例只覆盖了这个闭环的一部分。

Stream-X 在这类基础更新外加稳定化组件。2024 版使用 ObGD、资格迹、在线观测与奖励尺度统计、无可学习仿射参数的 LayerNorm，以及稀疏初始化。2026 修订版保留这一组织方式，但更换了更新控制器并修改初始化；两版的优化器需要分别理解。

$$
\bar o_{t,i}=\frac{o_{t,i}-\mu_{t,i}}{\sqrt{\sigma_{t,i}^2+\varepsilon}},\qquad U_t=\gamma U_{t-1}+r_t,\qquad \bar r_t=\frac{r_t}{\sqrt{\operatorname{Var}_t(U)+\varepsilon}}
$$

观测按各坐标中心化并缩放；奖励只缩放，不减均值。奖励迹使用已到来的奖励，是未来 return 尺度的因果代理。2026 实现先更新统计并缩放当前奖励，再在 episode 边界清零奖励迹；累计均值和方差保留。

固定正数缩放全部奖励会按同一比例改变回报，保留原来的策略排序；随经验变化的缩放没有这条直接保证。比如两个行为的两步奖励分别为 (2,0) 与 (0,3)，未折扣总分原来偏好后者，若第二步统一缩小为原来的十分之一，排序便反转。在线尺度处理首先是更新机制的一部分，应保留原始奖励作评价，并检查它与 critic、actor、终止及温度的共同作用。

LayerNorm 在单个观测的一层预激活上计算均值和方差，因而不需要 batch。稀疏初始化则改变参数更新对不同输入的共享程度。二者不相同：前者控制层内尺度，后者改变初始连接。2024 的逐层固定稀疏率在低输入维度下可能将一层全部置零；2026 的 layer-aware 初始化分配全网非零连接预算，复现时应连同 sparse_init.py 一起使用。

| 分支 | 当前误差或方向 | 必须保留的时序 |
| --- | --- | --- |
| Stream TD(λ) | 固定策略 δ=r+γv′−v | 价值输出梯度累积，再交给对应版本的更新控制器 |
| Stream Sarsa(λ) | δ=r+γq(s′,a′)−q(s,a) | 下一动作来自实际目标策略 |
| Stream Q(λ) | max 下一动作目标；Watkins 痕迹语义 | 执行非贪心动作时截断此前不再匹配贪心延续的痕迹 |
| Stream AC(λ) | critic δ 配合 ∇logπ 的 actor trace | 行为动作按更新前策略采样；actor/critic 采用同一旧 δ，熵项另核对 |

Stream-X 的 2026 修订版在模拟 Ant 实验中反复切换两种地面摩擦；实体机器人航向跟踪则按 120 秒回合重置，并保留学习状态。前者考察反复变化后的恢复，后者展示传感与控制闭环中的在线学习。这两种实验都可满足无重放协议，环境重置权限则按各自实验设置记录。

作者 2024 实现先对 $-v$ 和负 log-probability 求梯度，再由 optim.py 做减法；2026 实现改为正输出梯度和加更新。应避免先对平方 TD 损失求导，再把 $\delta$ 额外乘入优化器。actor 的熵项还含 $\operatorname{sign}(\delta)$，经本步乘误差后，其即时熵方向与 $|\delta|$ 成比例；这与直接加一个固定权重熵梯度不同。

**算法：对应固定提交的 2026 stream_ac_continuous.py 与 optimizer.py。两网络独立；共享 trunk 时需另行规定合并梯度和写入时刻。**

1. 初始化独立 actor、critic，以及各自的资格与尺度统计
1. 获取并因果归一化当前观察；每条真实转移：
  1. 用当前 actor 采样动作并执行，收到新观察、原始奖励、terminated、truncated
  1. 更新新观察与奖励的运行统计，得到本次学习输入
  1. 用旧 critic 计算当前值和后继值；只有 terminated 才将后继值置零
  1. 保存同一个 δ；求当前价值梯度和含 sign(δ) 熵项的 actor 输出梯度
  1. actor 与 critic 各自积累资格、计算完整增量和尺度，分别写入参数
  1. terminated 或 truncated 时，完成更新后清迹，再重置环境
  1. 否则沿新观察继续；下一动作由已经更新的 actor 采样

其中 δ 在任何参数写入前已保存；不能先更新 critic，再把新的 δ 给 actor。当前归一化观察沿上一轮传入，作者并未保存原始旧观察再用新统计重新归一化。2026 实现对时间截断保留尾值，但因后续环境重置而清迹；2024 入口把 terminated 与 truncated 合成 done 并一同去掉尾值。这个版本差异会改变学习目标，复现时须与对应环境的时限含义一起检查。

<a id="lesson-current-streamx"></a>

## 5 · 2026 更新控制器：从整体缩放到逐坐标边界

深入阅读：在理解 actor–critic 的逐步循环后，进一步检查逐坐标更新边界及其成立条件。

ObGD 用全网 $\|z\|_1$ 形成一个共同步长。若许多坐标同时具有非零痕迹，网络越大，每个坐标获得的更新可能越小。修订版 StreamingOptimizer 转而对完整增量 $u_t=\delta_t z_t$ 的每个坐标维护衰减最大值：

$$
v_{t,i}=\max\{\beta v_{t-1,i},|u_{t,i}|\},\qquad w_{t+1,i}=w_{t,i}+\alpha\frac{u_{t,i}}{v_{t,i}+\varepsilon}
$$

初始化最大值为零，保留比例满足 $0\leq\beta<1$。这里的 $v_{t,i}$ 是优化器统计，不是状态价值；代码中名为 max_v。更新后若 episode 结束，只清资格迹，保留这个尺度统计。

$$
|\Delta w_{t,i}|=\alpha\frac{|u_{t,i}|}{v_{t,i}+\varepsilon}\leq\alpha\frac{|u_{t,i}|}{|u_{t,i}|+\varepsilon}<\alpha
$$

有限增量、$\alpha>0$ 与 $\varepsilon>0$ 下，单坐标位移严格小于名义步长。这个结论不需要前一节的 TD 梯度尺度假设，但它只约束参数位移，并不保证 TD 误差减小、策略改善或无长期发散。

**算法：StreamingOptimizer 的 TD 形式，对应作者 2026 optimizer.py**

1. 初始化资格迹 $z=0$ 和逐坐标最大值 $v=0$
1. 每条转移：
  1. 用旧参数计算 $\delta$ 和输出梯度 $g$
  1. 更新 $z\leftarrow\gamma\lambda z+g$
  1. 形成完整增量 $u\leftarrow\delta z$
  1. 更新 $v\leftarrow\max(\beta v,|u|)$（逐坐标）
  1. 更新 $w\leftarrow w+\alpha u/(v+\varepsilon)$
  1. 若将开始新 episode，清零 $z$，保留 $v$

手算一个尖峰：上一最大值为 $2$，$\beta=0.9$，新增量为 $100$，则新最大值立即变成 $100$，位移约为 $\alpha$。下一步增量降为 $1$ 时，最大值仍为 $90$，位移约为 $\alpha/90$。所以它立即限制尖峰，而后逐步恢复更新尺度。

2026 的 Stream-RAC 还采用重参数化 actor：通过动作对 soft Q 目标求导，而不是把 TD 误差乘 log-policy 梯度。作者实现每步清 actor trace，critic 则使用 soft next-action target 与资格迹。它仍没有 replay 和 target network，但不能当作 Stream-AC 只换一个函数名。

<a id="lesson-output-steps"></a>

## 6 · Intentional Updates：按输出变化选择步长

深入阅读：从参数位移转向预测输出的改变，推导 Intentional Updates 的局部尺度选择。

Intentional-AC 属于 ICML 2026 的 Intentional Updates for Streaming Reinforcement Learning。其起点是输出变化：已选定参数方向 $d$，希望标量输出 $y(w)$ 改变 $\Delta$。一阶展开给出 $y(w+\alpha d)-y(w)\approx\alpha\nabla y^\top d$，因此可以反解步长。这里不追踪参数对过去步长的敏感度，也不建立元参数的资格迹；“步长依数据变化”与“元梯度学习步长”是不同机制。

$$
\alpha\approx\frac{\Delta}{\nabla y(w)^\top d},\qquad d=\delta\nabla V_w(s),\qquad\Delta=\eta\delta\quad\Longrightarrow\quad\alpha=\frac{\eta}{\|\nabla V_w(s)\|^2}
$$

令 $\delta=r+\gamma V_w(s')-V_w(s)$，这里目标 $r+\gamma V_w(s')$ 在本次更新中固定。在线性 TD(0)、梯度非零且不加裁剪时，当前预测向这个固定目标移动恰好 $\eta\delta$。对非线性网络，它只是局部近似；对重新计算过 bootstrap 的新 TD 误差，也没有相同收缩等式。

例如 $w=0.5$、当前特征 $x=2$、下一特征 $x'=1$、奖励 $r=1.5$、$\gamma=0.9$、$\eta=0.4$。旧目标为 $1.95$，误差为 $0.95$；步长为 $0.4/4=0.1$，更新后 $w^+=0.69$。到旧目标的误差是 $1.95-1.38=0.57=0.6\times0.95$，但重新计算 bootstrap 得到 $1.5+0.9\times0.69-1.38=0.741$。区分两个误差是理解其保证范围的关键。

资格迹使更新方向包含过去梯度，逐坐标缩放又改变了几何。记当前输出梯度为 $g_t$，逐坐标缩放矩阵为 $D_t=\operatorname{diag}(\rho_t)$，资格迹为 $z_t=qz_{t-1}+g_t$，且 $q=\gamma\lambda<1$。对过去预测变化的加权平方和应用 Cauchy–Schwarz，可得到涉及 $z_t^\top D_tz_t$ 与历史梯度范数的尺度。作者实现用偏差校正的指数平均跟踪后者，而不是储存所有过去梯度。

$$
\begin{aligned}\nu_t&=\beta_2\nu_{t-1}+(1-\beta_2)g_t^2,\qquad \rho_t=\left(\sqrt{\nu_t/(1-\beta_2^t)}+\varepsilon\right)^{-1},\\\sigma_t&=g_t^\top D_tg_t,\qquad s_t=qs_{t-1}+(1-q)\sigma_t,\qquad\bar\sigma_t=\frac{s_t}{1-q^t},\\\alpha_t&=\frac{\eta}{\max\{\sqrt{\bar\sigma_t\,z_t^\top D_tz_t},\varepsilon\}},\qquad w^+=w+\alpha_t\widetilde\delta_tD_tz_t.\end{aligned}
$$

平方、开方和倒数均逐坐标进行。$\bar\sigma$ 是平均而非未归一化的历史和，不能将推导中的历史求和与实现直接等同；$D_t$ 还随时间改变，因此历史统计量也是在线代理。分母只确定局部更新尺度，不给任意长期训练轨迹提供收敛保证。

完整实现先用原始 $\delta^2$ 的偏差校正指数平均给出裁剪幅度，得到 $\widetilde\delta$。critic 使用它更新价值；actor 再除以裁剪后误差绝对值的运行平均，以限制奖励尺度的影响。对于无迹 actor，$g=\nabla\log\pi(a\mid s)$ 时，目标是让已采样动作的对数概率变化约为 $\eta\widetilde A/\overline{|A|}$。这既不是精确的全分布 KL trust region，也不保证每个动作概率都只改变固定百分比；动作相关步长还会重加权期望策略梯度。

$$
\begin{aligned}c_t&=\beta_c c_{t-1}+(1-\beta_c)\delta_t^2,\qquad B_t=C\sqrt{c_t/(1-\beta_c^t)},\\\widetilde\delta_t&=\operatorname{clip}(\delta_t,-B_t,B_t),\\a_t&=\beta_a a_{t-1}+(1-\beta_a)|\widetilde\delta_t|,\qquad \widehat A_t=\frac{\widetilde\delta_t}{\max\{a_t/(1-\beta_a^t),10^{-12}\}}.\end{aligned}
$$

critic 使用 $\widetilde\delta_t$，actor 使用 $\widehat A_t$。作者默认 $C=20$、$\beta_c=\beta_a=0.9998$。统计量包含当前误差，并采用裁剪后的误差更新 actor 尺度；交换顺序会得到不同算法。

$$
g_t^{\rm actor}=\nabla_\theta\left[\log\pi_\theta(a_t\mid s_t)+\xi\,\mathcal H(\pi_\theta(\cdot\mid s_t))\,\operatorname{sg}\!\left(\operatorname{sign}(\delta_t)\right)\right]
$$

这是作者 actor 实际送入优化器的梯度。TD 误差由更新前的 critic 计算，符号作为常数，不沿 critic 反传；裁剪与正尺度归一化保持符号，因此也可用处理后优势的符号。该梯度一起进入 RMSProp 统计与资格迹，随后再乘归一化优势。无迹时，符号修正使熵贡献乘上优势的绝对值；有迹时还混合了过去的熵梯度，不能据此断言每次更新都增加当前状态熵。它不同于在最终更新中另加一项不经过资格迹的熵梯度。

**算法：Intentional optimizer 的运算顺序；网络与行为策略由 actor–critic 主循环提供**

1. 用旧 critic 计算 $\delta$，固定它的符号。
1. critic 使用 $g=\nabla V(s)$；actor 使用 $g=\nabla[\log\pi+\xi\mathcal H\,\operatorname{sign}(\delta)]$。
1. 二者有独立优化器状态：
  1. 更新梯度二阶矩，得到 $D$。
  1. 更新资格迹 $z$、当前梯度范数 $\sigma$ 和偏差校正平均 $\bar\sigma$。
  1. 根据 $\sqrt{\bar\sigma\,z^\top Dz}$ 计算步长。
  1. 更新误差裁剪统计，得到 $\widetilde\delta$。
  1. actor 额外更新 $|\widetilde\delta|$ 的尺度统计并归一化。
  1. 执行参数更新；若是真正 episode 边界，在本次更新后清空资格迹。

作者默认归一化顺序的标准库实现，以及能精确分析的未归一化 TD(0) 特例。

```python
@dataclass
class IntentionalStep:
    """Vector optimizer matching the full author's default normalization order.

    The caller supplies +grad V for the critic. For the author's actor use
    +grad_theta[log pi(a|s) + xi*H(pi(.|s))*sign(delta)], with the sign of the
    pre-update critic's delta held constant. Entropy enters the trace and the
    RMSProp statistics with the score gradient; it is not added after the step.
    Clipping and positive-scale normalization preserve delta's sign. Omitting
    this sign reverses the immediate entropy contribution for negative delta.
    With nonzero traces, past entropy terms may have different signs; this is
    not a guarantee that each update increases the current state's entropy.
    'policy' additionally normalizes the clipped TD error by its running L1 scale.
    No environment, neural network, or author benchmark is reproduced here.
    """
    size: int
    eta: float = 0.5
    q: float = 0.72  # gamma*lambda, must be in [0, 1)
    beta2: float = 0.999
    beta_clip: float = 0.9998
    beta_norm: float = 0.9998
    clip_mult: float = 20.0
    policy: bool = False
    t: int = 0
    sigma: float = 0.0
    delta_sq: float = 0.0
    delta_abs: float = 0.0
    v: list = field(default_factory=list)
    e: list = field(default_factory=list)

    def __post_init__(self):
        if self.size < 1 or not 0 <= self.q < 1:
            raise ValueError('positive size and 0 <= q < 1 required')
        if any(not 0 <= b < 1 for b in (self.beta2, self.beta_clip, self.beta_norm)):
            raise ValueError('EMA decays must lie in [0,1)')
        self.v, self.e = [0.0] * self.size, [0.0] * self.size

    def step(self, g, delta, reset=False):
        if len(g) != self.size or not all(math.isfinite(x) for x in [*g, delta]):
            raise ValueError('finite gradient with declared dimension required')
        self.t += 1
        self.v = [self.beta2 * v + (1 - self.beta2) * gi * gi
                  for v, gi in zip(self.v, g)]
        rho = [1 / (math.sqrt(v / (1 - self.beta2 ** self.t)) + 1e-8)
               for v in self.v]
        sigma_now = sum(gi * gi * ri for gi, ri in zip(g, rho))
        self.e = [self.q * e + gi for e, gi in zip(self.e, g)]
        self.sigma = self.q * self.sigma + (1 - self.q) * sigma_now
        sigma_bc = self.sigma / (1 - self.q ** self.t)
        trace_norm = sum(e * e * ri for e, ri in zip(self.e, rho))
        alpha = self.eta / max(math.sqrt(sigma_bc * trace_norm), 1e-8)
        self.delta_sq = self.beta_clip * self.delta_sq + (1 - self.beta_clip) * delta**2
        cap = self.clip_mult * math.sqrt(self.delta_sq / (1 - self.beta_clip ** self.t))
        safe = math.copysign(min(abs(delta), cap), delta)
        if self.policy:
            self.delta_abs = self.beta_norm * self.delta_abs + (1 - self.beta_norm) * abs(safe)
            norm = self.delta_abs / (1 - self.beta_norm ** self.t)
            safe /= max(norm, 1e-12)
        update = [alpha * safe * ri * e for ri, e in zip(rho, self.e)]
        if reset:
            self.e = [0.0] * self.size
        return dict(update=update, alpha=alpha, safe_delta=safe, sigma=sigma_bc)


def intentional_td0(w, x, reward, next_x, gamma, eta):
    """Unpreconditioned, unclipped TD(0): freeze the old bootstrap target."""
    target = reward + gamma * dot(w, next_x)
    delta = target - dot(w, x)
    norm_sq = dot(x, x)
    if norm_sq == 0:
        return list(w), target, delta
    new_w = [wi + eta * delta * xi / norm_sq for wi, xi in zip(w, x)]
    return new_w, target, delta
```

偏差校正中的 $t$ 从优化器第一次更新开始计数，初始二阶矩、误差统计和梯度范数统计均为零。episode 结束时，在完成本次更新之后清空 $z$；其余幅度统计与更新计数保留。actor 与 critic 分别维护这些状态，不能共用一个尺度估计。

比较这些方法时，应保持网络、资格迹、奖励处理和交互预算相同：固定步长提供基线；Metatrace 增加对学习过程的敏感度与元信用分配；Intentional updates 改变每步输出尺度；Stream-X 则是一套为 streaming 深度控制设计的组件组合。任何一条的正结果都不足以单独判定其他组件无关。

<a id="experiment-extended-intentional"></a>

### 实验：实验 · 局部输出尺度控制，不是全程策略稳定性的证书

critic 与 actor 都控制更新尺度之后，已经学到的贪心行为是否一定保持？

**环境与可用信息。** 五位置链从最左端开始，每步可左移或右移。到达最右端奖励 1，其余奖励 −0.02；到达终点或用完 12 步都是真实终止。输入包含位置 one-hot 与剩余步数。critic 线性、策略为线性 logits 的 softmax，初始参数全零。

**设置。** 五种子各 1200 个转移。Intentional 使用 γ=1、λ=0.8，value η=0.2、policy η=0.03，保留 RMS、误差裁剪、策略误差归一化和熵系数 0.01。对照为 α=0.03 的固定步长 AC(λ)，critic 与 actor 组合梯度中策略项权重为 0.5。

**检验的机制。** 两个 Intentional 优化器读取同一个旧 TD 误差，再分别根据梯度尺度和迹选择增量。它是多项更新规则的组合；与固定 AC 的区别不只有单一标量步长。因此这里比较整组组件，不能把结果只归因于其中一个归一化。

**测量。** 主图冻结当前 logits，执行确定性 argmax 策略并累加原始奖励。四步到达终点得 0.94；一直未到终点得 −0.24。它不是训练中的随机策略期望回报。应同时读原始日志中的 entropy、value_step 和 policy_step；单独的步长系数也不等于实际参数增量。

```bash
python3 implementations/extended_adaptation/intentional.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/extended-intentional/curves.svg)

横轴：environment_steps。纵轴：frozen_greedy_return。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 60 步两组五个种子的贪心回报都是 0.94。第 1200 步，Intentional 有 3 条仍为 0.94、2 条退为 −0.24，均值 0.468；固定 AC 五条均为 0.94。该配置展示了“先成功、后退化”，不能用早期成功替代全程评价。

**结论边界。** 这是线性离散控制组件，不是原论文连续控制或 Atari 实验。贪心指标很离散，没有计算随机策略精确价值；也未完成熵、归一化和超参数的因果消融。

**继续实验。** 首先增加冻结随机策略的精确评价，区分 argmax 跳变与随机行为真正恶化。再逐项消融熵项、策略误差归一化和 η，记录实际输出变化，不能仅比较 policy_step 的大小。

[源码](../implementations/extended_adaptation/intentional.py) · [逐种子记录](https://yingwen.io/crl-code/results/extended-intentional/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/extended-intentional/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/extended-intentional/curves.json)

<a id="lesson-example"></a>

## 7 · 两步链的流式执行

只保存资格迹，终点奖励究竟还能改到哪里？再把同一条轨迹交给一个可以重放的学习器，分别数它们读了几次数据、做了几次更新。下图把新经验固定为完全相同的两条，避免把不同采样结果混进这个比较。

状态特征为 $x_0=(1,0)$、$x_1=(0,1)$；转移 $s_0\to s_1$ 的奖励为 0，$s_1\to\mathrm{terminal}$ 的奖励为 1。初始化 $w=z=(0,0)$，取 $\gamma=0.9$、$\lambda=0.8$、$\alpha=0.1$。第一步 $\delta_0=0$、$z_0=(1,0)$，权重不变。第二步 $\gamma_{\rm next}=0$，但 $\gamma_{\rm current}=0.9$，因此 $z_1=(0.72,1)$、$\delta_1=1$；最终 $w=(0.072,0.1)$。

![A到B到终点G：资格迹使同一个终点误差点亮A、B；重放沿旧经验卡回到A；批更新先堆叠两条样本再取平均。](https://yingwen.io/crl-figures/concept-streaming-memory.svg)

节点下方是最终对应权重。资格迹直接分配当前误差；重放先更新 B，再用旧 A→B 转移计算新误差；批均值在同一旧参数下计算两条 TD(0) 增量后平均。三种协议的更新次数不同，初值、奖励、折扣和名义步长相同。原创精确算例。

若 $\lambda=0$，同一趟只更新第二个状态，得到 $w=(0,0.1)$；起点要等下次访问才利用新的后继预测。资格迹使用同一个 $\delta$ 乘保存的历史敏感性，将一次误差信号分配给过去，无需重放终点样本。

允许 replay 的 TD(0) 学习器，可以在第二步以后重新读取第一条转移。这次目标已经变成 $0+0.9\times0.1$，所以第三次更新的误差是 $0.09$，最终得到 $w=(0.009,0.1)$。它利用的是重新计算的旧样本目标。若改为等两条都到达，再用初值 $w=0$ 计算两项 TD(0) 增量并取平均，则一次批更新得到 $w=(0,0.05)$。三种答案分别来自不同的运算顺序。

“每条经验只更新一次”数的是学习操作，并没有规定只能改一个权重。图中第二个误差在一次向量更新里改变两个参数。参数共享还能让当前特征本身涉及多个坐标：把第二个状态的特征改成 $(1,1)$，TD(0) 就得到 $(0.1,0.1)$；带迹版本的最终资格为 $(1.72,1)$，权重为 $(0.172,0.1)$。此时第二个状态的预测为 $0.272$。其中 $0.1$ 的第一坐标更新来自当前特征，额外的 $0.072$ 才来自历史资格。

流式学习器在两步之间保留权重和资格，各两个数；当前转移处理完便释放。到了终点，先完成这次更新，再清空资格，已学权重留到下一回合。运行均值、方差与优化器尺度也可继续保留，是否重置由算法逐项规定。只说“重置 agent”会把这些不同操作混在一起。

两种常见错误可以直接手算定位：在终点更新前清零资格，会丢掉起点的 0.072；把当前终止转移的零折扣用于衰减过去资格，也会得到相同错误。配套测试把到达当前状态的折扣与指向下一状态的折扣分别传入。

<a id="lesson-learner-state"></a>

## 8 · 保存和恢复的是哪一个学习器？

两步链的特征可以直接从状态编号取出，因此保存权重和资格迹已经足够执行那个例子。若特征还依赖已经发生的观察，情况便不同了。考虑固定策略产生的三条转移 A→B→C→终点，奖励依次为 2、0、1；到达 A、B、C 时的观察依次为 1、0、0。下面所有恢复对照都接收这条相同的数据流。

活动状态按 $h_t=\tfrac12h_{t-1}+o_t$ 递推，取 $h_{-1}=0$，因此活动依次为 $1,0.5,0.25$。预测是 $\hat v_t=w_th_t$。这里递推系数固定，只学习读出权重 $w$，所以当前输出对 $w$ 的梯度为 $h_t$；并未训练递归网络，也不需要计算活动对递归参数的敏感度。活动 $h$ 决定本次预测使用什么特征，资格迹 $z$ 决定当前误差分给哪些过去的读出梯度。

把这些量连同尺度统计合记为学习器状态 $L_t$，规则可写为 $L_{t+1}=F_\eta(L_t,\xi_t)$；$\xi_t$ 是本次到来的经验，$\eta$ 控制规则。例如 $\eta$ 可以决定步长。若在给定经验流上保持同一个 $\eta$，在规则的可微处，对学习过程求导需要 $H_t=\partial L_t/\partial\eta$，并递推 $H_{t+1}=(\partial F_\eta/\partial L_t)H_t+\partial F_\eta/\partial\eta$。这与累积输出梯度的资格 $z_t$ 是不同导数；仅让分母 $b_t$ 随数据变化没有执行这个递推。非线性学习器常需近似或截断该敏感度，真实控制还要考虑行为改变采样的路径，见[元学习章](meta.md)。本节恢复实验仅继续原有更新规则。

沿用逐坐标最大值更新器的一维形式，记尺度统计为 $b$，对应代码中的 max_v。取 $\gamma=0.9$、$\lambda=0.8$、$\alpha=0.1$、$\beta=0.9$，并取方便手算的 $\varepsilon=0.1$。这组数值服务于本节算例，不是作者默认配置。所有预测使用更新前的同一个 $w_t$，终点预测设为零。

$$
\begin{aligned}h_{t+1}&=\tfrac12h_t+o_{t+1}\quad\text{（非终止转移）},\\\delta_t&=R_{t+1}+\gamma_{t+1}w_th_{t+1}-w_th_t,\\z_t&=\gamma_t\lambda z_{t-1}+h_t,\qquad u_t=\delta_tz_t,\\b_t&=\max\{\beta b_{t-1},|u_t|\},\qquad w_{t+1}=w_t+\alpha\frac{u_t}{b_t+\varepsilon}.\end{aligned}
$$

初始化 $w_0=z_{-1}=b_{-1}=0$。$\gamma_t$ 衰减进入当前状态以前的资格，$\gamma_{t+1}$ 控制本次 bootstrap；第一步取 $\gamma_0=0$，终止转移取 $\gamma_3=0$。$u$ 是尚未缩放的完整增量，最终参数位移还需经过 $b$ 与 $\alpha$。

第一条转移有 $\delta_0=2$、$z_0=1$、$u_0=2$、$b_0=2$，所以 $w_1=0.1\times2/(2+0.1)=2/21$。到达 B 后活动为 $h_1=0.5$。若此时暂停，需保存的学习器状态是 $w_1=2/21$、$h_1=0.5$、$z_0=1$、$b_0=2$，还要记录进入 B 的折扣、已完成一次更新及下一条数据的位置。恢复完整状态后，读入 B→C：

$$
\begin{aligned}\delta_1&=0+0.9\frac{2}{21}\frac14-\frac{2}{21}\frac12=-\frac{11}{420},\\z_1&=0.9\times0.8\times1+0.5=1.22,\\u_1&=-\frac{11}{420}\times\frac{61}{50}=-\frac{671}{21000},\\b_1&=\max\{1.8,|u_1|\}=1.8,\\\Delta w_1&=\frac{0.1u_1}{1.8+0.1}=-\frac{671}{399000},\qquad w_2\approx0.093556391.\end{aligned}
$$

这次更新不再读取 A→B。A 的影响通过活动、资格和尺度统计进入三个不同的运算位置；它们不能从权重一个数中还原。

![A、B、C的活动随观察衰减；在B保存相同权重，完整恢复或分别清空资格迹、尺度统计、活动后，B到C的下一次位移不同。](../assets/crl-figures/streaming-state-checkpoint.svg)

上部圆点大小辅助表示活动 h，紫框标出第一条转移更新后在 B 的暂停位置。下部四行共用权重轴：空圆是同一个旧权重，方块是第二次更新后的权重，箭头是这一次位移。活动清零时两标记重合。数值由附带脚本精确复算；这是原创有限机制算例。

| 在 B 处改变什么 | 第二次更新中首先改变的量 | 更新后的 w |
| --- | --- | --- |
| 完整恢复 | 沿上式得到 δ=−11/420、z=1.22、b=1.8 | 0.093556391 |
| 只清资格迹 z | δ 相同；z 从 1.22 变为 0.5 | 0.094548872 |
| 只清尺度统计 b | δ、z 相同；b 从 1.8 变为 0.031952381 | 0.071023010 |
| 只清活动 h | 当前及下一活动均为 0；δ=0，仍有历史资格 z=0.72 | 0.095238095 |
| 只清权重 w | 活动、资格与尺度保留；两个预测均为 0，δ=0 | 0 |

只清资格迹减少了历史梯度参与当前误差的份额；只清尺度统计则放大同一个 $u_1$ 的位移。只清活动时，第二步的误差变成零，却仍留有此前的资格。第三步终点奖励到来后，衰减后的历史资格仍能更新权重。若只加载权重文件，并把 $h,z,b$ 都从零启动，本例余下两步没有非零活动或资格，最终权重停在 $2/21$；完整恢复则最终得到约 $0.157626609$。预测参数相同不保证后续学习过程相同。

终点 C 的奖励到来时，完整分支仍要先用 $z_2=0.72\times1.22+0.25=1.1284$ 更新。完成这次更新后，本例清空活动和资格，让下一回合从新观察构造活动；权重与尺度 $b_2=1.62$ 保留，更新计数继续增长。在终点更新前清迹，会把本次使用的资格错误地降到 $0.25$。重置环境、重置活动、清资格和重新初始化参数必须分别规定；只写 reset 没有指明执行了哪一个操作。

这个小更新器的计数 k 只记录完成了几次更新，不进入分母。Adam 与上文 Intentional optimizer 的偏差校正还使用更新次数；恢复它们时，矩统计和计数必须保持对应。在线归一化也要保存均值、方差、样本数等递推状态。活动状态通常只跨连续轨迹保留，是否在任务切换时清空，则由实际的信息连续性与算法定义决定。

恢复还需要衔接同一个外部过程。本例的输入已给定，保存 cursor=1 就能从 B→C 接着读；真实控制中，还需对应的环境或设备状态、行为随机数状态和更新配置。若设备在暂停期间继续移动，加载学习器文件不会把外部世界带回 B。接下来的经验已经改变，应按实际继续运行的协议评价。

同一三条数据还可以单独检查 replay 权限：固定这些活动特征，暂用固定 $\alpha=0.1$ 的 TD(0)，逐条一次更新得到 $w=0.221017188$。若保存三条特征转移，随后重读第一条 A→B，新误差为 $2+(0.9\times0.5-1)w=1.878440547$，第四次更新得到 $w=0.408861242$。环境仍只产生三条转移，学习器多做了一次旧数据更新。保存供抽取的旧特征转移也是 replay；特征固定时可重算这个目标，若编码器可学习，旧特征还没有提供用新编码器重算原始观察的权限。

中途保存一个活动与统计摘要，是让原更新过程继续；从保存的旧转移重新形成目标，则增加了经验使用权限。本节据此检查学习器在给定协议下执行了什么。信用分配章进一步讨论误差应影响哪些过去的量，元学习章进一步讨论怎样改变学习规则。完整学习器的表现还要回到[生命期评价](objectives.md#lesson-maze-evaluation)，观察这些更新怎样改变后续行动和真实奖励；[带噪声的完整生命期算例](objectives.md#lesson-noisy-lifetime)进一步比较在线学习期间的所得与最后冻结状态的得分。

<a id="lesson-components"></a>

## 9 · 同一条经验流中的预测、技能与长期目标

现在让机器人在同一次行动后学习多个对象：外部奖励价值、到达充电点的预测，以及当前技能的后果。真实经验仍只到来一次。各模块从这次经验形成自己的目标，随后释放原始转移；是否符合预算取决于模块总数、各自的状态与计算成本。一个模块的更新次数不能代表整个智能体的成本。

对于第 $j$ 个 [GVF](gvf.md)，先给定 cumulant $C^j$、continuation $\gamma^j$ 和目标策略 $\pi_j$，再从同一行为 $b$ 的样本形成 $\delta_t^j=C_{t+1}^j+\gamma_{t+1}^j\hat v_j(S_{t+1};W_t)-\hat v_j(S_t;W_t)$。普通 IS 半梯度例使用各自的 $z_t^j=\rho_t^j[\gamma_t^j\lambda_jz_{t-1}^j+\nabla_W\hat v_j(S_t;W_t)]$。若共享网络，先用同一个旧 $W_t$ 算齐各方向，固定步长时再写入 $\Delta W_t=\sum_j\alpha_j\delta_t^jz_t^j$。这正是已有共享 GVF 小实现的时序，不等于具有线性 GTD 保证的神经算法。某个预测在充电点令 continuation 为零，只终止它自己的累计，其他预测与真实控制仍然继续。

[Option](options.md) 则改变高层决策时刻。固定技能执行期间，可只保留累计奖励 $C=0$、累计折扣 $\Gamma=1$ 与时长 $\tau=0$；每条原始转移依次执行 $C\leftarrow C+\Gamma R$、$\Gamma\leftarrow\Gamma\gamma_{\rm next}$、$\tau\leftarrow\tau+1$。技能停止时，用 $C+\Gamma\sum_{o'}\mu(o'\mid S')Q(S',o')$ 评价给定高层策略 $\mu$。技能停止允许继续选择，只有外部问题真的终止才去掉尾值。累计摘要不要求保存整段转移，但完整段 SMDP 更新等待技能结束；若协议要求每个原始步就修正高层估计，还需 intra-option 备份。学习内部策略、停止规则和技能模型又各有自己的更新。

若任务按[平均奖励](average.md)评价，令奖励率估计为 $\hat g_t$，差分价值为 $\hat h$。Differential TD 使用 $\delta_t=R_{t+1}-\hat g_t+\hat h(S_{t+1};w_t)-\hat h(S_t;w_t)$，价值与奖励率都用同一个旧误差更新。对持续 $\tau$ 步的完整技能，冻结一个奖励率参照 $g$ 才得到段目标 $\sum_{k=1}^{\tau}R_{t+k}-g\tau+\hat h(S_{t+\tau})$；奖励单位是每原始时间步，不能用一次技能调用替代 $\tau$。若逐步记录时用的是不断变化的 $\hat g_{t+k-1}$，累计的是另一组中心化奖励，不能不加说明地当作最后一个奖励率下的段目标。

这时目标的 continuation 可为 1，仍可选 $\lambda<1$ 使资格衰减；$\lambda$ 控制信用范围，不重新定义长期奖励率。冻结 gain 与价值时可建立有限前向关系，在线同时变化时再分析其偏差。把折扣 Stream-AC 的 $\gamma$ 直接改成 1，既没有指定奖励率学习器，也没有解决差分价值的加性规范；完整推导由平均奖励章承担。

这些接口最后要落实为下一次行动的读取规则：哪些预测进入 actor 输入，哪些技能价值影响高层选择，哪些后果模型进入规划？模型产生的额外备份不增加真实经验，却占用学习时钟；因此还要明确协议是否允许固定次数的模型计算。共享表示更新也会同时改变预测、策略与模型的输入，冻结一个 head 并未冻结其完整函数。[持续智能体架构](architectures.md)沿同一交互安排这些写入与读取，并以实际外部收益检查组合结果。

<a id="lesson-code"></a>

## 10 · 机制实验与作者实现

下载下方脚本后运行；仅标准库。分数算术逐项检查中途恢复、单状态重置、终点时序与 replay。

```bash
python3 tutorials/streaming_state_walkthrough.py
python3 tutorials/streaming_state_walkthrough.py --test
```

在站点源码目录运行：JavaScript 独立复算并核对图片。

```bash
node --test tests/crl-streaming-state-walkthrough.test.mjs
node scripts/generate-crl-streaming-state-walkthrough.mjs --check
```

可下载[状态过程脚本](../tutorials/streaming_state_walkthrough.py)与[图的计算模块](https://yingwen.io/crl-code/figures/streaming-state-walkthrough.mjs)。Python 使用 Fraction 保存精确分数，输出逐条预测误差、资格、未缩放增量、尺度与最终位移；保存位置只从后续转移继续读取。图中保留诊断记录是为了核对过程，这些记录没有再次作为严格流式分支的更新输入。脚本没有动作学习、可训练递归网络或深度控制训练。

**算法：在站点源码目录运行：独立检查两步链、共享特征、批平均、重放与双时钟，再核对 SVG 数据**

1. node --test tests/crl-concept-streaming.test.mjs
1. node scripts/generate-crl-concept-streaming.mjs --check

两幅图的[计算模块](https://yingwen.io/crl-code/figures/concept-streaming.mjs)保留了完整的更新、样本保存与时间记录。运行它可以逐项重算图中的权重和完成时刻；样本是否保留、是否再次参与更新，分别由对应协议规定。

线性 TD(λ)、Welford 与 2024 ObGD 的可执行机制实验。

```python
@dataclass
class Welford:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def observe(self, x):
        self.count += 1
        old_delta = x - self.mean
        self.mean += old_delta / self.count
        self.m2 += old_delta * (x - self.mean)
        return self.mean, self.variance

    @property
    def variance(self):
        return self.m2 / (self.count - 1) if self.count > 1 else 1.0


def td_lambda_step(w, trace, x, xp, reward, gamma_current,
                   gamma_next, lam=0.8, alpha=0.1, kappa=None):
    """Linear TD: gamma_current decays past trace; gamma_next bootstraps.
    kappa enables the ObGD algebra, NOT a general nonlinear safety theorem.
    """
    if not (len(w) == len(trace) == len(x) == len(xp)):
        raise ValueError("dimension mismatch")
    v = sum(a*b for a, b in zip(w, x))
    vp = sum(a*b for a, b in zip(w, xp))
    error = reward + gamma_next * vp - v
    z = [gamma_current * lam * old + feature
         for old, feature in zip(trace, x)]
    step = alpha
    if kappa is not None:
        mass = alpha * kappa * max(abs(error), 1.0) * sum(map(abs, z))
        step = alpha / max(1.0, mass)  # defined also when z == 0
    return [a + step * error * b for a, b in zip(w, z)], z, error, step


def streaming_demo():
    outcomes = {}
    for lam in (0.0, 0.8):
        w, z, _, _ = td_lambda_step([0., 0.], [0., 0.], [1., 0.],
                                    [0., 1.], 0, 0, 0.9, lam)
        w, z, _, _ = td_lambda_step(w, z, [0., 1.], [0., 0.],
                                    1, 0.9, 0, lam)
        outcomes[str(lam)] = [round(v, 6) for v in w]
    stats = Welford()
    for x in (1., 2., 3.):
        stats.observe(x)
    bounded = td_lambda_step([0.], [0.], [10.], [0.], 1, 0, 0,
                             alpha=0.1, kappa=2)
    print("streaming", {"two_step_credit": outcomes,
          "mean_variance": (stats.mean, stats.variance),
          "obgd_illustration_weight": bounded[0],
          "warning": "gradient 10 violates the <=1 heuristic bound"})
```

仅标准库；包含 ObGD 尺度条件不满足时的过冲反例

```sh
python examples/lifelong_algorithms_lab.py streaming
python examples/lifelong_algorithms_lab.py test
```

输出中 $\lambda=0$ 为 $[0,0.1]$，$\lambda=0.8$ 为 $[0.072,0.1]$；输入 $1,2,3$ 的均值为 $2$、样本方差为 $1$。尺度反例的更新后权重为 $0.5$。下载脚本覆盖这些机制；深度控制实验使用下方作者实现。2024 分支对应 ObGD，2026 分支对应 StreamingOptimizer，二者的配置不能混用。

Intentional 输出尺度实验及回归测试；仅标准库

```sh
python3 examples/meta_frontier_lab.py intentional
python3 examples/meta_frontier_lab.py test
```

IntentionalStep 实现梯度缩放、资格迹、误差裁剪和 actor 误差归一化。它接收网络梯度与旧 TD 误差，不包含神经网络和环境；无迹线性特例用于检查输出变化，带迹与熵项的测试用于检查更新顺序。完整控制结果需使用下方作者训练入口。

要追踪实际非线性执行，可打开[六步链神经 TD(λ)](../implementations/streaming_composition/neural_td_trace.py)及其[网络与资格递推](../implementations/streaming_composition/_common.py)：value_gradient 用旧网络返回当前值与全部参数梯度，旧后继值形成 δ，trace_step 用进入当前状态的折扣更新数值资格，net.add 一次应用完整方向，最后才处理终止。网络为 2–6–1 tanh，学习输入不含真值；全状态解析误差只进入诊断。它没有 actor 或 Stream-X 优化器。共享 GVF、冻结 trunk 与 option 时长对照的范围和入口见[流式组合实现说明](https://yingwen.io/crl-code/streaming-composition.md)。

- 诊断量一：每步 |δ|、trace 的 L1/L2 norm、实际参数更新 norm 和 αused/α，分别看是目标、历史累积还是步长造成尖峰。
- 诊断量二：原始及归一化观测/奖励的分位数，LayerNorm 前后的激活统计；不要只记录平均值。
- 诊断量三：峰值内存、单步 p50/p95/p99 延迟；用严格相同计算预算比较是否允许 replay 的方法。
- 消融顺序：线性 TD0→线性 traces→网络 TD0→network traces→逐个稳定化组件；每次只增加一个新的故障来源。

<a id="lesson-branches"></a>

## 11 · 从局部更新到持续学习问题

深入阅读：把流式协议带入机器人适应等研究，比较经验使用权限、预算和评价窗口。

局部算例解释了一次更新如何完成，机器人研究还要回答改变之后能否恢复。Vitchutripop 等（2026）的模拟实验先用 PPO 得到预训练策略，再经热启动切换到新的身体、环境或目标条件，并观察 150 万步适应。五个随机种子的断腿场景中，AdaptiveObGD 的平均峰值成功率为 96.8%，适应期均值为 74.7%；二者分别回答“曾恢复到多高”和“这段时间总体表现如何”。

这项研究的动作和环境允许回合重置，考察的是一次突变后的适应。它也保留了不成功的情况：推方块任务的带 LayerNorm 方法达到 78.4% 峰值，但适应期均值只有 23.3%，后续性能下降。优化器对照同时改变了信用机制——Adam 用 TD(0)，ObGD 系列用 TD(λ)——所以结果属于整套更新配置。若要解释差异，下一项实验应固定资格迹、网络与预算，再单独改变优化器。

| 实际问题 | 首个可解释基线 | 不应跳过的下一步 |
| --- | --- | --- |
| 短期信用难 | TD(λ) / Sarsa(λ) | λ 与 trace 截断；固定参数前向/后向核对 |
| Off-policy 预测发散 | 重要性比 TD 与梯度 TD 对照 | 覆盖、trace 方差与投影目标；不能只减步长 |
| 输入数值尺度改变 | 在线统计和线性尺度实验 | 统计漂移是否改变目标，是否有未来泄露 |
| 深网每步不稳定 | 完整 Stream-X 组件消融 | 初始化、网络、优化器、归一化作为整体记录 |
| 平均收益目标 | differential error 与独立 rate learner | γ=1 的 trace 管理、平均奖励理论不能照搬折扣保证 |

研究“仅一次真实交互”时，还要把动作延迟带来的环境状态变化纳入协议。若一次复杂更新占用了十个控制周期，却仍只按一条 transition 计时，它与固定时钟世界不是同一问题。

<a id="lesson-check"></a>

## 12 · 习题与答案线索

- 为什么不把资格迹等同于 Adam 的一阶矩？答：痕迹积累的是输出梯度；当前 δ 到来时才分配误差。Adam 的矩积累的是各个时刻已经乘各自误差的 loss gradient。
- 为什么 terminal 的 γnext=0 不应清掉本步全部历史信用？答：terminal 奖励属于刚结束的这一串决策，仍应沿到达当前状态的历史痕迹传播。
- 为何一个方法没有 buffer，却可能不满足给定预算？答：固定架构 RTRL 不随生命期扩张，但大敏感性矩阵仍可能超出设备预算；长历史重算或无界模型增长还可能随生命期增加成本。
- 能否用归一化证明长期稳定？答：归一化只控制部分尺度，目标漂移、off-policy 投影、非线性与策略反馈仍然存在。

动手题：在两步链加入第三个延迟状态，手算起点更新为 $α(γλ)^2$；再将特征整体乘 10，比较固定 α、按平方缩放 α 与 ObGD。将同一样本误差改变量和长期预测误差分开画，观察它们何时不一致。

恢复题：B 处只清活动与只加载权重，第二次更新都保持原权重，为什么第三次更新不同？答：前者还保存资格，终点误差仍可沿历史资格更新；后者资格也清零。再把 max_v 保留而清零计数 k，本例会改变参数位移吗？答：不会，因为本例没有偏差校正；换为使用计数的优化器时需重新计算。

接口题：机器人走到充电点，能耗 GVF 停止累计，当前 option 停止，环境继续运行。哪些尾值应置零？答：该 GVF 用零 continuation；高层技能价值仍保留下一次选择的尾值；外部持续控制没有因这两个内部边界而终止。再检查下一步哪些资格需要清除，并说明规则来自哪个预测或控制目标。

## 本章的实验设计

同时记录交互次数、每步延迟和持久内存。均值延迟可能掩盖超时步骤。

设定：一条连续数据流逐步到达，限定每步计算和持久内存。改变未来后缀、特征尺度和控制周期，检查学习前缀及超时行为。

- 未来奖励与标签改变不影响此前已完成的更新。
- 归一化只使用协议允许的当前和过去信息。
- 声明的数据重访次数、状态大小及 fallback 与日志一致。

对照：调好的固定步长与简单尺度控制；Stream-X/Intentional 组件的同基座消融；同交互与同控制时钟分别比较

记录：每步延迟分布和超时率；持久状态大小、样本读取与梯度次数；完整生命期收益和变化后恢复

[具体测试规程](../docs/experiment-handbook.md#handbook-streaming)

## 学习与研究衔接

严格流式学习限制经验到达后怎样再次使用。活动、资格迹和优化器统计可以留存，但都占用资源；保存完整序列以重算新状态则需要另一种数据权限。模型因技能学习而失准时，有限统计量未必足以修复，可能必须重新执行技能。因而流式方法既要控制每次更新，也要检验失去回放后重新获得信息的成本。

[分册导读](../docs/learning-route-continual-rl.md) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-streaming) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=streaming) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=streaming)

## 从本章进入实践

[估计与适应](https://yingwen.io/zh/continual-rl/code/#practice-tracking)：旧经验越来越多时，怎样仍对新变化作出反应？

[策略梯度与控制](https://yingwen.io/zh/continual-rl/code/#practice-policy-control)：优化器确实降低了损失，为什么行动仍可能变差？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

给定状态后可以估计价值；观测不足时，还要学习保留哪些历史。GVF 规定预测什么，RTRL 计算递归敏感度，资格迹组织时间信用。应分别检验信息是否进入状态、反馈能否教会这种保留，以及有限预测预算怎样分配，而不是把三者当作替代算法。

- [Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks](https://yingwen.io/zh/continual-rl/research/#recent-columnar-constructive-networks)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Streaming Deep Reinforcement Learning Finally Works](https://yingwen.io/zh/continual-rl/research/#recent-stream-x)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)
- [Deep Reinforcement Learning with Gradient Eligibility Traces](https://yingwen.io/zh/continual-rl/research/#recent-deep-gradient-eligibility-traces)
- [Real-Time Recurrent Learning using Trace Units in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-real-time-trace-units)
- [Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning](https://yingwen.io/zh/continual-rl/research/#recent-streaming-rtu-rtrl-2026)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)
- [An Idiosyncrasy of Time-discretization in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-openmind-time-discretization)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Extending Differential Temporal Difference Methods for Episodic Problems](https://yingwen.io/zh/continual-rl/research/#recent-openmind-episodic-differential)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Step-size Optimization for Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-step-size-optimization)
- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters](https://yingwen.io/zh/continual-rl/research/#recent-openmind-metaoptimize)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Revisiting Adam for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-revisiting-streaming-adam)
- [Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots](https://yingwen.io/zh/continual-rl/research/#recent-openmind-physical-atari)
- [The Open Ant: A Robot Platform for Reinforcement Learning Research](https://yingwen.io/zh/continual-rl/research/#recent-openmind-ant-platform)

### Scalable Real-Time Recurrent Learning Using Columnar-Constructive Networks

Khurram Javed, Haseeb Shah, Richard S. Sutton, Martha White

JMLR 24 · 2023 · 支持方法与理论

#### 研究问题

如果每次观测只处理一次，如何学习包含历史信息的状态，而不保存一段序列做反向传播？

#### 关键机制

一般递归网络的实时递归学习需要维护庞大的参数—状态敏感度。CCN 限制列之间的递归依赖，并逐步构造新特征，使敏感度可以局部计算。它通过改变网络结构和构造过程降低求导成本，而不是把任意稠密 RNN 的完整导数免费变小。

#### 证据

论文分析受限结构的计算性质，并在动物学习启发的预测问题和 Atari 策略评价中检验预测效率。这里的 Atari 结果主要是预测已有策略的回报，不等于从头训练完整控制智能体。

#### 条件与限制

结构约束、构造顺序和被冻结的旧特征共同限制函数类。监督预测和策略评价上的优势，还需要在会主动改变数据分布的控制闭环中检验。

#### 阅读与实验

先写出递归状态对参数的敏感度递推，再检查哪些跨列项被结构消除。比较 CCN、截断 BPTT 与 RTU 时，同时计入状态、梯度缓存和每步计算。

#### 原文与相关入口

- [JMLR 原文与论文入口](https://www.jmlr.org/papers/v24/23-0367.html)：从网络结构、敏感度传播与预测实验三部分阅读。

### Real-Time Recurrent Learning using Trace Units in Reinforcement Learning

Esraa Elelimy, Adam White, Michael Bowling, Martha White

NeurIPS 2024 · 2024 · 支持方法与理论

#### 研究问题

递归状态既要保存长时信息，又要在在线强化学习中以可控成本更新，怎样设计其递归结构？

#### 关键机制

RTU 使用有结构的递归连接，并维护状态关于参数的在线敏感度。复杂的递归动力学可以用实值运算实现。其关键是让状态更新与梯度迹具有相容的计算结构，减少一般 RTRL 的高阶成本；这与仅给 TD 误差加一条资格迹不同。

#### 证据

论文在部分可观测任务中与常见递归网络比较预测与控制表现。作者代码包含 RTU、其他递归基线、实时 actor–critic 以及部分可观测环境配置。

#### 条件与限制

计算优势依赖特定递归参数化，不能外推为任意记忆问题上的表达能力优势。PPO 版本和严格逐步更新版本的经验协议不同，应分别比较。

#### 阅读与实验

在同一部分可观测任务中固定隐状态维度，再比较完整运行内存和每步更新时间。检查 actor、critic 与递归状态的参数更新是否共享同一条敏感度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1e616bde0438cb10cb6adf076ae7d336-Abstract-Conference.html)：结构、在线导数与实验协议。
- [作者代码](https://github.com/esraaelelimy/rtus)：从 src/nets、src/agents 和实验配置追踪递归状态到控制更新。

#### 作者代码

[论文作者维护的实现。](https://github.com/esraaelelimy/rtus)

RTU 网络、实时学习器与论文实验配置。

### Deep Reinforcement Learning with Gradient Eligibility Traces

Esraa Elelimy, Brett Daley, Andrew Patterson, Marlos C. Machado, Adam White, Martha White

RLC 2025 / RLJ · 2025 · 支持方法与理论

#### 研究问题

资格迹怎样与明确的梯度目标结合，而不是直接把线性半梯度规则搬到深度网络？

#### 关键机制

论文从广义投影 Bellman 误差出发构造多步目标，推导带资格迹的梯度学习方法。前向视角连接多步回报与经验重放，后向视角通过递推迹分配信用。目标函数、辅助估计器和迹的更新共同决定算法，不只是选择一个较大的 λ。

#### 证据

作者给出多种算法并在 MuJoCo、MinAtar 等任务中比较。代码同时提供相关梯度算法与实验设置，可以把推导中的量映射到实际更新。

#### 条件与限制

线性 GTD 的收敛条件不能自动赋予非线性实现全局收敛保证。重放版本与流式版本的数据使用预算也不能混为一谈。

#### 阅读与实验

从一段短轨迹分别计算前向多步目标和后向迹。随后对照原代码检查辅助网络、目标与主网络参数使用的是更新前还是更新后的值。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_302.pdf)：目标、算法推导与实验。
- [作者算法库](https://github.com/esraaelelimy/gtd_algos)：论文提供的梯度 TD 与资格迹实现。

#### 作者代码

[原论文链接的作者仓库。](https://github.com/esraaelelimy/gtd_algos)

论文梯度算法、资格迹和实验配置。

### Streaming Deep Reinforcement Learning Finally Works

Mohamed Elsayed, Elena Sorina Lupu, Gautham Vasan, A. Rupam Mahmood

arXiv（2024 首稿；2026 v3） · 2026 · 直接研究持续学习

#### 研究问题

不保存经验重放、不使用目标网络或训练批次时，深度 RL 能否逐步稳定学习？

#### 关键机制

Stream-X 把信号归一化、表示初始化、资格迹和受控更新尺度组织为一组流式学习方法。各组件处理的是不同问题：奖励尺度、激活与梯度传播、延迟信用，以及一次更新造成的输出变化。去掉重放并不意味着这些问题会自动消失。

#### 证据

2026 年第三版扩展到 Atari、控制与机器人等实验，并包含持续变化设置。论文和代码都有过版本变化，比较结果时需要同时标明论文版本和算法实现。

#### 条件与限制

广泛任务上的流式可行性不等于所有非平稳问题都已解决。不能把旧版较弱 Adam 基线推广成对所有流式 Adam 方法的否定；后续研究专门检验了这一点。代码许可证也应独立于本教材许可证处理。

#### 阅读与实验

按归一化、资格迹、更新控制分别做消融，并保持每步算力一致。先验证严格一次使用经验，再研究长期变化，而不是仅把小批量大小改成一。

#### 原文与相关入口

- [2026 年第三版论文](https://arxiv.org/abs/2410.14606v3)：作者名单、任务范围与算法版本以该版为准。
- [作者代码版本](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：固定实现版本，避免把不同年份的更新规则混在一起。

#### 作者代码

[原作者仓库的固定版本。](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)

Stream-X 算法、变换、优化器及实验；使用前阅读仓库许可证。

### Intentional Updates for Streaming Reinforcement Learning

Arsalan Sharifnassab, Mohamed Elsayed, Kris De Asis, A. Rupam Mahmood, Richard S. Sutton

ICML 2026 · 2026 · 支持方法与理论

#### 研究问题

能否先规定本次更新应产生多大作用，再反推合适的参数更新尺度？

#### 关键机制

Intentional 方法以局部线性近似连接参数变化和预测变化。critic 以减少一定比例的 TD 误差为目标，actor 控制策略输出变化的局部代理量；再结合资格迹和逐坐标尺度，求出这一次更新的强度。这是有目标的局部更新控制，不是对长期表现求导的元梯度。

#### 证据

论文给出推导和流式控制比较，ICML 2026 正式论文入口与作者实现均可用。实现将优化器与 actor–critic 交互区分开，便于检查更新时序。

#### 条件与限制

Taylor 近似在大更新时可能失准。采样动作上的对数概率变化不等于精确的全分布 KL 上界；熵项与 TD 误差符号也必须按原算法处理。

#### 阅读与实验

在一次更新前后直接测量预测变化，并与线性估计比较。分别测试正、负 TD 误差和很小梯度的情形，不要只检查参数是否有限。

#### 原文与相关入口

- [ICML 2026 原文](https://proceedings.mlr.press/v306/sharifnassab26a.html)：正式会议版本与更新意图的定义。
- [作者实现](https://github.com/sharifnassab/Intentional_RL)：重点对照 optimizer.py 与 intentional_ac.py。

#### 作者代码

[原论文作者提供的实现。](https://github.com/sharifnassab/Intentional_RL)

Intentional 更新与流式 actor–critic。

### Revisiting Adam for Streaming Reinforcement Learning

Florin Gogianu, Luțu Adrian-Cătălin, Razvan Pascanu

RLC 2026 / RLJ 预会议版 · 2026 · 支持方法与理论

#### 研究问题

流式 RL 的不稳定来自 Adam 本身，还是目标导数、方差与超参数的组合？

#### 关键机制

论文重新分析自适应更新的信噪比，将 Adam 的稳定项与目标导数尺度联系起来，并研究有界导数的回报分布学习及多步更新。它改变的是目标与更新的配合，而非简单沿用批量训练时的默认配置。

#### 证据

作者在大规模 Atari 流式实验中展示了具有竞争力的结果，并重新比较早期流式方法。正式 RLJ 入口收录为 RLC 2026 预会议论文。

#### 条件与限制

主体实验采用经典回合式 Atari 的流式学习协议，不是任意非平稳终生适应的证据。这些结果也不否定归一化、资格迹或更新约束在其他任务中的价值。版本、调参预算和目标分布必须对齐。

#### 阅读与实验

建立二维对照：固定目标换优化器，固定优化器换目标。将调参种子与最终测试分开，再判断改进来自哪一个因素。

#### 原文与相关入口

- [RLC 2026 论文入口](https://rlj.cs.umass.edu/2026/papers/Paper131.html)：会议收录信息与论文。
- [作者预印本](https://arxiv.org/abs/2605.06764)：Adam 尺度分析、回报分布目标与实验协议。

### Step-size Optimization for Continual Learning

Thomas Degris, Khurram Javed, Arsalan Sharifnassab, Yuxin Liu, Richard S. Sutton

arXiv 预印本 · 2024 · 支持方法与理论

#### 研究问题

误差变大时，应该减小步长过滤噪声，还是增大步长追踪真实变化？

#### 关键机制

论文区分梯度归一化与步长优化。IDBD 类方法以 $\alpha_i=\exp(\beta_i)$ 保证步长为正，并用权重对过去步长的敏感度估计改变 $\beta_i$ 是否有利。持续学习中，静止的无关方向适合很小步长，而持续变化的有用方向需要保留追踪能力。

#### 证据

作者用权重翻转和带噪追踪等线性学习问题比较机制，显示相似的误差幅度可以要求相反的步长反应。

#### 条件与限制

这些可分析任务不是深度控制上的普适优越性证据。元步长、近似敏感度与输入尺度仍会影响结果；步长自适应并没有消除全部外部设计参数。

#### 阅读与实验

分别增加观测噪声和目标漂移速度，检查步长是否采取不同反应。若只记录平均误差，就看不到噪声过滤与追踪之间的区别。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2401.17401)：步长优化与归一化的对照实验。

### Learning from experience instead of curated datasets

Oak Lab

Oak Lab 技术博文 · 2026 · 支持方法与理论

#### 研究问题

有用信号稀疏且大量输入是噪声时，在线学习规则如何分配不同方向的更新能力？

#### 关键机制

博文从含稀有有效特征的线性预测问题出发，对比统一步长与 IDBD 的逐权重适应，再展示 NetworkIDBD 在非线性带噪观测中的例子。核心主张是让长期学习效果影响信用和步长分配，而不只依据当前梯度幅度归一化。

#### 证据

公开页面提供受控噪声特征任务和 NoisyMNIST 示例。它们是机制演示，便于理解有效信号密度与输入规模的关系。

#### 条件与限制

该页面不是完整 CRL 控制论文，也未给出可直接复现所有图表的完整代码和算法推导。监督噪声任务的结果不能证明一般 SGD 或所有深度 RL 都无法从经验学习。

#### 阅读与实验

先复现线性噪声特征问题，分开改变有效特征稀疏度与噪声维数。进入控制前，再加入策略改变数据分布这一因素。

#### 原文与相关入口

- [Oak Lab 原始博文](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)：2026 年 7 月 13 日；受控实验、NetworkIDBD 示例与研究动机。

### Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning

Mohamed Elsayed, A. Rupam Mahmood

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

同一网络里，哪些方向应当保护，哪些方向应当获得更强的新学习与扰动？

#### 关键机制

UPGD 用移除权重或特征的反事实损失变化定义效用，并以 Taylor 近似在线估计。平滑、缩放后的效用同时调制梯度与随机扰动，让近期高效用方向变化较小、低效用方向更活跃。

#### 证据

主体证据包括未知边界的非平稳流式监督任务；另外包含长时间 PPO 实验。两类证据应分别理解，不能把监督任务数量写成 RL 任务覆盖。

#### 条件与限制

近期分布上的效用不保证稀有旧知识的重要性；一阶和二阶近似、权重级和特征级版本不同。PPO 仍使用 rollout 与重复更新，不因 optimizer 在线就成为严格流式 RL。

#### 阅读与实验

用可精确消融的小网络检查效用估计，再拆开保护梯度、保护噪声和 weight decay 三种作用；独立报告新学习与旧功能。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。
- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

#### 作者代码

[论文首页明确链接的作者仓库；README 的短实现是一个指定变体。](https://github.com/mohmdelsayed/upgd)

权重/特征效用实验、流式任务及 PPO 实现。

### Streaming Reinforcement Learning under Partial Observability with Real-Time Recurrent Learning

Noah Farr, Aryaman Reddi, Carlo D’Eramo, Jan Peters

arXiv预印本（2026-07-07 v2） · 2026 · 支持方法与理论

#### 研究问题

严格逐步更新的智能体怎样同时学习递归记忆、分配延迟信用并控制计算？

#### 关键机制

将RTU结构的RTRL敏感度接入QRC与流式actor–critic。敏感度给出当前输出对记忆参数的导数，资格迹再组合过去输出的回报信用；两条递推保持分工。

#### 证据

v2在MemoryChain、五项POPGym和masked MuJoCo上报告5-seed结果，另用KMemoryChain比较在线敏感度与当前参数重算参考，并检验Taylor修正。

#### 条件与限制

masked MuJoCo仍落后批量PPO。固定参数精确RTRL不代表在线变参敏感度始终等于当前参数重算；诊断保存整个episode，须计为额外评价资源。尚未确认作者公开代码。

#### 阅读与实验

在相同递归容量下独立改变记忆跨度、回报λ与参数步幅，同时测敏感度误差和回报。诊断改善不能单独当作控制改进证据。

#### 原文与相关入口

- [2026年v2原文](https://arxiv.org/html/2605.24709v2)：方法、5-seed实验、masked MuJoCo负边界与staleness诊断。

### MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters

Arsalan Sharifnassab, Saber Salehkaleybar, Richard S. Sutton

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

怎样根据后来真正发生的损失，评价过去采用的步长，而不预知未来？

#### 关键机制

先写未来损失的元目标，再以历史敏感度建立因果后向更新。优化器状态也进入递推。完整联合敏感度、分块尺度及 Hessian-free 近似承担不同计算代价。

#### 证据

正式论文包含静态图像、语言建模与非平稳 CIFAR100 实验；部分近似可接近精心选择的学习率计划。分块步长不在所有实验中优于标量版本。

#### 条件与限制

这些实验主要检验优化与监督学习，不能自动推出持续 actor–critic 的回报改善。元优化器仍有步长和结构选择；元目标折扣也不是环境回报折扣。

#### 阅读与实验

先在元学习章手算一次步长对后续误差的影响，再对照作者代码辨认内层状态、元状态和被删去的导数路径。

#### 原文与相关入口

- [ICML 正式论文](https://proceedings.mlr.press/v267/sharifnassab25a.html)：正式年份为 2025；2024 是早期预印本年份。
- [定稿推导与实验](https://arxiv.org/html/2402.02342v6)：未来元目标、后向代理及多种计算近似。

#### 作者代码

[正式论文链接的作者仓库。](https://github.com/sabersalehk/MetaOptimize)

原作者提供的优化器组合与实验实现；不同配置不是同一条更新规则。

### Extending Differential Temporal Difference Methods for Episodic Problems

Kris De Asis, Mohamed Elsayed, J. He

RLC 2026 / RLJ · 2026 · 支持方法与理论

#### 研究问题

中心化怎样加速回合任务的学习，又不因为回合长度不同而改变目标？

#### 关键机制

处理终止边界的中心化尾项，再用共享价值偏置重参数化。区分奖励单位的中心与价值单位的偏置，使最终终止的无折扣回合也能使用适当形式。

#### 证据

论文分别研究固定中心化的策略不变性、在线偏置学习的线性 TD 分析，以及流式深度实验。教材给出遗漏终止补偿导致排序翻转的两动作反例。

#### 条件与限制

这里沿用原回合目标，不是把目标改成长期平均奖励。线性预测条件不保证非线性控制全局收敛；采样截断也不是自然终止。

#### 阅读与实验

计算立即终止和延迟终止两条路径的原始、错误中心化、正确补偿回报。再辨认代码里的偏置、终止分支和更新前 TD 误差。

#### 原文与相关入口

- [RLC 正式记录](https://rlj.cs.umass.edu/2026/papers/Paper33.html)：问题、保证与实验分别阅读。
- [原文](https://arxiv.org/html/2605.04368v1)：终止补偿、共享偏置与回合式扩展。

### An Idiosyncrasy of Time-discretization in Reinforcement Learning

Kris De Asis, Richard S. Sutton

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

同样的物理奖励流，为什么会因奖励和折扣放在区间的不同位置而得到不同目标？

#### 关键机制

从连续时间回报的右端点近似出发，让区间奖励与后继价值按到达时间共同折扣。固定间隔时只差一个比例，不等间隔时这个比例一般无法提出求和。

#### 证据

原文给出时间离散化分析与实验。教材用恒定奖励率的两段时间计算，比较左右端点近似与精确积分。

#### 条件与限制

奖励率采样与已经积分的区间奖励不同。该修正不能消除动作延迟或低采样率遗漏事件，也不是任意 SMDP 接口都应照搬的公式。

#### 阅读与实验

固定每秒的目标而非每步折扣，再改变采样周期及抖动。报告积分误差、每秒更新次数和控制收益。

#### 原文与相关入口

- [RLC 正式记录](https://rlj.cs.umass.edu/2024/papers/Paper164.html)：正式出版入口。
- [原文推导](https://arxiv.org/html/2406.14951v2)：式 7 与不均匀时间步的回报定义。

### Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots

Khurram Javed, Joseph Modayil, Gloria Kennickell, Richard S. Sutton, John Carmack

RLC 2026 · 2026 · 评价与实验协议

#### 研究问题

在真实延迟、视觉观测与不同身体下，经典游戏任务能提供怎样的控制学习证据？

#### 关键机制

机器人实际操纵手柄，摄像头读取运行中的 Atari 游戏。先发动作后学习的调度，分离了身体响应、观测和更新的时间。

#### 证据

平台论文报告六个游戏中多次试验的累计运行与跨身体性能变化；作者公开硬件及学习工程。

#### 条件与限制

累计运行时间不是单条终生学习轨迹。作者学习器仍有经验回放和目标网络；平台可靠性与长期知识增长是不同主张。

#### 阅读与实验

同时比较环境步数和物理小时下的学习曲线，并记录动作延迟、身体差异与人工干预。

#### 原文与相关入口

- [作者项目与论文](https://keenagi.com/research/physical-atari/)：项目已从旧 GitHub Pages 地址迁移到 Keen 官方域名。

#### 作者代码

[作者团队发布的完整工程。](https://github.com/Keen-Technologies/physical-atari-rlc)

身体搭建、传感控制、智能体和实验脚本。运行需实物设备。

### The Open Ant: A Robot Platform for Reinforcement Learning Research

Elena Sorina Lupu, Patrick Spieler, Khurram Javed, Kris De Asis, John D. Martin, Martha Steenstrup, Joseph Modayil

RLC 2026 · 2026 · 评价与实验协议

#### 研究问题

能否在有限场地中直接从身体经验学习，并明确比较仿真、实机和维护条件？

#### 关键机制

开放硬件四足平台配合模拟器、传感接口和学习器。越界后切换目标方向，使任务无需每走到边界就结束回合。

#### 证据

论文比较 SARSA(λ) 与 SAC 的实机学习，给出模拟—实机对照。公开工程包含身体设计、组装演示和运行入口。

#### 条件与限制

缆线仍可能需要人工解缠。两学习器的动作及经验协议不同；真机运行、无回合任务与无人维护的持续学习不能混称。

#### 阅读与实验

先列状态与动作权限，再核对时间戳、奖励方向和恢复记录。用同一真实时间预算检验新增预测或规划是否值得其计算成本。

#### 原文与相关入口

- [原文](https://arxiv.org/abs/2607.18488)：平台、任务、实机实验与局限。

#### 作者代码

[Openmind 官方工程仓库。](https://github.com/Openmind-Research-Institute/open-ant)

硬件、MuJoCo 模拟、SARSA/SAC 与主控制入口。


<a id="chapter-code"></a>

## 下载与运行

流式 TD、在线统计与 ObGD 的机制检查；meta_frontier_lab.py 提供 Intentional 输出尺度实验。不包含完整神经网络控制训练。

[下载 lifelong_algorithms_lab.py](../examples/lifelong_algorithms_lab.py)

```sh
python examples/lifelong_algorithms_lab.py streaming
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [配套章节：时间信用分配](credit.md)：系统讨论多步回报、资格迹和跨时间信用；流式学习只是其中部分方法可以满足的数据协议。

- [配套章节：学习规则的适应](meta.md)：在线元梯度使用学习过程的敏感度调整规则；与本章预设的更新尺度控制区分。

- [Sharifnassab 等：Intentional Updates（ICML 2026）](https://proceedings.mlr.press/v306/sharifnassab26a.html)：Intentional-TD、Intentional-Q 与 Intentional-AC；按预期输出变化选择局部步长。

- [Intentional-AC 作者优化器源码](https://github.com/sharifnassab/Intentional_RL/blob/e86e26fd8613ac212e9a52c3fed8a01d0a31f685/optimizer.py)：二阶矩、sigma 偏差校正、误差裁剪、policy 归一化及更新后 trace reset 的具体顺序。

- [Intentional-AC 作者 actor–critic 主循环](https://github.com/sharifnassab/Intentional_RL/blob/e86e26fd8613ac212e9a52c3fed8a01d0a31f685/intentional_ac.py#L79)：update_params 用旧 critic 构造 TD 误差；actor 熵项乘该误差的符号，随后将合成梯度送入独立的 policy 优化器。

- [Sutton & Barto — Reinforcement Learning: An Introduction，第二版](http://incompleteideas.net/book/the-book-2nd.html)：§2.5 非平稳追踪、§8.4–8.5 模型与规划分布、§10.3–10.4 平均奖励、§17.3 状态与未来方向。

- [Stream-AC · 2026 固定版本主循环](https://github.com/mohmdelsayed/streaming-drl/blob/9326fc3e23a401f28087ae2e41b635888740586b/stream_ac_continuous.py)：update_params 第 69–94 行保存旧 δ，分别求 actor/critic 输出梯度；main 第 109–122 行显示更新后再采样。截断保留 bootstrap，同时在更新后清迹。

- [Kingma & Ba · Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)：Algorithm 1 与 §2–3：一阶矩、二阶矩和更新计数共同决定偏差校正；只保存参数没有保存优化器的继续过程。

- [van Seijen & Sutton · True Online TD Learning](https://proceedings.mlr.press/v32/seijen14.html)：线性在线前向/后向精确等价的适用范围，区别于普通 accumulating traces。

- [Elsayed, Vasan & Mahmood · Stream-X，2024 版](https://arxiv.org/abs/2410.14606v2)：ObGD 的有效步长动机、资格迹与网络稳定化组合；与修订版区分。

- [Stream-X · 2024 作者实现](https://github.com/mohmdelsayed/streaming-drl/tree/407dca7a8b584c1c20bc649053557f66e270b1e6)：optim.py 使用 ObGD；从 stream_ac_continuous.py 核对输出梯度、熵项和误差乘法。

- [Elsayed, Lupu, Vasan & Mahmood · Stream-X，2026 修订版](https://arxiv.org/abs/2410.14606v3)：2026-09-21 v3；逐坐标 StreamingOptimizer、layer-aware 初始化及 Stream-RAC。附录 C 对照早期版本，§4–5 给出变化环境与设备端学习协议。

- [Vitchutripop 等 · An Analysis of Streaming Deep Reinforcement Learning for Adaptive Continual Learning in Robotics](https://arxiv.org/abs/2609.28807v1)：2026-09-23 预印本。§III–V 与表 I、III：预训练后一次改变的模拟适应，区分峰值和适应期均值，并核查优化器与资格迹同时变化的对照。

- [Stream-X · 2026 作者实现](https://github.com/mohmdelsayed/streaming-drl/tree/9326fc3e23a401f28087ae2e41b635888740586b)：依次阅读 obs_reward_transforms.py、sparse_init.py、optimizer.py 和相应 agent；代码采用 CC BY-NC 4.0 许可证。

- [Welford · Note on a Method for Calculating Corrected Sums of Squares and Products](https://doi.org/10.1080/00401706.1962.10490022)：在线均值与方差递推的原始统计方法。

- [Dohare et al. — Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：将持续追踪困难与长期可塑性问题分开；标量追踪例不用于解释全部深网老化机制。

- [De Asis & Sutton · An Idiosyncrasy of Time-discretization in RL · RLC 2024](https://rlj.cs.umass.edu/2024/papers/Paper164.html)：正式 RLJ/RLC 条目，区分连续奖励率采样与已经积分的区间奖励。

- [Time-discretization · 原文式 (7)](https://arxiv.org/html/2406.14951v2)：更早折扣、时长缩放与不等间隔的右端点回报；固定间隔仅差共同比例。

- [Kris De Asis · Design for Learning · 2025](https://kris.pengy.ca/designforlearning)：作者关于硬件与学习共同设计的立场文章；与实证论文分开。
