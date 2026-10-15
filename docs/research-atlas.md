# 持续强化学习：研究问题、方法与实验

长期学习涉及怎样评价行为、形成知识、改善选择，以及保持学习能力。下面六组问题分别讨论这些困难；可以选择熟悉的一章继续深入，也可以比较几组问题怎样相互影响。

- [Ⅰ · 目标与设定](research-atlas.md#research-group-objectives)：评价的是最终策略，还是整个学习生命期？
- [Ⅱ · 状态与预测](research-atlas.md#research-group-knowledge)：历史中的哪些信息应当成为可用知识？
- [Ⅲ · 控制与学习机制](research-atlas.md#research-group-learning)：行为、信用与学习规则怎样持续更新？
- [Ⅳ · 子任务与规划](research-atlas.md#research-group-construction)：怎样构造可复用行为，并用其后果做决策？
- [Ⅴ · 长期适应与探索](research-atlas.md#research-group-adaptation)：怎样保留有用能力，并继续获得可学习的经验？
- [Ⅵ · 完整智能体与长期学习](research-atlas.md#research-group-systems)：各部分的学习怎样共同改善长期行为？

## 专题：状态与技能共同学习

状态与技能同时学习时，旧预测可能因输入含义、真实后果或经验覆盖改变而失准。专题长篇用一个贯穿任务说明怎样区分这些原因、已有方法能解决哪些部分，以及怎样检验修正预测是否值得付出代价。

[学习改变了记忆与行为，已有知识还适用吗？](research-joint-learning.md)

以下是可检验的实验提纲，不是已完成的实验或可直接运行的完整协议。实施前应预先确定目标效应、最小有意义差异、独立随机单位、预算和停止条件。区间过宽或包含零说明证据不足，不等于证明没有作用；只有能排除所声称的改善幅度，或出现与机制预测相反的结果，才真正限制该主张。

## Ⅰ · 目标与设定

<a id="research-group-objectives"></a>

评价的是最终策略，还是整个学习生命期？

两种学习器最后可能采用同一策略，其中一种却已经在试错中损失了大半生命期。评价学习过程，需要同时规定奖励表达什么偏好、奖励怎样随时间累计，以及重置或外部反馈有什么成本。奖励模型也在学习时，还要用独立的任务评价区分“更会完成任务”与“更会取得模型给出的高分”。这一组先确定什么算进步，后面的机制比较才有共同尺度。

<a id="research-reward-design"></a>

### 奖励表示与奖励学习：目标怎样进入智能体？

设计者希望智能体完成任务，却只能通过有限观测构造奖励，或用片段比较提供反馈。控制器学会新行为后，可能进入奖励模型没有见过的区域；训练分数上升仍可能伴随实际偏好满足程度下降。研究需分别规定真实评价、训练奖励以及反馈模型能观察的信息。

期望效用表示与逐步加性奖励表示需要不同条件。满足相应边界条件的势函数塑形可保持策略排序；偏好学习则从比较数据估计奖励，其校准依赖反馈和覆盖。这两种操作分别改变学习信号的表达与我们对目标的估计。

若目标和环境保持不变，只让控制器产生新行为，奖励误差便可能增加。需要判断哪些新片段值得查询，以及奖励模型更新后怎样修正仍在使用旧标签的经验。再独立改变真实目标，才能区分模型外推失败与偏好本身变化；仅把模型预测得更准，尚未解释控制器是否更接近任务目标。

#### 已有方法怎样处理这个问题

- **表示条件与奖励状态**：用反例检查原状态是否丢失时间结构；必要时显式增加奖励自动机或记忆。
  条件与代价：更大状态不保证任意偏好都可由有限计算表示。
- **偏好学习与重标记**：比较损失更新奖励模型；控制器使用更新后的信号并产生新经验。
  条件与代价：反馈噪声、片段选择和奖励模型版本都属于协议。

#### 相互竞争的解释

- 收益可能来自更接近真实目标，也可能只是更稠密的反馈。
- 奖励下降可能来自目标变化、模型外推错误或控制器失去可塑性。

#### 一种可能的研究办法

假设：相同反馈预算下，版本一致的经验重标记可减少奖励模型更新后的控制滞后。

设计：固定环境与真实评价，仅改变奖励模型；再独立改变真实目标。设置训练分布内与新策略片段两种偏好测试。

对照：真奖励诊断、固定模型、不重标记、全部重标记；匹配查询与计算预算。

测量：原任务收益、偏好校准、奖励投机行为、适应成本与反馈数。

什么结果会反驳该解释：若优势只来自更高反馈量，或训练奖励提高但真实评价下降，则不支持目标学习改进。

#### 基础准备

- [MDP、回报与价值：序列决策的数学对象](../foundations/tabular/mdps.md)：先明确状态、奖励与回报的定义，再问设计者的偏好能否由这一状态上的奖励表达。
- [约束强化学习：占据测度、拉格朗日与可行策略](../foundations/deep/constraints.md)：将回报目标与成本可行性分开，才能判断改变奖励是在表达偏好，还是只用惩罚近似另一项约束。

#### 教材与研究者

[奖励假设与奖励设计](../textbook/reward-design.md) · [交互、奖励与优化目标](../textbook/objectives.md) · [智能体状态与递归学习](../textbook/state.md) · [持续控制与学习智能体比较](../textbook/control.md) · [目标条件化与子任务构造](../textbook/goals.md) · [实验设计、统计与算法测试](../textbook/experiments.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [David Abel](https://yingwen.io/zh/continual-rl/resource/S08/) · [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/) · [Satinder Singh](https://yingwen.io/zh/continual-rl/resource/S41/) · [Will Dabney](https://yingwen.io/zh/continual-rl/resource/S65/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。

#### 原文与代码

- [Bowling et al. · Settling the Reward Hypothesis](https://proceedings.mlr.press/v202/bowling23a.html)（ICML 2023）：区分期望效用表示与逐步 Markov 奖励表示；时间一致性是额外条件。
  此条目提供原文阅读入口。
- [Abel et al. · On the Expressivity of Markov Reward](https://arxiv.org/abs/2111.00876)（NeurIPS 2021）：给定状态和奖励输入后，并非所有任务偏好都能由固定 Markov 奖励表达。
  此条目提供原文阅读入口。
- [Lee et al. · PEBBLE](https://arxiv.org/abs/2106.05091)（2021）：用预训练与奖励重标记提高偏好反馈的利用率。反馈模型与策略的数据分布共同变化。
  [B-Pref / PEBBLE 作者代码](https://github.com/rll-research/BPref)：奖励模型、查询策略、回放重标记与训练配置。

#### 同一问题的跨场景检验

当反馈来自测试、偏好或人工接管时，反馈模型也会变化。还需区分代理评分提高与独立长期效用提高。

- [奖励与反馈学习：偏好变化时，优化目标怎样保持可信？](research-atlas.md#horizon-human-feedback) — 人机协作、个性化服务与机器人反馈

---

<a id="research-lifetime-objective"></a>

### 01 · 什么目标能够评价一个始终在学习的智能体？

一个学习器较快得到可用策略，另一个学习器经过很长练习后表现更好。有限部署时长下，后者未必获得更多总奖励；若实验每次失败后免费重置，它又可能显得比真实部署更有利。比较前因此要规定奖励、原始时间单位、评价时域和重置权限，并将学习所用的存储与计算纳入资源条件。

折扣回报、有限生命期累计奖励和长期平均奖励对发生时间的权重不同。长期平均奖励忽略有限前缀，适合比较持续行为，却可能抹去很长的启动损失。Abel 的定义进一步相对于行为基础讨论最优智能体是否持续搜索；history-process 研究则提供比较整个交互过程的形式化途径。它们要求说明评价对象，不能由训练时间长直接推出持续学习。

固定策略的长期平均奖励忽略有限前缀；一个会更新的智能体却可能花很长时间学习或恢复。具体问题是：给定可接受的部署时长，哪些适应代价足以改变方法排序，这种改变来自评价准则还是算法的数值条件？把折扣设为接近一不自动得到差分价值算法，平均奖励表现好也不单独证明知识仍在积累。

#### 已有方法怎样处理这个问题

- **Differential TD/Q → RVI-SAC**：差分 TD 用 $R_{t+1}-\bar r_t$ 替代未中心化奖励，联合更新奖励率与相对价值。RVI-SAC 进一步把平均奖励目标接到熵正则 actor–critic，并处理 reset cost。
  条件与代价：表格收敛定理不直接覆盖非线性深网或任意漂移。RVI-SAC 的 replay 和目标网络也不属于严格 streaming 协议。
- **Reward centering 与协议比较**：从折扣 critic 中分离共同奖励偏移，减少无关基准对数值尺度的影响；同时显式比较三种 reset 协议。改变估计方法不必然改变任务目标。
  条件与代价：奖励中心化、平均奖励优化和延长 episode 是不同干预，实验不能将三者同时改变后只归因于平均奖励。
- **History process / deviation regret**：用交互历史刻画问题，并通过允许的行为偏离比较实际学习过程。评价对象不再只是训练完成后的一个策略。
  条件与代价：必须规定偏离集合、可观测信息和估计条件；这是一条形式化路线，而非已统一整个领域的标准。

#### 相互竞争的解释

- 方法收益可能来自更合适的任务目标，也可能只是奖励中心化改善了数值条件。
- 高终局成绩可能伴随更长探索期；高在线平均分也可能来自初始先验，而非更强持续学习。

#### 一种可能的研究办法

假设：算法排序的一部分差异由目标与 reset 权限造成，而不是网络表达能力造成。

设计：从教材循环奖励例子开始，解析比较折扣与平均奖励的策略排序；随后固定同一控制环境，分别改变折扣、奖励常数偏移和 reset 成本，每次只改一项。

对照：同网络、数据预算、种子与调参次数；记录 reset 是否由 agent 决策。设置不学习、固定步长学习和对应平均奖励方法。

测量：整段累计奖励、窗口奖励率、冻结策略诊断、恢复时间、真实交互秒数和重置次数。

什么结果会反驳该解释：在匹配目标、中心化与重置条件后排序差异消失，就不支持“目标形式本身带来普遍收益”的解释。

#### 基础准备

- [MDP、回报与价值：序列决策的数学对象](../foundations/tabular/mdps.md)：折扣和与有限期回报已经给出不同的比较对象；评价持续学习器时，还需把初始学习和恢复代价放回这些时间范围。
- [持续控制与平均奖励](../foundations/approximation/average-control.md)：周期链的奖励率与暂态价值说明长期极限会忽略有限前缀，由此可问适应成本何时改变实际部署期的排序。

#### 教材与研究者

[交互、奖励与优化目标](../textbook/objectives.md) · [持续控制与学习智能体比较](../textbook/control.md) · [平均奖励与差分价值](../textbook/average.md) · [实验设计、统计与算法测试](../textbook/experiments.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [A. Rupam Mahmood](https://yingwen.io/zh/continual-rl/resource/S05/) · [David Abel](https://yingwen.io/zh/continual-rl/resource/S08/) · [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [DeepRL-continuing-tasks](https://github.com/facebookresearch/DeepRL-continuing-tasks)：无重置、预设重置、智能体控制重置的连续控制。 奖励偏移、回报目标、重置成本对方法排序的影响。 边界：continuing 不自动包含任务漂移或无界学习需要。
- [CSuite](https://rl-csuite.readthedocs.io/en/latest/)：持续交互的小型诊断环境。 低成本测试适应与学习机制，再接复杂平台。 边界：每个任务测量的困难有限；不可用单一分数代表整个 CRL。

#### 原文与代码

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)（ICML 2021）：用 TD 误差更新平均奖励率，建立无参考状态的差分预测、控制与规划更新。收敛条件针对论文规定的表格问题。
  [作者实验代码](https://github.com/abhisheknaik96/average-reward-methods)：从 prediction_agents.py、control_agents.py 对照奖励率和价值的联合更新。
- [Hisaki & Ono · RVI-SAC](https://proceedings.mlr.press/v235/hisaki24a.html)（ICML 2024）：把相对价值迭代、平均奖励 soft policy improvement 和 reset cost 接到深度连续控制。
  [作者 RVI-SAC 实现](https://github.com/yhisaki/average-reward-drl)：检查 rvi_sac.py 的 critic 基准、actor 目标和重置成本，不仅修改 discount。
- [Wan et al. · An Empirical Study of Deep RL in Continuing Tasks](https://arxiv.org/abs/2501.06937)（2025 论文）：比较无重置、预设重置和智能体控制重置的任务，分析 reward centering 对深度方法的影响。
  [作者环境与实验](https://github.com/facebookresearch/DeepRL-continuing-tasks)：协议、奖励中心化和各基线配置同时构成实验对象。
- [Elelimy et al. · Rethinking the Foundations for Continual RL](https://rlj.cs.umass.edu/2025/papers/Paper243.html)（RLC 2025）：以 history process 和 deviation regret 研究学习过程的评价。比较对象与可实现偏离集合仍需明确，不能只把指标名称换掉。
  论文给出形式化与实验；此处没有可确认的作者原始代码入口。
- [Abel et al. · A Definition of Continual Reinforcement Learning](https://arxiv.org/abs/2307.11046)（NeurIPS 2023）：用智能体对行为基础的持续搜索定义持续性。它与“环境显式切换若干任务”的定义不相同。
  理论定义论文；没有需要复现的单一控制算法。

#### 同一问题的跨场景检验

真实部署中的恢复、等待和人工帮助改变生命期收益；跨个体汇总后的模型迭代应与个体内学习分开评价。

- [单生命与安全：怎样学习，而不耗尽以后的学习机会？](research-atlas.md#horizon-safe-autonomy) — 真实机器人与持续服务

## Ⅱ · 状态与预测

<a id="research-group-knowledge"></a>

历史中的哪些信息应当成为可用知识？

只看到当前位置时，智能体可能无法判断该走哪条路线；记住先前提示才能区分后果。状态构造研究怎样用有限活动保留这种差别，预测知识研究应该维持哪些关于未来的问题。两者相互帮助，却各有未解条件：一个预测集合可能包含所需信息但难以学准，一个记忆也可能保留许多与决策无关的细节。应分别考察信息是否足够、答案能否学到，以及决策怎样使用答案。

<a id="research-agent-state"></a>

### 02 · 有限的内部状态应当保留哪些历史信息？

同一个当前位置，在先前收到不同提示时可能要求不同选择。完整历史能够区分这两种情况，有限的内部状态则必须学会保留相关提示。困难不仅在能保存多少信息，还在怎样逐步更新，以及晚到的反馈怎样改变保存方式。环境隐藏变量是预测对象的一部分；网络活动、参数和保存的经验承担不同计算作用，应分别计入资源。

状态更新 $h_{t+1}=f_\theta(h_t,A_t,O_{t+1})$ 只规定如何压缩历史，不保证压缩后具有 Markov 性。控制充分性允许丢掉与最优决策无关的信息；完整预测充分性通常更强。递归结构和用于训练它的目标需要分别选择。

若决策线索只在很早以前出现，失败可能是状态没有保存线索，也可能是更新规则没有把晚到的反馈传回线索出现时。问题因此分成两步：给定相同历史权限，哪些预测目标能保留决策所需信息；保留方式改变后，原 critic 和模型需要付出多少重学代价？更长记忆不是这两步的共同答案。

#### 已有方法怎样处理这个问题

- **历史堆叠 / GRU / LSTM**：历史堆叠显式保留固定窗口；门控递归通过活动延续信息。BPTT/TBPTT 则决定参数能够接收到多远的训练信用，不能由 hidden state 不清零推断信用也没有截断。
  条件与代价：要匹配观测历史权限和实际训练窗口。完整回放序列、缓存 hidden state 与纯逐步处理具有不同成本。
- **GVFN：用预测约束状态**：让递归状态中的量估计指定策略下的未来信号，以这些预测的误差训练状态更新。相同活动既用于下一步递推，也承担预测，因此问题选择会影响保留的信息和学习难度。
  条件与代价：原文区分表示所需信息与及时学准信息。选择的问题必须可覆盖、可学习并对下游有用；有限预测集合不会自动成为充分状态。
- **RTU → 部分可观测 streaming RTRL**：结构化递归使参数对状态的敏感度可以逐步维护。近期工作把该敏感度与控制资格迹结合，同时研究在线权重变化造成的敏感度陈旧。
  条件与代价：高效精确计算依赖递归结构；不是任意 RNN 都能在线性成本下完成 RTRL。2026 工作仍是预印本。

#### 相互竞争的解释

- 提高回报可能来自额外历史信息，而不是更好的表示学习目标。
- RNN 比前馈网络好，可能只是在隐式推断任务标签；也可能真正记住了任务内关键事件。

#### 一种可能的研究办法

假设：预测式内部状态在相同容量下更容易保留决策所需的稀疏历史事件。

设计：先做延迟线索记忆链，分别改变线索延迟、无关观测和任务切换；再在 Forager 中只改变一个观测模式。对状态做冻结线性探针，但不把探针数据反馈给 agent。

对照：同参数量、动作信息与每步延迟预算；前馈、固定窗口、GRU/RTU、加预测损失各自单独比较。

测量：在线回报、延迟事件预测、记忆消融后的行为变化、状态漂移、敏感度误差与峰值内存。

什么结果会反驳该解释：匹配历史窗口后收益消失，或能预测关键事件却不改善决策，就不能归因于更好的控制状态。

#### 基础准备

- [MDP、回报与价值：序列决策的数学对象](../foundations/tabular/mdps.md)：Markov 条件规定状态需要支持哪些预测；当前观察不满足这一条件时，才需要研究内部状态应保留哪些历史。
- [不完全可观测：信念状态、信息行动与递归记忆](../foundations/deep/partial-observability.md)：精确信念更新与信息行动提供可检查的参照，进而可以辨认有限递归状态丢失的是哪项决策信息。
- [深度 RL 的训练循环：经验、目标与更新](../foundations/deep/practice.md)：将 recurrent 活动状态与跨时间参数梯度分开，才能区分线索没有被保存和保存机制没有学会。

#### 教材与研究者

[智能体状态与递归学习](../textbook/state.md) · [通用价值函数与预测知识](../textbook/gvf.md) · [时间信用分配与资格迹](../textbook/credit.md)

[Martha White](https://yingwen.io/zh/continual-rl/resource/S03/) · [Adam White](https://yingwen.io/zh/continual-rl/resource/S04/) · [Esraa Elelimy](https://yingwen.io/zh/continual-rl/resource/S34/) · [Matthew Schlegel](https://yingwen.io/zh/continual-rl/resource/S35/) · [Jan Peters](https://yingwen.io/zh/continual-rl/resource/S72/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [POPGym](https://github.com/proroklab/popgym)：可控记忆难度的部分可观测任务集合。 记忆长度、噪声与序列信用。 边界：通常有 episode；记忆基准成绩不能直接证明单生命期适应。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。

#### 原文与代码

- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)（NeurIPS 2024）：通过结构化递归单元降低前向敏感度维护成本，使逐步递归学习可以直接与控制任务结合。
  [作者 RTU 实现](https://github.com/esraaelelimy/rtus)：linear_rtus.py 的状态与敏感度 carry；复杂度优势依赖单元结构。
- [Schlegel et al. · General Value Function Networks](https://arxiv.org/abs/1807.06763)（GVFN 原论文）：让递归状态的坐标对应预测问题，以预测约束帮助状态学习。问题设计和递归求导是两个独立部分。
  [作者 Julia 实验](https://github.com/mkschleg/GVFN)：分别阅读预测问题定义、递归状态和训练更新；不是通用任意 GVF 集合的充分性证明。
- [Farr et al. · Streaming RL under Partial Observability with RTRL](https://arxiv.org/abs/2605.24709)（2026 预印本）：将实时递归学习用于部分可观测流式控制。网络记忆、信用传播与流式优化需要共同评价。
  此处未提供已确认的该论文作者代码；RTU 仓库对应上面的另一篇论文。
- [Forager: a Lightweight Testbed for Continual Learning with Partial Observability in RL](https://arxiv.org/abs/2605.01131)（2026 预印本）：在长期交互中隔离观测与状态构造问题，研究局部感知下的资源获取。
  [作者 Forager 环境](https://github.com/andnp/forager)：配置 observation_mode 和地图；start()/step() 接口不同于 Gymnasium 五元组。

#### 同一问题的跨场景检验

语言上下文、持久记忆和伙伴状态保留不同的信息。应测试它们何时仍有效，而不只测试是否能存入更多历史。

- [选择性观察与多粒度信用：局部反馈怎样连接长程后果？](research-atlas.md#horizon-language-actions) — 语言工具与软件维护
- [长期记忆：怎样判断一条经验值得保存、调用或遗忘？](research-atlas.md#horizon-memory) — 语言记忆与跨任务经验
- [相互适应与非平稳性：怎样区分自己、伙伴和共同约定的变化？](research-atlas.md#horizon-social-learning) — 人机合作与多机器人开放团队

---

<a id="research-predictive-knowledge"></a>

### 03 · 哪些预测值得学习，谁来使用这些预测？

一条路线经验既能训练到达概率，也能训练时长、累计消耗和途中事件的预测。每多维护一个问题，就增加更新和存储需求；当前行为也可能很少尝试该问题要求的动作。于是，除了怎样估计答案，还要决定哪些问题值得继续维护，以及预测通过显式读出还是共享表示影响控制。

一个 GVF 由累积信号 $c$、延续条件 $\gamma$ 和目标策略 $\pi$ 指定。TD、GTD 与 emphatic 方法是回答该问题的算法。GVF 的定义不要求其参与状态，也不要求其信号就是任务奖励。

给定一组问题后可以研究如何估计答案；撤掉“问题由设计者选好”的条件，智能体还要决定有限预算应维持哪些预测。预测容易学准、学习进展快、对后续决策有用是三个不同准则。应先固定候选问题池比较选择规则，再研究如何生成新问题，否则生成质量和资源分配的贡献无法区分。

#### 已有方法怎样处理这个问题

- **Horde / off-policy prediction**：为多个问题维护各自的目标与权重，把一条 transition 共享给多个学习器。重要性比率修正行为与目标策略差异；梯度 TD 或 emphatic weighting 处理特定离策略稳定性问题。
  条件与代价：行为策略必须覆盖目标策略，方差和表示误差仍可能很大。线性稳定性不等于深度预测任意稳定。
- **辅助预测与 GVFN**：用观测重构、潜在自预测或指定未来信号提供训练目标。GVFN 进一步把预测纳入递归状态。两者都需要独立检验预测是否被决策使用。
  条件与代价：RLC 2024 的分析表明目标用途依赖干扰和与 TD 的组合，不能把一个辅助 loss 的下降当作通用知识增益。
- **学习预测语义与更新目标**：DiscoRL 不预先给全部预测指定 GVF 语义，而是让元网络产生学习目标，再通过 agent 回报选择规则。这把“预测什么”与“怎样更新”一并纳入搜索。
  条件与代价：语义自由度增加时，可解释性、元训练成本及未见分布的可检验性也成为研究问题。

#### 相互竞争的解释

- 收益来自预测所携带的知识，还是来自额外梯度产生的正则化？
- 辅助任务有效，是因为其预测目标有因果用途，还是因为它更容易优化且共享了更多参数？

#### 一种可能的研究办法

假设：在固定候选 GVF 池和维护预算下，依据已经观察到的下游改进选择问题，比只追求低预测误差更能减少后续控制的再学习成本。

设计：用小 MDP 的解析答案核对问题语义；给所有选择器相同候选与数据覆盖，每次只允许维护其中固定数量。先检验奖励变化，再检验未参与选择的奖励组合。未来测试答案不能用于估计用途。

对照：随机选择、预测误差、学习进展与下游用途选择；等计算的额外主任务更新。部署诊断在同一冻结检查点中替换预测读出并匹配输入尺度；训练诊断则从同一初始化重训，单独切断辅助目标对共享表示的梯度。

测量：预测校准、覆盖、维护成本、主任务在线回报与再学习样本；分别报告部署读出的边际作用和训练辅助梯度的作用。

什么结果会反驳该解释：在足够精确的比较中，用途选择不比误差选择节省预定的经验量，就削弱该选择规则的主张。删除读出不改变行为只能限制当前显式读出解释，不能否定过去辅助训练已改变表示；无关目标同样有效则削弱特定预测语义的必要性。

#### 基础准备

- [函数逼近预测：从回归到 TD 固定点](../foundations/approximation/prediction.md)：先理解固定预测问题下的误差、投影与半梯度答案，再研究有限预算应该用来维持哪些预测问题。
- [离策略函数逼近：覆盖、发散与稳定更新](../foundations/approximation/off-policy.md)：同一经验流学习多个策略条件预测时，覆盖与稳定性决定哪些答案可学；答案可学以后，还要寻找使用它的消费者。

#### 教材与研究者

[价值预测与资格迹](../textbook/value.md) · [通用价值函数与预测知识](../textbook/gvf.md) · [智能体状态与递归学习](../textbook/state.md) · [元学习与学习规则的适应](../textbook/meta.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Martha White](https://yingwen.io/zh/continual-rl/resource/S03/) · [Adam White](https://yingwen.io/zh/continual-rl/resource/S04/) · [Patrick M. Pilarski](https://yingwen.io/zh/continual-rl/resource/S18/) · [Joseph Modayil](https://yingwen.io/zh/continual-rl/resource/S32/) · [Matthew Schlegel](https://yingwen.io/zh/continual-rl/resource/S35/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。

#### 原文与代码

- [Sutton et al. · Horde: A Scalable Real-time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction](https://www.ifaamas.org/Proceedings/aamas2011/papers/A6_R70.pdf)（AAMAS 2011）：同一经验服务多个由累积量、延续条件和目标策略定义的预测；学习问题与回答问题的更新器分离。
  [原研究系统 RLPark](https://github.com/rlpark/rlpark)：Horde.java 调度多个 demon，GTDLambda.java 实现离策略预测；旧 Java/机器人栈。
- [Voelcker et al. · When does Self-Prediction Help?](https://rlj.cs.umass.edu/2024/papers/Paper197.html)（RLC 2024）：比较自预测目标及其与价值学习的组合，说明辅助预测的用途依赖表示、数据和主任务。
  [作者仓库（仅论文说明）](https://github.com/adaptive-agents-lab/understanding_auxiliary_tasks)：公开 main 分支只有 README 与 .gitignore，没有可运行的实验源码。
- [Schlegel et al. · General Value Function Networks](https://arxiv.org/abs/1807.06763)（GVFN 原论文）：让递归状态的坐标对应预测问题，以预测约束帮助状态学习。问题设计和递归求导是两个独立部分。
  [作者 Julia 实验](https://github.com/mkschleg/GVFN)：分别阅读预测问题定义、递归状态和训练更新；不是通用任意 GVF 集合的充分性证明。
- [Oh et al. · Discovering State-of-the-Art RL Algorithms](https://www.nature.com/articles/s41586-025-09761-x)（Nature 2025）：通过元网络产生策略与预测的学习目标，从多智能体经验中搜索更新规则；学习出的预测未必具有预先指定的 GVF 语义。
  [作者 DiscoRL 工程](https://github.com/google-deepmind/disco_rl)：区分已发布规则的使用、训练 agent 与重新发现规则；三者需要不同预算。

#### 同一问题的跨场景检验

预测问题可以是动作完成时间、权限失败或恢复概率。无需先获得完整世界模拟器，但预测必须在指定行动条件下接受检验。

- [可行动的模型：需要预测整个世界，还是足以作决定的后果？](research-atlas.md#horizon-action-models) — 机器人组合任务与语言工具接口

## Ⅲ · 控制与学习机制

<a id="research-group-learning"></a>

行为、信用与学习规则怎样持续更新？

即使目标与所需信息已经确定，经验也不会自动产生合适的更新。延迟反馈需要找到早先相关的参数；一次很大的更新可能毁掉其他预测；统一步长又可能同时妨碍快速追踪和抑制噪声。信用分配、流式更新与元学习分别处理这些困难。比较时先保持预测问题和数据相同，辨认更新究竟改变了什么，再让策略继续采样，观察这些变化怎样影响长期控制。

<a id="research-continual-control"></a>

### 04 · 旧价值什么时候应该复用，什么时候应该快速改写？

一条曾经有利的路线突然变差，智能体需要迅速修正；若这条路线后来恢复，完全丢掉旧知识又可能浪费经验。类似的误差上升也会在策略首次访问陌生区域时出现。旧价值是否该改写，因而取决于规律、行为和覆盖中究竟哪一项变了，以及保留它会怎样影响下一次选择。

Q-learning、actor–critic 和最大熵控制提供不同的策略改善步骤。持续控制沿用这些基础，却必须反复面对估计速度与变化速度不匹配的问题。增大步长可以加快追踪，也会放大噪声；保留旧分量可以减少回访重学，也会延续过时偏差。需要在相同控制骨干下测量这些取舍，才能解释某种记忆安排的作用。

给定变化边界，可以安排重置或整合；边界未知时，误差上升既可能来自规律改变，也可能来自策略刚访问了陌生状态。值得问的是：哪些可观察信号足以触发快速改写，又不丢掉仍然有效的旧价值？先在同一数据上辨认估计滞后，再放开行为检查探索，才能区分两类困难。

#### 已有方法怎样处理这个问题

- **Permanent–Transient 价值分解**：用 $V=V^{\mathrm{permanent}}+V^{\mathrm{transient}}$ 表示长期知识与短期残差，分别安排更新和知识整合。变化出现时不必把全部价值重新学一遍。
  条件与代价：原文区分有边界和持续实验。两个时间尺度增加超参数，也可能把过时知识长期保留下来。
- **滑窗估计 / 乐观探索 / 变化预算**：滑窗减少过时转移的影响；置信集加宽与乐观规划避免窗口过短导致探索停止。理论将变化量与可达到的动态 regret 联系起来。
  条件与代价：需要明确有限状态、可观察信息和变化预算；不是对任意非平稳深度 actor–critic 的保证。
- **重置在线学习器并蒸馏**：Reset & Distill 将新任务的快速学习与过去行为保存放到不同网络和训练阶段，直接处理旧初始化带来的负迁移。
  条件与代价：依赖任务化协议、重置和额外蒸馏成本；不能直接移植为单一无边界 agent 而保持原结论。

#### 相互竞争的解释

- 快速适应可能来自减少陈旧价值，也可能来自探索策略重新增加覆盖。
- 从头训练优于接着训练，可能是表示塑性下降，也可能是旧策略把 agent 困在错误数据分布中。

#### 一种可能的研究办法

假设：在反复出现的奖励模式中，双时间尺度价值能够兼顾首次适应和回访复用。

设计：设置 A→B→A 奖励变化与独立的动力学变化两条序列。先固定 replay 数据比较估计，再放开在线行为比较控制；逐渐移除任务边界提示。

对照：单价值固定步长、步长网格、边界重置、随机探索重启；参数量和总更新次数匹配。

测量：首次适应、回访收益、旧价值误差、状态覆盖、在线累计奖励及负迁移。

什么结果会反驳该解释：固定数据后差异消失且只在重新探索时有效，就应将解释归于数据获取而非价值记忆机制。

#### 基础准备

- [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](../foundations/tabular/temporal-difference.md)：区分 SARSA、Expected SARSA 与 Q-learning 的目标和实际行为，才能把变化后的估计滞后与探索不足分别检查。
- [特征、泛化与半梯度控制](../foundations/approximation/features-control.md)：走廊中的共享特征让一次价值更新改变下一次选择，说明旧价值的复用还会改变随后得到的数据。
- [深度价值学习：DQN、Double DQN 与目标的时间顺序](../foundations/deep/deep-value.md)：Replay 中的样本年龄与目标网络的参数版本会延缓价值改写；环境变化后，需要判断这些滞后何时有益、何时妨碍适应。

#### 教材与研究者

[持续控制与学习智能体比较](../textbook/control.md) · [深度价值学习](../textbook/deep-value.md) · [策略梯度与 actor–critic](../textbook/policy.md) · [最大熵控制](../textbook/soft-control.md) · [知识保留与再适应](../textbook/retention.md)

[Doina Precup](https://yingwen.io/zh/continual-rl/resource/S07/) · [Csaba Szepesvári](https://yingwen.io/zh/continual-rl/resource/S15/) · [Hado van Hasselt](https://yingwen.io/zh/continual-rl/resource/S61/) · [Wang Chi Cheung](https://yingwen.io/zh/continual-rl/resource/S98/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。
- [Continual World](https://github.com/awarelab/continual_world)：基于 Meta-World 的机器人任务序列。 遗忘、前向迁移及回访表现。 边界：任务边界、重置和任务标识应明确；不是严格无重置协议。

#### 原文与代码

- [Anand & Precup · Prediction and Control in Continual RL](https://arxiv.org/abs/2312.11669)（NeurIPS 2023）：把价值拆为长期 permanent 与快速 transient 分量，研究长期积累和短期适应的兼容性。
  [作者预测与控制实验](https://github.com/NishanthVAnand/prediction-and-control-in-continual-reinforcement-learning)：prediction_semi_crl 与 control 分开；有边界和持续协议要分别运行。
- [Cheung et al. · Reinforcement Learning for Non-Stationary MDPs](https://proceedings.mlr.press/v119/cheung20a.html)（ICML 2020）：用滑动窗口、置信集加宽及变化预算处理探索与历史失效；理论对象是规定条件下的非平稳 MDP。
  以论文算法和定理条件为实现依据；此处没有可确认的作者代码入口。
- [Ahn et al. · Prevalence of Negative Transfer in Continual RL](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)（ICLR 2025）：Reset & Distill 在新任务中重置在线学习器，并离线蒸馏在线策略与过去 expert，研究负迁移和知识保留。
  [作者 Reset & Distill](https://github.com/hongjoon0805/Reset-Distill)：任务边界、网络重置、离线蒸馏与额外 expert 存储均应纳入实验协议。

#### 同一问题的跨场景检验

长期工具调用与真实操作都把恢复和未来信息纳入控制；他者也学习时，还要区分自身改善与对方迁就。

- [选择性观察与多粒度信用：局部反馈怎样连接长程后果？](research-atlas.md#horizon-language-actions) — 语言工具与软件维护
- [单生命与安全：怎样学习，而不耗尽以后的学习机会？](research-atlas.md#horizon-safe-autonomy) — 真实机器人与持续服务
- [奖励与反馈学习：偏好变化时，优化目标怎样保持可信？](research-atlas.md#horizon-human-feedback) — 人机协作、个性化服务与机器人反馈
- [相互适应与非平稳性：怎样区分自己、伙伴和共同约定的变化？](research-atlas.md#horizon-social-learning) — 人机合作与多机器人开放团队

---

<a id="research-temporal-credit"></a>

### 05 · 远处的反馈应该怎样更新早先的决策与内部计算？

智能体在路线起点写入提示，经过许多无关观测后才收到奖励。反馈可能需要修改起点动作的价值，也可能需要修改更早的记忆写入。前者涉及过去预测或动作的信用，后者还涉及参数怎样通过递归活动影响今天的输出。是否保存旧序列、是否允许回放，决定能够怎样计算这些影响。

冻结参数时，$\lambda$-return 的总增量可改写为 TD 误差乘资格迹的后向更新。传统在线 TD($\lambda$) 在每步改变参数后不再精确保持该恒等式。True-online TD 重新定义在线前向目标并加入荷兰迹与修正。

在线学习撤掉了“历史梯度始终在同一参数处计算”的条件；递归控制又让早先参数影响今天的内部状态。具体问题是：在固定每步成本下，怎样保存对延迟反馈仍有用的方向，而不传播过时或高方差的信用？需要把非线性近似、离策略修正和状态敏感度分开研究，不把任意一条 trace 当成全部答案。

#### 已有方法怎样处理这个问题

- **n-step / λ-return / true-online TD**：多步目标在不同 bootstrap 长度之间混合。后向资格迹压缩过去预测的梯度方向；true-online 的额外预测修正使指定线性算法对应逐前缀前向参考。
  条件与代价：精确等价针对定义好的线性、固定特征、步长等条件。更大 λ 会延长信用，也会增加误差与方差传播。
- **控制中的资格迹**：SARSA 的 bootstrap 使用实际选择的下一动作；Watkins Q(λ) 使用 max target，并在下一行为非贪心时剪断后续迹。剪迹发生在当前反馈完成信用分配之后。
  条件与代价：两种方法目标不同；不能把任意 off-policy actor–critic 都用同一个剪迹规则替换。
- **RTRL 敏感度 × 价值/策略时间信用**：RTRL 追踪参数对当前内部状态的影响；外层 TD 或策略更新还需要把延迟反馈与预测/动作联系起来。结构化递归研究使这两个层次可以在有限预算下组合。
  条件与代价：在线参数变化使全历史敏感度产生陈旧误差；截断和低秩近似又带来不同偏差或方差。

#### 相互竞争的解释

- 提高 λ 后变好，可能只是有效步幅变大，而不是利用了更久以前的信息。
- 更长反传窗口有效，可能来自更多重复训练，而非更准确的历史因果路径。

#### 一种可能的研究办法

假设：在匹配参数更新尺度后，正确的历史信用仍能改善稀疏延迟反馈学习。

设计：先运行教材独立在线前向 oracle；再控制线索到奖励的延迟和干扰数量。分别改变 λ、BPTT 窗口与状态递归，保持其余部分固定。

对照：TD(0)、传统 TD(λ)、true-online 线性参考；递归实验设置 detach 但不清空活动的对照。

测量：逐前缀权重误差、有限差分梯度误差、回报随延迟曲线、实际更新范数及每步成本。

什么结果会反驳该解释：收益在匹配更新范数或训练次数后消失，就不能将其归因于更准确的长时间信用。

#### 基础准备

- [多步学习：n-step、Tree Backup 与 Q(σ)](../foundations/tabular/multistep.md)：多步目标明确等待多少真实奖励、从哪里自举，为研究有限等待和计算下的延迟信用建立参照。
- [多步回报、资格迹与 True-online TD](../foundations/approximation/traces.md)：在线前向参考、荷兰迹和预测差修正说明精确等价依赖哪些版本条件，随后才能研究非线性与变动表示。
- [策略梯度：从轨迹概率到 GAE 与 actor–critic](../foundations/deep/policy-gradient.md)：GAE 将价值误差组成动作优势，其 bootstrap 与跨序列边界决定远处反馈怎样进入策略更新。

#### 教材与研究者

[时间信用分配与资格迹](../textbook/credit.md) · [价值预测与资格迹](../textbook/value.md) · [智能体状态与递归学习](../textbook/state.md) · [策略梯度与 actor–critic](../textbook/policy.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [A. Rupam Mahmood](https://yingwen.io/zh/continual-rl/resource/S05/) · [Khurram Javed](https://yingwen.io/zh/continual-rl/resource/S33/) · [Esraa Elelimy](https://yingwen.io/zh/continual-rl/resource/S34/) · [Anna Harutyunyan](https://yingwen.io/zh/continual-rl/resource/S66/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [POPGym](https://github.com/proroklab/popgym)：可控记忆难度的部分可观测任务集合。 记忆长度、噪声与序列信用。 边界：通常有 episode；记忆基准成绩不能直接证明单生命期适应。

#### 原文与代码

- [van Seijen et al. · True Online Temporal-Difference Learning](https://www.jmlr.org/papers/v17/15-599.html)（JMLR 2016）：荷兰迹与预测修正实现线性在线前向视图；这不同于冻结参数下的传统前后向总增量恒等式。
  [作者原始 true-online 实现](https://github.com/armahmood/totd-rndmdp-experiments/blob/4b6d459890a0e9da898f6cf8b849ba7870ea044a/pysrc/algorithms/tdprediction/onpolicy/totd.py)：此源码将步长包含在 trace 中；与教材未缩放 trace 对照时需换算。
- [Sutton & Barto · Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)（第二版，2018）：第 7、11、12 章连接多步目标、离策略学习和资格迹；用于核对预测与控制算法的不同条件。
  作者教材和伪代码入口；本网站实验为独立教学实现。
- [Elelimy et al. · Real-Time Recurrent Learning using Trace Units](https://proceedings.neurips.cc/paper_files/paper/2024/file/1e616bde0438cb10cb6adf076ae7d336-Paper-Conference.pdf)（NeurIPS 2024）：通过结构化递归单元降低前向敏感度维护成本，使逐步递归学习可以直接与控制任务结合。
  [作者 RTU 实现](https://github.com/esraaelelimy/rtus)：linear_rtus.py 的状态与敏感度 carry；复杂度优势依赖单元结构。
- [Farr et al. · Streaming RL under Partial Observability with RTRL](https://arxiv.org/abs/2605.24709)（2026 预印本）：将实时递归学习用于部分可观测流式控制。网络记忆、信用传播与流式优化需要共同评价。
  此处未提供已确认的该论文作者代码；RTU 仓库对应上面的另一篇论文。

#### 同一问题的跨场景检验

语言动作同时具有 token 生成与外部动作两个粒度；把二者直接按同一时间折扣，可能改变原本要优化的目标。

- [选择性观察与多粒度信用：局部反馈怎样连接长程后果？](research-atlas.md#horizon-language-actions) — 语言工具与软件维护

---

<a id="research-streaming-stability"></a>

### 06 · 不存 replay、每步只处理新经验时，怎样避免更新失稳？

数据到达后立即用于学习，过去的 transition 不存下来重新训练。一次罕见的大误差既可能提示环境变化，也可能只是噪声；更新过猛会扰动其他预测，更新过小又会错失必要适应。资格迹、递归活动和 optimizer 状态可以保留，但须计入内存与每步计算。方法还必须在动作 deadline 之前完成更新。

逐步更新时，样本间的相关性和瞬时尺度不会被 batch 平均掉。归一化、网络尺度、资格迹与 actor–critic 时序会共同影响稳定性。“batch size=1”只是数据粒度，不是一套完整算法。

批次训练可以重新采样并平均瞬时误差；严格流式学习不能事后再取同一条经验。由此产生的具体问题是：面对稀有但有用的大误差，怎样限制破坏性的输出变化，又不把必要的快速适应一并压掉？应比较实际输出改变与以后学习速度，不能只比较参数范数或一次更新耗时。

#### 已有方法怎样处理这个问题

- **Stream-X：共同更新配方**：信号归一化与表示稳定化配合受控参数更新。2024 ObGD 以全局尺度限制步幅；2026 实现维护 $v_i=\max(\beta v_i,|\delta e_i|)$，并按坐标缩放更新。两者不是同一个 optimizer。
  条件与代价：逐坐标上界限制参数变化，不保证 TD 误差绝不越过零；版本、网络和归一化都需匹配。
- **Intentional TD / Policy Gradient**：在局部近似下先指定希望减少多少 TD 误差，或允许多大策略 KL 变化，再求更新尺度。eligibility 与对角缩放改变方向，输出约束决定其长度。
  条件与代价：近似依赖局部函数几何，真实输出改变可能偏离目标；须记录实际 TD-error 与 KL，而不能只看参数范数。

#### 相互竞争的解释

- 方法提高回报，是归一化减少了数值异常，还是输出尺度控制更有效？
- 吞吐量变高，是更新更便宜，还是省略了基线需要完成的学习工作？

#### 一种可能的研究办法

假设：相较固定参数步长，输出变化控制能在奖励缩放后维持更接近的学习行为。

设计：在同一 Stream-AC 骨干上改变奖励比例、干扰特征和少量异常奖励；一次替换一个尺度机制，再测试组合。

对照：匹配架构、trace、初始化、归一化、调参预算和每秒环境步；旧版 ObGD 与新版 StreamingOptimizer 分列。

测量：实际输出变化、TD 误差越过零比例、策略 KL、崩溃率、峰值内存、动作 deadline 违约率和在线回报。

什么结果会反驳该解释：在匹配归一化与有效步幅后不再改善，就不支持输出约束是主要收益来源的解释。

#### 基础准备

- [函数逼近预测：从回归到 TD 固定点](../foundations/approximation/prediction.md)：共享参数的几何表明相同参数步长可能产生不同输出变化，为流式更新的尺度控制提供直接检查对象。
- [离策略函数逼近：覆盖、发散与稳定更新](../foundations/approximation/off-policy.md)：覆盖充分仍可能发生的 TD 发散说明，小步长或尺度归一化与修正期望更新方向是两项不同工作。
- [深度价值学习：DQN、Double DQN 与目标的时间顺序](../foundations/deep/deep-value.md)：DQN 的采样、回放与优化时钟明确了数据复用权限；撤掉 replay 后，需要重新检查相关性和每步更新幅度。

#### 教材与研究者

[流式更新与稳定性](../textbook/streaming.md) · [时间信用分配与资格迹](../textbook/credit.md) · [策略梯度与 actor–critic](../textbook/policy.md) · [最大熵控制](../textbook/soft-control.md)

[A. Rupam Mahmood](https://yingwen.io/zh/continual-rl/resource/S05/) · [Mohamed Elsayed](https://yingwen.io/zh/continual-rl/resource/S107/) · [Gautham Vasan](https://yingwen.io/zh/continual-rl/resource/S108/) · [Khurram Javed](https://yingwen.io/zh/continual-rl/resource/S33/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。
- [DeepRL-continuing-tasks](https://github.com/facebookresearch/DeepRL-continuing-tasks)：无重置、预设重置、智能体控制重置的连续控制。 奖励偏移、回报目标、重置成本对方法排序的影响。 边界：continuing 不自动包含任务漂移或无界学习需要。

#### 原文与代码

- [Elsayed et al. · Streaming Deep Reinforcement Learning Finally Works](https://arxiv.org/abs/2410.14606)（2024 初稿；2026 修订）：将信号归一化、表示稳定化和受控参数更新组成 Stream-X。2024 与 2026 的优化器不同。
  [作者 Stream-X 实现](https://github.com/mohmdelsayed/streaming-drl)：2024 分支为 ObGD；2026 分支使用逐坐标 StreamingOptimizer，并包含 Stream-RAC。必须与所用论文版本对应。
- [Sharifnassab et al. · Intentional Updates for Streaming RL](https://proceedings.mlr.press/v306/sharifnassab26a.html)（ICML 2026）：先规定 TD 误差或策略输出的期望变化，再近似求合适步幅；输出约束不同于参数范数约束。
  [作者 Intentional RL](https://github.com/sharifnassab/Intentional_RL)：optimizer.py 中的方向缩放、输出变化近似、迹与截断共同决定更新。

#### 同一问题的跨场景检验

真实机器人的动作时钟不会必然等待梯度更新。端侧更新、远端推理和批次再训练的时延与信息权限不同。

- [快状态与慢知识：未知变化应由哪个学习过程承担？](research-atlas.md#horizon-embodied-adaptation) — 机器人本体变化、sim-to-real 与 VLA

---

<a id="research-learning-rules"></a>

### 07 · 哪些学习参数应当适应，怎样评价学出来的更新规则？

相同的预测误差可能来自可追踪的漂移，也可能来自不可约噪声；合适的步长调整方向因而不同。元学习利用后续表现判断过去的更新是否有益，再调整步长、return 参数或整个更新规则。在线元学习只能使用这段持续经验；跨任务元 RL 还可能使用任务分布和重置，两者所能学习到的规则受不同信息条件约束。

IDBD 维护步长对权重的敏感度，用后续预测误差调整逐特征学习尺度；TIDBD 在自举预测中处理这一问题。Metatrace 进一步结合元敏感度与时间信用，Meta-gradient RL 对内层更新求导以调整 return 等参数。这些方法共同面对一个选择：外层究竟评价哪段未来，内层又允许改变什么。局部误差下降不自动等于长期控制收益。

当一组特征携带漂移信号、另一组只携带噪声时，两者可能需要相反的步长变化。统一归一化解决不了这种选择。问题是：有限长度的后续误差能否辨认哪些过去更新有用，并使学到的尺度规则迁移到未见漂移速度？短展开、跨任务搜索和更长部署时域应分别评价。

#### 已有方法怎样处理这个问题

- **逐特征步长 → Metatrace**：维护 $\partial w_t/\partial\beta$ 等敏感度，用后续误差判断过去步长是否有益。Metatrace 再累积元资格，使延迟反馈能够影响步长参数。
  条件与代价：监督误差、TD semi-gradient 和完整 TD-error 导数不是同一目标。归一化与截断也改变算法。
- **Meta-gradient RL**：内层更新生成新权重，外层用固定评价问题衡量这些权重，并沿更新过程反传到元参数。内层 discount 可学习，不代表外层评价目标也应随它任意改变。
  条件与代价：完整敏感度含跨步与跨参数耦合；截断、停止梯度和随机轨迹导数均需说明。
- **规则搜索与 DiscoRL**：将损失、目标或更新规则参数化，通过元训练评价其学习效果。RLC 2025 比较搜索途径与成本；DiscoRL 提供表达更丰富的目标生成规则及公开元训练/元评价入口。
  条件与代价：发现规则的总算力与应用已发布规则的成本分开；发布实验不等于每个研究者都能同预算复现原发现规模。

#### 相互竞争的解释

- “学会学习”的收益可能只是更大的超参数搜索预算，或更适合某个评价时域的手工偏置。
- 元网络依赖的是任务结构，还是 reward scale、episode 长度等容易泄漏的任务标记？

#### 一种可能的研究办法

假设：学习尺度或更新规则能够在未见变化速度上迁移，而非只记住开发任务的时域。

设计：先用标量与向量问题做有限差分；随后在信号相关性、噪声和漂移速度的网格中划分元训练/测试区域，测试更长生命期。

对照：相同搜索总预算的固定步长、Adam/RMSProp、简单尺度归一化；冻结元更新、重置元状态两项消融。

测量：在线误差与回报、元梯度误差、元训练总成本、长时域失稳、未见分布性能和部署更新成本。

什么结果会反驳该解释：增益仅存在于元训练时域内，或基线获得同等调参预算后差距消失，就不足以支持通用学习规则改进。

#### 基础准备

- [多臂老虎机：估计、探索与直接策略学习](../foundations/tabular/bandits.md)：样本平均与常数步长在同一变化奖励流中的差别，先展示跟踪速度与噪声的取舍，再引出怎样从经验选择步长。
- [多步回报、资格迹与 True-online TD](../foundations/approximation/traces.md)：资格迹规定本次误差怎样改变权重；学习步长或迹长度则要继续追踪这些更新怎样影响后续评价。

#### 教材与研究者

[元学习与学习规则的适应](../textbook/meta.md) · [时间信用分配与资格迹](../textbook/credit.md) · [流式更新与稳定性](../textbook/streaming.md) · [实验设计、统计与算法测试](../textbook/experiments.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Matthew E. Taylor](https://yingwen.io/zh/continual-rl/resource/S19/) · [Jakob Foerster](https://yingwen.io/zh/continual-rl/resource/S58/) · [Shimon Whiteson](https://yingwen.io/zh/continual-rl/resource/S59/) · [Hado van Hasselt](https://yingwen.io/zh/continual-rl/resource/S61/) · [Luisa Zintgraf](https://yingwen.io/zh/continual-rl/resource/S69/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。

#### 原文与代码

- [Sutton · Adapting Bias by Gradient Descent](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf)（AAAI 1992）：IDBD 对每个特征的对数步长维护敏感度，使学习尺度由后续预测效果调整。
  原文包含算法；这里没有 1992 年作者原始代码发布入口。
- [Kearney et al. · TIDBD: Adapting Step-sizes Through Stochastic Meta-descent for Temporal-Difference Learning](https://arxiv.org/abs/1804.03334)（2018 论文）：将逐特征步长适应从监督预测扩展到 TD，并处理 bootstrap 误差与资格迹；不同梯度和归一化版本须逐式区分。
  原文 Algorithm 1 与教材线性实现可以对应；此处没有可确认的完整作者原始实验仓库。
- [Young, Wang & Taylor · Metatrace Actor-Critic](https://www.ijcai.org/proceedings/2019/0581.pdf)（IJCAI 2019）：把步长敏感度和时间信用结合；scalar、vector、mixed 形式具有不同的适应粒度和归一化。
  未找到可确认的作者原始代码；元学习教材提供独立线性特例，不含原文全部控制实验。
- [Xu, van Hasselt & Silver · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)（NeurIPS 2018）：对内层学习更新求导，以后续评价调整 return 等学习参数；元目标和内层目标必须分开。
  以论文伪代码和教材敏感度实验为入口；此处没有可确认的作者原始训练工程。
- [Goldie et al. · How Should We Meta-Learn RL Algorithms?](https://rlj.cs.umass.edu/2025/papers/Paper218.html)（RLC 2025）：在不同算法对象上比较元优化途径，并比较泛化、可解释性、样本开销与训练成本。
  [作者算法搜索实验](https://github.com/AlexGoldie/learn-rl-algorithms)：按元训练任务与元测试任务恢复配置，不能只计发现后规则的部署成本。
- [Oh et al. · Discovering State-of-the-Art RL Algorithms](https://www.nature.com/articles/s41586-025-09761-x)（Nature 2025）：通过元网络产生策略与预测的学习目标，从多智能体经验中搜索更新规则；学习出的预测未必具有预先指定的 GVF 语义。
  [作者 DiscoRL 工程](https://github.com/google-deepmind/disco_rl)：区分已发布规则的使用、训练 agent 与重新发现规则；三者需要不同预算。

#### 同一问题的跨场景检验

学习规则可以是步长递推，也可以是产生程序修改的策略；必须排除候选更多、先验更强和评测反复使用带来的解释。

- [学习规则与自修改：什么证据支持改变学习过程？](research-atlas.md#horizon-self-modification) — 可编辑程序与语言智能体
- [快状态与慢知识：未知变化应由哪个学习过程承担？](research-atlas.md#horizon-embodied-adaptation) — 机器人本体变化、sim-to-real 与 VLA

## Ⅳ · 子任务与规划

<a id="research-group-construction"></a>

怎样构造可复用行为，并用其后果做决策？

拥有一套技能之后，高层可以比较怎样调用；技能集合本身尚未给定时，还要决定什么值得学习。子目标规定练习的结果，option 实现行为，后果模型预测它的奖励、终点和时长，规划据此选择。每一步都可能成功而下一步仍失败：技能能达到子目标，却很少用于主任务；模型平均误差很小，却在两个候选接近时排错顺序。因此需要用下游用途检验发现、建模和计算分配，同时计入各自的学习成本。

<a id="research-useful-options"></a>

### 08 · 什么样的时间抽象值得加入技能库？

已经给定一套 option 时，可以学习何时调用它们；技能发现还要决定哪些行为值得先花经验学会并长期维护。一个跨房间技能可能帮助探索，却在当前奖励下从不被规划器采用；另一个短技能可能覆盖很少，却经常节省决策计算。比较时需分别测量探索、执行、信用和规划的用途，而不是只数技能或观察轨迹多样性。

Machado 的谱发现路线把环境连通结构变成内在奖励，再变成长行为。现代技能方法可从可区分性、后果模型或时间距离组织行为。它们改变的目标不同，产生的技能也未必适合相同任务。

给定技能集合可以学习如何调用它；固定容量的持续技能库却必须放弃某些已会的行为。应先问：一个技能的探索、执行或规划用途，能否抵偿发现和维护它的成本？再问下游用途如何指导保留、合并与替换。只统计技能执行成功率，会把成功但从不被需要的行为也当成进步。

#### 已有方法怎样处理这个问题

- **Eigenoptions → ALLO**：谱方向描述由经验转移诱导的连通结构；沿特征坐标变化构造奖励，学习长距离移动。ALLO 更准确地区分谱方向与特征值，给技能和后续规划提供结构化坐标。
  条件与代价：图取决于采样行为与状态表示。慢变化方向不必然是当前奖励最有用的方向。
- **METRA 的度量感知技能**：使沿技能方向的表示变化成为内在奖励，并限制相邻状态的表示距离。约束使表示更接近时间可达性结构，而非任意放大尺度获得高奖励。
  条件与代价：原目标和源码中的按维度缩放、slack 与 dual 更新需对应。状态覆盖好仍不是外部任务有效性的充分条件。
- **奖励感知谱结构**：Default Representation 把默认动力学与奖励结构共同纳入后果表示，进一步影响谱特征、塑形和 option 发现。它提供与纯动力学 SR 不同的技能偏置。
  条件与代价：奖励改变会改变表示本身的有效性；奖励感知不等于对所有未来任务更可迁移。

#### 相互竞争的解释

- 技能收益来自更好的探索覆盖，还是来自更少的高层决策与更有效的规划？
- 谱技能看似更好，可能只是执行时长或可到达范围比随机技能更大。

#### 一种可能的研究办法

假设：结构化技能在计入发现成本后，仍能降低跨房间目标变化的适应成本。

设计：在同一网格世界采样固定经验，比较随机、谱和奖励感知技能。先冻结低层技能只改变高层目标，再改变通道位置检验技能失效。

对照：primitive-only、时长与终点覆盖匹配的随机技能；同技能数、总数据、决策预算；有无模型规划分别报告。

测量：技能发现总成本、启动成功率、时长分布、主任务累计奖励、模型误差和通道改变后的恢复。

什么结果会反驳该解释：等时长随机技能或等覆盖原始探索得到相同收益，则不足以证明谱结构本身带来可复用抽象。

#### 基础准备

- [学习与规划：Dyna、优先扫描和执行时搜索](../foundations/tabular/planning.md)：区分实际执行、模型预测与规划 backup 的用途，才能评价一个长行为节省的经验或计算是否抵偿其维护成本。
- [探索与不确定性：后验、乐观估计和时间一致行动](../foundations/deep/exploration.md)：时间一致探索解释了长行为的一种用途，同时要求把覆盖改善与技能自身的执行成功率分别评价。

#### 教材与研究者

[Options 与技能发现](../textbook/options.md) · [目标条件化与子任务构造](../textbook/goals.md) · [通用价值函数与预测知识](../textbook/gvf.md) · [时间抽象与规划](../textbook/planning.md)

[Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [Doina Precup](https://yingwen.io/zh/continual-rl/resource/S07/) · [George Konidaris](https://yingwen.io/zh/continual-rl/resource/S09/) · [Pierre-Luc Bacon](https://yingwen.io/zh/continual-rl/resource/S37/) · [André Barreto](https://yingwen.io/zh/continual-rl/resource/S40/) · [Benjamin Eysenbach](https://yingwen.io/zh/continual-rl/resource/S53/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [OGBench](https://github.com/seohongpark/ogbench)：离线目标条件强化学习的环境、数据与基线。 长时域、轨迹拼接、观测与数据覆盖。 边界：离线成功率不能直接说明在线技能发现或持续控制；规划时间另计。

#### 原文与代码

- [Machado et al. · A Laplacian Framework for Option Discovery in RL](https://arxiv.org/abs/1703.00956)（ICML 2017）：用状态转移图的谱方向构造内在奖励，并学习沿这些方向移动的时间扩展行为。
  [Machado 作者原始 options 工程](https://github.com/mcmachado/options)：main.py、谱表示和 option 学习相连；环境封装和旧依赖属于原实验条件。
- [Gomez, Bowling & Machado · Proper Laplacian Representation Learning](https://arxiv.org/abs/2310.10833)（ICLR 2024）：ALLO 的约束与对偶动态恢复有序谱方向及相应特征值，减少只学到任意旋转子空间的问题。
  [作者 ALLO 实现](https://github.com/tarod13/laplacian_dual_dynamics)：同时读原始变量、对偶变量和正交约束；只保留相邻状态平滑项不是完整算法。
- [Park, Rybkin & Levine · METRA](https://arxiv.org/abs/2310.08887)（ICLR 2024）：在转移上的表示距离约束下，最大化技能方向与状态变化的对齐，再以该信号学习技能策略。
  [作者 METRA 实现](https://github.com/seohongpark/METRA)：iod/metra.py 的内在奖励、距离约束、对偶更新和 SAC 技能训练需一并对照。
- [Tse, Chandrasekar & Machado · Reward-Aware Proto-Representations in RL](https://arxiv.org/abs/2505.16217)（NeurIPS 2025）：分析 Default Representation 的动态规划、TD 学习与特征形式，把奖励结构引入后果表示。
  [作者 DR 实验](https://github.com/httse9/Reward-Aware-Proto-Representations)：rep_utils.py 区分 SR/DR；ROD_DR.py 连接采样学习、谱分解与发现。

#### 同一问题的跨场景检验

技能可以是低层控制策略，也可以是工具程序。可执行性不保证适用条件稳定，技能更新后还要重新检验后果模型。

- [目标形成与开放式学习：怎样选择值得学的下一件事？](research-atlas.md#horizon-open-ended-goals) — 开放世界、生成式课程与程序技能

---

<a id="research-goal-construction"></a>

### 09 · 子目标从哪里来，为什么学习它能帮助总体任务？

总体任务奖励给定后，智能体仍可以选择练习到达某个位置、实现某种可观察后果，或改进某项预测。提出一个目标、学会实现它、决定何时练习或调用它，需要不同信息。若设计者直接提供恰好有用的中间目标，实验主要检验目标条件学习；要研究自主构造，还须让目标的产生与选择承担成本。

UVFA 提供目标条件价值接口；HER 从失败轨迹重标记可达目标；HIQL 将长期任务分解为高层目标和低层动作。它们主要解决已给目标下的学习和组合，不能单独回答哪些新目标值得创造。

给定目标时，成功率可以评价低层学习；选择练习什么时，高成功率却可能只表示目标很容易。应先固定候选和低层学习器，问已有主任务经验是否足以估计一次练习的后续用途，再研究如何扩展候选。由设计者提供未来任务分布，是另一种信息条件，不能作为自主发现的隐含前提。

#### 已有方法怎样处理这个问题

- **HER：增加已达到目标的学习信号**：从同一轨迹选取 achieved goal，重算目标相关奖励，使原本失败的轨迹成为另一个控制问题的训练样本。它修改 replay 样本的目标，不改变真实执行结果。
  条件与代价：需要可重算目标奖励和适合离策略学习的数据；目标相关终止与随机性可能影响重标记语义。
- **HIQL：分层提取目标条件策略**：先学习目标价值，再分别利用一步和跨步优势拟合低层动作与高层中间目标。优势作为停止梯度的权重，避免策略更新反向扭曲其评价标尺。
  条件与代价：离线数据必须支持相关状态与目标；更短有效时域不能解决完全没有覆盖的行为。
- **Reward-respecting 与 AI-feedback 技能设计**：STOMP 在子任务执行期间沿用环境奖励，并用基于主任务价值的停止值鼓励达到某项特征；由此得到的技能仍需按真实后果评价。MaestroMotif 用语言描述、偏好奖励和规则组织技能，提供另一种目标来源。
  条件与代价：停止奖励和环境实际收益分开记录；已有特征、人工或语言模型的描述、反馈与程序生成都属于方法的信息和成本。

#### 相互竞争的解释

- 目标生成有效，可能是更准确地选择学习难度，也可能只是引入了额外任务信息。
- 层级方法提升，可能源于降低价值估计噪声，而非发现了可迁移语义。

#### 一种可能的研究办法

假设：在相同候选目标与低层学习器下，用已发生的主任务改进估计练习用途，比只按成功率选目标更节省总经验。

设计：先在在线小迷宫固定候选。候选用途由练习后预定时间窗内的主任务改进估计，并记录反馈延迟；开发序列上选规则，未见组合上测试。只有选择规则有效后，再单独加入候选生成；OGBench 用于检查离线覆盖，不替代在线选择实验。

对照：均匀选择、适中难度、学习进展与用途选择；相同交互、候选维护和低层更新预算。已知未来任务分布仅作特权诊断，不提供给主方法。

测量：主任务净收益、用途估计校准、目标覆盖、失败练习和选择成本；报告未见组合的首次表现及后续学习。

什么结果会反驳该解释：在预定精度下，用途选择的总经验开销不优于难度选择，或增益只在提供未来任务标签后出现，就削弱当前用途估计机制；不能仅凭练习成功率上升支持目标构建。

#### 基础准备

- [探索与不确定性：后验、乐观估计和时间一致行动](../foundations/deep/exploration.md)：内在奖励与学习进展改变智能体练习什么；研究目标构造时，还需检查这种练习是否改善后续外部任务。

#### 教材与研究者

[目标条件化与子任务构造](../textbook/goals.md) · [交互、奖励与优化目标](../textbook/objectives.md) · [Options 与技能发现](../textbook/options.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [Peter Stone](https://yingwen.io/zh/continual-rl/resource/S10/) · [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/) · [Benjamin Eysenbach](https://yingwen.io/zh/continual-rl/resource/S53/) · [Tom Schaul](https://yingwen.io/zh/continual-rl/resource/S63/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [OGBench](https://github.com/seohongpark/ogbench)：离线目标条件强化学习的环境、数据与基线。 长时域、轨迹拼接、观测与数据覆盖。 边界：离线成功率不能直接说明在线技能发现或持续控制；规划时间另计。
- [TeachMyAgent](https://github.com/flowersteam/TeachMyAgent)：teacher 可以选择环境参数的自动课程实验。 学习进展估计和经验分配。 边界：必须计算 teacher 权限、失败尝试和环境重置，不能视作完全自主单生命期。

#### 原文与代码

- [Andrychowicz et al. · Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)（NeurIPS 2017）：把轨迹实际达到的目标重标记为训练目标，以稀疏反馈学习目标条件控制。
  [作者团队 HER 实现](https://github.com/openai/baselines/tree/master/baselines/her)：her_sampler.py 中的目标替换与 reward_fun；结束标志与目标依赖需按任务处理。
- [Park et al. · HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://arxiv.org/abs/2307.11949)（NeurIPS 2023）：用目标价值分别提取高层子目标策略和低层动作策略，缩短每层面对的有效时域。
  [作者 HIQL 实现](https://github.com/seohongpark/HIQL)：hiql.py 的 expectile 价值学习、停止梯度优势权重和 high/low actor 更新。
- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)（Artificial Intelligence 2023）：STOMP 连接子任务、option、真实后果模型与规划；子任务停止奖励和环境真实回报属于不同接口。
  原文提供更新规则和实验；此处没有可确认的完整作者 STOMP 工程。
- [Klissarov et al. · MaestroMotif: Skill Design from AI Feedback](https://arxiv.org/abs/2412.08542)（ICLR 2025）：结合技能描述、偏好奖励和程序化组合规则，训练并组织 NetHack 技能。语言提供了额外的先验和任务设计信息。
  [作者 MaestroMotif 实现](https://github.com/mklissa/maestromotif)：preference、code_generation、rl_baseline 分别对应奖励、组合与技能学习。

#### 同一问题的跨场景检验

在既有世界中选择练习目标，与教师直接生成环境和奖励，是不同的问题；二者不能共享未说明的干预权限。

- [目标形成与开放式学习：怎样选择值得学的下一件事？](research-atlas.md#horizon-open-ended-goals) — 开放世界、生成式课程与程序技能

---

<a id="research-reusable-models"></a>

### 10 · 模型需要预测什么，才能在变化后继续支持决策？

高层要比较两条路线，需要预测沿途奖励、耗时和停止后的可用状态。模型训练误差很低，也可能没有保留这个比较需要的信息。状态表示、技能策略与环境规律又可以分别改变，所以模型有效性必须相对于所预测的行为和将要执行的规划查询来定义。

若下游价值对终点特征线性，期望终点特征可用于计算相应的期望尾值。若下游要读取非线性价值或风险，单一均值一般不够。折扣 option 模型还要把随机时长放进终点权重。世界模型、后继特征与 option model 保存的对象因此不同；是否可复用，要逐项检查动力学、行为、特征和奖励条件。

模型有效性相对于所预测的行为和规划查询定义。撤掉“表示和技能固定”的条件，旧模型可能失效，也可能仍然足够准确。重要问题不是版本号是否相同，而是哪些变化会改变备份或动作排序，以及怎样用有限真实执行发现这些变化。高平均预测精度可能掩盖罕见但决定选择的错误。

#### 已有方法怎样处理这个问题

- **Option 的奖励、终点与时长模型**：折扣模型需预测执行期间奖励及 $\gamma^\tau$ 加权的终点后果，再将终点价值接入备份。平均奖励模型还需计入持续时间成本，不能直接复用同一个折扣模型公式。
  条件与代价：必须说明模型针对哪个内部策略和终止规则；子任务 stopping bonus 不能混入真实环境奖励模型。
- **SR / SF 与 Default Representation**：Successor 类表示预测指定策略下的未来占用或特征累积，在奖励线性族中支持快速重估。DR 将奖励纳入默认后果表示，使表示本身更偏向特定奖励结构。
  条件与代价：动力学、默认行为或奖励条件改变时，可复用范围不同。不能把奖励变化、策略变化和转移变化视作同一迁移问题。
- **Dreamer 的潜在世界模型**：模型在真实序列上学习潜在动力学和预测头，随后在想象序列上训练行为。它直接服务 actor/critic，而不是要求所有像素都同样准确。
  条件与代价：模型训练数据、rollout 长度和想象分布决定偏差；跨域表现并不自动证明长期模型保留。

#### 相互竞争的解释

- 模型改善回报，可能因预测更准确，也可能因生成了更多训练目标。
- 更低 reconstruction loss 可能只来自拟合背景，而非保留了决策相关后果。

#### 一种可能的研究办法

假设：按规划相关后果的失准触发局部重学，比每次技能版本改变就全量重学更节省维护成本。

设计：在小 MDP 中分别改变 option 内部策略、奖励和环境转移。设置两类改变：版本变化但奖励、时长与终点后果不变；后果变化且足以改变动作排序。保持规划器固定，再引入表示坐标变化。

对照：真模型诊断、冻结模型、周期重学、版本触发与后果误差触发；匹配真实探测和总模型更新预算。额外容量不代替预算对照。

测量：奖励、时长与终点误差，备份误差、动作排序、失准发现延迟、真实回报和维护成本。

什么结果会反驳该解释：触发规则只降低平均模型误差却未减少规划错误，或漏掉预定的关键后果变化，就限制其决策用途；后果不变时无需重学是负对照，不是系统失效。

#### 基础准备

- [模型学习与规划：MPC、短模型 rollout 和潜在想象](../foundations/deep/model-based.md)：模型误差经过 rollout 和末端价值进入动作选择，因而模型应保留什么信息取决于它的决策消费者。
- [分布强化学习：Bellman 分布、分位数与风险目标](../foundations/deep/distributional.md)：相同回报均值可以对应不同的分布和风险排序，说明只预测平均后果何时不足以支持下游决策。

#### 教材与研究者

[转移模型与后果模型](../textbook/models.md) · [Dyna 与模型学习](../textbook/dyna.md) · [通用价值函数与预测知识](../textbook/gvf.md) · [智能体状态与递归学习](../textbook/state.md) · [平均奖励与差分价值](../textbook/average.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [Matthew Schlegel](https://yingwen.io/zh/continual-rl/resource/S35/) · [André Barreto](https://yingwen.io/zh/continual-rl/resource/S40/) · [Danijar Hafner](https://yingwen.io/zh/continual-rl/resource/S104/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [OGBench](https://github.com/seohongpark/ogbench)：离线目标条件强化学习的环境、数据与基线。 长时域、轨迹拼接、观测与数据覆盖。 边界：离线成功率不能直接说明在线技能发现或持续控制；规划时间另计。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。

#### 原文与代码

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)（Artificial Intelligence 2023）：STOMP 连接子任务、option、真实后果模型与规划；子任务停止奖励和环境真实回报属于不同接口。
  原文提供更新规则和实验；此处没有可确认的完整作者 STOMP 工程。
- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)（ICML 2021）：用 TD 误差更新平均奖励率，建立无参考状态的差分预测、控制与规划更新。收敛条件针对论文规定的表格问题。
  [作者实验代码](https://github.com/abhisheknaik96/average-reward-methods)：从 prediction_agents.py、control_agents.py 对照奖励率和价值的联合更新。
- [Barreto et al. · Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)（NeurIPS 2017）：在相同动力学与线性奖励族下分离后果特征和奖励权重，通过 generalized policy improvement 复用多个策略。
  [后续作者 Option Keyboard 工程](https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard)：展示 SF 与策略组合的接口；对应后续 Option Keyboard，不是该 2017 论文的原始实验代码。
- [Tse, Chandrasekar & Machado · Reward-Aware Proto-Representations in RL](https://arxiv.org/abs/2505.16217)（NeurIPS 2025）：分析 Default Representation 的动态规划、TD 学习与特征形式，把奖励结构引入后果表示。
  [作者 DR 实验](https://github.com/httse9/Reward-Aware-Proto-Representations)：rep_utils.py 区分 SR/DR；ROD_DR.py 连接采样学习、谱分解与发现。
- [Hafner et al. · Mastering Diverse Domains through World Models](https://www.nature.com/articles/s41586-025-08744-2)（Nature 2025）：DreamerV3 通过真实序列学习潜在世界模型，再用想象轨迹训练策略与价值；行动时不必执行树搜索。
  [作者维护的公开重实现](https://github.com/danijar/dreamerv3)：README 将其称为 reimplementation；不是原内部训练工程。模型、想象学习和真实数据比例均需匹配。
- [Fu et al. · Knowledge Retention in Continual Model-Based RL](https://proceedings.mlr.press/v267/fu25f.html)（ICML 2025）：DRAGO 用生成经验保持动力学，再用内在奖励引导回访旧状态；任务共享状态空间与动力学。
  [作者 DRAGO 实现](https://github.com/YixiangSun/drago)：合成回放和真实回访各自消融；“不存旧任务数据”不等于没有当前任务 replay。

#### 同一问题的跨场景检验

共享表征漂移和技能版本变化都会使旧模型失效。组合任务中的复用必须用旧模块消融检验，而非只看路由权重。

- [可行动的模型：需要预测整个世界，还是足以作决定的后果？](research-atlas.md#horizon-action-models) — 机器人组合任务与语言工具接口

---

<a id="research-planning-budget"></a>

### 11 · 什么时候值得规划，应该把计算花在哪里？

模型预测两种行为的收益接近时，多做一些计算可能改变选择；若两者差距已经很大，同样计算可能没有当前决策价值。模型也可能出错，使更深搜索更确信错误路线。真实交互、模型更新、搜索与想象训练共享预算，研究因而要判断下一段计算应花在哪里，以及等待计算是否延误行动。

Dyna 在模型中产生价值更新；prioritized sweeping 根据前驱与变化传播分配备份；MPC 在行动时优化短序列；Dreamer 在想象中训练策略。它们把计算放在不同阶段。

给定准确模型，可以比较同一预算下的备份安排；模型也在学习时，多想一步可能只是更充分地利用同一个错误。应先问哪些可获得信号能预测下一段规划的净收益，再比较把这段计算用于模型修正或真实行动的效果。旧搜索结构的维护、动作等待和知识发现都要进入同一预算。

#### 已有方法怎样处理这个问题

- **Dyna / STOMP：由模型支持价值改善**：Dyna 交替真实学习与模型备份；STOMP 进一步学习子任务、option 及其后果模型，将更长时间的行为作为可规划对象。
  条件与代价：抽象模型需与当前技能一致；计入 option 发现、模型训练及规划全部成本。
- **TD-MPC2：短模型路径加终点价值**：用短时域潜在预测和终点 critic 避免完全依赖长 rollout，策略先验帮助生成候选动作序列。
  条件与代价：模型误差和终点价值误差可以互相补偿，也可以共同误导；短时域不是无偏性保证。
- **ALPS：谱结构支持决策时分层规划**：先从离线数据学谱表示和模型，以区域图提出中间目标，再通过先验引导的短时域优化行动。它把全局连通结构和局部模型搜索分开。
  条件与代价：ICML 2026 的主要证据来自离线目标条件任务。它没有同时解决完整在线 STOMP 链条，二者不是新旧替代关系。

#### 相互竞争的解释

- 更好表现来自时间抽象，还是仅仅多用了测试时计算？
- 规划带来的收益来自更好的状态距离，还是更好的候选行为先验？

#### 一种可能的研究办法

假设：在固定总计算下，可靠短时域模型与抽象子目标比一味延长 rollout 更有效。

设计：先使用真模型比较规划器，再替换为相同数据学出的模型。扫描规划深度、候选数量和抽象层数；改变环境后观察恢复。

对照：无规划、同计算的额外策略更新、真模型 oracle；固定离线数据与动作 deadline。

测量：真实回报对总计算的曲线、模型误差、决策延迟、搜索失败率和环境变化后的恢复。

什么结果会反驳该解释：匹配测试时计算后优势消失，或真模型下仍无法获得收益，应分别否定效率解释或规划器设计。

#### 基础准备

- [动态规划：评价、改善与最优递推](../foundations/tabular/dynamic-programming.md)：在准确模型上比较相同 backup 数量的更新次序，可先隔离计算分配，再引入模型也会出错的困难。
- [学习与规划：Dyna、优先扫描和执行时搜索](../foundations/tabular/planning.md)：前驱传播、优先扫描与执行时搜索把计算用于不同位置，由此可问下一次模型查询的净收益。
- [模型学习与规划：MPC、短模型 rollout 和潜在想象](../foundations/deep/model-based.md)：MPC、短模型 rollout 与潜在想象在不同阶段使用模型，比较规划预算时需同时记录模型误差与真实交互成本。

#### 教材与研究者

[Dyna 与模型学习](../textbook/dyna.md) · [转移模型与后果模型](../textbook/models.md) · [时间抽象与规划](../textbook/planning.md) · [Options 与技能发现](../textbook/options.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [George Konidaris](https://yingwen.io/zh/continual-rl/resource/S09/) · [Matthew E. Taylor](https://yingwen.io/zh/continual-rl/resource/S19/) · [Matthew Schlegel](https://yingwen.io/zh/continual-rl/resource/S35/) · [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/) · [Danijar Hafner](https://yingwen.io/zh/continual-rl/resource/S104/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [OGBench](https://github.com/seohongpark/ogbench)：离线目标条件强化学习的环境、数据与基线。 长时域、轨迹拼接、观测与数据覆盖。 边界：离线成功率不能直接说明在线技能发现或持续控制；规划时间另计。

#### 原文与代码

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)（Artificial Intelligence 2023）：STOMP 连接子任务、option、真实后果模型与规划；子任务停止奖励和环境真实回报属于不同接口。
  原文提供更新规则和实验；此处没有可确认的完整作者 STOMP 工程。
- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)（ICML 2021）：用 TD 误差更新平均奖励率，建立无参考状态的差分预测、控制与规划更新。收敛条件针对论文规定的表格问题。
  [作者实验代码](https://github.com/abhisheknaik96/average-reward-methods)：从 prediction_agents.py、control_agents.py 对照奖励率和价值的联合更新。
- [Hansen, Su & Wang · TD-MPC2](https://arxiv.org/abs/2310.16828)（ICLR 2024）：结合短时域潜在模型、终点价值与策略先验做决策时优化，降低长模型 rollout 的需求。
  [作者 TD-MPC2 实现](https://github.com/nicklashansen/tdmpc2)：模型损失、规划函数和策略先验共同构成方法；预训练迁移不等于持续在线积累。
- [Shehmar et al. · Laplacian Representations for Decision-Time Planning](https://proceedings.mlr.press/v306/shehmar26a.html)（ICML 2026）：ALPS 用谱表示组织区域子目标，并结合已学模型和短时域优化完成离线目标条件规划。
  [作者 ALPS 实现](https://github.com/machado-research/ALPS)：hierarchical.py 的区域路径与 optimizer.py 的 CEM 分开；训练数据是离线数据。
- [Gomez, Bowling & Machado · Proper Laplacian Representation Learning](https://arxiv.org/abs/2310.10833)（ICLR 2024）：ALLO 的约束与对偶动态恢复有序谱方向及相应特征值，减少只学到任意旋转子空间的问题。
  [作者 ALLO 实现](https://github.com/tarod13/laplacian_dual_dynamics)：同时读原始变量、对偶变量和正交约束；只保留相邻状态平滑项不是完整算法。

#### 同一问题的跨场景检验

模型想象、工具查询和真实动作可以耗费不同时间。规划更深只有在计入这些成本后改善行为，才有实际意义。

- [可行动的模型：需要预测整个世界，还是足以作决定的后果？](research-atlas.md#horizon-action-models) — 机器人组合任务与语言工具接口

## Ⅴ · 长期适应与探索

<a id="research-group-adaptation"></a>

怎样保留有用能力，并继续获得可学习的经验？

有限容量的学习器无法永久保留全部经验。它要判断哪些旧能力值得维护，哪些过时反应应该改写，以及是否还学得动新目标。这些问题又依赖行为：不再访问某个区域，便可能既无法发现旧模型失效，也无法发现那里的新机会。这一组将零更新保留、新目标学习速度和主动获得经验分别测量，再研究它们在同一生命期中的取舍。

<a id="research-retention-transfer"></a>

### 12 · 怎样保留旧能力，而不把过时知识强加给新任务？

离开一种情境一段时间后，智能体再次遇到它，旧行为可能立即恢复，也可能必须重学。保存轨迹、函数输出或独立模块会占用资源，还可能把过时反应强加给当前任务。保留研究因此需要同时规定未来回访是否可知、任务边界是否提供，以及智能体能否主动取得旧经验。

Replay 保留数据，蒸馏保留函数输出，参数正则保留局部参数几何，生成模型保留经验分布。它们保存的对象不同。低遗忘不等于高前向迁移，快速新学习也不等于旧知识保存良好。

若未来会回访什么已知，设计者可以据此分配保留资源；未知回访和固定容量使这一分配成为学习问题。需要区分“旧规律仍真但暂时不用”“旧规律已经失效”和“旧知识以后可以廉价重学”。可先固定经验池，检验调用频率、失败代价与重学成本各自能否改善保留选择，再允许经验池自行变化。

#### 已有方法怎样处理这个问题

- **CLEAR：经验与行为共同保留**：混合当前和历史经验进行学习，同时在旧经验上约束行为与价值输出。单纯增加 reservoir 容量不等于实现完整 CLEAR。
  条件与代价：允许存储和再访问旧 transition；历史行为概率及离策略修正是接口的一部分。
- **DRAGO：模型的生成回放与真实回访**：合成旧经验用于维护动力学模型，内在奖励引导重新访问熟悉区域，使保留机制同时作用于训练数据和行为。
  条件与代价：论文任务共享动力学与状态空间。模型生成误差、当前任务 replay 和任务边界不能从成本中省略。
- **Reset & Distill：分开新任务学习与保留**：新任务在线学习器重新开始，expert 蒸馏保留过去行为。它明确面对旧参数初始化造成的负迁移，而非只压制遗忘。
  条件与代价：网络重置、离线训练和多个信息来源有额外权限与资源。

#### 相互竞争的解释

- 回访更好来自知识仍在，也可能来自更快重新学习；需要零更新诊断区分。
- 保留正则有效可能只是降低总体学习速度，新任务变化慢时恰好受益。

#### 一种可能的研究办法

假设：保存决策相关行为或后果，比同容量随机旧数据更有效地兼顾回访与新任务学习。

设计：使用 A→B→A 与 A→B→C 两条序列；在回访前先冻结诊断，再恢复更新。分别改变奖励与动力学，不合并解释。

对照：从头训练、连续微调、等容量 reservoir、等计算蒸馏；有任务边界与无边界分别比较。

测量：零更新保留、再学习速度、前向迁移、在线总效用、存储字节和额外计算。

什么结果会反驳该解释：只在恢复训练后才出现回访收益，就不能称为零更新知识保留；新任务成本抵消收益时也不支持总效用提升。

#### 基础准备

- [深度价值学习：DQN、Double DQN 与目标的时间顺序](../foundations/deep/deep-value.md)：Replay 保存旧样本，目标网络保存较慢变化的函数副本；这些保存对象不同，不能用一个遗忘分数解释所有作用。
- [特征、泛化与半梯度控制](../foundations/approximation/features-control.md)：共享特征使新经验同时改变未访问状态的预测，提供了检查旧能力受干扰以及何时可复用的最小情形。
- [离线强化学习：数据支持、策略评估与保守改进](../foundations/deep/offline.md)：保留旧经验并不补齐新行为的数据支持，因而复用旧数据时还要检查价值外推和独立策略评价。

#### 教材与研究者

[知识保留与再适应](../textbook/retention.md) · [持续控制与学习智能体比较](../textbook/control.md) · [转移模型与后果模型](../textbook/models.md) · [实验设计、统计与算法测试](../textbook/experiments.md)

[Doina Precup](https://yingwen.io/zh/continual-rl/resource/S07/) · [George Konidaris](https://yingwen.io/zh/continual-rl/resource/S09/) · [Sarath Chandar](https://yingwen.io/zh/continual-rl/resource/S13/) · [Razvan Pascanu](https://yingwen.io/zh/continual-rl/resource/S14/) · [Michael L. Littman](https://yingwen.io/zh/continual-rl/resource/S42/) · [Samuel Kessler](https://yingwen.io/zh/continual-rl/resource/S100/) · [Piotr Miłoś](https://yingwen.io/zh/continual-rl/resource/S103/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [Continual World](https://github.com/awarelab/continual_world)：基于 Meta-World 的机器人任务序列。 遗忘、前向迁移及回访表现。 边界：任务边界、重置和任务标识应明确；不是严格无重置协议。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。

#### 原文与代码

- [Rolnick et al. · Experience Replay for Continual Learning](https://papers.nips.cc/paper_files/paper/2019/hash/fa7cdfad1a5aaf8370ebeda47a1ff1c3-Abstract.html)（NeurIPS 2019）：CLEAR 将新经验学习、旧经验离策略学习与行为/价值克隆组合，以维持已有能力。
  原始论文为方法依据。AGI-Labs/continual_rl 提供后续比较框架中的复现，不是 CLEAR 原论文作者代码。
- [Fu et al. · Knowledge Retention in Continual Model-Based RL](https://proceedings.mlr.press/v267/fu25f.html)（ICML 2025）：DRAGO 用生成经验保持动力学，再用内在奖励引导回访旧状态；任务共享状态空间与动力学。
  [作者 DRAGO 实现](https://github.com/YixiangSun/drago)：合成回放和真实回访各自消融；“不存旧任务数据”不等于没有当前任务 replay。
- [Ahn et al. · Prevalence of Negative Transfer in Continual RL](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)（ICLR 2025）：Reset & Distill 在新任务中重置在线学习器，并离线蒸馏在线策略与过去 expert，研究负迁移和知识保留。
  [作者 Reset & Distill](https://github.com/hongjoon0805/Reset-Distill)：任务边界、网络重置、离线蒸馏与额外 expert 存储均应纳入实验协议。

#### 同一问题的跨场景检验

错误记忆也可能被有效保留。应区分旧知识仍有用、规律已过时和情境暂时改变，而不把遗忘一律视为坏事。

- [长期记忆：怎样判断一条经验值得保存、调用或遗忘？](research-atlas.md#horizon-memory) — 语言记忆与跨任务经验

---

<a id="research-plasticity"></a>

### 13 · 为什么训练越久，学习新东西反而越慢？

训练很久的网络继续产生梯度，却可能比新初始化更慢地学会同一个新目标。在线回报低还可能因为旧策略没有访问有用数据，不能直接归因于网络学不动。先给 aged 与 fresh 网络相同数据、目标和更新机会，检查可塑性；再恢复在线采样，观察这种学习差异是否仍影响控制。

休眠单元、参数尺度、有效学习率、损失几何和预测函数剧烈变化都可能参与可塑性损失。单一活动度或参数范数是诊断变量，不是可塑性的完整定义。

固定新目标能隔离网络是否还学得动，却不能回答被替换特征以后是否有用。原始 CBP 主要按当前数据估计效用，并不解决遗忘。新的问题是：在相同回收预算下，能否保护低频但仍有效的功能，同时为真正新目标保留学习空间？这要分别在新颖序列和旧情境回访中检验。

#### 已有方法怎样处理这个问题

- **ReDo / Continual Backprop**：ReDo 以低活动度发现可回收单元；CBP 结合效用、成熟期和替换预算引入新特征。替换涉及输入、输出连接与 optimizer 状态。
  条件与代价：单位替换量、初始扰动和实际参数预算必须匹配；激活率提高不证明新任务学得更快。
- **Normalize-and-Project**：归一化网络中，权重范数增长会改变有效学习率。NaP 将归一化与权重投影结合，使这一隐含调度可被显式研究。
  条件与代价：保持有效学习率不必在所有任务最好；原文也分析隐含衰减何时有帮助。
- **C-CHAIN：减少函数 churn**：在参考状态上限制近期更新造成的全动作 Q 变化，以减轻一个数据片段对其他预测的干扰。它干预的是函数变化，而不是直接重置单元。
  条件与代价：需要参考数据与近期网络；正则过强可能妨碍合理策略变化，且不满足无 replay 的最严格协议。

#### 相互竞争的解释

- 新任务学得慢是优化退化，还是新数据没有覆盖需要学习的区域？
- 重置有效是因为产生了新特征，还是因为重置了学习率、动量或价值尺度？

#### 一种可能的研究办法

假设：在固定训练数据上控制有效学习率后，特征回收仍能独立改善长期新学习能力。

设计：保存不同训练年龄的网络，在同一新目标上做短期学习探针；再将同一干预放回在线控制。设置 feature reset 与 optimizer reset 的交叉消融。

对照：fresh、aged、随机部分重置、只重置 optimizer、NaP/步长调整；每组相同参数和调参预算。

测量：相同初始误差附近的下降速度、旧功能扰动、有效学习率、函数 churn、在线回报与长期失稳。

什么结果会反驳该解释：调整有效学习率即可消除 aged/fresh 差异，或回收只改善活动指标却不改善学习曲线，就应缩小特征退化解释。

#### 基础准备

- [函数逼近预测：从回归到 TD 固定点](../foundations/approximation/prediction.md)：固定特征与固定新目标提供简单的学习参照，帮助将表示不足、共享干扰与更新规则的困难分开。
- [深度 RL 的训练循环：经验、目标与更新](../foundations/deep/practice.md)：训练与评价协议中的参数、优化器和数据权限必须明确，才能比较老网络、新初始化及局部重置后是否还能学会新目标。

#### 教材与研究者

[可塑性与特征更新](../textbook/plasticity.md) · [知识保留与再适应](../textbook/retention.md) · [深度价值学习](../textbook/deep-value.md)

[Clare Lyle](https://yingwen.io/zh/continual-rl/resource/S11/) · [Shibhansh Dohare](https://yingwen.io/zh/continual-rl/resource/S12/) · [Razvan Pascanu](https://yingwen.io/zh/continual-rl/resource/S14/) · [Will Dabney](https://yingwen.io/zh/continual-rl/resource/S65/) · [Evgenii Nikishin](https://yingwen.io/zh/continual-rl/resource/S105/) · [Ghada Sokar](https://yingwen.io/zh/continual-rl/resource/S106/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [MinAtar](https://github.com/kenjyoung/MinAtar)：低成本、简化视觉的 Atari 风格环境。 深度控制、函数变化和多 seed 消融。 边界：任务切换须另行规定；切换版本与普通 MinAtar 不是同一协议。
- [AgarCL 与作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整持续世界与分解小游戏。 探索、记忆、信用及可塑性的共同作用。 边界：死亡后局部重生不等于整套学习器清空；混合动作与预算需保持一致。

#### 原文与代码

- [Sokar et al. · The Dormant Neuron Phenomenon in Deep RL](https://proceedings.mlr.press/v202/sokar23a.html)（ICML 2023）：ReDo 根据相对活动度回收休眠单元，重置输入权重并控制输出影响。
  [作者 Dopamine/ReDo](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：weight_recyclers.py 的活动度阈值、替换频率与权重处理。
- [Dohare et al. · Loss of Plasticity in Deep Continual Learning](https://www.nature.com/articles/s41586-024-07711-7)（Nature 2024）：Continual Backprop 以效用、成熟期和替换预算控制特征更新，研究长期新学习能力。
  [作者 CBP 实现](https://github.com/shibhansh/loss-of-plasticity)：gnt.py 和 AdamGnT.py 处理单元替换及 optimizer 状态；不同效用变体要分开。
- [Lyle et al. · Normalization and Effective Learning Rates in RL](https://papers.nips.cc/paper_files/paper/2024/hash/c04d37be05ba74419d2d5705972a9d64-Abstract-Conference.html)（NeurIPS 2024）：研究归一化、权重尺度和有效学习率之间的关系，为可塑性下降提供区别于死神经元的解释。
  本条以原文算法与分析为入口；此处没有可确认的作者原始实现。
- [Tang et al. · Mitigating Plasticity Loss in Continual RL by Reducing Churn](https://arxiv.org/abs/2506.00592)（ICML 2025）：C-CHAIN 对参考状态上的函数变化施加约束，降低当前更新对其他预测的扰动。
  [作者 C-CHAIN 实现](https://github.com/bluecontra/C-CHAIN)：crl_minatar/agents/double_dqn_c_chain.py 包含参考 batch、网络队列和全动作 Q 正则。
- [Mohamed et al. · The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/abs/2505.18347)（2025 初稿；2026 修订）：用完整世界与分解小游戏共同研究探索、记忆、信用与可塑性；一个局部机制难以解释全部困难。
  [作者环境](https://github.com/machado-research/AgarCL)：配套训练在 machado-research/AgarCL-benchmark；动作、重生和观察配置须与实验配对。

#### 同一问题的跨场景检验

预训练策略可能擅长已见变化却难以学会新动力学；固定适应模块的表达不足与在线优化退化需要不同诊断。

- [快状态与慢知识：未知变化应由哪个学习过程承担？](research-atlas.md#horizon-embodied-adaptation) — 机器人本体变化、sim-to-real 与 VLA

---

<a id="research-experience-selection"></a>

### 14 · 应当探索什么、练习什么，以及如何保留未来交互与学习的机会？

智能体可以重复一条熟悉路线、试探未知出口，也可以练习失败后的返回。三者产生不同经验，并占用同一段生命期。新奇区域未必可学，可学规律未必有用，暂时没有奖励的恢复练习又可能保留以后交互的机会。若课程 teacher 可以改环境或重置，它还拥有部署智能体没有的经验选择权限。

预测误差可以鼓励访问未熟悉状态，学习进展可以优先分配可改善任务，恢复策略可以减少陷入失能状态的风险。但噪声环境可能永远产生大误差，短期安全也不等于长期可恢复。

固定数据上的误差下降说明某种经验可学习；智能体自己选数据时，还须判断这种改善是否值得取得。一个精确学会但永不使用的规律可能不如一条稀有恢复经验有用。先把可学性、任务用途与恢复代价分别控制，才能问学习进展是否是合适的探索依据；增加新奇奖励并未完成这种选择。

#### 已有方法怎样处理这个问题

- **RND：用预测误差生成新奇奖励**：固定随机 target 网络，训练 predictor；未熟悉输入产生较大误差。归一化和内外在价值估计决定该信号怎样改变行为。
  条件与代价：误差受输入噪声与表示尺度影响，不自动等价于信息增益或真正的学习进步。
- **ALP-GMM：按学习进展分配课程**：根据环境参数附近表现变化估计绝对学习进展，用混合模型选择下一批训练条件。它调整的是经验分布，而非低层 RL 更新。
  条件与代价：teacher 需要选择训练条件和反复试验的权限；变化的成功率也可能来自噪声。
- **Single-Life RL / QWALE**：以先前经验指导一次部署中的恢复与适应。QWALE 用价值加权的分布匹配信号，帮助 agent 从未见状态回到有用经验支持区域。
  条件与代价：允许先验数据与预训练，但部署中不能随意重置；返回旧分布不等于任意环境中的安全保证。

#### 相互竞争的解释

- 课程收益来自学习进展估计，还是来自 teacher 排除了太难或危险的任务？
- 恢复成功来自在线适应，还是来自预训练已经覆盖了测试变化？

#### 一种可能的研究办法

假设：区分可学习变化与不可约噪声，能够减少无效探索而不损失任务发现。

设计：在小环境同时设置稀疏有用线索、纯噪声区域和可恢复/不可恢复陷阱。比较新奇和学习进展；若有 teacher，另开同权限实验。

对照：随机探索、固定内在权重、oracle 可学区域仅作诊断上界；匹配先验数据和外部救援次数。

测量：有用覆盖、噪声区域停留、首次任务发现、恢复成功、失能时间、外在累计奖励和人工干预。

什么结果会反驳该解释：去掉 teacher 的环境筛选权限后优势消失，或探索仅增加覆盖而未改善后续学习能力，则需更换机制解释。

#### 基础准备

- [多臂老虎机：估计、探索与直接策略学习](../foundations/tabular/bandits.md)：探索在当期收益与信息之间取舍；估计更准确却未必选择更好，说明取得的信息还需要任务用途。
- [探索与不确定性：后验、乐观估计和时间一致行动](../foundations/deep/exploration.md)：环境噪声与知识不足会产生不同的信息价值，时间一致探索进一步检验哪些经验值得为之连续行动。
- [约束强化学习：占据测度、拉格朗日与可行策略](../foundations/deep/constraints.md)：平均成本约束不等于逐步安全或长期可恢复性，选择探索经验时需要分别规定这些未来交互条件。

#### 教材与研究者

[探索与经验选择](../textbook/exploration.md) · [目标条件化与子任务构造](../textbook/goals.md) · [交互、奖励与优化目标](../textbook/objectives.md)

[Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/) · [Peter Stone](https://yingwen.io/zh/continual-rl/resource/S10/) · [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/) · [Chelsea Finn](https://yingwen.io/zh/continual-rl/resource/S50/) · [Tim Rocktäschel](https://yingwen.io/zh/continual-rl/resource/S60/) · [Ian Osband](https://yingwen.io/zh/continual-rl/resource/S64/) · [Pierre-Yves Oudeyer](https://yingwen.io/zh/continual-rl/resource/S71/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [TeachMyAgent](https://github.com/flowersteam/TeachMyAgent)：teacher 可以选择环境参数的自动课程实验。 学习进展估计和经验分配。 边界：必须计算 teacher 权限、失败尝试和环境重置，不能视作完全自主单生命期。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。
- [AgarCL 与作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整持续世界与分解小游戏。 探索、记忆、信用及可塑性的共同作用。 边界：死亡后局部重生不等于整套学习器清空；混合动作与预算需保持一致。

#### 原文与代码

- [Burda et al. · Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)（ICLR 2019）：以固定随机网络的预测误差构造新奇奖励，减少显式密度建模需求，但误差不是学习进展本身。
  [作者 RND 实现](https://github.com/openai/random-network-distillation)：观察内在奖励归一化、内外价值估计及 PPO 训练，不仅复制预测器损失。
- [Portelas et al. · Teacher Algorithms for Curriculum Learning of Deep RL](https://arxiv.org/abs/1910.07224)（CoRL 2019 / PMLR 2020）：ALP-GMM 用目标/环境参数附近的绝对学习进展分配训练分布。teacher 可以选择环境条件。
  [作者团队课程实验平台](https://github.com/flowersteam/TeachMyAgent)：支持 ALP-GMM 等方法的后续比较框架；不是无 teacher 的单生命期协议。
- [Chen et al. · You Only Live Once: Single-Life Reinforcement Learning](https://arxiv.org/abs/2210.08863)（NeurIPS 2022）：研究一次部署生命期内利用先前经验完成适应与恢复，突出无法随意外部重置的限制。
  [作者 Single-Life RL / QWALE](https://github.com/anniesch/single-life-rl)：envs/、data/ 和 train.py 分别提供环境、先验数据及部署学习；比较时应计算预训练和先验数据。

#### 同一问题的跨场景检验

探索会消耗设备状态、人类注意力和可恢复性。保持未来交互机会是行动后果，不只是奖励中的一个新奇项。

- [单生命与安全：怎样学习，而不耗尽以后的学习机会？](research-atlas.md#horizon-safe-autonomy) — 真实机器人与持续服务
- [目标形成与开放式学习：怎样选择值得学的下一件事？](research-atlas.md#horizon-open-ended-goals) — 开放世界、生成式课程与程序技能

## Ⅵ · 完整智能体与长期学习

<a id="research-group-systems"></a>

各部分的学习怎样共同改善长期行为？

把有效的局部机制放在一起，可能改变每一部分原有的学习条件。状态更新会改变模型输入，技能更新会改变预测后果，规划又会决定下一段真实经验。完整架构的研究要找出这些作用何时相互帮助、何时产生滞后或干扰；评价则要检验局部改善是否抵偿计算、探索和维护成本。共同任务可以揭示一种相互作用，跨任务检验再判断它在哪些条件下仍成立。

<a id="research-integrated-architecture"></a>

### 15 · 状态、知识、技能与规划怎样共同构成持续智能体？

技能学会更常到达有利终点后，旧模型仍可能低估它；高层因此很少调用，模型又缺少纠正所需的经验。反过来，状态记住一个关键提示，也可能让多个模型同时更容易预测。这类相互作用在单独训练模块时不会出现，需要把输入、行为、目标和更新时序放进同一个持续交互过程研究。

Dyna 给出直接学习、模型学习和规划的基本接口；Alberta Plan 与 OaK 把长期知识构建、时间抽象及规划纳入更广路线。它们提出研究组织方式，但模块相连本身不能保证收益相加。

STOMP 的实验分阶段学习，尚未证明所有模块同步变化时仍能有效协作。撤掉阶段间冻结后，状态、技能、模型和规划相互改变学习条件。一个可先解决的问题是：在固定维护预算下，下游规划收益能否反向决定重学、保留或停用哪些知识？先用给定候选验证这一反馈，再加入自动构造。

#### 已有方法怎样处理这个问题

- **STOMP 的模块链**：子任务规定练习目标，option 学习如何实现，模型估计执行的真实奖励和停止后果，规划用这些估计改善任务价值。原文的阶段实验分别建立这些关系，同时运行会进一步引入目标与数据随学习改变的问题。
  条件与代价：子任务停止奖励用于行为学习，模型预测环境实际收益；停止条件和时间尺度需要明确对应。原文讨论的同时学习及下游用途反馈仍是进一步研究方向。
- **Alberta Plan / OaK 的研究路线**：把有限资源下的实时学习与经验构造知识连接起来，强调可由经验检验、可支持规划的时间抽象。研究焦点是模块如何在生命期中产生并改进。
  条件与代价：研究纲领和讲座不等于完整通用 agent 的可复现性能报告；能力与效率需要对应实现和测试。
- **Oak 的选择性信用与事件驱动方向**：公开研究文章用稀疏相关信号与大量干扰说明选择性步长的动机，并展示 NetworkIDBD 的预测演示。事件驱动方向涉及计算分配，但对应页面尚无技术正文。
  条件与代价：不能从 NoisyMNIST 图示推出持续控制收益；没有完整更新与运行包时，不能复现或比较其能耗主张。

#### 相互竞争的解释

- 整机提高回报可能只是参数和计算更多，或某一个强模块掩盖了其他模块无效。
- 模块之间看似协同，可能来自共同访问了额外任务标签或真实环境状态。

#### 一种可能的研究办法

假设：有限维护预算优先用于改变规划选择的失准知识，比均匀更新各模块更能减少持续系统的变化后损失。

设计：先固定状态与少量技能候选，只让技能后果和模型更新速度可控地变化。比较哪些模型应重学、暂时停用或保留，再逐步放开状态和技能学习。另设版本改变但后果不变的负对照，避免把编号一致误当作必要条件。

对照：均匀、周期、预测误差和规划用途分配；固定总容量、真实数据与更新预算。分别冻结模块，并区分训练期间撤掉模块与部署时屏蔽其读出。

测量：整段效用、规划动作排序、接口失准与恢复延迟、各模块维护成本，以及停用后重新需要该知识时的重学成本。

什么结果会反驳该解释：用途分配未减少预定范围的恢复损失，或关键模型虽更新却未改变规划错误，就削弱这一闭环解释。某次部署删除模块无影响只说明该时点未依赖其读出，不足以否定历史训练贡献。

#### 基础准备

- [学习与规划：Dyna、优先扫描和执行时搜索](../foundations/tabular/planning.md)：Dyna 中同一经验的直接学习、模型学习和规划三种用途，构成检查模块接口与相互影响的最小闭环。
- [大规模训练：算法与系统怎样共同设计](../foundations/deep/systems.md)：行为、学习与参数版本各有时间顺序；检查这些接口后，才能研究多个持续更新的模块能否协同改善行为。

#### 教材与研究者

[持续学习的智能体架构](../textbook/architectures.md) · [智能体状态与递归学习](../textbook/state.md) · [通用价值函数与预测知识](../textbook/gvf.md) · [目标条件化与子任务构造](../textbook/goals.md) · [Options 与技能发现](../textbook/options.md) · [转移模型与后果模型](../textbook/models.md) · [时间抽象与规划](../textbook/planning.md) · [流式更新与稳定性](../textbook/streaming.md)

[Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/) · [Marlos C. Machado](https://yingwen.io/zh/continual-rl/resource/S06/) · [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/) · [Patrick M. Pilarski](https://yingwen.io/zh/continual-rl/resource/S18/) · [Joseph Modayil](https://yingwen.io/zh/continual-rl/resource/S32/) · [Khurram Javed](https://yingwen.io/zh/continual-rl/resource/S33/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。
- [AgarCL 与作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整持续世界与分解小游戏。 探索、记忆、信用及可塑性的共同作用。 边界：死亡后局部重生不等于整套学习器清空；混合动作与预算需保持一致。

#### 原文与代码

- [Sutton et al. · Reward-Respecting Subtasks](https://arxiv.org/abs/2202.03466)（Artificial Intelligence 2023）：STOMP 连接子任务、option、真实后果模型与规划；子任务停止奖励和环境真实回报属于不同接口。
  原文提供更新规则和实验；此处没有可确认的完整作者 STOMP 工程。
- [Sutton, Bowling & Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173v3)（2022 首稿；2023 v3 研究纲领）：从实时学习、状态与预测走向子任务、模型和规划，并提出以知识用途反馈决定替换的研究路线；路线可修订，不是已完成系统的性能结论。
  研究纲领，不是一套完整 agent 发布包。
- [Sutton · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)（RLC 2025 讲座入口）：以经验中的时间抽象、可检验知识和规划构造智能体；适合分析模块之间的信息与资源接口。
  公开入口为架构讲座；没有在此页面发布完整 OaK agent 代码。
- [Oak Lab · Learning from Experience instead of Curated Datasets](https://oaklab.ai/posts/learning-from-experience-instead-of-curated-datasets)（2026 官方研究文章）：用信号、干扰特征和噪声目标的受控预测实验讨论选择性信用，展示 IDBD 与 NetworkIDBD。
  文章未给出完整 NetworkIDBD 更新规则与公开复现包。NoisyMNIST 演示不等于长期控制实验。
- [Oak Lab · Event-Driven Neural Networks with Batch-Size One Learning Algorithms](https://oaklab.ai/posts/event-driven-computation-with-batch-size-one)（官方主题入口）：页面仍为 Coming soon；仅说明研究主题，不能由标题推导算法、能耗或性能。
  无公开技术正文或实现入口。

#### 同一问题的跨场景检验

持续对象可以是权重、记忆、程序或更新规则。多种对象相连不证明协同，需要追踪版本、证据和各自代价。

- [学习规则与自修改：什么证据支持改变学习过程？](research-atlas.md#horizon-self-modification) — 可编辑程序与语言智能体

---

<a id="research-measurement"></a>

### 16 · 什么实验能区分“仍在更新”与“仍在有效学习”？

长曲线后半段变好，可能因为学习器积累了知识，也可能因为任务变容易或它避开了困难区域。冻结测试可以检查当前行为，却撤掉了之后继续更新的过程；回访和新目标测试又回答不同问题。评价应先规定要估计哪种改善、独立运行如何产生，以及诊断是否会改变主生命期。

在线奖励评价实际学习过程；冻结策略诊断回答某个时间点会做什么；回访与新目标探针分别测试保留与可塑性。一个指标不能代替另外几个。跨种子区间与多任务稳健聚合有助于避免只看最好结果。

冻结策略的测试撤掉了继续学习的过程，而连续在线回报又混合了初始能力、探索和适应。需要问：在相同起点与资源下，所声称的机制究竟改善了哪一段生命期，改善幅度是否足以抵偿代价？有限实验可以排除特定范围的失败解释，但既不能由长曲线证明永远改善，也不能由不显著结果证明机制无用。

#### 已有方法怎样处理这个问题

- **生命期指标与诊断分离**：把实际交互累计效用作为主要结果；冻结、回访和新任务探针在独立拷贝中进行，避免诊断数据改变主生命期。
  条件与代价：拷贝只用于离线诊断且不把结果回流到部署 agent；无法复制真实系统时，需要另行设计安全评估。
- **rliable 与分层统计**：以独立训练运行和任务为重采样层级，报告区间、分布和稳健聚合，避免少数异常高分支配平均数。
  条件与代价：相邻窗口不是独立 seed；只挑显著任务或事后选择最优时间点会使区间失去原解释。
- **机制诊断 → 简化平台 → 复杂世界**：解析任务验证公式，Forager 检查信息与状态，AgarCL 的小游戏及完整环境检查组合难点。不同平台提供互补证据，不按难度大小简单替代。
  条件与代价：某一平台上的失败可限制具体主张，但不能直接宣布一种机制普遍无用。

#### 相互竞争的解释

- 后期变好可能因为任务自然变容易，或评价状态更接近训练分布。
- 极小均分提升可能来自种子方差、任务权重或选择性报告，而不是算法机制。

#### 一种可能的研究办法

假设：一个被称作“持续学习改进”的方法，应在共同预算下至少改善预先指定的生命期指标，而不是仅改善代理诊断。

设计：预先固定任务顺序、变化时点、主指标和停止条件。开发集调参；测试集运行完整生命期。使用配对任务种子，保留失败和超时运行。

对照：固定策略、不学习、从头训练、持续基础算法与单机制消融；匹配所有环境干预。

测量：在线效用及区间、适应/保留/可塑性诊断、峰值内存、时延分位数、崩溃和干预频次；显示每任务结果。

什么结果会反驳该解释：主指标无可靠改善，或改善仅存在于获得额外 reset/搜索预算的条件，就不支持原定端到端主张；负结果仍可支持更窄机制结论。

#### 基础准备

- [多臂老虎机：估计、探索与直接策略学习](../foundations/tabular/bandits.md)：同一奖励流中估计误差与选取动作的收益可以给出不同结论，要求评价指标先对应明确对象。
- [深度 RL 的训练循环：经验、目标与更新](../foundations/deep/practice.md)：分开训练过程、冻结评价、随机重复和计算预算，才能判断一个机制改善了哪一段学习过程。
- [离线强化学习：数据支持、策略评估与保守改进](../foundations/deep/offline.md)：策略学习、超参数选择和独立离线评价使用不同数据角色，避免把用于选择的方法再次当作独立验证。

#### 教材与研究者

[实验设计、统计与算法测试](../textbook/experiments.md) · [交互、奖励与优化目标](../textbook/objectives.md) · [持续学习的智能体架构](../textbook/architectures.md) · [可塑性与特征更新](../textbook/plasticity.md) · [知识保留与再适应](../textbook/retention.md)

[Martha White](https://yingwen.io/zh/continual-rl/resource/S03/) · [Adam White](https://yingwen.io/zh/continual-rl/resource/S04/) · [David Abel](https://yingwen.io/zh/continual-rl/resource/S08/) · [Csaba Szepesvári](https://yingwen.io/zh/continual-rl/resource/S15/) · [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/) · [Rishabh Agarwal](https://yingwen.io/zh/continual-rl/resource/S109/)

#### 实验平台

- [解析小 MDP 与短序列](experiments.md)：已知转移、已知目标、固定种子的低维问题。 精确最优值、有限差分、前后向等价、机制反例。 边界：用来验证算法与解释，不估计复杂任务性能。
- [CSuite](https://rl-csuite.readthedocs.io/en/latest/)：持续交互的小型诊断环境。 低成本测试适应与学习机制，再接复杂平台。 边界：每个任务测量的困难有限；不可用单一分数代表整个 CRL。
- [Forager](https://github.com/andnp/forager)：部分可观测的持续资源获取。 信息历史、状态容量、更新成本与长期行为。 边界：感知模式和地图差异会改变问题；不只比较网络名字。
- [Continual World](https://github.com/awarelab/continual_world)：基于 Meta-World 的机器人任务序列。 遗忘、前向迁移及回访表现。 边界：任务边界、重置和任务标识应明确；不是严格无重置协议。
- [AgarCL 与作者基线](https://github.com/machado-research/AgarCL-benchmark)：完整持续世界与分解小游戏。 探索、记忆、信用及可塑性的共同作用。 边界：死亡后局部重生不等于整套学习器清空；混合动作与预算需保持一致。

#### 原文与代码

- [Abel et al. · A Definition of Continual Reinforcement Learning](https://arxiv.org/abs/2307.11046)（NeurIPS 2023）：用智能体对行为基础的持续搜索定义持续性。它与“环境显式切换若干任务”的定义不相同。
  理论定义论文；没有需要复现的单一控制算法。
- [Elelimy et al. · Rethinking the Foundations for Continual RL](https://rlj.cs.umass.edu/2025/papers/Paper243.html)（RLC 2025）：以 history process 和 deviation regret 研究学习过程的评价。比较对象与可实现偏离集合仍需明确，不能只把指标名称换掉。
  论文给出形式化与实验；此处没有可确认的作者原始代码入口。
- [Agarwal et al. · Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264)（NeurIPS 2021）：用区间估计、分层 bootstrap 和稳健聚合处理多任务、多随机种子的评价不确定性。
  [作者 rliable](https://github.com/google-research/rliable)：独立随机单位是 run/任务而非同一曲线的相邻时间点；统计工具不能修复不公平协议。
- [Forager: a Lightweight Testbed for Continual Learning with Partial Observability in RL](https://arxiv.org/abs/2605.01131)（2026 预印本）：在长期交互中隔离观测与状态构造问题，研究局部感知下的资源获取。
  [作者 Forager 环境](https://github.com/andnp/forager)：配置 observation_mode 和地图；start()/step() 接口不同于 Gymnasium 五元组。
- [Mohamed et al. · The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/abs/2505.18347)（2025 初稿；2026 修订）：用完整世界与分解小游戏共同研究探索、记忆、信用与可塑性；一个局部机制难以解释全部困难。
  [作者环境](https://github.com/machado-research/AgarCL)：配套训练在 machado-research/AgarCL-benchmark；动作、重生和观察配置须与实验配对。

#### 同一问题的跨场景检验

有状态系统的收益需与同初始能力、同资源的无状态或冻结对照比较；同一对局的双方适应也不是独立样本。

- [相互适应与非平稳性：怎样区分自己、伙伴和共同约定的变化？](research-atlas.md#horizon-social-learning) — 人机合作与多机器人开放团队

<a id="research-horizons"></a>

## 跨场景检验：持续强化学习问题在不同世界中的表现

前面的研究主线从目标、状态、预测、信用和资源约束提出问题。这里改变的是检验情境：工具动作具有内部结构，机器人恢复会消耗真实时间，他者还可能同时学习。这些变化使部分教材条件不再成立，但大模型、机器人或多智能体这些名称本身没有规定新的学习目标。每个场景都应回到一个可以隔离的基础困难。

温颖在《从现代深度强化学习到持续强化学习》中提出一个研究视角：把预训练视为起点，把运行中可检验的第一人称经验视为后续知识来源；状态、预测、技能、模型和学习规则都可以成为更新对象。以下沿此视角提出问题，同时区分已有论文证据、邻近研究和仍待检验的设想。这不是对 CRL 的唯一界定，也不要求所有系统采用同一种架构。

[作者观点原文](https://yingwen.io/zh/blog/from-modern-deep-rl-to-continual-rl/)

读一项工作，先问什么在持续、反馈从哪里来、谁被重置、占用哪些资源，以及评价谁的收益。先用最小对照辨认候选机制，再扩大场景；不把复杂系统的总分直接解释成所有模块都有效。下面是尚待实施的实验提纲，需另行确定效应大小、独立运行数和停止条件，不是本网站已经完成的实验。

### 先区分：什么在更新，在哪个生命期中更新

| 研究协议 | 持续对象 | 反馈 | 不能据此推出 |
| --- | --- | --- | --- |
| 离线后训练 | 更新模型权重；数据与训练任务由外部管线提供。 | 偏好、答案检查或任务回报。 | 可以使用 RL，但不因此证明部署中的同一个体能够持续学习。 |
| 单次推理与上下文适应 | 搜索树、工作上下文或递归状态随当前交互更新。 | 内部评分，或当前试次获得的外部反馈。 | 固定参数可以支持学习式适应；需要另问知识是否跨试次保留，以及再次计算的成本。 |
| 持续记忆与程序技能 | 保留经验摘要、效用估计、可执行技能及其适用条件。 | 后续调用的成功、失败、时间和副作用。 | 跨任务持久化不等于有效迁移；历史错误也可能持续传播。 |
| 同一个体内在线更新 | 状态、模型、策略或学习规则在部署期间改变。 | 当前行动带来的真实结果；可能延迟或缺失。 | 不能把暂停世界、额外重放或人工恢复视为零成本。世界可以平稳，学习仍可能必要。 |
| 车队或平台级迭代 | 汇总多个个体的数据，训练并发布新模型版本。 | 部署日志、人工标注与批量评价。 | 系统层面确实在改善，但不等同于某个机器人现场独立适应；信息共享与发布延迟是协议的一部分。 |
| 修改学习系统本身 | 工具、检索、验证、规划甚至生成下一次修改的程序。 | 独立评测与部署表现。 | 改写程序不自动属于 RL；基准搜索收益也不等于开放部署下持续净收益。 |

### 语言环境的五个分析层次

| 层次 | 例子 | 需要回答 |
| --- | --- | --- |
| 外部世界 | 真实文件、服务、用户和未结束的工作。 | 动作改变了什么？哪些影响不能由重开会话消除？ |
| 运行时 | 权限、工具版本、模型版本、超时、可用内存。 | 失败来自世界规律，还是接口与执行条件改变？ |
| 内部任务 | 查证、定位错误、试验修补方案、决定是否求助。 | 这个中间目标由谁产生？它的成功是否支持总体目标？ |
| 证据 | 进程返回值、独立测试、后续用户反馈。 | 实际执行过什么？工具成功返回不等于结果正确，测试通过也不等于长期有用。 |
| 更新载体 | 上下文、事实记忆、程序技能、价值估计、权重。 | 哪一处因证据而改变？保持多久？如何撤销过时或错误的更新？ |

这五层是分析语言环境的区分，不是规定智能体必须具备的五个模块。例如，一条失败日志只有与动作、前提、版本和后续用途联系起来，才可能成为可学习的经验。世界继续运行的时间也不能用生成 token 的数量替代。

### 持续控制、信用与反馈

局部反馈是否支持生命期改善，学习还能否保留未来行动的机会？

<a id="horizon-language-actions"></a>

#### 选择性观察与多粒度信用：局部反馈怎样连接长程后果？

检验场景：语言工具与软件维护

一个语言智能体能否从自己执行过的行动中，学会以后更有效地观察、行动与核验？

具体情境：修复服务故障时，读取日志、改配置和重启进程是不同的外部动作。一个 shell 命令又包含多个 token。命令返回零可能只是成功写入了错误配置；真正损失可能在下一次部署才出现。

问题设定：语言模型生成结构化动作，工具改变部分可观测的世界。奖励可以来自独立测试、用户验收或后续运行，但必须预先说明如何组合。需要同时处理动作内部的生成信用与动作之间的时间信用。内部思考步不一定推进世界，工具执行和等待通常会。

持续对象：保留项目经验、工具后果模型或更新后的策略；分别比较，不把它们合成一个“记住了”的标签。

反馈：原始 stdout、测试结果和学习奖励分开记录。验证器只能看到它能够检查的性质。

生命期：跨 issue 的知识可以保留；测试容器是否重建、工作成果是否延续需单列。

资源：共同预算包含输入/输出 token、工具调用、等待时间、训练更新和失败后的恢复。

核心困难：序列末尾的分数无法直接说明哪次观察或哪段命令有贡献。工具版本变化又使旧轨迹失效。若奖励只覆盖显式测试，优化器可能学会绕过测试。把每个 token 都按同一个世界时间折扣，还会引入由措辞长度造成的目标差异。

##### 已有证据与适用边界

- [POAD：Reinforcing Language Agents via Policy Optimization with Action Decomposition](https://arxiv.org/abs/2405.15821) · Muning Wen、Ziyu Wan、Jun Wang 等 · NeurIPS 2024

  把语言动作分解为 token 决策，推导动作内与动作间一致的 Bellman backup，再用于 PPO。它处理的是语言生成与环境动作两个粒度的信用。

  边界：证据来自规定交互任务，不是持久软件服务的长期在线部署。分解信用也不能补回验证器没有观察到的副作用。

  [作者链接的 ADRL 仓库](https://github.com/morning9393/ADRL)：提供 mappo 目录与依赖；仓库首页缺少完整运行说明。不能把“代码可访问”写成已独立复现。

- [Learning CLI Agents with Structured Action Credit under Selective Observation](https://arxiv.org/abs/2605.08013) · Haoyang Su、Ying Wen · 2026 预印本

  把 token 预算内的观察选择与结构化动作信用分开；A3 综合轨迹回报、AST 子链残差和轨迹树比较，ShellOps 提供可验证 CLI 任务。

  边界：观察选择发生在推理时，策略信用用于训练。论文未因此证明跨长期部署的持续权重更新；AST 结构也不是真实因果贡献的直接观测。

  [A3 与 ShellOps 作者代码](https://github.com/Hoyant-Su/Agentic-RL-A3)：基于 verl/verl-agent，需模型训练资源和仓库执行环境。先对齐数据划分、动作解析与验证器。

##### 仍需回答

- 能否把“取得有用证据”作为可学的行动价值，而不是奖励更多读取和更长推理？
- 当反馈跨多次任务才出现时，怎样追踪信用、权限和版本，而不把未来信息泄漏给早期决策？

##### 可检验的实验提纲

假设：固定学习器和知识载体后，依据行动后果选择观察能减少无效诊断，而跨调用信用能减少局部修补造成的后续回归；两种收益应可分别辨认。

设计：构造共享隐藏依赖的仓库问题序列，其中短期测试通过的修补可能破坏后续功能。先做观察选择与信用规则的交叉实验，两者各自只有开、关两种条件；保留每组自己的工作空间，使用独立检查服务。

对照：相同初始模型、记忆形式和参数更新规则，匹配工具调用、上下文及等待总预算。识别这两个因素后，再单独改变持久记忆或在线权重更新。不同策略会生成不同历史，不强行复制经验。

测量：观察选择对无效读取的作用，信用规则对延迟回归的作用，以及两者的交互；另报整段修复净收益、求助和端到端成本。

反驳条件：预算匹配后观察选择未减少预定幅度的诊断成本，或信用变化未减少延迟回归，就分别削弱相应机制；组合总分上升不能替代这两项检验。

##### 学习与实验入口

[智能体状态与递归学习](../textbook/state.md) · [时间信用分配与资格迹](../textbook/credit.md) · [持续控制与学习智能体比较](../textbook/control.md) · [奖励假设与奖励设计](../textbook/reward-design.md)

环境：[CL-Bench：跨实例知识利用](https://github.com/pgasawa/continual-learning-bench)：可从 codebase adaptation 入手；其实例协议不自动具有所有动作后果的永久延续。

实现：[多步 SARSA 教学实现](https://yingwen.io/zh/continual-rl/code/nstep_sarsa/)：用于检查多步回报与时间信用；不处理动作内部 token，也不是 POAD 的替代实现。

相关问题：[长期记忆：怎样判断一条经验值得保存、调用或遗忘？](research-atlas.md#horizon-memory) · [可行动的模型：需要预测整个世界，还是足以作决定的后果？](research-atlas.md#horizon-action-models) · [学习规则与自修改：什么证据支持改变学习过程？](research-atlas.md#horizon-self-modification)

<a id="horizon-safe-autonomy"></a>

#### 单生命与安全：怎样学习，而不耗尽以后的学习机会？

检验场景：真实机器人与持续服务

没有免费的外部重置时，探索、恢复、求助和任务执行应如何共同评价？

具体情境：机械臂把物体推到够不到的角落。短期任务失败不是主要问题；真正困难是它失去了之后的有效交互。如果人把物体放回原位，这次救援提供了能力与成本，不能在学习曲线上消失。

问题设定：世界状态由前面的行动延续。允许的帮助、恢复动作、安全控制器和初始经验必须写入协议。Single-life 的一次部署、reset-free 的连续训练、长期服务的反复任务是相关但不同的问题；都不等于“无限长 episode”。

持续对象：世界资产、损伤、知识和学习器均有状态；设备恢复不是学习器重置。

反馈：外在收益之外单列风险、失能、恢复与求助。硬安全约束不能只靠任意惩罚权重替代。

生命期：独立随机世界可用于多次评测；每个受评个体内部不得暗中重生。

资源：报告人类分钟数、恢复操作、风险暴露、设备停机以及全过程真实时间。

核心困难：恢复需要访问训练未覆盖的状态；但尝试恢复也可能造成损伤。安全限制改变可访问经验，从而影响学习可识别性。若未知系统允许一次错误就不可逆失效，不能同时无条件保证任意探索与零失效。

##### 已有证据与适用边界

- [You Only Live Once：Single-Life Reinforcement Learning](https://arxiv.org/abs/2210.08863) · Annie S. Chen、Archit Sharma、Sergey Levine、Chelsea Finn · NeurIPS 2022

  定义借助先前经验、在一个测试试次中应对新情况并完成任务的设定。QWALE 用价值加权的分布匹配，帮助回到已有经验支持的有用区域。

  边界：一次成功任务不等于长期多目标服务；先验数据不可省略，也没有任意环境中的安全恢复保证。

  [Single-Life RL 作者实现](https://github.com/anniesch/single-life-rl)：可追踪预训练经验与测试时更新的权限，不能只计算最后一次试次的训练成本。

- [Reset-free Reinforcement Learning with World Models](https://arxiv.org/abs/2408.09807) · Zhao Yang、Thomas Moerland、Mike Preuss、Aske Plaat、Edward Hu · TMLR 2025

  MoReFree 在世界模型学习中让探索和目标条件行为更多覆盖任务相关状态，学习返回与再出发，而不是依赖每次外部物理 reset。

  边界：主要证据来自可控模拟任务，训练的 reset-free 与评测 episode 要分开；回到指定区域不是一般不可逆环境的安全证书。

  [MoReFree 官方实现](https://github.com/yangzhao-666/MoReFree)：有任务配置、目标采样与模型学习代码。任务相关状态的先验权限是复现实验的一部分。

- [Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning](https://arxiv.org/abs/2410.21845) · Jianlan Luo、Charles Xu、Jeffrey Wu、Sergey Levine · 2024 预印本；Science Robotics 2025

  用演示、二元奖励分类器和训练中的人工纠正支持真实机器人学习。它使干预成为可研究的数据来源，而不是假定训练完全无人协助。

  边界：人为布置、奖励样本和接管均需计入。高成功率不能单独证明无人恢复或长期损伤风险可接受。

  [HIL-SERL 作者系统](https://github.com/rail-berkeley/hil-serl)：包括控制与采集集成；需要真实硬件、安全配置和操作者，不适合将示例直接当成无人值守部署。

##### 仍需回答

- 何时应该花费奖励去取得关于可恢复性的证据，何时应立即请求帮助？
- 把帮助次数降到更少时，风险是否转移给了后续任务、设备或人工维修？

##### 可检验的实验提纲

假设：学习可恢复性并在不确定时求助，能在同一帮助预算下提高净服务收益。

设计：从带可恢复角落与受约束风险的模拟操作开始，显式建模每次求助的时间和物理后果。先验数据固定；任务结束不重置物体。真实验证只在已有安全边界内开展。

对照：任务策略、固定恢复规则、学得恢复策略及有限帮助策略；记录每组的帮助和失败，不给某组免费 reset。

测量：有效工作占比、失能持续时间、恢复率、累计收益、帮助分钟数与风险事件分布。

反驳条件：收益依赖未记账的人工摆放，或以更少小失败换来更多重大失效，就不能支持安全长期改善。

##### 学习与实验入口

[持续控制与学习智能体比较](../textbook/control.md) · [探索与经验选择](../textbook/exploration.md) · [平均奖励与差分价值](../textbook/average.md) · [Options 与技能发现](../textbook/options.md)

环境：[环境持续性、重置与真实时间](worlds.md)：选择平台前先核对失能、身体死亡、重新生成与外部重置的不同语义。

实现：[持续控制与学习器比较](../textbook/control.md)：先掌握生命期评价与可行比较器；教学环境不是机器人安全认证。

相关问题：[快状态与慢知识：未知变化应由哪个学习过程承担？](research-atlas.md#horizon-embodied-adaptation) · [奖励与反馈学习：偏好变化时，优化目标怎样保持可信？](research-atlas.md#horizon-human-feedback) · [目标形成与开放式学习：怎样选择值得学的下一件事？](research-atlas.md#horizon-open-ended-goals)

<a id="horizon-human-feedback"></a>

#### 奖励与反馈学习：偏好变化时，优化目标怎样保持可信？

检验场景：人机协作、个性化服务与机器人反馈

智能体怎样在不过度索取人类注意力的条件下，校准自己正在优化的目标？

具体情境：家庭机器人学会快速清空桌面，却把主人希望保留的物品收走。早期两段视频的偏好标签不足以刻画后来出现的情境。再次询问有成本；不询问可能持续优化错误目标。

问题设定：偏好、纠正、示范与任务结果是不同反馈渠道。学习器同时改变策略和奖励估计，因而会访问新的、标签稀缺区域。要区分固定偏好下估计不确定性、人类偏好实际变化，以及不同人之间的合理差异。

持续对象：用户条件、奖励模型、反馈来源与策略；不把不同用户的私有经验混为共享真值。

反馈：允许拒答、延迟、噪声和不一致；模型裁判属于代理反馈，不等同于人的真实评价。

生命期：从短期片段偏好到长期后果的评价需另行设计；历史标签可保留但未必永远有效。

资源：以标注时长、被打断次数和纠错负担衡量帮助，不只统计标签数。

核心困难：策略会利用奖励模型的误差。只在熟悉轨迹上查询可以得到高预测准确率，却放过危险的新行为。单纯加强旧偏好约束也可能妨碍对新情境的合理调整。

##### 已有证据与适用边界

- [PEBBLE：Feedback-Efficient Interactive Reinforcement Learning](https://arxiv.org/abs/2106.05091) · Kimin Lee、Laura Smith、Pieter Abbeel · ICML 2021

  结合无监督预探索、偏好奖励学习与 replay 重新标注，使更新后的奖励估计可用于过去经验。它明确展示反馈学习与策略学习之间的耦合。

  边界：对片段的偏好与长期人类效用仍有距离；重新标注也依赖保存过去数据的权限。

  [作者 B-Pref 中的 PEBBLE 与反馈模型](https://github.com/rll-research/BPref)：包含噪声、错误、短视、跳过等模拟教师与 PEBBLE 脚本；模拟教师不是实际用户研究。

- [Improving Reward Models with Proximal Policy Exploration for Preference-Based Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2025/file/9cd2a03f427acc03b6ddbb9c8f3be57c-Paper-Conference.pdf) · Yiwen Zhu、Jinyi Liu、Pengjie Gu 等 · NeurIPS 2025

  指出偏好缓冲区覆盖不足会使奖励模型在当前策略附近的新区域失准；通过邻近策略探索与混合查询改善覆盖和反馈利用。

  边界：控制任务上的奖励模型改善，不证明真实偏好漂移、多用户冲突或长期安全已解决。更多覆盖也有实际试错成本。

  [PPE 作者实现](https://github.com/yiwenzhu-evan/PPE)：提供 DMControl、MetaWorld 与偏好学习基线。先匹配查询数量和交互预算，再比较查询策略。

##### 仍需回答

- 怎样识别“我的奖励模型错了”，而不是通过更强策略把同一错误优化得更好？
- 个性化学习如何在保留用户控制、数据隔离与明确撤回机制的同时共享可复用知识？

##### 可检验的实验提纲

假设：基于决策不确定性而非单纯预测误差选择查询，可以降低人类负担并减少奖励投机。

设计：先用可控偏好模型分开测试噪声、情境依赖与真实漂移；再在获同意的人类研究中检验关键结论。设置局部看似成功但长程有代价的行为。

对照：随机、分歧、覆盖感知与不查询；标签数量和人类时间分别匹配，不能只选有利的一种预算。评价器使用独立保留反馈。

测量：策略效用、奖励校准、错误优化事件、用户纠错时间、旧情境回访与漂移后的响应延迟。

反驳条件：只提高训练偏好预测而不改善独立效用，或以显著更多人类负担换取收益，就不支持反馈效率提升。

##### 学习与实验入口

[奖励假设与奖励设计](../textbook/reward-design.md) · [持续控制与学习智能体比较](../textbook/control.md) · [探索与经验选择](../textbook/exploration.md) · [知识保留与再适应](../textbook/retention.md)

环境：[B-Pref 反馈模型与实验协议](https://github.com/rll-research/BPref)：适合隔离反馈误差；不能用单一合成教师替代人的长期偏好。

实现：[Bradley–Terry 偏好奖励教学实现](https://yingwen.io/zh/continual-rl/code/preference_reward/)：用于理解成对偏好怎样训练奖励模型；不包含完整人机闭环。

相关问题：[选择性观察与多粒度信用：局部反馈怎样连接长程后果？](research-atlas.md#horizon-language-actions) · [单生命与安全：怎样学习，而不耗尽以后的学习机会？](research-atlas.md#horizon-safe-autonomy) · [相互适应与非平稳性：怎样区分自己、伙伴和共同约定的变化？](research-atlas.md#horizon-social-learning)

### 状态、记忆与预测知识

什么经验值得保留，什么后果值得预测，以及它们何时失效？

<a id="horizon-memory"></a>

#### 长期记忆：怎样判断一条经验值得保存、调用或遗忘？

检验场景：语言记忆与跨任务经验

过去经验是否在相同预算下改善后续决策，而不只是让上下文变长？

具体情境：数据库迁移失败后，智能体保存“先删除索引再改字段”。若失败其实由旧版本权限配置造成，这条语义上相关的记忆会在新版本持续误导操作。需要保存适用条件，也需要允许证据推翻它。

问题设定：生成模型可以固定，经验库及检索策略持续改变。事实记忆、操作程序和成功效用不是同一个对象。评价既包含从旧经验获益，也包含旧规则失效后的撤销速度；未来可能回访旧情境，因此不能把最近经验总当真理。

持续对象：记忆内容、来源、适用条件和效用估计；容量以字节与检索开销计，不只数条目。

反馈：调用后的结果提供效用线索；未调用记忆的反事实效果通常未知。

生命期：跨任务保留记忆；独立测试不得写回主记忆。版本回访与新结构迁移分开。

资源：压缩、写入、检索、嵌入服务和验证都占预算；冻结 LLM 不意味着学习成本为零。

核心困难：相关不等于有用。被检索的记忆更容易获得反馈，形成选择偏差；错误摘要也会改变后续证据。价值下降可能表示环境改变，也可能只是新任务更难。需要把容量管理、可靠性和因果效用区分。

##### 已有证据与适用边界

- [MemRL：Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory](https://arxiv.org/abs/2601.03192) · Shengtao Zhang 等 · 2026 预印本

  冻结生成模型，用语义召回加效用排序选择记忆；通过反馈更新记忆的效用。具体机制采用回报均值式更新，而不是要求对生成模型做梯度更新。

  边界：在所测任务上支持记忆选择的作用，不能推出任意变化下无遗忘。记忆效用依赖检索情境和生成器；不断变化的记忆池并不自动满足固定 MDP 的收敛条件。

  [MemRL 官方实现](https://github.com/MemTensor/MemRL)：包括 HLE、BigCodeBench、ALFWorld 与 LLB 入口；需模型/嵌入服务。LLB 发布实现覆盖 db/os，不含全部原基准任务。

- [Continual Learning Bench：Evaluating Frontier AI Systems in Real-World Stateful Environments](https://arxiv.org/abs/2606.05661) · Parth Asawa、Christopher Glaze 等 · 2026 预印本

  以共享潜在结构的实例序列，比较有状态系统与相应无状态对照。报告中简单上下文保留优于若干专门记忆系统，提醒我们不能由模块复杂度推断学习收益。

  边界：测的是跨实例经验利用；六类任务的世界延续方式不同。归一化增益依赖参考系统，需与绝对任务得分及成本一起看。

  [CL-Bench 官方运行框架与数据](https://github.com/pgasawa/continual-learning-bench)：包含任务、系统、调度与日志。部分任务依赖 Docker 或模型 API；固定这些版本才能比较记忆机制。

##### 仍需回答

- 只对被调用的记忆观察到结果时，怎样区分记忆的用途和检索器原有的选择偏差？哪些安全诊断允许比较读与不读同一条记忆？
- 有限容量下，应保存常用、低频高损失，还是难以重新获得的经验？旧规则失效时，能否更新适用条件而不是整条删除？

##### 可检验的实验提纲

假设：记忆效用依赖适用情境；在容量和检索预算相同时，条件化效用应减少过时经验的误用，并保留旧情境返回时的收益。

设计：设置 A→B→A 版本序列及 A→C 未见组合，隐藏版本标签但提供可探测线索。在可安全复制的离线诊断世界中，从同一检查点随机允许或屏蔽候选记忆，比较独立后续结果；主生命期仍按各自策略运行。

对照：无记忆、等字节原始轨迹、摘要、纯相似度与条件化效用。诊断匹配上下文长度和调用成本，不把测试结果写回主记忆；不可复制或高风险世界不直接进行这种干预。

测量：随机读出干预的后续效用差、效用排序校准、过时建议采用率、撤销延迟、回访零更新表现和整段净收益。

反驳条件：在可识别的诊断情境中，估计的高效用记忆并不比低效用记忆改善行动，或条件化优势只来自更多存储与调用，就削弱其选择机制。该诊断不证明所有未访问情境中的反事实用途。

##### 学习与实验入口

[智能体状态与递归学习](../textbook/state.md) · [知识保留与再适应](../textbook/retention.md) · [价值预测与资格迹](../textbook/value.md) · [实验设计、统计与算法测试](../textbook/experiments.md)

环境：[CL-Bench 文档与任务协议](https://continual-learning-bench.com/docs/)：先区分共享潜在结构、实例重置和持久状态，再选择任务。

实现：[知识保留的机制实验](../textbook/retention.md)：教材实验隔离旧知识保留与再学习；并非语言记忆系统性能评估。

相关问题：[选择性观察与多粒度信用：局部反馈怎样连接长程后果？](research-atlas.md#horizon-language-actions) · [奖励与反馈学习：偏好变化时，优化目标怎样保持可信？](research-atlas.md#horizon-human-feedback) · [学习规则与自修改：什么证据支持改变学习过程？](research-atlas.md#horizon-self-modification)

<a id="horizon-action-models"></a>

#### 可行动的模型：需要预测整个世界，还是足以作决定的后果？

检验场景：机器人组合任务与语言工具接口

怎样形成可复用、可校准的行动后果知识，并在状态与技能变化后及时修正？

具体情境：软件 agent 不必模拟整个互联网，才能预测“在当前权限和版本下，这个部署技能多久结束、失败概率多大、是否需要回滚”。机器人也可能更需要抓取后的接触与可达性预测，而非每个像素都逼真的长视频。

问题设定：模型的输入包含行动或技能及其条件，输出是决策所需的未来事件、时长、奖励或状态分布。预测准确度与控制价值分别评价。这里的模型可以是 GVF、option 后果模型或隐空间动力学；它们的条件、停止时间和用途不同。

持续对象：状态表示、行动模型、技能版本及不确定性；模型须注明针对哪个执行策略。

反馈：真实执行检验预测；自生成视频或自述推理不是额外外部证据。

生命期：保留旧机制并接纳新机制；技能或接口变化后，旧预测需重校准。

资源：模型训练、想象、真实探测和规划调用共享预算；比较不能只匹配真实步数。

核心困难：低预测误差可能只反映常见但无关的观测。规划会主动寻找模型高估区域。共享编码器漂移会使旧模型失效，技能更新又改变行动条件；单步拟合好不保证长期决策可靠。

##### 已有证据与适用边界

- [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) · Danijar Hafner、Jurgis Pasukonis、Jimmy Ba、Timothy Lillicrap · Nature 2025

  DreamerV3 在隐状态模型中想象未来并训练行为，提供跨多个领域复用学习设计的具体系统。它是研究“模型如何服务控制”的强基线。

  边界：跨领域采用相同算法不等于同一学习器顺序积累所有领域。图像重建、replay 和模拟回合协议均不能从预算中删去。

  [DreamerV3 官方代码](https://github.com/danijar/dreamerv3)：有完整模型与训练系统；要改成持续协议需另外规定重置、旧经验保留和模型版本。

- [Benchmarking World Models for Continual Learning on Compositional Tasks](https://arxiv.org/abs/2609.22055) · Haoyu Zhou、Joe Watson、Anson Lei、Ingmar Posner · 2026 预印本

  用先学习组成机制、再学习组合任务的序列，分开检查动作与感知重组。研究提醒：旧模型是否真正被复用，需要干预旧模块而不是只看路由权重。

  边界：模块化优势依赖见过全部任务演示的冻结编码器诊断；去掉这一特权时未优于对应整体模型。容量增长、任务头重置和模拟 reset 限制外推。

  [作者项目与附录](https://object814.github.io/Compositional-Continual-Learning/)：有任务说明与补充结果；该入口未提供可直接运行的完整作者仓库链接，不据此宣称可一键复现。

##### 仍需回答

- 能否按“改变决策需要哪些后果”学习模型，而不依赖设计者预先给出全部问题？
- 一个技能更新之后，怎样判断哪些模型仍有效？能够局部修复，而不是整套重新学习吗？

##### 可检验的实验提纲

假设：带行动条件与版本信息的后果模型，在组合变化后能比静态通用模型更快恢复规划可靠性。

设计：设置共享部件但不同组合的操作或工具世界。分别改变视觉线索、动力学和技能实现；保持三种变化可独立控制。对规划选中的动作单独测校准。

对照：模型自由控制、静态模型、持续整体模型和条件化模型；匹配总容量、实际数据、重放及规划次数。真实状态仅供诊断。

测量：任务收益、决策相关事件校准、旧机制复用消融、失效发现时间和每次有效规划的总成本。

反驳条件：模型预测变好却没有改善决策，或收益只来自先见全部任务的编码器，就不支持经验构建了可复用控制知识。

##### 学习与实验入口

[通用价值函数与预测知识](../textbook/gvf.md) · [转移模型与后果模型](../textbook/models.md) · [时间抽象与规划](../textbook/planning.md) · [Options 与技能发现](../textbook/options.md)

环境：[组合机制与开放世界的比较](worlds.md)：Alchemy、XLand-MiniGrid 等隔离隐藏机制与试次内适应；需另设持续保留协议。

实现：[Option 后果模型教学实现](https://yingwen.io/zh/continual-rl/code/option_model/)：可先检查真实奖励与折扣终点核；它不是视频世界模型或互联网模拟器。

相关问题：[选择性观察与多粒度信用：局部反馈怎样连接长程后果？](research-atlas.md#horizon-language-actions) · [快状态与慢知识：未知变化应由哪个学习过程承担？](research-atlas.md#horizon-embodied-adaptation) · [目标形成与开放式学习：怎样选择值得学的下一件事？](research-atlas.md#horizon-open-ended-goals)

### 适应、可塑性与学习规则

哪些变化可由当前状态处理，哪些需要改变知识或学习过程？

<a id="horizon-self-modification"></a>

#### 学习规则与自修改：什么证据支持改变学习过程？

检验场景：可编辑程序与语言智能体

修改工具、工作流程或学习规则的收益，能否在未参与选择的任务上保留？

具体情境：一个编程智能体把“运行一次测试”改成“先定位受影响模块，再执行相应测试”。这可能提高效率，也可能漏掉跨模块回归。若它同时能修改评分脚本，较高分数就不再是独立的效果证据。

问题设定：可变对象从模型参数扩展到程序。修改者提出候选，评价过程决定保留、回滚或继续搜索。修改策略本身也可以变化，但可信执行记录、权限约束和最终评价不能仅由被评系统自行声明。

持续对象：程序版本、学习策略、经验与候选档案；继承关系必须可追踪。

反馈：独立执行的任务结果，不采用自报“已经通过”。开发评分与保留测试分离。

生命期：离线分支搜索与真正部署序列分别评估；候选克隆不是免费延长同一生命。

资源：把全部失败候选、并行分支、评测、模型调用和回滚成本计入。

核心困难：反复选择会过拟合评测。更好的任务执行器不必是更好的修改者；元层收益可能只是搜索更多候选。程序接口变化还可能破坏旧知识、验证器和安全假设。

##### 已有证据与适用边界

- [Darwin Gödel Machine：Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954) · Jenny Zhang、Shengran Hu、Cong Lu、Robert Lange、Jeff Clune · 2025 研究论文

  让编码智能体提出自身代码修改，并通过任务评价形成可继续分支的候选档案。研究对象是可执行程序及其搜索历史，而非只有神经网络权重。

  边界：这是经验评价驱动的程序搜索，不是 Gödel 式形式证明，也不是无条件自我改善。作者另报告了评测投机实例；开放部署安全仍未由基准分数保证。

  [DGM 作者实现](https://github.com/jennyzzt/dgm)：包含自修改与编码任务评价，依赖模型、容器及外部任务资源；必须保留沙箱与独立评测边界。

- [Hyperagents](https://arxiv.org/abs/2603.19461) · Jenny Zhang、Bingchen Zhao、Jakob Foerster、Jeff Clune 等 · 2026 预印本

  将任务智能体和负责修改的元智能体置于可编辑程序，使“如何产生下一次改进”也成为实验对象。原文研究跨编码、评审、机器人奖励设计和数学评分的迁移。

  边界：有限领域与有限搜索预算的结果，不是对任何可计算任务都能无限自我改善的证明。候选档案扩张和元评测成本不能从持续学习账目中省略。

  [HyperAgents 官方代码](https://github.com/facebookresearch/HyperAgents)：可检查任务层与元层编辑边界及实验设置；不能将代码可编辑性本身视为有效元学习。

##### 仍需回答

- 怎样区分可迁移的改进规则与针对当前评测漏洞的特化？
- 什么组件应保持外部不可改，什么组件可在受约束条件下学习？这也是问题定义，不只是实现细节。

##### 可检验的实验提纲

假设：如果修改策略学到了可迁移的改进能力，它应在相同预算下更有效地改进未参与元训练的新执行器，而不只是交付一个更好的旧补丁。

设计：先在开发任务中学习修改者。随后把固定修改者与学得修改者应用于同一批未见初始执行器和新任务族；每个候选的保留由开发反馈决定，最终收益由独立保留集评价。另测最后一个执行器直接迁移，区分两项主张。

对照：相同初始执行器、候选数量、总计算与可编辑接口；另设只扩大搜索预算的固定修改者。验证器和审计记录不可修改，保留集结果不回流。

测量：在新执行器上产生有效修改的比例与成本、保留集净收益、旧能力破坏、回滚率、越权尝试及全部失败候选成本。

反驳条件：只有最终执行器迁移较好，而学得修改者不能更有效地改善新执行器，就只支持任务能力迁移；开发评分收益、核验绕过或额外搜索均不足以支持修改能力迁移。

##### 学习与实验入口

[元学习与学习规则的适应](../textbook/meta.md) · [持续学习的智能体架构](../textbook/architectures.md) · [奖励假设与奖励设计](../textbook/reward-design.md) · [实验设计、统计与算法测试](../textbook/experiments.md)

环境：[DGM 的任务与搜索设置](https://github.com/jennyzzt/dgm)：适合先隔离程序修改与评价选择偏差；不是无重置真实世界。

实现：[元学习的教学机制](../textbook/meta.md)：可先掌握“改变学习过程”的比较条件；元梯度不等于程序自修改。

相关问题：[长期记忆：怎样判断一条经验值得保存、调用或遗忘？](research-atlas.md#horizon-memory) · [选择性观察与多粒度信用：局部反馈怎样连接长程后果？](research-atlas.md#horizon-language-actions) · [奖励与反馈学习：偏好变化时，优化目标怎样保持可信？](research-atlas.md#horizon-human-feedback)

<a id="horizon-embodied-adaptation"></a>

#### 快状态与慢知识：未知变化应由哪个学习过程承担？

检验场景：机器人本体变化、sim-to-real 与 VLA

面对预训练未覆盖的身体与环境，机器人需要更新状态估计、控制参数，还是更慢的表示与技能？

具体情境：同一个抓取器逐渐磨损。视觉中的物体几乎不变，接触结果却改变。重新提示任务、增加相似演示和推断摩擦可能各有帮助，但不一定都能修复控制误差；磨损造成的真实失败也不能由模拟 reset 消除。

问题设定：机器人从预训练策略出发，在自身动力学、观测与任务约束下行动。至少区分零样本泛化、固定参数的历史条件适应、现场在线参数更新，以及多机器人采集后统一重训。它们能组合，但信息权限与时间成本不同。

持续对象：身体历史、隐变量估计、低层控制、VLA 或模型；逐项说明是否更新及能否保留。

反馈：位置、力、触觉、任务结果与人工纠正；监督演示和奖励不是同一种反馈。

生命期：个体内连续适应与车队版本迭代分别统计；记录传感器标定和设备更换。

资源：真实动作次数之外，计入人工布置、数据标注、训练停机、推理延迟和设备损耗。

核心困难：相同观测可能对应不同接触条件。离线演示偏向成功分布，现场失败又稀少且昂贵。快速改权重可能破坏安全先验；不改权重的适应器也可能无法表达新规律。视觉迁移好不能证明控制对未知动力学稳健。

##### 已有证据与适用边界

- [OpenVLA：An Open-Source Vision-Language-Action Model](https://openvla.github.io/) · Moo Jin Kim、Karl Pertsch 等 · 2024 研究论文

  把视觉与语言预训练接到机器人动作预测，提供多机器人数据训练的策略与适配入口。它说明可复用先验怎样降低新任务学习的起点成本。

  边界：动作监督训练与下游微调不等于自主在线 RL。数据、动作尺度和本体接口的匹配仍需处理。

  [OpenVLA 官方实现](https://github.com/openvla/openvla)：有权重、微调和部署说明；它是先验与基线入口，不是持续学习闭环成品。

- [RMA：Rapid Motor Adaptation for Legged Robots](https://ashish-kmr.github.io/rma-legged-robots/) · Ashish Kumar、Zipeng Fu、Deepak Pathak、Jitendra Malik · RSS 2021

  在仿真中训练基础策略与适应模块，部署时从近期本体感觉历史估计环境相关隐变量，不依赖现场微调。展示了快速状态适应与慢参数学习可以分离。

  边界：固定参数不等于固定行为，但短窗口隐变量适应不是自动获得可永久保留的新技能。结果受训练变化范围和适应器表达能力约束。

  [项目页链接的 locomotion 代码](https://github.com/antonilo/rl_locomotion)：检查 base policy 与 adaptation module 的训练阶段；真实部署还依赖机器人与底层控制接口。

- [pi-star 0.6：a VLA That Learns From Experience](https://arxiv.org/abs/2511.14759) · Physical Intelligence · 2025 研究论文

  RECAP 把自主执行、失败与人工纠正用于价值估计及优势条件化策略提取。原文通过多轮采集和重训研究成功率与任务吞吐的改善。

  边界：原文明确采用批量采集后的离线更新，依赖奖励标注、干预与 episode resets；不是机器人一边执行一边实时更新的完整协议。

  [官方 openpi：相关基础模型与适配工具](https://github.com/Physical-Intelligence/openpi)：所列仓库提供 pi0/pi0-FAST/pi0.5；不能据此声称 pi-star 0.6 与完整 RECAP 训练管线已开源。

##### 仍需回答

- 哪些变化适合即时隐变量估计，哪些需要新表征或权重更新？如何在可观察证据不足时识别两者？
- 能否在有限端侧计算下维持在线学习，并把云端更新的价值与延迟、带宽、隐私成本分开？

##### 可检验的实验提纲

假设：快速历史条件适应与受约束慢更新的组合，能在未见动力学下产生累计收益，而不仅提高最终成功率。

设计：先在仿真中分开改变载荷、摩擦、观测标定和任务目标；再在受控设备上做小范围验证。让旧条件返回，检查快速估计与持久知识各起什么作用。

对照：冻结先验、仅上下文适应、仅在线小参数更新、二者结合及批次重训。共享预训练检查点；并报告各组总数据、总计算和人工资源。

测量：变化后的失误与恢复时间、在线效用、旧条件回访、控制时延、人工介入和真实时间吞吐。

反驳条件：收益只来自更强预训练覆盖，或更新停机及失败成本抵消改善，就不能声称部署生命期更优。

##### 学习与实验入口

[智能体状态与递归学习](../textbook/state.md) · [元学习与学习规则的适应](../textbook/meta.md) · [持续控制与学习智能体比较](../textbook/control.md) · [可塑性与特征更新](../textbook/plasticity.md)

环境：[具身世界与基准目录](worlds.md)：从连续控制诊断到机器人与真实时间平台逐层增加难度；相同任务名称不保证相同观测权限。

实现：[非线性逼近与表示漂移实验](learning-dynamics.md)：先隔离表示变化和更新干扰；这些小实验不复现真实机器人结果。

相关问题：[单生命与安全：怎样学习，而不耗尽以后的学习机会？](research-atlas.md#horizon-safe-autonomy) · [可行动的模型：需要预测整个世界，还是足以作决定的后果？](research-atlas.md#horizon-action-models) · [相互适应与非平稳性：怎样区分自己、伙伴和共同约定的变化？](research-atlas.md#horizon-social-learning)

### 探索、目标与可复用行为

如何产生值得练习的目标，并把经验变成以后可用的能力？

<a id="horizon-open-ended-goals"></a>

#### 目标形成与开放式学习：怎样选择值得学的下一件事？

检验场景：开放世界、生成式课程与程序技能

智能体能否发现有用的学习目标，而不把环境设计者的课程当成自己的发现？

具体情境：工坊暂时没有高收益订单。智能体可练习修理设备、调查未知配方，或重复已经掌握的廉价动作。三个行为都可能被“探索奖励”鼓励，但只有部分经验会帮助未来服务。

问题设定：总体评价由任务或设计者规定；内部练习目标可以由智能体产生。另有研究允许教师生成新环境甚至奖励。两种权限必须分开：改造整个训练世界与在同一个世界中选择下一次尝试，不是同一控制问题。

持续对象：技能、模型、课程档案或适应状态；明确哪些跨目标和世界保留。

反馈：新奇、可学习性、成功与外部用途分开。生成更多目标不是最终效用。

生命期：试次内新规则适应、跨环境课程训练和同一世界长期探索分别评估。

资源：统计教师/LLM 先验、世界生成、失败课程、重置及全部训练预算。

核心困难：新奇不等于可学，可学不等于有用。技能库可能无限膨胀却难以组合；课程生成器可能只挑容易验证的目标。奖励与任务的无限数量也不证明能力复杂度持续增长。

##### 已有证据与适用边界

- [Human-Timescale Adaptation in an Open-Ended Task Space](https://arxiv.org/abs/2301.07608) · Adaptive Agent Team · ICML 2023

  AdA 在广泛任务分布上结合元 RL、注意力记忆与自适应课程，研究未见规则下从交互快速适应。结果支持训练分布、记忆和规模共同影响适应能力。

  边界：主要是训练后上下文内适应，不是部署权重无限在线更新。XLand 训练规模与环境权限不能默认为个人可复现。

  [AdA 官方项目与任务示例](https://sites.google.com/view/adaptive-agent/)：提供原文与演示；公开 XLand-MiniGrid 是受启发的独立平台，不是 AdA 的完整作者训练工程。

- [Voyager：An Open-Ended Embodied Agent with Large Language Models](https://voyager.minedojo.org/) · Guanzhi Wang、Yuqi Xie、Yunfan Jiang 等 · 2023 研究论文

  在 Minecraft 中结合自动课程、可执行程序技能库和执行反馈迭代。技能能保存并在新任务中复用，说明知识积累不必只发生在权重中。

  边界：核心生成器通过模型 API 调用，不进行显式梯度式 RL 训练。动作 API、已有世界知识和技能库增长是重要条件；迁移测试也会重建世界。

  [Voyager 作者代码](https://github.com/MineDojo/Voyager)：包括课程、技能库与 Minecraft 接口；模型服务成本和版本需要固定并报告。

- [OMNI-EPIC：Environments Programmed in Code](https://arxiv.org/abs/2405.15568) · Maxence Faldor、Jenny Zhang、Antoine Cully、Jeff Clune · ICLR 2025

  用基础模型生成环境代码与奖励，再依据可学习性和有趣性选择后续任务。把“训练问题从哪里来”变成显式研究对象。

  边界：生成器拥有改写环境与奖励的权限；这不是固定外部世界中的自主目标形成。有趣性判据还包含预训练的人类先验。

  [OMNI-EPIC 作者实现](https://github.com/maxencefaldor/omni-epic)：含环境生成、Dreamer 训练与档案；所有生成失败和子任务训练均应计入成本。

##### 仍需回答

- 能否以未来可用性而非语义新颖程度评价技能？评估目标又如何避免偷偷给出未来任务？
- 在固定容量与不可重置世界中，哪些目标值得放弃，哪些技能值得重新组合？

##### 可检验的实验提纲

假设：根据已发生的主任务改进估计练习用途，比只追求新奇更能在有限练习预算内改善未见组合任务。

设计：先固定候选目标和低层学习器，从练习后预定时间窗内的主任务收益估计用途，明确延迟反馈如何归属。加入可学无用机制与不可约噪声；测试组合不参与选择。候选生成及教师改写环境另开实验。

对照：随机目标、固定课程、新奇、学习进展与用途选择；相同交互和技能容量。知道未来任务分布的教师仅作诊断上界，不与自主选择混合。

测量：用途估计与后续净收益的关系、新组合首次表现、技能实际复用、准备与选择成本；报告失败和被放弃目标。

反驳条件：用途估计不能优于简单进展准则，或收益只在给出未来任务后存在，就不支持该自主选择机制；目标与技能数量增加本身不是能力改善。

##### 学习与实验入口

[目标条件化与子任务构造](../textbook/goals.md) · [探索与经验选择](../textbook/exploration.md) · [Options 与技能发现](../textbook/options.md) · [元学习与学习规则的适应](../textbook/meta.md)

环境：[XLand-MiniGrid 官方环境与基线](https://github.com/dunnolab/xland-minigrid)：适合隐藏规则与组合适应；试次边界和记忆重置须按研究问题重新规定。

实现：[内在奖励元梯度教学组件](https://yingwen.io/zh/continual-rl/code/intrinsic_meta_gradient/)：展示内在奖励如何影响后续策略更新；单步组件不等于开放式课程系统。

相关问题：[可行动的模型：需要预测整个世界，还是足以作决定的后果？](research-atlas.md#horizon-action-models) · [单生命与安全：怎样学习，而不耗尽以后的学习机会？](research-atlas.md#horizon-safe-autonomy) · [相互适应与非平稳性：怎样区分自己、伙伴和共同约定的变化？](research-atlas.md#horizon-social-learning)

### 与其他学习者共同交互

当他者也根据经验变化时，怎样识别学习与协调的贡献？

<a id="horizon-social-learning"></a>

#### 相互适应与非平稳性：怎样区分自己、伙伴和共同约定的变化？

检验场景：人机合作与多机器人开放团队

与陌生伙伴合作、辨认正在变化的伙伴，以及共同形成新约定，是否需要不同的学习机制？

具体情境：两个仓库机器人都曾学会在路口向右让行。新伙伴遵循另一规则，而原伙伴随后又根据我的行为改变策略。失败既可能来自陌生约定，也可能来自同时适应造成的追逐；把所有变化视为固定环境噪声会掩盖区别。

问题设定：他者的行动影响我的经验，他者也可能根据我改变。区分训练中的种群扩张、测试时零样本协调、试次内伙伴推断和部署期间持续共同学习。合作之外还包括竞争、资源共享与混合动机，团队奖励并非普遍存在。

持续对象：伙伴模型、约定、角色和策略；身份是否可见、记忆是否跨伙伴保留必须明确。

反馈：个体与团队收益、通信、人的纠正；中心化训练信息不应在分散执行时泄漏。

生命期：允许伙伴加入离开、再次出现或改变行为；旧伙伴回访与未见伙伴泛化分别测。

资源：通信、种群训练、伙伴模拟和人类适应的成本分别记账。

核心困难：收益上升可能是对方学会迁就，而不是我的模型改善。双方同时更新改变数据分布；过快适应可破坏原有约定。自博弈的共同惯例还可能妨碍与群体外成员协作。

##### 已有证据与适用边界

- [Cooperative Open-ended Learning Framework for Zero-shot Coordination](https://arxiv.org/abs/2302.04831) · Yang Li、Shao Zhang、Jichen Sun、Yali Du、Ying Wen 等 · ICML 2023；JAIR 2024 扩展

  COLE 通过合作能力相关的种群结构扩展训练伙伴，研究与未见策略及人协调。它把伙伴多样性与合作兼容性引入训练目标。

  边界：零样本协调不是部署中的持续参数学习。作者的人类实验也显示可预测的固定策略有时更易被人配合，单一平均团队分数不足以解释原因。

  [COLE 作者平台与训练分支](https://github.com/liyang619/COLE-Platform)：包含人机交互平台、基线和 COLE 训练分支；需区分真实人、代理策略及评价角色。

- [Multi-Robot Open Adaptive Teaming Across Unseen Environments, Partners, and Scales](https://arxiv.org/abs/2607.04972) · Yang Li、Feng Xue、Fan Mo、Yunhao Liu 等 · 2026 预印本

  用超图形式博弈表达多方合作关系，并扩展训练中的伙伴与环境。评价包含试次内成员变化，以及未见伙伴、环境和队伍规模下的实际机器人协调。

  边界：论文强调无需现场微调的迁移；它不是持续更新权重的证据。超图形式博弈也不能直接说成一种图神经网络。

  [原文：开放团队协议与硬件设置](https://arxiv.org/html/2607.04972v1)：该原文未列出完整作者工程入口；可先复用 COLE 或公开多智能体环境搭建小规模判别任务，不宣称已复现 HOLA。

##### 仍需回答

- 如何区分自己学习、伙伴学习和共同约定形成各自带来的收益？
- 能否对不熟悉的伙伴保留适应能力，同时不遗忘旧伙伴、不泄漏他人的私有经验，也不通过牺牲对方效用获得高分？

##### 可检验的实验提纲

假设：显式区分伙伴推断与慢策略更新，能够在成员变化时降低协调损失，并保留对返回伙伴的知识。

设计：让 A、B 两类伙伴交替出现，并设固定伙伴和同时学习伙伴两条协议。先在可控合作游戏检查，再加入混合收益与通信成本。身份标签仅作为诊断上界。

对照：冻结策略、仅递归伙伴状态、仅策略更新、两者结合；交叉冻结双方，另配未见伙伴池。人类实验需随机化顺序以控制熟练效应。

测量：个体与团队整段收益、冲突率、约定恢复、对方负担、通信成本及面对群体外伙伴的表现。

反驳条件：改善完全来自伙伴对系统的迁就，或仅在给出真实伙伴身份时存在，就不支持自主持续伙伴学习。

##### 学习与实验入口

[智能体状态与递归学习](../textbook/state.md) · [元学习与学习规则的适应](../textbook/meta.md) · [持续控制与学习智能体比较](../textbook/control.md) · [实验设计、统计与算法测试](../textbook/experiments.md)

环境：[多智能体与社会环境目录](worlds.md)：选择 Overcooked、Melting Pot 等平台时，还需明确伙伴池与测试时更新规则。

实现：[实验设计：学习器与评价单位](../textbook/experiments.md)：将双方学习运行作为相互依赖的随机过程，不把同一对局的窗口当独立样本。

相关问题：[目标形成与开放式学习：怎样选择值得学的下一件事？](research-atlas.md#horizon-open-ended-goals) · [奖励与反馈学习：偏好变化时，优化目标怎样保持可信？](research-atlas.md#horizon-human-feedback) · [快状态与慢知识：未知变化应由哪个学习过程承担？](research-atlas.md#horizon-embodied-adaptation)

### 从一个方向形成可检验的研究问题

这些方向共享状态、信用、探索、保留与计算分配等机制，却不共享完全相同的评价目标。可从一个具体失败出发：谁缺少什么信息，哪种更新可能消除失败，它会引入什么新成本？然后再回到教材选择机制。应用场景越复杂，越需要先在小问题中区分解释，而不是只搭建更大的系统。

先确定更新载体和生命期，再选择基线。冻结权重但保留记忆不是“完全不学习”；清空记忆但保留更新后的权重也不是从头开始。冻结诊断最好在独立拷贝中进行。交互策略不同时，后续世界轨迹也会不同，因此配对初始世界不等于获得完全相同的经验。

未来的工作还包括持续科学实验、医疗与教育等反馈更慢的决策场景，以及能源、生态和社会系统中的资源分配。它们需要领域特有的伦理与安全约束。这里的九条问题线是扩展研究的入口，不是穷尽领域的分类，也不是把邻近研究统一改名为 CRL。
