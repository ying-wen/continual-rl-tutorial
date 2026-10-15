# 章节、实验、资源与学者

## 交互、奖励与优化目标

折扣与平均奖励是目标选择。持续学习不能仅靠把回合接长来定义。

先用可解析的循环 MDP 检查策略排序，再讨论更复杂环境。

[教材](../textbook/objectives.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-objectives)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：奖励、回报与持续交互的统一问题表述。
- Andrew G. Barto｜自适应控制、内在动机与层级学习：自适应控制与强化学习问题的形成。
- David Abel｜从表示抽象到“学习究竟在哪里”：持续学习定义与智能体内部学习过程。
- Michael Bowling｜博弈、评测与 agency：比较固定策略与完整学习过程，明确奖励表示、研究假说和形式化提案的不同角色。
- Martha White｜可靠 off-policy 学习到长期控制：将环境、任务、关注分布与学习目标分别定义。
- Esraa Elelimy｜高效在线递归学习：区分冻结策略评价与持续学习过程评价。
- Khurram Javed｜有限算力下的持续构造与信用分配：把大世界立场转化为可被反驳的研究问题。
- Patrick M. Pilarski｜预测知识、人机共适应与身体：智能增强的系统目标

## 奖励假设与奖励设计

偏好、奖励机制、回报与辅助信号不是同一对象。保持最优策略、加快学习与符合设计者意图需要不同证据。

先检验顺序目标、塑形边界和偏好梯度；再用 B-Pref 与安全诊断环境检验反馈成本和代理失效。

[教材](../textbook/reward-design.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-reward-design)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：奖励假设与从经验学习的总体问题。
- David Abel｜从表示抽象到“学习究竟在哪里”：奖励表达能力与持续学习的概念基础。
- Michael Bowling｜博弈、评测与 agency：奖励表示的公理条件与设计者目标。
- Will Dabney｜价值分布与目标／agent 基础：奖励表达与偏好表示。
- Satinder Singh｜内在动机、时间抽象与 agency：奖励、内在动机与智能能力的关系。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：检验目标能被奖励表达的条件。
- Matthew E. Taylor｜迁移、教学与人类输入：人类输入的语义

## 平均奖励与差分价值

调用一次技能可能只走一步，也可能走许多步。评价奖励率要按真实持续时间累计奖励；一般不能把各次调用的奖励除以时长后再取平均。长期极限适合描述持续行为，却会忽略有限的启动损失。研究一个仍在学习的系统时，应同时规定有限评价时域，显式保留练习、模型修正和恢复所花的时间。

已知平稳分布的循环 MDP；再用持续控制任务检查暂态、长期奖励率与漂移。

[教材](../textbook/average.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-average)

- Martha White｜可靠 off-policy 学习到长期控制：平均奖励、可靠 off-policy 学习与长期控制。
- Adam White｜从实时预测知识到可信实验：平均奖励学习与经验研究。
- Richard S. Sutton｜从时间信用分配到持续经验型智能体：差分价值与 continuing control。
- Kris De Asis｜多步价值学习与面向真实时间的机器人：从真实时间理解持续任务的回报。

## 智能体状态与递归学习

状态递推让智能体在提示消失后仍能利用它。判断一个摘要是否够用，要看它是否把需要不同答案的历史合并了；维度更大、重构观测更准都不能单独回答这一点。联合学习还增加一个问题：摘要改变后，旧读出是否仍理解它。已知可逆变换可检验读出适配；历史信息已丢失时，则需要重新学习记忆本身。

POPGym 隔离记忆需求；Forager 检查长期交互。对照完整状态 oracle 时标明额外信息权限。

[教材](../textbook/state.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-state)

- Matthew Schlegel｜预测式状态与表示支持规划：预测状态与可规划表示。
- Esraa Elelimy｜高效在线递归学习：递归网络的在线训练。
- Leslie Pack Kaelbling｜部分可观测与层级机器人规划：部分可观测决策与信念状态。
- Amy Zhang｜可泛化与分层决策所需的表示：面向决策的状态表示。
- Joseph Modayil｜从感知结构到可验证预测知识：从观测关系构造内部状态。
- Randy Goebel｜知识表示、推理与可检验的解释：分开状态的预测充分性与人类可理解性。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：研究内部状态与外部痕迹的边界。
- Dale Schuurmans｜表示、优化与可规划状态：表示要支持预测与决策；部分可观测的理论条件应和实现一起阅读。
- Martha White｜可靠 off-policy 学习到长期控制：比较预测语义、表示可更新性与循环计算成本。
- Adam White｜从实时预测知识到可信实验：同时评价部分可观测性、记忆语义和更新代价。
- Khurram Javed｜有限算力下的持续构造与信用分配：从计算依赖解释可在线训练的循环架构。

## 价值预测与资格迹

表格 TD 的局部更新推广为共享特征上的参数更新。GVF 再改变预测问题，而非只改变网络。

Random Walk 与小型马尔可夫奖励过程：有解析真值，便于分开估计误差和程序错误。

[教材](../textbook/value.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-value)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：TD 学习与资格迹。
- Andrew G. Barto｜自适应控制、内在动机与层级学习：价值学习与演员—评论家框架。
- Peter Dayan｜预测表征与计算神经科学：Successor representation 连接预测与表征。
- Arsalan Sharifnassab｜更新几何、步长适应与流式学习：比较 Bellman 目标与更新几何。
- Csaba Szepesvári｜统计效率和算法边界：把表格 TD、函数逼近与随机逼近保证的条件逐一列出。
- Martin Müller｜长期价值与临时搜索记忆：TD 不只可用于真实轨迹学习，也可作为模拟搜索中的局部价值更新。

## 通用价值函数与预测知识

GVF 用目标策略、累积量与延续系数指定一个可以由经验检验的问题。同一历史可以回答许多这样的提问，有限预算却只允许维护一部分。表示更新时，应先确认问题规格是否保持不变；技能策略变了，所预测的未来也可能变了。进一步的困难是依据已经观察到的用途选择问题，而不是依赖设计者预知以后会需要什么。

可解析 off-policy MRP 检查预测真值；机器人信号预测需独立测决策效用。

[教材](../textbook/gvf.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-gvf)

- Joseph Modayil｜从感知结构到可验证预测知识：以通用价值函数组织预测知识。
- Patrick M. Pilarski｜预测知识、人机共适应与身体：机器人预测知识与人机共适应。
- Adam White｜从实时预测知识到可信实验：并行离策略预测与实时经验学习。
- Martha White｜可靠 off-policy 学习到长期控制：稳定的离策略预测方法。
- Matthew Schlegel｜预测式状态与表示支持规划：预测问题及其发现

## 持续控制与学习智能体比较

固定策略、记忆递推和完整学习器是不同评价对象。CRL 可定义历史条件价值，但不能不加条件地沿用固定 MDP 的策略排序。

解析例子检验时域排序、相同策略下不同学习规则、不可逆后果与偏离遗憾；再扩展到单次生命期评价。

[教材](../textbook/control.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-control)

- Michael Bowling｜博弈、评测与 agency：持续学习器的评价与可行偏离比较。
- Martha White｜可靠 off-policy 学习到长期控制：适应过程、学习目标与评测。
- David Abel｜从表示抽象到“学习究竟在哪里”：历史过程与持续学习定义。
- Richard S. Sutton｜从时间信用分配到持续经验型智能体：GPI 为局部策略改善提供基础。
- Csaba Szepesvári｜统计效率和算法边界：固定问题中的策略优化率不能代替整个持续学习器的生命期评价。
- Dieter Büchler｜高动态机器人中的在线适应：动态运动与物理时间

## 深度价值学习

DQN 保留 TD 目标。网络、回放与目标网络增加新的时间尺度，长期训练时也可能引入陈旧数据与可塑性问题。

CartPole 做管线检查；MinAtar 或 ALE 做视觉控制。不能把 CartPole 成功当作 Atari 复现。

[教材](../textbook/deep-value.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-deep-value)

- Hado van Hasselt｜稳定深度价值学习与持续 RL 基础：Double Q-learning 与深度价值估计偏差。
- Marc G. Bellemare｜表示、分布式价值与可靠环境：分布式价值学习与 Atari 评测。
- Will Dabney｜价值分布与目标／agent 基础：分位数价值分布。

## 策略梯度与 actor–critic

从 REINFORCE 到 actor–critic，评论家提供低方差学习信号。PPO 的批量多轮更新与严格流式协议不同。

离散 bandit 检查梯度；CartPole 检查完整采样更新；连续控制再检查动作概率密度。

[教材](../textbook/policy.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-policy)

- Sergey Levine｜从现实机器人经验到可泛化控制：策略优化、机器人学习与深度 RL 教学。
- Pieter Abbeel｜技能学习、元学习与通用机器人：策略搜索、机器人控制与元学习。
- Gautham Vasan｜增量策略梯度与在线连续控制：增量策略梯度和连续控制。
- Dale Schuurmans｜表示、优化与可规划状态：把真梯度优化理论与含估计误差的 actor–critic 更新区分。
- Martha White｜可靠 off-policy 学习到长期控制：从明确目标推导 actor 更新，再分析近似与分布错配。
- Shibhansh Dohare｜持续更新特征，而非只保护参数：区分表征可训练性和策略分布坍塌。

## 最大熵控制

连续动作中，策略承担动作搜索。熵、双评论家与回放各有作用，不能合并为一个“稳定化技巧”。

Pendulum 做单位与动作范围检查；MuJoCo / DMC 控制需固定版本、动作重复与观测处理。

[教材](../textbook/soft-control.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-soft-control)

- Sergey Levine｜从现实机器人经验到可泛化控制：最大熵控制与现实经验学习。
- Hado van Hasselt｜稳定深度价值学习与持续 RL 基础：价值估计误差与双估计器方法的基础。
- Dale Schuurmans｜表示、优化与可规划状态：用 soft consistency 理解策略、价值与熵正则化目标的共同结构。

## 时间信用分配与资格迹

记忆保存了提示，并不意味着晚到的反馈能够更新当时的写入方式。资格迹记录过去预测或动作的更新方向，递归敏感度记录参数对当前活动的影响，元梯度再追踪过去学习怎样影响后续表现。比较这些机制时应明确求导对象，并用截断信用但保留活动的对照，区分信息没有留下与信息留下却学不会使用。

固定轨迹的代数与导数检查 → Markov/混叠路径反例 → 延迟反馈和部分可观测控制；MinAtar、MuJoCo 与严格流式协议分别比较。

[教材](../textbook/credit.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-credit)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：多步回报、资格迹与在线信用分配。
- A. Rupam Mahmood｜把学习放回物理时间：True-online TD 与在线更新。
- Khurram Javed｜有限算力下的持续构造与信用分配：资源受限的递归信用分配。
- Esraa Elelimy｜高效在线递归学习：高效在线递归学习。
- Kris De Asis｜多步价值学习与面向真实时间的机器人：连接抽样、期望与多步备份。

## 流式更新与稳定性

严格流式学习限制经验到达后怎样再次使用。活动、资格迹和优化器统计可以留存，但都占用资源；保存完整序列以重算新状态则需要另一种数据权限。模型因技能学习而失准时，有限统计量未必足以修复，可能必须重新执行技能。因而流式方法既要控制每次更新，也要检验失去回放后重新获得信息的成本。

固定每步算力的连续控制；同时记录延迟尾部、学习状态大小、输出变化与回报。

[教材](../textbook/streaming.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-streaming)

- A. Rupam Mahmood｜把学习放回物理时间：严格增量深度 RL 与在线稳定化。
- Mohamed Elsayed｜严格增量的深度强化学习：Streaming deep RL 的实现与评测。
- Gautham Vasan｜增量策略梯度与在线连续控制：增量 actor–critic。
- Joseph Modayil｜从感知结构到可验证预测知识：检查并行预测的每步计算。
- Kris De Asis｜多步价值学习与面向真实时间的机器人：逐步更新必须满足运行时预算。
- Arsalan Sharifnassab｜更新几何、步长适应与流式学习：从预期函数变化理解单样本稳定化。
- Sorina Lupu｜自适应控制与直接从经验学习的机器人：真实机器人中的延迟与逐步计算。
- Khurram Javed｜有限算力下的持续构造与信用分配：按每步成本而不只按参数量比较方法。

## 元学习与学习规则的适应

不同特征可能需要不同更新速度：稳定信号应保留，正在漂移的信号需要追踪，噪声又不宜追得过快。元学习用后续表现调整这些学习参数，但短时间窗可能无法评价较早更新的长期作用。状态与技能同时改变时，还应分别在共同数据和真实交互中比较，判断学到的规则是否改善更新，以及这种改善怎样改变采样。

漂移预测测试步长适应；跨任务元 RL 必须分训练任务、开发任务与未见测试任务。

[教材](../textbook/meta.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-meta)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：增量步长适应与元学习。
- Martha White｜可靠 off-policy 学习到长期控制：学习规则适应和稳定性。
- Chelsea Finn｜快速适应与机器人元学习：MAML 与快速适应。
- Luisa Zintgraf｜用潜变量推断进行快速适应：基于隐变量的元强化学习。
- Arsalan Sharifnassab｜更新几何、步长适应与流式学习：追踪学习规则对未来误差的影响。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：通过后续学习效果适应规划采样。
- Sorina Lupu｜自适应控制与直接从经验学习的机器人：区分离线元学习与运行时适应。
- A. Rupam Mahmood｜把学习放回物理时间：跟踪逐特征学习速度，同时追问元参数和非平稳性。
- Khurram Javed｜有限算力下的持续构造与信用分配：区分学习表示的外层过程与在线更新率的自适应。

## 目标条件化与子任务构造

给定一个子目标，可以研究怎样更快达到；决定练习哪个子目标，还必须估计练习对以后主任务的作用。容易达到的目标可能从不被调用，困难目标也可能带来罕见但有用的恢复行为。研究应先固定候选与低层学习器，比较目标选择的净收益，再加入候选生成，辨认新知识究竟来自哪一步。

先给定目标集合，再比较随机、人工与学得目标。计入发现和技能训练的成本。

[教材](../textbook/goals.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-goals)

- Satinder Singh｜内在动机、时间抽象与 agency：内在动机、子任务与智能体自主性。
- George Konidaris｜从技能出发构造符号世界：技能与可规划抽象。
- Khimya Khetarpal｜affordance、抽象与 CRL 框架：Affordance 与任务相关抽象。
- Benjamin Eysenbach｜从未来状态预测理解自监督控制：目标条件、自监督控制与未来状态。

## Options 与技能发现

Option 使经过多个原始动作才完成的行为成为高层选择对象。要评价它，需要知道从哪里启动、内部怎样行动、何时停止，以及停止后继续做什么。学习修改其中任何一项，都可能改变奖励、时长或终点分布；仅冻结策略参数也不够，因为输入状态仍可能改变。旧模型是否可用，应由相关后果是否保持来判断。

Four Rooms 检查覆盖与打断条件；迁移目标时同时报告预训练成本和主任务交互。

[教材](../textbook/options.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-options)

- Marlos C. Machado｜表示—技能—经验的循环：SR、eigenoptions 与表示驱动的技能发现。
- Doina Precup｜从 options 到可持续的抽象智能体：Options、时间抽象与层级控制。
- Pierre-Luc Bacon｜时间抽象与优化视角：Option-Critic 的端到端优化。
- André Barreto｜可组合的价值知识与技能：可组合价值知识与技能。
- Richard S. Sutton｜从时间信用分配到持续经验型智能体：理解随机时长行为的接口。
- Levi H. S. Lelis｜让学到的知识指导搜索：比较程序组件、可执行技能与有明确启动／终止语义的 options。
- Dieter Büchler｜高动态机器人中的在线适应：带时间的子目标

## Dyna 与模型学习

一次真实经验既更新模型，也支持额外规划。持续环境中应同时测模型陈旧程度和规划收益。

Blocking Maze / Shortcut Maze：固定真实交互，另报规划备份次数与墙钟。

[教材](../textbook/dyna.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-dyna)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：Dyna：直接学习、模型学习与规划的统一。
- David Silver｜规划、强化学习与经验规模：搜索、价值学习与模型支持的决策。
- Martin Müller｜长期价值与临时搜索记忆：比较真实经验、模型模拟和双记忆的不同更新来源。

## 转移模型与后果模型

模型需要保留什么，取决于规划随后会问什么。线性尾值可以读取期望终点特征，非线性尾值通常还需要更多分布信息。状态或技能改变后，应分别检查输入含义、真实后果与采样覆盖；只重新拟合一个起点的备份，可能掩盖其他条件问题已经无法区分。模型维护由此既有统计问题，也有信息是否仍在的问题。

解析模型作 oracle；学得模型与无模型同骨干对照。区分重建、价值和闭环误差。

[教材](../textbook/models.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-models)

- Dale Schuurmans｜表示、优化与可规划状态：表示、优化与模型的可规划性。
- Danijar Hafner｜潜在世界模型与行为想象：潜在世界模型与想象学习。
- Samuel Kessler｜task-agnostic 世界模型的持续适应：任务无关的持续世界模型。
- Matthew Schlegel｜预测式状态与表示支持规划：预测表示与规划接口。
- Randy Goebel｜知识表示、推理与可检验的解释：讨论学得模型表达了什么以及何时失效。
- Sorina Lupu｜自适应控制与直接从经验学习的机器人：辨认模型误差与闭环控制需求。

## 时间抽象与规划

规划通过模型比较行为，所以误差是否跨过候选间的价值差距，比误差是否非零更直接地影响当前选择。还要区分模型错误与搜索不足：在真模型下追加计算是否有用，换回学到的模型后是否仍有用？状态和技能持续学习时，规划计算还要与模型修正共享预算，最终比较它们对实际交互收益的贡献。

目标条件任务比较不同搜索深度；在固定计算下测模型误差与计划执行偏差。

[教材](../textbook/planning.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-planning)

- David Silver｜规划、强化学习与经验规模：搜索与价值学习。
- George Konidaris｜从技能出发构造符号世界：抽象技能的符号规划。
- Levi H. S. Lelis｜让学到的知识指导搜索：学习引导搜索。
- Martin Müller｜长期价值与临时搜索记忆：搜索中的临时记忆与长期价值。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：分配有限的模型查询与备份。
- Michael Bowling｜博弈、评测与 agency：将 DeepStack 的决策时重求解与跨经验的参数学习分账。
- Csaba Szepesvári｜统计效率和算法边界：区分可查询模型中的模拟效率、模型误差和真实交互效率。
- Dale Schuurmans｜表示、优化与可规划状态：联合评价学得的表示与可执行的规划，而不是只考察编码质量。
- Marlos C. Machado｜表示—技能—经验的循环：表示如何支持决策时规划
- Matthew Schlegel｜预测式状态与表示支持规划：表示支持多尺度规划

## 知识保留与再适应

回访旧情境时，先前行为可能立即恢复，也可能需要重新学习。冻结更新、恢复旧线索或适配读出的诊断有助于区分知识丢失与调用条件改变；恢复训练后的速度则测量重新学习。有限存储还要求进一步选择保留什么：访问频率之外，失败代价、重学成本以及旧规律是否仍有效，都可能改变保留价值。

Continual World / COOM：逐任务学习矩阵、任务身份权限、首次习得和再学习速度。

[教材](../textbook/retention.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-retention)

- Sarath Chandar｜稳定适应与持续学习的方法共同体：稳定适应与持续学习。
- Eric Eaton｜可组合知识的 lifelong RL：终身策略学习与知识组合。
- Jorge A. Mendez｜模块化与知识组合：模块化持续学习。
- Alessandro Lazaric｜知识迁移的条件与负迁移：迁移条件与负迁移。
- Shibhansh Dohare｜持续更新特征，而非只保护参数：同时观察旧知识保留与新知识学习，避免只报告其中一个。
- Matthew E. Taylor｜迁移、教学与人类输入：迁移与负迁移

## 可塑性与特征更新

旧问题仍答得好，不能说明网络还学得会新问题。可以先给训练年龄不同的网络相同新数据和目标，比较它们的学习速度，再回到在线交互检查探索影响。替换特征或调整有效步长若有帮助，还要测量旧功能的损失，以及技能和模型重新适配的成本；这些结果共同决定干预是否值得长期使用。

稳定世界中的训练年龄实验和外部变化实验分开；控制梯度尺度、优化器状态与特征容量。

[教材](../textbook/plasticity.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-plasticity)

- Shibhansh Dohare｜持续更新特征，而非只保护参数：持续反向传播与特征替换。
- Clare Lyle｜持续可训练性的机制解释：可塑性衰退的机制。
- Evgenii Nikishin｜早期经验偏置与部分重置：早期经验偏置与网络重置。
- Ghada Sokar｜休眠神经元与 ReDo：休眠神经元与 ReDo。
- Rishabh Agarwal｜统计可靠的 RL 比较与可塑性诊断：可靠比较与可塑性诊断。
- Richard S. Sutton｜从时间信用分配到持续经验型智能体：检验训练年龄对继续学习的影响。
- A. Rupam Mahmood｜把学习放回物理时间：把短期数值稳定与长期学习能力分成两个指标。

## 探索与经验选择

若两种预测在已访问区域完全相同，继续收集同类经验无法分辨它们。探索应尝试能产生不同后果的行为，同时考虑该信息对后续选择的用途。噪声区域也能持续制造大误差，却未必提供可学规律；恢复经验可能很少出现，却关系到能否继续交互。经验选择因此需要同时研究可学性、用途和取得经验的代价。

Procgen / MiniGrid 分离训练关卡和测试关卡；加入不可学习噪声目标检验课程选择。

[教材](../textbook/exploration.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-exploration)

- Peter Stone｜技能迁移、协作与自主课程：自动课程与技能迁移。
- Tim Rocktäschel｜开放式学习与环境生成：无监督环境设计与开放式学习。
- Ian Osband｜深度探索与认识不确定性：不确定性与深度探索。
- Pierre-Yves Oudeyer｜学习进步驱动的自主发展：学习进展驱动的自主发展。
- Jeff Clune｜从质量多样性到开放式能力增长：质量多样性与开放式搜索。
- Csaba Szepesvári｜统计效率和算法边界：用 bandit 上下界理解反馈不足和比较器选择，再进入状态会随动作改变的 RL。
- Marlos C. Machado｜表示—技能—经验的循环：技能如何改变探索
- Dieter Büchler｜高动态机器人中的在线适应：身体决定探索空间

## 持续学习的智能体架构

状态、预测、技能和规划一面共享经验，一面改变彼此的输入与目标。一个技能刚学好时，旧模型可能低估它，使规划很少调用它，进而延缓模型修正；状态改进也可能帮助多个预测同时辨认历史。完整架构应通过分别冻结、恢复和继续学习来辨认这些作用，并检验发现与维护知识的成本是否换来更多生命期收益。

从两模块因子消融到完整持续环境。多智能体以独立训练团队为单位，补充 cross-play。

[教材](../textbook/architectures.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-architectures)

- Richard S. Sutton｜从时间信用分配到持续经验型智能体：从经验构造知识的整体架构。
- David Abel｜从表示抽象到“学习究竟在哪里”：智能体结构与持续学习定义。
- Marlos C. Machado｜表示—技能—经验的循环：表征、技能与经验生成的耦合。
- 温颖 Ying Wen｜相互影响、递归推理与合作决策：递归推理与多智能体相互影响。
- 汪军 Jun Wang｜博弈论、多智能体交互与规模化决策：多智能体交互与博弈。
- 俞扬 Yang Yu｜经验复用到持续多智能体协调：经验复用与多智能体协调。
- 章宗长 Zongzhang Zhang｜不确定环境中的策略迁移：部分可观测决策与策略迁移。
- 郝建业 Jianye Hao｜多智能体适应与可迁移深度 RL：多智能体适应与可迁移控制。
- 高阳 Yang Gao｜机器人学习与知识积累：机器人学习与知识积累。
- 朱军 Jun Zhu｜Bayesian 持续学习与主动遗忘：Bayesian 持续学习与遗忘机制。
- 张伟楠 Weinan Zhang｜可读 RL 教学、决策学习与 agent：深度 RL 教学与决策学习。
- 杨耀东 Yaodong Yang｜大规模合作、博弈与适应：合作、博弈与多智能体适应。
- Joseph Modayil｜从感知结构到可验证预测知识：让预测知识服务后续决策。
- Randy Goebel｜知识表示、推理与可检验的解释：连接学习模块、推理模块与解释接口。
- John D. Martin｜奖励表达、规划计算与环境中的记忆：把身体、环境和计算纳入系统边界。
- Levi H. S. Lelis｜让学到的知识指导搜索：把慢速表示学习、快速程序搜索和执行过程定义为不同接口。
- Martin Müller｜长期价值与临时搜索记忆：检验局部搜索与长期学习的接口，保留模块组合失败的负证据。
- Patrick M. Pilarski｜预测知识、人机共适应与身体：人机双方共同适应
- Matthew E. Taylor｜迁移、教学与人类输入：交互中人的适应

## 实验设计、统计与算法测试

从精确测试到受控学习曲线，再到多运行基准。每一层支持不同强度的结论。

预先定义随机单位、主指标与失败规则。持续学习的窗口不是独立训练样本。

[教材](../textbook/experiments.md) · [实验](https://yingwen.io/zh/continual-rl/labs/#experiment-experiments)

- Adam White｜从实时预测知识到可信实验：实验设计、性能变异与调参。
- Martha White｜可靠 off-policy 学习到长期控制：算法比较与基准方法论。
- Rishabh Agarwal｜统计可靠的 RL 比较与可塑性诊断：统计可靠的深度 RL 评价。
- Joseph Modayil｜从感知结构到可验证预测知识：分别测预测误差与控制效用。
- Randy Goebel｜知识表示、推理与可检验的解释：为解释的忠实性设计干预和对照。
- Kris De Asis｜多步价值学习与面向真实时间的机器人：比较控制频率、延迟与恢复成本。
- Arsalan Sharifnassab｜更新几何、步长适应与流式学习：拆分预条件、尺度控制与元学习的作用。
- Sorina Lupu｜自适应控制与直接从经验学习的机器人：同时报告学习、恢复和平台成本。
- Michael Bowling｜博弈、评测与 agency：从 ALE 与扑克评测学习平台设计、统计方差和可比较条件。
- Csaba Szepesvári｜统计效率和算法边界：让统计效率结论与实际实验的数据访问、资源预算保持一致。
- Levi H. S. Lelis｜让学到的知识指导搜索：测试组合泛化与跨任务摊销，同时保留预训练、建库和模拟成本。
- Martin Müller｜长期价值与临时搜索记忆：精确求解证据、有限样本胜率和持续适应曲线各自回答不同问题。
- A. Rupam Mahmood｜把学习放回物理时间：将机器人设置与生命期成本纳入比较。
- Esraa Elelimy｜高效在线递归学习：联合操纵可观测性与变化机制，而非只延长训练。
- Shibhansh Dohare｜持续更新特征，而非只保护参数：用同难度长期序列与替换对照识别机制。
- Marlos C. Machado｜表示—技能—经验的循环：固定技能与在线发现的对照
- Patrick M. Pilarski｜预测知识、人机共适应与身体：交互价值与使用者负担
- Matthew E. Taylor｜迁移、教学与人类输入：计入教师与源任务成本
- Dieter Büchler｜高动态机器人中的在线适应：恢复、损耗和外部干预

## 经典与深度强化学习：按正文查阅

### 经典强化学习

#### [多臂老虎机：估计、探索与直接策略学习](../foundations/tabular/bandits.md)

没有状态转移时，仍需一边估计动作收益，一边决定下一次尝试什么。这个最小问题把估计误差、探索代价和策略更新分开。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-bandits) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-bandits) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-bandits)

#### [MDP、回报与价值：序列决策的数学对象](../foundations/tabular/mdps.md)

动作会改变后续状态时，需要评价整个未来。本章从随机交互过程推导价值与 Bellman 方程，明确后续算法共同使用的数学对象。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-mdps) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-mdps) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-mdps)

#### [动态规划：评价、改善与最优递推](../foundations/tabular/dynamic-programming.md)

已知环境模型时，怎样通过局部计算得到长期价值和策略？本章将 Bellman 方程转化为迭代，并证明评价与改善之间的联系。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-dynamic-programming) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-dynamic-programming) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-dynamic-programming)

#### [Monte Carlo：完整回报、探索控制与离策略评价](../foundations/tabular/monte-carlo.md)

不知道模型时，可以将完整回报作为样本。估计还会改变下一次行动：需要看清回报来自哪一版策略、探索怎样影响收益，以及长回合怎样消耗有效覆盖。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-monte-carlo) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-monte-carlo) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-monte-carlo)

#### [TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q](../foundations/tabular/temporal-difference.md)

如何在完整回报尚不可用时学习？TD 用下一预测补足未来；控制算法再根据不同的下一动作处理方式，形成不同的学习目标。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-temporal-difference) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-temporal-difference) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-temporal-difference)

#### [多步学习：n-step、Tree Backup 与 Q(σ)](../foundations/tabular/multistep.md)

学习目标可以在一步 bootstrap 与完整回报之间选择，也可以在动作采样与动作期望之间选择。这是两条不同的设计维度。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-multistep) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-multistep) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-multistep)

#### [学习与规划：Dyna、优先扫描和执行时搜索](../foundations/tabular/planning.md)

真实经验既能直接改进价值，也能训练后果模型。规划使用这个模型继续计算，关键是模型语义、backup 的成本以及计算应分配到哪里。

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-tabular-planning) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-tabular-planning) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-tabular-planning)

#### [函数逼近预测：从回归到 TD 固定点](../foundations/approximation/prediction.md)

共享少量参数以后，MC、TD 和最小二乘方法究竟在求解什么？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-prediction) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-prediction) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-prediction)

#### [特征、泛化与半梯度控制](../foundations/approximation/features-control.md)

特征怎样改变学习行为，Sarsa 又怎样在共享参数下改善策略？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-features-control) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-features-control) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-features-control)

#### [持续控制与平均奖励](../foundations/approximation/average-control.md)

智能体没有自然回合终点时，怎样定义和学习长期控制目标？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-average-control) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-average-control) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-average-control)

#### [离策略函数逼近：覆盖、发散与稳定更新](../foundations/approximation/off-policy.md)

行为数据足够覆盖目标策略，为什么 TD 仍可能发散，又能怎样修复？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-off-policy) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-off-policy) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-off-policy)

#### [多步回报、资格迹与 True-online TD](../foundations/approximation/traces.md)

当前到来的奖励怎样更新过去的预测，同时保留正确的在线更新语义？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-traces) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-traces) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-traces)

#### [策略梯度、基线与 Actor–Critic](../foundations/approximation/policy-gradient.md)

直接学习策略时，哪一个目标的梯度能由经验估计，近似从哪里进入？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：与 Barto 合著教材；从本章对应的定义、推导与例子开始阅读。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：与 Sutton 合著教材；把本章算法放回预测、控制与经验学习的问题中。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-approximation-policy-gradient) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-approximation-policy-gradient) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-approximation-policy-gradient)

### 深度强化学习

#### [深度价值学习：DQN、Double DQN 与目标的时间顺序](../foundations/deep/deep-value.md)

把表格 Q-learning 换成网络后，损失、数据与目标为什么都需要重新组织？

- [Hado van Hasselt](https://yingwen.io/zh/continual-rl/resource/S61/)：Double Q-learning 与 Double DQN：区分动作选择和动作评价。
- [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/)：深度价值学习与强化学习课程；对照数据、目标与策略的时间顺序。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-deep-value) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-deep-value) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-deep-value)

#### [策略梯度：从轨迹概率到 GAE 与 actor–critic](../foundations/deep/policy-gradient.md)

延迟奖励怎样改变动作概率？有限 rollout、critic 与停止梯度分别改变哪一项估计？

- [Richard S. Sutton](https://yingwen.io/zh/continual-rl/resource/S01/)：策略梯度定理与 actor–critic 的基础。
- [Andrew G. Barto](https://yingwen.io/zh/continual-rl/resource/S02/)：演员—评论家的基本结构及其教材论述。
- [Pieter Abbeel](https://yingwen.io/zh/continual-rl/resource/S51/)：策略优化与深度强化学习课程。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-policy-gradient) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-policy-gradient) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-policy-gradient)

#### [策略更新的尺度：TRPO 与 PPO](../foundations/deep/trust-region.md)

旧策略的数据能支持多远的策略更新？怎样从局部代理走到采样、更新与独立评价？

- [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/)：TRPO 的共同作者；从代理目标与状态分布近似理解更新。
- [Pieter Abbeel](https://yingwen.io/zh/continual-rl/resource/S51/)：TRPO 的共同作者；区分理论约束、数值求解和 PPO 近似。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-trust-region) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-trust-region) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-trust-region)

#### [连续动作的价值优化：DDPG 与 TD3](../foundations/deep/deterministic-control.md)

不能枚举连续动作时，如何用 critic 的梯度改进 actor？

- [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/)：确定性策略梯度；连续动作的价值导数。
- [Hado van Hasselt](https://yingwen.io/zh/continual-rl/resource/S61/)：双估计器与价值估计偏差，为比较 actor 所依赖的 critic 提供背景。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-deterministic-control) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-deterministic-control) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-deterministic-control)

#### [最大熵连续控制：SAC 的价值、密度与温度](../foundations/deep/entropy-control.md)

随机 actor 不只是加噪声：熵如何进入 Bellman 方程与自动微分？

- [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/)：SAC 的共同作者；最大熵目标、随机 actor 与离策略更新。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-entropy-control) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-entropy-control) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-entropy-control)

#### [深度 RL 的训练循环：经验、目标与更新](../foundations/deep/practice.md)

一条真实经验怎样经过采样、保存、构造目标和求导，最后改变下一次行动？

- [Rishabh Agarwal](https://yingwen.io/zh/continual-rl/resource/S109/)：深度 RL 的统计可靠性；从运行与采样单位理解结果比较。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-practice) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-practice) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-practice)

#### [大规模训练：算法与系统怎样共同设计](../foundations/deep/systems.md)

环境、推理和学习并行以后，怎样把更多计算变成更快的策略改善？

- [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/)：A3C 的共同作者；并行采样与异步更新的算法接口。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-systems) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-systems) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-systems)

#### [不完全可观测：信念状态、信息行动与递归记忆](../foundations/deep/partial-observability.md)

当前观察不能决定未来时，智能体应记住什么，信息又如何影响行动？

- [Leslie Pack Kaelbling](https://yingwen.io/zh/continual-rl/resource/S43/)：POMDP 中的状态不确定性、信念更新与规划。
- [Michael L. Littman](https://yingwen.io/zh/continual-rl/resource/S42/)：POMDP 的求解与学习；对照历史、信念和行动条件。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-partial-observability) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-partial-observability) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-partial-observability)

#### [探索与不确定性：后验、乐观估计和时间一致行动](../foundations/deep/exploration.md)

为什么每一步都随机，并不等于有效获取长期有用的信息？

- [Ian Osband](https://yingwen.io/zh/continual-rl/resource/S64/)：后验采样与深度探索；时间一致的探索行为。
- [Benjamin Van Roy](https://yingwen.io/zh/continual-rl/resource/S46/)：信息、后验采样与不确定决策。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-exploration) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-exploration) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-exploration)

#### [分布强化学习：Bellman 分布、分位数与风险目标](../foundations/deep/distributional.md)

学习完整回报分布，与学习均值、评估风险和估计知识不确定性分别有什么关系？

- [Marc G. Bellemare](https://yingwen.io/zh/continual-rl/resource/S39/)：分布强化学习；回报分布与 Bellman 更新。
- [Will Dabney](https://yingwen.io/zh/continual-rl/resource/S65/)：分位数回归及价值分布的表示。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-distributional) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-distributional) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-distributional)

#### [离线强化学习：数据支持、策略评估与保守改进](../foundations/deep/offline.md)

不能补采数据时，怎样判断策略好坏，怎样避免利用没有证据的高价值动作？

- [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/)：CQL、IQL 等离线 RL 工作；数据支持与策略改进。
- [Nan Jiang](https://yingwen.io/zh/continual-rl/resource/S55/)：离策略评价及其信息和统计条件。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-offline) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-offline) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-offline)

#### [模型学习与规划：MPC、短模型 rollout 和潜在想象](../foundations/deep/model-based.md)

模型在哪里进入决策，预测误差又怎样变成控制误差？

- [Danijar Hafner](https://yingwen.io/zh/continual-rl/resource/S104/)：Dreamer 系列；潜在动力学中的行为学习与想象。
- [Sergey Levine](https://yingwen.io/zh/continual-rl/resource/S49/)：PETS、MBPO 等模型控制工作；模型误差与数据使用。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-model-based) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-model-based) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-model-based)

#### [约束强化学习：占据测度、拉格朗日与可行策略](../foundations/deep/constraints.md)

“回报高且代价不超过预算”与“每一步都安全”之间差了哪些条件？

- [Pieter Abbeel](https://yingwen.io/zh/continual-rl/resource/S51/)：Constrained Policy Optimization 的共同作者；约束、近似和实际策略更新。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-constraints) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-constraints) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-constraints)

#### [多智能体合作：结构化探索与信用分配](../foundations/deep/multi-agent.md)

团队共享一个奖励时，怎样从联合经验中学习可执行的协作策略，并正确处理同伴更新？

- [温颖 Ying Wen](https://yingwen.io/zh/continual-rl/resource/S110/)：合作决策与 MAT 等工作；联合动作结构与学习目标。
- [汪军 Jun Wang](https://yingwen.io/zh/continual-rl/resource/S111/)：多智能体合作学习与联合决策。
- [Jakob Foerster](https://yingwen.io/zh/continual-rl/resource/S58/)：反事实基线与合作信用分配。
- [Shimon Whiteson](https://yingwen.io/zh/continual-rl/resource/S59/)：价值分解和合作多智能体学习。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-multi-agent) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-multi-agent) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-multi-agent)

#### [自对弈与开放式多智能体学习：评估、目标与策略种群](../foundations/deep/multi-agent-populations.md)

自对弈怎样产生课程和训练标签，又怎样通过历史保留、交互评价与策略种群发现值得继续学习的问题？

- [温颖 Ying Wen](https://yingwen.io/zh/continual-rl/resource/S110/)：开放式多智能体学习及合作、竞争场景。
- [汪军 Jun Wang](https://yingwen.io/zh/continual-rl/resource/S111/)：种群学习与博弈中的评估和目标构建。
- [Michael Bowling](https://yingwen.io/zh/continual-rl/resource/S17/)：博弈评估与不完全信息决策。
- [David Silver](https://yingwen.io/zh/continual-rl/resource/S62/)：AlphaGo 系列：自对弈、搜索和策略价值学习。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-marl-populations) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-marl-populations) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-marl-populations)

#### [对手建模与递归推理：预测谁，回应什么？](../foundations/deep/multi-agent-reasoning.md)

给定参与者和评价目标，怎样利用行为预测、条件响应与有限递归改善决策，并检验模型是否可信？

- [温颖 Ying Wen](https://yingwen.io/zh/continual-rl/resource/S110/)：PR2、GR2 等递归策略推理工作。
- [汪军 Jun Wang](https://yingwen.io/zh/continual-rl/resource/S111/)：相互影响与递归博弈建模。
- [Stefano V. Albrecht](https://yingwen.io/zh/continual-rl/resource/S68/)：其他智能体的行为模型与适应。

[配套阅读](https://yingwen.io/zh/continual-rl/library/?chapter=study-deep-marl-reasoning) · [相关学者](https://yingwen.io/zh/continual-rl/people/?chapter=study-deep-marl-reasoning) · [实验](https://yingwen.io/zh/continual-rl/labs/#foundation-study-deep-marl-reasoning)
