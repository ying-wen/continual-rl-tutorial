# 子目标构建：从经验到技能与规划

怎样从已经取得的经验提出子目标，学会执行它，再让真实后果改变规划与下一次行动？

## 本章内容

- 区分从经验选择候选、定义子任务、逐步学策略、冻结 option、学习后果模型与规划六种操作。
- 沿同一运输任务比较三次与四次真实练习，解释为什么有目标、有一次成功记录仍不等于起点已经会执行。
- 从首次到达回报推到目标条件 Q-learning，再理解 UVFA、HER 与课程分别改变什么。
- 推导带 stopping value 的子任务 target，解释奖励相关子任务与到达子目标的关系。
- 计入练习、探索、模型采集和重置成本，并把学得技能的局部证据与自主状态/目标构造的未解问题分开。

## 阅读这条论证

子目标规定一项值得学会的行为，技能是对这项问题的解，模型则预测实际执行这个解的后果。先在同一个运输任务中分清这些对象，再撤掉给定候选与给定技能：用经验提出问题，用真实练习学习动作，最后让外部后果进入规划。

[子目标规定什么](goals.md#lesson-setting) → [分清三种评价](goals.md#lesson-shared-world) → [从已见经验产生候选](goals.md#lesson-goal-discovery) → [逐步学出技能并用于行动](goals.md#lesson-learning-cycle) → [推导可复用的子任务](goals.md#lesson-subtasks) → [选择发现准则](goals.md#lesson-discovery-criteria) → [核对实现](goals.md#lesson-code) → [连接其他学习过程](goals.md#lesson-branches)

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [通用价值函数与预测知识](gvf.md)：理解如何指定可由经验学习的问题，再讨论选择哪些问题。


### 状态、动作、目标

$s$ 是此刻用于决策的信息，$a$ 是可执行动作，$g$ 是希望达到的结果规格。目标可以用坐标、特征、图像或奖励参数表达；表示之外，还需要明确成功条件或奖励函数。

### Q-learning 的含义

$Q(s,a,g)$ 估计先做动作 $a$、随后尽力完成目标 $g$ 的累计回报。一次真实转移提供当前奖励和下一状态；最大值对应后续选择价值最高的动作。

$$
Q(s,a,g)\leftarrow Q(s,a,g)+\alpha\left[r_g+\gamma(1-d_g)\max_{a'}Q(s',a',g)-Q(s,a,g)\right]
$$

### Bootstrap 与半梯度

用自己对未来的估计构造当前标签叫 bootstrap。神经网络训练时通常对 target 停止梯度，只对当前 Q 的输出求导；target network 可减缓标签与拟合器同时变化。

### Option 与子任务不是同一物

子任务规定评价某种行为的标准；option 是一种可执行的解，由内部策略 π 与停止概率 β（以及允许启动的位置）组成。一个子任务可能有多个解，也可能始终解不好。

<a id="problem-definition"></a>

## 本章的问题定义

同一动力学下需要应对一族结果目标，并决定经验如何共享、下一目标如何选择、哪些子任务有长期用途。

### 给定条件与符号

- 候选可以从经验构造，但候选表示、生成与筛选准则、成功和停止语义需要明确。给定目标族的情形另指定评价分布。
- 目标无关动力学、实际转移、可重算标签的信息，以及真实练习、模型验证和重置预算。

### 需要求解的对象

从经验提出可检验的子目标，学出完成它的策略，预测实际技能的外部后果，再由主任务判断其用途；UVFA、HER与课程分别处理其中不同环节。

### 信息与数据权限

$g$ 是当前指定目标；保存状态、动作、后继、实际达到结果与物理终止。事后目标 $g'$ 只能重算评价标签，不能改写已发生转移。

$$
\max_\pi\ \mathbb E_{g\sim p_G}\mathbb E_{\pi(\cdot\mid s,g)}\!\left[\sum_{t=0}^{T_g-1}\gamma^t r_g(S_t,A_t,S_{t+1})\right]
$$

本式规定给定子任务族的求解目标。$p_G$ 是声明的目标评价分布，$r_g$ 为该目标奖励，$T_g$ 为按目标语义定义的停止时刻，$\gamma$ 为折扣。它不唯一规定应发现哪些问题；候选还需按其技能与模型对主任务的用途评价。UVFA共享一族价值，重标记分布不等于 $p_G$；STOMP另以沿途cumulant与停止价值定义更一般的子任务。

### 成立条件与解的含义

- 重标记要求动力学不因目标改变；必要标签可由保留信息重算。
- 随机环境中按实际结果选目标可能条件化噪声；标准HER不普遍无偏。子任务停止价值计时遵从明确的return约定。

判断准则：在原先指定且未用于选择的目标上测成功率、步数和收益；重标记检验奖励与终止重算；子任务还需测其技能/模型对主任务规划的用途。

### 适用边界

- hindsight成功标签不等于原真实目标已经成功。
- 子任务到达率高不证明有主任务价值。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 特例：增加条件 · [持续控制与学习智能体比较](control.md)：相对完整持续控制，本章限定目标参数索引的任务族和可共享动力学；目标输入、重标记权限与目标分布是这类控制问题的具体结构。

- 组合不同学习问题 · [Options 与技能发现](options.md)：子任务是行为评价规格，option是包含执行与停止的可复用解。

- 组合不同学习问题 · [探索与经验选择](exploration.md)：课程选择改变真实经验分配，不能由目标条件控制或HER自动完成。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

提出一个值得到达的地方，并没有给出到达策略；到达成功，也未必对主任务有用。

### 本章的核心思路

分开候选评分、内部控制和外部后果，从同一条经验追踪三者怎样改变。

1. [由已见经验提出候选](goals.md#lesson-goal-discovery)：只用已访问地点及尚未尝试的当地动作生成到达问题；候选评分用于分配练习，不提供成功路径。

2. [从真实动作学习内部策略](goals.md#lesson-learning-cycle)：按到达信号和停止条件推导Q更新，检查终点的成功怎样在后续访问中传回起点。

3. [用实际执行的后果比较技能](goals.md#lesson-learning-cycle)：冻结已学策略后真正执行，完整奖励、时长和终点形成模型；失败也进入模型，规划读取外部后果选择后续行为。

4. [让子任务尊重沿途收益](goals.md#lesson-subtasks)：到达信号可能忽略危险代价，reward-respecting子任务保留沿途奖励并以停止价值表达可复用后果。

5. [区分共享、重标记与发现](goals.md#lesson-derive)：UVFA共享给定问题的参数，HER重用标签；二者本身都不决定哪些新问题值得提出。

结论与条件：固定目标恢复普通控制定义；HER有效性取决于动力学和重标记条件，STOMP停止规则及折扣计时不能凭普通option公式替换。

### 相关方法改变了什么

- UVFA：共享一族目标价值，不决定练习目标。

- HER：重用已发生经验的目标标签，不改变实际交互成功。

- 课程/STOMP：课程选择经验分配；STOMP提出有沿途收益与停止价值的技能子任务。


<a id="lesson-setting"></a>

## 1. 目标如何进入一个强化学习问题

小车已经走过几条运输道路，还看见沿途有未试过的按钮。与其始终等待一个外部目标，可以从这些记录提出“先学会到那个地点，再试一项未知动作”。但是选中了地点不等于会到达：到达的价值还要通过真实经验传回分岔路口，学出的策略要实际执行，后果才有资格进入规划。本章沿同一运输图追踪这几次变化，再讨论目标条件共享、经验重标记和更一般的子任务。

| 研究对象 | 输入与输出 | 没有被顺带解决的事 |
| --- | --- | --- |
| 目标表达 | 由 g 定义奖励 $r_g$、成功条件与可选的终止条件 | 并不保证目标可达或可观察 |
| 目标条件控制 / UVFA | 给 s,a,g，预测 Q；给 s,g，输出动作 | 不决定下一次练哪个 g |
| HER / 重标记 | 给已发生的轨迹，构造另一 g 下的训练转移 | 不把失败事实改成真实外部成功 |
| 目标生成 / 课程 | 给学习进度与经验，选择下一批 g | 不等于已学到可靠技能或它的模型 |
| 子任务 / STOMP | 给沿途 cumulant 与停止价值，学习行为及停止 | 学到行为以后还要学习后果模型并检查规划用途 |

先考虑有限、可观察、动力学不依赖目标的 MDP：环境转移 $P(s'|s,a)$ 不因事后改写 $g$ 而改变。真实数据包含 $(s,a,r,s')$；为了重算目标奖励，还要保存 achieved goal（实际达到的结果）、物理终止与必要的附加信息。下面允许经验重放和环境重置。扩展到 CRL 时，还需明确存储和重置限制，以及目标分布、动力学和目标语义分别怎样变化。

这些能力在一个智能体里各有位置：状态表示使两个到达结果可被区分；候选规则从经验选择值得尝试的结果；子任务规定怎样评价到达过程；在线控制学习改变执行动作；后果模型回答这段行为给主任务带来什么；规划将答案用于下一选择。GVF 可以表达其中的预测问题，却不自动给出该问哪些问题。[原书 §17.1–17.3](http://incompleteideas.net/book/the-book-2nd.html)依次把辅助预测、时间抽象与状态构造连接起来；本章先固定可辨认的地点，隔离目标和行为怎样从经验形成。

<a id="lesson-shared-world"></a>

### 1.1 同一条运输记录，换目标时究竟改什么

先把仓库缩成六个可观察地点。S 可以走上路 S→A→X，也可以走下路 S→B→C→X；所有边单向，X 与 Y 是物理终点。初始世界的上路出口通往 X，Y 暂时没有入边。智能体知道地点名和可用动作，但只能从记录学习动作的后果。两条预先指定的示范路线共提供五次物理转移，中间允许一次回到 S 的重置。目标由外部给定，本例不训练目标生成器。

这组贯穿目标、模型和规划的算例采用到达目标奖励 $1$、其余奖励 $0$，$\gamma=0.9$，成功立即结束；与下一节演示 HER 时采用的 $0/-1$ 奖励约定分别计算。外部给定的 $g$ 在此定义整项任务的奖励；若另有固定的主任务奖励，内部子目标的成功标签不能替代它。事后重标记只构造训练问题，不改写原任务已经取得的真实回报。原始记录保存 $(s,a,s',d_{\rm phys})$，任务标签由下式生成。

$$
r_g(s,a,s')=\mathbf1\{s'=g\},\qquad d_g=d_{\rm phys}\lor\mathbf1\{s'=g\},\qquad y_g=r_g+\gamma(1-d_g)\max_bQ_g(s',b)
$$

只对仍可行动的起点计算；$s=g$ 或物理终点没有该任务的下一动作。奖励在到达时发出，终止后的价值为零。

![同一个六地点运输世界，目标A时上路在中间地点停止，目标X时继续通过A到X。](../assets/crl-figures/goal-model-planning-walkthrough-goals.svg)

橙线是精确规划所得路径，双圈标任务终止；X、Y 同时是物理终点。两图共享五条物理记录。目标 A 的一步回报为 $1$；目标 X 的上路两步回报为 $0+0.9\times1=0.9$，下路三步为 $0.9^2=0.81$。图由附带确定性算例生成。

| 同一记录 $S\to A$ | 目标 $g=X$ | 目标 $g=A$ |
| --- | --- | --- |
| 任务奖励 $r_g$ | $0$ | $1$ |
| 任务终止 $d_g$ | $0$ | $1$ |
| 精确尾值下的 target | $0+0.9\times1=0.9$ | $1$ |
| 之后的 $A\to X$ | 仍是可用行动记录 | 起点已终止，不进入该任务更新 |

这里重用的是物理后果。把目标从 X 换成 A，并没有使已经发生的 S→A 变成别的转移；改变的是它的奖励和继续条件。只改奖励而沿用原来的继续标志，会把 A 以后的价值错误接上。反过来，只保存旧任务的 done 而丢掉物理终止来源，也可能把本来仍可走的 A 永久当成物理终点。[UVFA 原文 §2–3](https://proceedings.mlr.press/v37/schaul15.pdf)正是用目标相关奖励与继续条件定义不同问题；[HER 原文 §3.2–3.3](https://arxiv.org/pdf/1707.01495)再说明如何给已有经验另配目标。

下一次若闸门改变了 A 的出口，旧 S→A 仍可复用，旧 A→X 就不再描述当前世界。重标目标无法把它自动变成 A→Y；那需要新观察。另一个容易漏掉的变化是技能本身：原来“经过 A 到 X”的技能若改成“到 A 即停”，它的执行时长和终点模型也变了。因此，目标规格、技能行为和物理动力学要分别注明版本。

同样，SF 中的累计特征依赖目标策略与累计到何时结束。本例从 X 改成 A，同时改变了首次成功停止条件；不能仅给旧 SF 换一个奖励权重，就默认得到了新任务的正确答案。原子转移模型保留了途中经过 A 的信息，可以重新求解；只保存“最终到 X”的技能摘要则未必保留中途命中 A 所需的信息。[模型章的同例](models.md#lesson-model-repair)继续检查这项复用边界。

这段记录仍由人指定 A、X 两个目标。后面的[目标发现算例](goals.md#lesson-goal-discovery)保留这些道路，再增加两个尚未尝试的动作，让候选地点由记录产生；先分清“给一个目标求答案”和“从经验提出下一问题”，才能判断撤掉了哪项假设。从本页下载独立脚本即可复算当前目标标签、模型更新和规划轨迹：[goal_model_planning_walkthrough.py](../tutorials/goal_model_planning_walkthrough.py)。脚本只用 Python 标准库；输出的数值来自确定性记录与有理数运算。

下载该单文件后，在仓库根目录运行；无第三方依赖

```sh
python3 tutorials/goal_model_planning_walkthrough.py
```

<a id="lesson-derive"></a>

## 2. 从一族控制问题推到 UVFA 与 HER

深入阅读：给定目标族以后，推导共享价值与事后目标重标记，区分求解与发现问题。

对每个目标 g 先固定奖励 $r_g$ 和终止 $d_g$。最常见的到达任务令成功奖励为 0、未成功为 −1；成功立即结束时，回报衡量到达前的折扣步数。也可以成功后继续交互，只在奖励里反映成功。这两种任务的 Bellman 方程不同，不能把 success、环境 terminated 和时间上限 truncated 混成一个 done。

$$
Q_g^*(s,a)=\mathbb E\!\left[r_g(s,a,S')+\gamma(1-d_g)\max_{a'}Q_g^*(S',a')\mid s,a\right]
$$

这是对固定 g 的普通最优 Bellman 方程。期望来自真实动力学；g 告诉我们怎样评价结果，不负责改变已发生的转移。

UVFA 让不同目标 $g$ 共享一个函数近似器。最简单实现把状态与目标特征拼接后输入网络，也可以分别编码再组合；每个目标仍遵循自己的 Bellman 方程。共享使相近目标互相提供训练信号，也可能产生干扰。泛化能力取决于数据覆盖和表示，需在未用于训练的目标上单独评价。

$$
Q_\theta(s,a,g),\qquad y=r_g+\gamma(1-d_g)\max_{a'}Q_{\bar\theta}(s',a',g)
$$

θ 是在线网络，θ̄ 是慢更新的目标网络；表格实现不需要两个网络。这里展示离散动作版本；连续动作可使用目标 actor 提供下一动作。

$$
\begin{aligned}L(\theta)&=\tfrac12\left(Q_\theta(s,a,g)-\operatorname{stopgrad}(y)\right)^2\\ \theta^+&=\theta+\alpha\left(y-Q_\theta(s,a,g)\right)\nabla_\theta Q_\theta(s,a,g)\end{aligned}
$$

对平方误差求导就得到更新。target 被视作本次拟合的固定标签，因此这是半梯度；并不是对完整 Bellman 残差两侧一起求导。

HER 再增加一个数据操作：先真实执行原目标 g，保留轨迹；抽到其中一步时，选一个后续实际达到的结果 g′；状态、动作、下一状态保持不变，只按 g′ 重算奖励和目标相关的终止。于是“没完成 g”仍能为“怎样完成 g′”提供监督。HER 要配合能利用其他行为策略数据的 off-policy 学习器；它不是直接把 PPO 的 on-policy 轨迹标签改掉就仍保留原保证。

**算法：算法伪代码**

1. 收集一段 ($s_t,a_t,s_{t+1}$，$\mathrm{achieved}_{t+1}$，`physical_done`，`info`)
1. 对抽到的时刻 t：
  1. 以一定概率保留原目标 g；否则从未来 $\mathrm{achieved}_{t+1:T}$ 采样 g′
  1. 若当前状态在新任务下已经终止，跳过这条动作转移
  1. 不改 $s_t,a_t,s_{t+1}$
  1. 重算 $r_{g^{\prime}}(s_t,a_t,s_{t+1})$
  1. 按所定义任务重算 $d_{g^{\prime}}$，物理终止仍保留
  1. 构造目标 y；更新 Q（以及连续动作时的 actor）
1. 独立评估：只用预先规定的真实目标，不用 hindsight 标签

在确定性且目标无关的环境中，只要当前状态在新任务下仍可行动，原转移就是同一状态—动作对的有效结果。成功即终止的任务不能使用“从新目标状态继续离开”的转移；若重标记完整轨迹或多步片段，还要在该片段的首次新成功或物理终止处截断，不把之后的动作拼进同一回报。独立单步样本则只需检查它自己的起点与终点语义，不能因记录中更早曾到达该目标而一律删除。随机环境里，按后来实际发生的结果选择新目标，可能条件化转移噪声：例如偶然成功的随机结果，被过度表示为可稳定达成的目标。此时标准 HER 不普遍给出无偏 Bellman 样本。课程分布还会改变训练目标的权重，因此重标记后的训练误差与原目标分布上的成功率需要分别评价。

<a id="experiment-her_replay"></a>

### 实验：HER究竟改了哪部分经验

失败轨迹能怎样成为另一个目标的成功轨迹，又不伪造世界的转移？

**环境与可用信息。** 七格确定性链，状态和目标均为0—6，左右动作在边界截断；动力学与目标无关。从3开始，每回合目标从其他六格随机选择。到达目标奖励0并终止该目标任务，其余奖励−1；12步上限是截断，不是物理终止。

**设置。** 5个种子，各1200个真实转移。表格目标条件Q零初始化，步长0.2、γ=0.95、ε=0.4。两方法都做一次当前转移更新，且每步至多2次回放。HER额外将完整轨迹中已经发生的未来后继选作目标；普通回放只保存原目标标签。

**检验的机制。** 原始状态、动作、后继和物理终止事实不变，只换目标并重算目标奖励与目标终止。若起点已经满足新目标，该样本跳过，不能训练已经终止的任务继续离开。时间截断允许bootstrap。

**测量。** 冻结贪心策略，在42个不同起点—目标组合上测12步内到达率。横轴为真实转移；回放次数与缓冲大小另记。训练目标、回放目标和评估目标分布需要分别描述。

```bash
python3 implementations/extended_knowledge/her_replay.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/her_replay/curves.svg)

横轴：environment_steps。纵轴：七目标冻结贪心到达率。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 600步HER到达率0.6619，普通回放0.7619；1200步分别为0.8333±0.0753与0.8095±0.0292。HER并未全程领先，末端差异也很小。图说明实际目标重标记流程可运行，不说明HER在此设置稳定更强。

**结论边界。** 表格小链，非机器人神经HER。回放缓存随数据增长，且HER保存更多标签；环境步和每步回放数匹配不等于存储量匹配。

**继续实验。** 挑一条未达原目标的三步轨迹，列出所有可合法选用的future目标。分别修改reward、目标done和物理done，核对只有前两者可以随目标改变；再以等缓存容量重跑比较。

[源码](../implementations/extended_knowledge/her_replay.py) · [逐种子记录](https://yingwen.io/crl-code/results/her_replay/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/her_replay/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/her_replay/curves.json)

<a id="lesson-subtasks"></a>

## 3. 从到达目标到有停止规则的子任务

外部给定目标只要求解决一个任务；子任务构造还要决定什么行为值得长期保留。考虑通向门口的两条路：一条两步但经过危险区，每步真实代价 −5；另一条四步但无危险。仅奖励“最短到门口”可能学到对主任务有害的技能。reward-respecting 的出发点是保留沿途真实奖励，同时用停止价值表达“这个结果有时值得准备”。

$$
G^{c,z}=\sum_{k=1}^{K}\gamma^{k-1}C_{t+k}+\gamma^{K-1}z(S_{t+K})
$$

K≥1 是本次子任务停止时刻，C 是累积信号，z 是停止价值。这采用 STOMP 原文式(2)的计时：停止价值与最后一次 cumulant 具有同一折扣指数，不是再过一个原始时间步。

将第一步单独拿出。到 $s'$ 后，以 $\beta(s')$ 的概率停止并获得 $z(s')$；否则从 $s'$ 的下一步继续积累，其全部未来相对当前多乘一个 $\gamma$。对这两个分支取期望，得到以下递推。停止的是子任务回报的累积，物理环境仍可继续运行。

$$
\begin{aligned}v(s)&=\mathbb E_\pi\!\left[C+\beta(s')z(s')+\gamma(1-\beta(s'))v(s')\mid s\right]\\ \delta&=C+\beta(s')z(s')+\gamma(1-\beta(s'))v(s')-v(s)\end{aligned}
$$

β=1 时 target=C+z；β=0 时 target=C+γv(s′)。真实环境终止使用 β=1 且 z=0。两个端点是移植实现时最值得先写的测试。

$$
z^i(s)=w^\top x(s)-w_i x_i(s)+\bar w^i x_i(s)
$$

STOMP 的 feature-attainment 子任务把某一特征的价值权重替换为一个乐观 bonus weight。它不是任意给 feature 一个大奖励：其他特征当前的价值仍保留，沿途 cumulant 也保持真实奖励。

固定目标策略与停止函数时，可以直接做 TD；行为由另一策略 $b$ 产生时，一步半梯度更新可乘 $\rho=\pi(a|s)/b(a|s)$。分母使用产生动作时记录的概率，且目标动作需要行为支持。这个比率修正当前动作分布；函数逼近与 off-policy bootstrap 的稳定性仍需另行分析。

$$
\begin{aligned}v(s)&\leftarrow v(s)+\alpha\rho\delta\\ \theta&\leftarrow\theta+\alpha_\pi\rho\delta\nabla_\theta\log\pi_\theta(a|s)\end{aligned}
$$

这里取 $\lambda=0$，同一 TD 误差 $\delta$ 在 critic 更新前缓存。重要性比修正当前动作分布；actor 所见的状态分布仍由采样过程决定，完整目标的梯度解释还取决于相应占用权重。

停止收益的计时还会影响停止规则。STOMP 式(9)采用 $\beta(s)=\mathbf1[z(s)\geq v(s)]$；若保持当前 $v$ 不变、单独最大化前面的单步 target，比较量却是 $z(s)$ 与 $\gamma v(s)$。例如 $z=9.5$、$v=10$、$\gamma=0.9$ 时，前一规则继续，后一规则停止。因此，复现该论文应保留其规则；设计另一种最优停止问题则应明确 return 与边界条件，再重新推导，不能仅凭符号相似互换。

**算法：算法伪代码**

1. 初始化每个子任务的价值 $v_i$、策略 $π_i$、停止价值规格 $z_i$
1. 每个真实转移：
  1. 记录旧价值、旧策略概率、行为概率 b(a|s)
  1. 由选定规则计算 $β_i(s^{\prime})$，环境终止强制 $β_i=1,z_i=0$
  1. 对每个 i 计算 $δ_i = r + β_i z_i + γ(1-β_i)v_i(s^{\prime}) - v_i(s)$
  1. 更新 critic；用已缓存 $δ_i$ 更新 actor
  1. 另行学习此 option 的真实 reward/end-state 模型
  1. 评价该模型用于主任务规划的改善，不能只看子任务成功率

这里新增了可学习的内部行为与停止比较。STOMP 的实验先学 options，再学它们的模型，最后规划；论文把这些过程同时运行、让规划用途反过来选择特征，列为后续问题。下面的运输例采用更小的固定首次到达子任务，用表格 Q-learning 学策略，停止规则仍由目标定义。它让学习—模型—规划的时序可以逐行检查，不将这项教学实现称为完整 STOMP。

<a id="lesson-goal-discovery"></a>

### 3.1 撤掉给定目标名单：从未试动作提出到达问题

前面的子任务有了评价方式，却还没有说明应该为哪个特征创建子任务。回到六地点运输世界：如果直接写下“学会到 A”和“学会到 C”，目标名单仍是设计者提供的。现在只给地点可辨认的特征，以及到达当地后可见的动作按钮；转移的终点和奖励必须执行以后才知道。让记录决定哪个地点值得成为下一次到达问题。

保留 §1.1 的五条单向道路与两条种子路线。作两项明确扩展：A、C 各多一个 probe 按钮；主任务改为有净收益的送货，普通到 X 收益 1，Y 收益 2，A 的 probe 道路另有费用 2。因此环境的完整规则是 A→Y 净奖励 0，C→Y 净奖励 2；这些新后果只供读者事后核对，选择器在执行前都看不见。X、Y 仍是物理终点，每次送货可重置到 S，折扣仍为 0.9。种子经验只走普通道路，共 5 原子步；Y 尚未被观察。

学习器保存实际记录 $(s,a,R,s',d_{\rm phys})$、尝试计数 $N_t(s,a)$ 和已观察地点的动作集合 $\mathcal A_t(s)$。这里 $R$ 是环境实际发出的净奖励，并没有向学习器提前提供费用或终点收益的分解。地点特征为已见地点的 one-hot 编码 $x_j(s)=\mathbf1\{s=j\}$；候选地点由记录筛选，不再预先列出 A、C，但基本状态特征与筛选准则仍由设计者规定。

$$
\begin{aligned}\mathcal E_t&=\{(s,a,s'):N_t(s,a)>0\},\\ U_t(s)&=\{a\in\mathcal A_t(s):N_t(s,a)=0\},\\ \mathcal C_t&=\{g\in\mathcal S_t^{\rm obs}:g\notin\mathcal T_t,\ U_t(g)\ne\varnothing,\ d_t(S,g)<\infty\},\\ g_t&\in\arg\max_{g\in\mathcal C_t}J_t(g),\qquad J_t(g)=\frac{1}{d_t(S,g)+1}.\end{aligned}
$$

$\mathcal T_t$ 为已观察物理终点；$d_t$ 是已观察有向图上的最短原子步数，由广度优先搜索计算。每次到达后只试一个未试动作，所以分子为 1 条新状态—动作记录，分母为到达步数加 1 次探测。平分时按地点名排序。分数单位是“新记录 / 原子步”，没有奖励单位。

![五条已观察道路中，A和C有未知后果的动作短桩；根据已观察路程，A的覆盖分数为二分之一，C为三分之一。](../assets/crl-figures/goal-discovery-walkthrough-candidates.svg)

灰色 ? 只表示当地观察到一个未试按钮，不透露其终点。所有已见地点都经过同一规则筛选：S、B 无未试动作，X 已物理终止，Y 尚未观察；于是产生候选 A、C。到 A 后试一次动作需 2 原子步，到 C 需 3 原子步。图和后两图均为附带程序生成的原创确定性算例。

这是本节自建的可检查基线：在经验覆盖的边界上提出目标。候选集合由数据计算，删去 A 的 probe 按钮就只产生 C；完成一次 A 的探测也会使 A 退出集合。它撤掉了人工枚举具体目标的要求，但仍保留了地点表示、动作可见性、首次到达任务族、覆盖评分和探索预算。若按钮也不可观察，或两个物理地点共用同一个特征，规则需要新增信息，不能沿用本例的可达性结论。

这个分数只估算“已经会走已见最短路时，再增加一条记录”的交互密度；它没有计入把技能学会的练习、失败尝试和重置成本，也没有考虑主任务价值。它会先选近处 A，即使探测最终没有送货收益。下面让执行能力从零学习，并把这些额外成本全部留下来。

<a id="lesson-learning-cycle"></a>

### 3.2 从真实练习到技能、模型和下一次送货

候选评分只决定练什么，不能把图中的最短路直接写进策略。这里给每个选中的目标新建一张零初值动作价值表，并固定当时已经真实尝试过的动作清单。练习从 S 开始，只在这份清单内先选该子任务练习次数较少的动作；并列按当地菜单顺序，上路在下路之前。这是明确给定的覆盖规则。它让新按钮留到技能到达后再探测，也使策略学习的数据来源与候选图计算分开；“试过”本身并不是一般环境中的安全保证。

目标 $g$ 规定首次到达问题。令 $\tau_g\geq1$ 为命中目标的原始步数，$\tau_{\rm phys}$ 为物理终止步数；从尚未停止的位置开始，固定策略 $\pi$ 的动作条件答案为

$$
F_g^\pi(s,a)=\mathbb E_\pi\!\left[\gamma^{\tau_g-1}\mathbf1\{\tau_g\leq\tau_{\rm phys}\}\mid S_0=s,A_0=a\right].
$$

永不到达贡献为 0。到达当步发出成功信号 1，所以一步成功的答案为 1，不是 $\gamma$。这是累积信号 $c^g=\mathbf1\{S'=g\}$、延续 $\Gamma_g=\gamma\mathbf1\{S'\ne g,\neg d_{\rm phys}\}$、策略 $\pi$ 所定义的 GVF。控制学习再比较可执行动作，寻找较大的答案。

拆出第一步：若到达目标，答案就是成功信号；若仍能继续，后续答案多折扣一次。用当前这条真实转移代替后果期望，再减去更新前的估计，得到表格 Q-learning。这里当前动作已是条件，不需给样本再乘当前动作的重要性比率；最大值选择的是下一步的内部行为。

$$
\begin{aligned}
      c_t^g&=\mathbf1\{S_{t+1}=g\},\qquad d_t^g=d_{\rm phys}\lor\mathbf1\{S_{t+1}=g\},\\
      y_t^g&=c_t^g+\gamma(1-d_t^g)\max_{b\in\mathcal A_{\rm tried}(S_{t+1})}F_g(S_{t+1},b),\\
      \delta_t^g&=y_t^g-F_g(S_t,A_t),\qquad F_g(S_t,A_t)\leftarrow F_g(S_t,A_t)+\alpha\delta_t^g.
   \end{aligned}
$$

尾值来自写入前的表，停止后为 0。本例固定 $\gamma=0.9,\alpha=1$，每条真实练习转移只更新一个元素，不重放种子记录，也不在模型上回溯。有限次练习的表只是估计；到达过目标并不保证起点动作已经选对。

到 A 的第一次练习走 S→A，就把 $F_A(S,\mathrm{upper})$ 从 0 改成 1。到 C 更能显出学习过程：第一次走上路，在 X 物理终止，内部成功为 0；第二次走 S→B→C，S→B 当时仍读到 B 的旧零值，只有随后 B→C 把 $F_C(B,\mathrm{forward})$ 写成 1。第三次覆盖上路；直到第四次实际重访 S→B，起点才读到 $0.9\times1$。这里变化的不是目标名字，而是控制器会读取的动作价值。

![真实练习在t20先走S到B仍读取零值，t21到达C把B的值写成1，t24再次S到B才把0.9写回起点。](../assets/crl-figures/learned-subgoal-propagation.svg)

图由 learned_subgoal_cycle.py 的四次练习分支生成。三帧对应新到来的真实样本，不是同一经验上的三轮模型备份；青色目标 C 的到达信号是内部成功 1，沿途外部奖励仍为 0。

| 冻结前的真实练习数 | 到 C 的起点动作值 | 冻结策略的实际执行 | 记录的后果 |
| --- | --- | --- | --- |
| 3 次 | upper = 0，lower = 0；并列选 upper | S→A→X；未到 C | 外部折扣奖励 0.9，终点 X，物理终止 |
| 4 次 | upper = 0，lower = 0.9 | S→B→C；到 C 停止 | 外部折扣奖励 0，终点 C，仍可行动 |

每个练习预算结束后，复制当前贪心策略为版本 1，从 S 发起一次真实执行；执行期间不修改它。三个练习得到的 C 技能虽然不成功，仍是一个明确的行为，模型必须记录它实际停在 X。它不能因为名称含有 C 就生成“到达 C”的标签。若只观察到尚未停止的前缀，则不写完整后果模型。本图是单向有限道路，每次尝试最多三步自然结束，没有暗中增加超时；若另加时间上限，它必须成为技能停止规格或明确的观测截断。

冻结版本把模型的问题固定下来：从 S 执行这个策略直到目标或物理终止，会得到什么？完整执行给出奖励和、时长与终点。使用外部奖励 $R$，而非内部成功 $c^g$，构造一次 Monte Carlo 模型样本：

$$
\hat R_o(S)=\sum_{j=1}^{K}\gamma^{j-1}R_j,\qquad
      \hat B_o(j|S)=\gamma^K\mathbf1\{S_K=j\},\qquad
      \hat q_V(S,o)=\hat R_o(S)+\sum_j\hat B_o(j|S)V(j).
$$

本例每个冻结技能先有一次完整样本。随机环境需继续估计这些量的期望；这里的确定性只让读者可以手算核对，程序仍须先执行才有该行。$V$ 在物理终点取 0，技能目标本身不一定是物理终点。多次样本平均，或模型章的逐步 GVF 递推，都是后续可选估计器。

四次练习版本中，到 A 的模型是 $(\hat R,K,\hat B_A)=(0,1,0.9)$，到 C 的模型是 $(0,2,0.81)$。在真正到达后，高层另选一个当地未试按钮。A 的 probe 实得净奖励 0，A 因已无未试动作而退出候选；C 的 probe 实得 2，C 随后也退出。这个新后果进入原子经验模型，使外部价值规划得到 $V(A)=1,V(C)=2$。候选的更新依赖尝试计数，主任务偏好的更新依赖实际奖励，它们不是同一个评分。

规划调用两个已执行技能的模型，得到 $0+0.9\times1=0.9$ 与 $0+0.81\times2=1.62$。随后不是只输出“应选 C”：程序在新的真实送货中调用冻结的 C 技能，实际首动作从先前送货的 upper 改成 lower，走 S→B→C，停止后再选 probe 到 Y。这次真实回报为 1.62。三次练习分支没有到达 C 去探测，两个技能当前外部模型的分数均为 0.9，按固定并列规则选 A；失败并未从数据中消失。

![三次练习冻结的C技能实际走上路到X失败，四次练习则到C形成对应后果模型，探测后规划在新回合实际部署S到B到C到Y。](../assets/crl-figures/learned-subgoal-execution.svg)

两支都执行当前表的真实贪心策略，不把失败技能改写成正确路径。X 是物理终点，其后续价值为零；C 只是技能停止点，还能接高层动作。三次与四次练习的交互和重置预算不同，本图解释学习传播与后果变化，不作同等成本的性能比较。

| 计入的真实操作 | 每目标 4 次练习 | 每目标 3 次练习 |
| --- | --- | --- |
| 两条种子路线 | 5 原始步 | 5 原始步 |
| 探测前实际送货 | 2 | 2 |
| 内部策略练习 | 16 | 11 |
| 冻结技能的后果采集 | 3 | 3 |
| 到达后的未知按钮探测 | 2 | 1 |
| 最后实际部署 | 3 | 2 |
| 合计原始步 / 允许的重置 | 31 / 13 | 24 / 11 |

重置把当前地点送回 S，是本例明示允许且单独计数的外部操作，不占一个普通动作，也没有附赠任务奖励。尤其是成功到 A 或 C 后的练习重置，并非物理环境终止所自动提供。两个预算分支说明一次额外练习怎样改变知识与行为，不是同等成本下的算法优劣实验；纯原子规划得到相同新记录也能选到 1.62，本例不把新信息的收益全归给时间抽象。

下载 [learned_subgoal_cycle.py](../tutorials/learned_subgoal_cycle.py)，在仓库根目录运行下列命令，也可使用文件绝对路径。GoalQ.update 只接收刚到达的练习记录；freeze 复制策略；complete_model_sample 核对完整连续执行、版本、停止位置与外部奖励；最后的 model_score 才把模型交给当前外部价值。基础地点、覆盖规则、练习预算和重置权限仍给定，尚未实现状态特征构造、任意停止规则学习或自主计算分配。

Python 标准库；输出逐条真实经验、TD 写入、冻结策略、实际技能后果、规划分数、真实部署和预算。

```sh
python3 tutorials/learned_subgoal_cycle.py --practice-episodes 4
python3 tutorials/learned_subgoal_cycle.py --practice-episodes 3
python3 tutorials/learned_subgoal_cycle.py --test
```

<a id="lesson-discovery-loop"></a>

### 3.3 对照：在已观察模型上求解同一到达问题

深入阅读：在相同经验图上直接求路径，作为主文逐步控制学习的模型求解对照。

如果允许充分重用已有模型，同一个候选还可不经过上述真实练习就被求解。这个对照先保存确定性转移，再从零计算首次到达答案 $F_g(s,a)$：到达 $g$ 的内部成功信号为 1，否则为 0；首次成功或物理终止就停止。网页计算对唯一状态—动作记录做同步全量备份；独立 Python 脚本枚举经验支持的路径并选最短路径。二者都在已观察模型上求解内部策略，不是刚才的逐步真实 TD 学习。

更精确地说，令 $\tau_g\geq1$ 为首次到达 $g$ 的原子步数，$\tau_{\rm phys}$ 为物理终止步数。固定策略的答案是 $\mathbb E[\gamma^{\tau_g-1}\mathbf1\{\tau_g\leq\tau_{\rm phys}\}\mid s,a]$，永不到达时贡献为 0；$F_g$ 求其中数据支持动作上的最优答案。指数少 1 来自“到达这一步发出成功 1”的计时，与之后首次到达价值采用的 $\gamma^{\tau_g}$ 约定相差一个 $\gamma$。

$$
\begin{aligned}c^g(s,a,s')&=\mathbf1\{s'=g\},\quad\beta_g(s')=\mathbf1\{s'=g\ \text{或}\ d_{\rm phys}=1\},\\ F_g^{k+1}(s,a)&=c^g(s,a,s')+\gamma(1-\beta_g(s'))\max_{b:N_t(s',b)>0}F_g^k(s',b),\\ \pi_g(s)&\in\arg\max_{a:N_t(s,a)>0}F_g(s,a).\end{aligned}
$$

每轮右侧全部读取旧表，空动作集合尾值为 0。起点已是 $g$ 或物理终点的记录不参与该任务更新。启动集取具有正到达值的非目标地点；不在此集合就不调用该 option。首次到达停止的函数族是给定的，$g$ 的选择改变停止地点；本例没有训练任意的 $\beta$。

目标 A 在第一次备份后就使 $F_A(S,\mathrm{upper})=1$，策略走 S→A。到 A 即停的 option 只消耗 1 原子步，途中真实奖励为 0。随后探索控制器接过行动权，按按钮顺序选择一个未试动作，实际执行 A 的 probe，得到 $(A,\mathrm{probe},0,Y,1)$。这条新记录使 $N(A,\mathrm{probe})$ 从 0 变成 1，A 退出候选集。此次两步真实回报为 0；不探测而走已知 S→A→X 本可得到 0.9。探索的损失没有被隐藏。

下一回合重置到 S，重新计算只剩 C 的候选集。为 C 求答案时，第一轮从 B→C 写入 $F_C(B,\mathrm{forward})=1$，S 仍读到旧的 0；第二轮才有 $F_C(S,\mathrm{lower})=0.9$。策略因此成为 S→B→C。第三轮仅用于确认表格不再变化。这解释了目标规格如何变成可执行策略，而不仅是把一面旗子放到 C。

![目标C固定后，第一次同步备份将到达成功写入B，第二次将价值传回S并确定经过B到C的动作。](../assets/crl-figures/goal-discovery-walkthrough-learning.svg)

节点旁的 F 数值显示目标 C 下的最佳到达值 $\max_a F_C(s,a)$，奖励在到达时发出；两次备份没有新增物理转移。橙箭头由计算后的动作价值选出。C 是 option 的停止点，仍可由高层选择下一原子动作；它没有因此变成物理终点。小图省略不影响到达 C 的探测支路。

后果模型接着回答另一问题：执行刚得到的策略，在现实任务中会收到多少奖励、经过多久、停在哪里？本例将已记录的原子后果沿固定策略组合，得到下面的精确经验模型；准确性来自这里的确定性、平稳性和路径完全覆盖，未观察动作仍没有模型。

$$
\hat R_o(s)=\sum_{j=1}^{K}\gamma^{j-1}R_{t+j},\qquad \hat B_o(s,s')=\gamma^K\mathbf1\{S_{t+K}=s'\},\qquad \hat q_V(s,o)=\hat R_o(s)+\sum_{s'}\hat B_o(s,s')V(s').
$$

$K$ 按原子步计量；$\hat B$ 是折扣后的终点权重，并非总和为 1 的概率。本节确定性模型省略期望符号。内部成功信号 $c^g$ 和 STOMP 停止 bonus 均不进入 $\hat R$。这里端点采用 $\gamma^K$，区别于上一节 STOMP 子任务 return 的 $\gamma^{K-1}z$。

| 从 S 启动的对象 | 到 A 的 option | 到 C 的 option |
| --- | --- | --- |
| 内部答案 $\max_a F_g(S,a)$ | 1 | 0.9 |
| 真实累计奖励模型 $\hat R$ | 0 | 0 |
| 时长 K（原子步） | 1 | 2 |
| 终点权重 $\hat B$ | A 上 0.9 | C 上 0.81 |
| 该停止规则的来源 | 给定首次到达族，目标 A 由候选规则选择 | 给定首次到达族，目标 C 由新记录后重新选择 |

实际执行 C 的 option 后，高层再试 C 的 probe，第一次收到 C→Y 的净奖励 2。此次真实路径 S→B→C→Y 的回报为 $0+0.9\times0+0.9^2\times2=1.62$。记录进入同一模型，已观察动作的外部 Bellman 规划得到 $V(A)=1$、$V(C)=2$。送货控制器现在读取两个 option 模型：$\hat q_V(S,o_A)=0+0.9\times1=0.9$；$\hat q_V(S,o_C)=0+0.81\times2=1.62$。它在新的真实回合选择并复用 $o_C$，停止后再做 probe，再收到奖励 2。

![先选择A探测得到零回报，再根据更新后的候选选择C探测得到1.62；外部模型消费者随后复用到C的option并完成送货。](../assets/crl-figures/goal-discovery-walkthrough-loop.svg)

上方记录两次实际探测，第一次的零收益保留为负例；下方条长表示外部规划的折扣回报，0.9 与 1.62 共用尺度。候选更新依赖刚发生的探测，送货选择依赖真实奖励模型。部署回合另计 3 原子步；图中没有将模型备份当作真实经验。

全过程用 5 步种子经验、5 步目标引导探测、3 步实际复用，共 5 回合、4 次重置；两次探测并不是免费评估。保存 7 个不同状态—动作后果及其计数、2 个 option。两个目标求解共做 23 次记录备份，含最后的收敛确认轮；广度优先搜索、后果模型组合与外部价值规划另行计算。这里验证了谁使用哪项知识，并未测量 options 相比原子动作的规划加速。最终外部价值使用的转移也足以让原子动作规划得到 1.62。

独立运行 [goal_discovery_walkthrough.py](../tutorials/goal_discovery_walkthrough.py)：Python 标准库版本用有理数和穷举有限路径求解，网页数值模块则用同步 Bellman 备份，两条路线互查目标答案、时长、外部回报及候选变化。脚本还只把 C 的 probe 奖励从 2 改为 0：候选、到 C 的策略与路程都不变，主任务最优回报却回到 0.9。

下载单文件后在其目录运行，或使用文件绝对路径；Python 3 标准库。输出候选、真实路径、两类价值、预算与精确断言。

```sh
python3 tutorials/goal_discovery_walkthrough.py
python3 tutorials/goal_discovery_walkthrough.py --json
```

<a id="lesson-discovery-criteria"></a>

### 3.4 更换发现准则时，问题与证据怎样改变

覆盖只是一个提出问题的准则。访问频率受到行为策略影响，常经过某处可能只是来回重复；图的瓶颈性描述连接结构，却没有算穿越危险区域的奖励代价；学习进度描述某个答案正在变得容易学习，也没有说明这个答案会被哪次控制使用。本例第一次选择 A 就把这个差别暴露出来：覆盖评分较高，到达也准确，探测回报仍为 0。

表示驱动的发现进一步改变候选的形状。以 [Laplacian eigenoptions 原文 §3](https://proceedings.mlr.press/v70/machado17a/machado17a.pdf) 为例，从转移图的归一化 Laplacian 取一个特征方向 $e$，由 $c^e(s,s')=e^\top(x(s')-x(s))$ 定义内部奖励；再求解该内部控制问题，并加入价值为 0 的 terminate 动作。若继续的最佳内部动作值不大于 0，就结束这个 option。特征方向提出了问题，策略与终止则是求这个问题的答案；一个特征向量本身还不是技能。

作者的 [main.py::discoverOptions](https://github.com/mcmachado/options/blob/master/main.py) 可逐项追到邻接矩阵、归一化 Laplacian、两个符号方向、内部奖励和策略迭代；[Learning.py](https://github.com/mcmachado/options/blob/master/Learning.py) 处理 terminate 及平分时的停止选择。该历史表格入口直接读取环境图和模型，拥有比本节记录限定机制更多的信息，不能当作“仅凭这五条经验已经发现全部技能”的实现证据。

当发现出的行为真的改变下一批经验，新的经验再改变表示，才形成表示驱动发现的循环。[Machado 的 ROD 讲座概要](https://ojs.aaai.org/index.php/AAAI/article/download/35100/37255)明确区分收集数据、学习表示、规定子任务、求解并使用 options 四个环节。本节闭合了候选—行为—新记录这条链；学习神经表示、随表示漂移维护旧技能，以及大规模探索仍是不同的研究工作。

STOMP 把注意力推进到模型的使用者：保留沿途外部奖励，用与特征相关的停止价值准备行为，再用真实后果模型参与规划。原论文的特征达到构造假设已有特征；四房间实验还明确指定四个门口特征。因此它提供了奖励相关的子任务设计与行为—模型—规划证据，并没有完成任意感官输入上的全自主目标生成。正文 §3 的停止价值约定与本例固定首次到达规则也应分别保留。[原文 §2、§5、§7](https://arxiv.org/html/2202.03466v3)。

由此可以提出一个可检验的下一步：保持相同种子记录、总真实交互和模型内存，分别按覆盖、图结构、近期答案改善或预测的外部规划贡献选择目标；把 A 的费用或 C 的收益作为唯一变化，记录实际选择、后续覆盖、模型误差以及真实任务回报。总预算必须包括技能练习、探测和实际复用，重置权限与次数也要匹配。如果只给探测计费，模型内直接求出的策略与真实学习策略就会被不公平地比较。

<a id="lesson-curriculum"></a>

## 4. 哪个目标值得现在学：显式课程到持续目标生成

深入阅读：进一步研究怎样分配练习，不把候选选择、内部策略与外部收益评价混为一谈。

先做一个透明的有限目标课程：每个目标维护尝试次数、成功次数和近期成功率，优先采样既非几乎必成、也非长期完全失败的目标。这是“适中难度”的可运行基线，不是 Goal GAN 的完整实现。Goal GAN 再学习一个生成器来覆盖适中难度目标区域；分类器用当前策略对目标的成功概率构造标签，生成器提出下一批练习目标。它增加的是目标分布学习，而不改变低层控制的 Bellman 方程。

$$
\hat p_t(g)=\frac{\text{近期成功次数}}{\text{近期尝试次数}},\qquad \mathcal G_{\rm learnable}=\{g:\ell\leq\hat p_t(g)\leq u\}
$$

近期窗口避免历史累计均值完全淹没近期变化；阈值与置信度、窗口长度共同决定课程。探索保底应保证暂时失败的目标并非永远被排除。

学习进度是另一种信号：比较同一目标在两个相邻窗口的成功率或预测误差改善。纯粹选进步最快的目标可能遗忘旧目标，也可能把噪声造成的波动当作进步；所以应保留复习预算、未探索目标预算和难度不确定性。CRL 中目标语义与环境可达性一起变化时，还需要区分“技能退化”与“世界变了”。这应成为实验中的独立变量，而非让课程默默选择更容易的任务。

目标少见，还有可能是从当前位置根本到不了。SUN（2026 预印本）因此把新颖度与目标条件可达价值相乘：从 replay 取候选，用当前策略的到达能力折减新颖目标的吸引力。为明确其含义，先考虑首次到达即停止计分的状态目标：

$$
V_{\rm hit}^{\pi_g}(s,g)=\mathbb E_{\pi_g}[\gamma^{\tau_g}\mid S_t=s],\qquad g_t\in\arg\max_{g\in\mathcal C_t}\widehat V_{\rm hit}(S_t,g)\nu_t(g)
$$

$\tau_g$ 是从现在到首次命中目标的步数，$0<\gamma<1$，约定 $\gamma^\infty=0$；$\mathcal C_t$ 是候选集，$\nu_t$ 是有限非负新颖度。精确值落在 $[0,1]$，结合到达概率与时间；若到达后继续累计奖励，它就不再是这个首次到达量。

SUN 还在当前可达价值相对目标选定时起点的估计下降时重新选目标。这个规则需要可靠的价值估计；神经近似可能给不可达目标正值。候选也限于已见经验。用于有限内存持续学习时，淘汰经验后怎样维护计数，以及重新开放的旧区域是否值得再探索，便成为目标选择的新问题。[原文 §3、§5 与附录 C](https://arxiv.org/html/2609.08642v1)。

进一步还要问：到达一个少见目标，获得了什么知识？Viteri 等（2026 预印本）构造了一个启动时的不可逆选择：检查模式告知闹钟以后是否会响；游戏模式提供已知规律的随机显示，放弃这次检查。两种模式以后都能听到闹钟。持续沉默却无法在所有可能延迟下区分“尚未响”与“永不响”，因此丰富的显示变化可以伴随永久的信息损失。

他们用统计实验比较经验：允许更长采集时间后，能否用同一个不知真实世界的随机变换，在所有候选世界中重现另一策略的有限历史分布？这是检验内在目标的一种标准，关注目标最优策略取得的证据。把它用于课程研究，可在匹配预算下另测新任务预测或控制：区分“覆盖变广”与“学到了可复用差别”，并将学习算法能否找到该策略作为另一个问题。[原文 §2–3](https://arxiv.org/html/2610.02159v1)。

| 分支 | 实际改变的对象 | 必要对照 |
| --- | --- | --- |
| 固定目标集 + 均匀采样 | 只控制训练目标分布 | 先确认控制器能解单目标 |
| 适中难度 / Goal GAN | 采样或生成可学习目标 | 相同交互预算下与均匀目标比较 |
| 学习进度驱动 | 按能力变化分配练习 | 区分噪声、遗忘和真正进步 |
| 奖励相关子任务 | 改变行为学习的 cumulant / stopping value | 与最短路径、随机技能比较规划收益 |
| 持续目标发现 | 候选产生、评估、保留与淘汰共同在线变化 | 计入候选学习与模型维护的算力/内存成本 |

<a id="experiment-learning_progress"></a>

### 实验：“现在学什么”先用可检查的小问题检验

最近预测误差变化大，是否就代表值得投入更多样本的可学习任务？

**环境与可用信息。** 这是任务选择子机制，不是完整MDP控制。每轮选择三个回归任务之一，观察均匀分布于[-1,1]的x及目标y；前两个为y=x和y=−2x，第三个y为单位高斯噪声。没有环境奖励、状态转移或终止；平方误差只是调度信号。

**设置。** 5个种子，各1200个训练样本。三个斜率参数零初始化，平方损失SGD步长0.05。每任务保存最近20个更新前误差，比较前后两个10样本窗口的均值差绝对值；收集满20个样本后用ε=0.2选择最高进展任务。对照均匀抽任务。

**检验的机制。** 调度器只看已到达样本的误差，不读取真实斜率。绝对窗口差也会把噪声造成的升降算作进展；因此高进展不必意味着可降低的误差或下游有用性。

**测量。** 图为冻结参数后三任务均匀平均的解析预测MSE，包括第三任务不可约噪声贡献1/3。另看各任务的采样次数，而不只看调度器偏爱的任务自身损失。

```bash
python3 implementations/extended_knowledge/learning_progress.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/learning_progress/curves.svg)

横轴：training_examples。纵轴：冻结三任务期望预测MSE。每种方法 1200 training_examples；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 最终自适应调度MSE为0.33626±0.00343，均匀调度为0.33798±0.00608，都接近1/3噪声下限。这个结果未显示明确课程优势，也提示任务容易时复杂调度可能没有必要。

**结论边界。** 没有目标可达性、动作执行成本、目标生成或机器人区域划分。不能把这个回归课程组件写成完整内在动机智能体。

**继续实验。** 只提高噪声任务的方差，保持两个可学习任务不变。检查它是否吸走更多预算。再用误差下降的有符号差替换绝对差，明确这改变了什么调度假设。

[源码](../implementations/extended_knowledge/learning_progress.py) · [逐种子记录](https://yingwen.io/crl-code/results/learning_progress/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/learning_progress/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/learning_progress/curves.json)

<a id="lesson-example"></a>

## 5. 一次经验如何产生两个不同但合法的更新

深入阅读：对同一实际转移重算目标标签，检查哪些终止与奖励可以改变。

把旗子从位置 4 移到位置 2，已经走过的那一步保持不变。改变的是对到达结果的评价，以及是否还要接上后续价值。

![两幅相同五格地图保留位置一到二的真实路径，只移动目标旗帜；原目标继续自举，事后目标在到达处截断。](https://yingwen.io/crl-figures/concept-crl-mechanisms-goals.svg)

两图是同一条经验的两种目标标签，不是重新走了两次。目标 4 尚未达到，虚线弧将状态 2 的后续价值用于更新状态 1；目标 2 已达到，金色竖线按“成功即终止”的任务定义切断自举。下方数轴共用尺度：从旧 $Q=-2$ 向各自目标走一半，分别落在 −1.95 和 −1。

小车从 $1$ 向右到 $2$，原目标为 $4$，$\gamma=0.9$、$\alpha=0.5$。当前动作价值为 $-2$，下一状态对目标 $4$ 的最大动作价值为 $-1$。任务成功即终止，成功奖励为 $0$，其余为 $-1$。原目标未达成，故 target 为 $-1+0.9\times(-1)=-1.9$，新值为 $-2+0.5\times0.1=-1.95$。

把同一转移重标为目标 $2$：物理过程不变，但目标已达成，$r'=0$、$d'=1$，不再 bootstrap。若当前动作对目标 $2$ 的价值原为 $-2$，更新后为 $-2+0.5\times(0-(-2))=-1$。两个值回答不同的条件问题。若任务规定成功后继续，$d'$ 则不自动变为 $1$，应保留相应后续价值。

| 本次计算 | 原目标 4 | 重标记目标 2 |
| --- | --- | --- |
| reward | −1 | 0 |
| 目标终止 | 否 | 是 |
| target | −1.9 | 0 |
| 更新前 Q | −2 | −2 |
| 更新后 Q | −1.95 | −1 |

子任务例子中，$v(s)=2$、$v(s')=6$、$r=1$、$z=4$、$\beta=0.25$。停止分支贡献 $0.25\times4=1$，继续分支贡献 $0.9\times0.75\times6=4.05$，target 合计 $6.05$。取 $\alpha=0.1$、$\rho=1$，新价值为 $2.405$。这里的 $z$ 是子任务停止价值，区别于主任务中“换一个 option 继续”的高层价值。

<a id="lesson-code"></a>

## 6. 实现与实验：目标标签和终止条件

主线先运行 [learned_subgoal_cycle.py](../tutorials/learned_subgoal_cycle.py) 的三次/四次练习对照。比较 rounds 中的逐步 td、freeze 后的 skill、实际 model_execution 与最后 deployment。保持四次练习，再用 --c-probe-reward 0 改变实验环境的 C 探测净奖励：候选和到达策略仍相同，外部规划却回到 A。这能检查内部目标分数是否被错误当成主任务收益；奖励参数只由环境读取，规划器仍需真实探测。旧 [goal_discovery_walkthrough.py](../tutorials/goal_discovery_walkthrough.py) 留作经验模型求解对照，不能用它的路径枚举替代真实策略学习。

目标重标记、表格 goal-Q、future HER、子任务 TD 与有限目标课程。

```python
def relabel_goal(next_position: int, goal: int, physical_terminal: bool = False):
    """Our goal task ENDS on success; another task may not use this convention."""
    success = next_position == goal
    reward = 0.0 if success else -1.0
    terminal = physical_terminal or success
    return reward, terminal


def goal_q_update(q, state, action, next_state, goal, alpha=0.5, gamma=0.9,
                  physical_terminal=False):
    # A successful state is terminal in THIS task. Its zero continuation value
    # is not an action value to train using a transition out of that state.
    if state == goal:
        raise ValueError("A terminal goal state has no outgoing task transition")
    reward, terminal = relabel_goal(next_state, goal, physical_terminal)
    next_value = 0.0 if terminal else max(q.get((next_state, a, goal), 0.0)
                                          for a in (-1, 1))
    key = (state, action, goal)
    old = q.get(key, 0.0)
    target = reward + gamma * next_value
    q[key] = old + alpha * (target - old)
    return reward, terminal, target, q[key]


def future_her(trajectory, original_goal, rng):
    """trajectory: (s,a,s_next,physical_terminal) in one recorded episode.

    Outputs (s,a,s_next,goal,physical_terminal), preserving physical termination
    for both original and hindsight goals. Goal success must never erase it.
    Dynamics are deterministic and goal independent in this teaching example.
    These are independent one-step replay samples, not a relabeled rollout.
    Skip a hindsight goal equal to s: this task terminates as soon as it succeeds.
    A valid one-step sample may still exist after an earlier visit to its new goal;
    relabeling a whole multi-step rollout would instead truncate at first success.
    """
    replay = []
    for t, (s, a, sn, physical_terminal) in enumerate(trajectory):
        if s == original_goal:
            raise ValueError("Recorded episode acts from its terminal goal")
        if t + 1 < len(trajectory) and (physical_terminal or sn == original_goal):
            raise ValueError("Recorded episode continues after termination")
        replay.append((s, a, sn, original_goal, physical_terminal))
        future_index = rng.randrange(t, len(trajectory))
        hindsight_goal = trajectory[future_index][2]
        if s != hindsight_goal:
            replay.append((s, a, sn, hindsight_goal, physical_terminal))
    return replay


def subtask_td(value, state, next_state, reward, stopping_value, beta,
               alpha=0.1, gamma=0.9, rho=1.0):
    """STOMP Eq.(5) timing; beta is supplied, not optimized by this function."""
    old_here, old_next = value[state], value[next_state]
    target = reward + beta * stopping_value + gamma * (1.0 - beta) * old_next
    value[state] = old_here + alpha * rho * (target - old_here)
    return target


def stop_rules(stopping_value, continuation_value, gamma=0.9):
    paper_rule = stopping_value >= continuation_value  # STOMP Eq.(9)
    greedy_one_step = stopping_value >= gamma * continuation_value
    return paper_rule, greedy_one_step


def intermediate_goals(success_counts, attempt_counts, low=0.2, high=0.8):
    """Transparent finite curriculum; not the Goal GAN training algorithm."""
    return [g for g in success_counts if attempt_counts[g] > 0
            and low <= success_counts[g] / attempt_counts[g] <= high]
```

下载脚本后运行；仅需 Python 3.10+ 标准库

```sh
python3 examples/knowledge_algorithms_lab.py goals
python3 examples/knowledge_algorithms_lab.py test
```

运行结果中，原目标的新值为 $-1.95$，HER 新值为 $-1$，子任务 target 为 $6.05$；两种停止比较在 $z=9.5$ 时分别返回继续与停止。physical_terminal 随原样本和重标记样本一起传递。把物理终止后状态的 $Q$ 人为设成 $100$，未达成目标的转移 target 仍为 $-1$，而非错误 bootstrap 得到的 $89$。这个例子同时检查了“目标是否成功”和“物理过程是否结束”两个独立条件。

- 实验 A：把“成功即终止”改为“成功后仍继续”。同时修改 relabel_goal 和对应测试，观察若只改 reward、不改 bootstrap 会怎样。
- 实验 B：为相同 (s,a) 加随机分支，观察 future relabel 后的成功结果是否过度代表真实概率；分别在原目标评估与重放 loss 上作图。
- 实验 C：固定两条通向同一目标的路径，一条短但有负奖励。比较最短路径子任务与真实奖励加 stopping value 的排序，同时记录到达率和沿途代价。

OpenAI Baselines 的 her_sampler.py 展示了“选未来 achieved goal → 替换 desired goal → 调用 reward_fun”的数据流程；随后可读 ddpg.py 的 target 和 replay buffer。原工程采用自己的定长 episode 与成功处理约定，移植到成功即终止任务时，需要同步调整 bootstrap。Goal GAN 的 rllab-curriculum 则按目标采样、成功率标签、策略更新三个环节阅读。两套历史工程适合在独立依赖环境中运行。

<a id="lesson-branches"></a>

## 7. 如何组合目标学习的几个组成部分

目标相关方法可以组合，因为它们修改不同接口：UVFA 表示一族价值函数，HER 重用经验，课程分配交互，子任务定义评价，option 实现行为。一个系统可以同时包含它们，也可能只需要其中一个。组合并不自动有效：HER 的训练目标分布可能与课程提出的目标失配；新生成目标可能无法由现有表示判断成功；有用行为如果没有准确后果模型，仍不能用于可靠规划。

| 你观察到的失败 | 优先检查的量 | 可比较的方法线 |
| --- | --- | --- |
| 目标一换就从零学 | Q 是否以目标为条件，是否共享可迁移表示 | UVFA / goal-conditioned actor-critic |
| 几乎没有成功样本 | 奖励可否从记录数据重算，动力学是否目标无关 | HER / relabeling |
| 只会练容易任务 | 课程覆盖、成功率估计、探索保底 | 均匀采样 / 适中难度 / 学习进度 |
| 到达子目标却损害主任务 | 沿途真实 reward 是否被删掉 | shortest-path / reward-respecting |
| 技能很多而规划没收益 | 模型精度、技能适用性、维护预算 | STOMP 的行为—模型—规划联合评价 |

再撤掉给定地点特征，困难会发生在更早的位置。若状态学习把 A 与 C 压成同一个坐标，当前表示就无法判定两个到达目标；若同一特征后来改变含义，旧的成功标签、技能和终点模型也可能失配。可先固定目标判定的一份表示版本，同时学习新的候选表示；检验后再提交新版本，并重新核对它所条件化的策略和后果预测。这是一种明确的维护设计，不是说任何辅助预测一变准就自动获得了充分状态。

一条真实转移可以同时供主任务、到达 GVF 和模型预测使用，但每个问题的累积信号、继续条件和目标策略不同。在线资格迹负责让一个问题的后续误差影响此前预测；内部策略学习负责改变该问题怎样行动；探索选择负责决定下一条真实证据从哪里来；规划则使用已学后果重算决策。资源有限时，增加一个候选也增加练习、预测头、模型和维护成本，评价应回到匹配预算后的真实后续控制。

本例的目标是让上述依赖可见：基础状态、候选规则和练习调度仍固定，真正从经验变化的是候选集合、到达策略、外部后果与最终选择。[状态构造](state.md)另行研究历史中哪些差别值得保留，[探索](exploration.md)研究如何取得这些差别所需的证据，[完整架构](architectures.md)再区分已经实现的部件和仍待联合验证的更新。两项分别可运行的学习例，不等于已经合成一个自主构造全部部件的系统。

<a id="lesson-offline-goals"></a>

## 8. 目标很远时：从平坦价值函数到 HIQL

深入阅读：在经验集合固定且目标很远时，比较 HIQL 的分层价值与策略。

前面的 HER 假设能继续交互，错误的价值还有机会被新经验纠正。离线目标条件 RL 只给定一个已有轨迹集，学习者不能随时尝试一个看起来很好的动作。如果目标很远，好坏动作的价值差很小，估值噪声就可能盖过真正的控制信号。分层的一个作用是把“一步选哪个动作”改成“先到哪个较近的中间状态”。

HIQL（NeurIPS 2023）使用一个目标条件状态价值 $V(s,g)$，然后提取两个策略：高层 $\pi_h(z|s,g)$ 预测中间状态的潜在表示，低层 $\pi_\ell(a|s,z)$ 产生到达这个中间目标的动作。这里的高层动作不是环境按钮，而是数据中未来状态的表示；低层才承担可执行控制。

先说明价值如何学习。离线数据没有覆盖的动作，不能靠实际尝试来检验；直接在所有动作上取最大估值，可能选中估计错误的动作。标准 IQL 对数据中的动作价值 $Q(s,a)$ 做不对称平方回归，偏重估值较高的动作；$Q$ 的学习先对动作后果的随机性取期望。HIQL 在此采用不需要动作标签的版本，直接从 $(s,s')$ 和抽样目标 $g$ 构造一步价值目标。这个改变省去了动作输入，也引入了后面要讨论的随机性歧义。下面先用单个价值头说明原理。

$$
\begin{aligned}d_g(s)&=\mathbf1[s=g],\qquad r_g(s)=d_g(s)-1,\\y&=r_g(s)+\gamma(1-d_g(s))\bar V(s',g),\\u&=y-V_\theta(s,g),\\L_V(\theta)&=\mathbb E_{\mathcal D,g}\!\left[\ell_\xi(u)\right],\qquad \ell_\xi(u)=|\xi-\mathbf1[u<0]|u^2,\quad \tfrac12\leq\xi<1.\end{aligned}
$$

$\bar V$ 是求本次梯度时固定的目标网络；$\xi$ 是 expectile 参数，不是折扣。本节按原方法以当前状态是否为目标定义奖励：目标处为 0 且不自举，其他状态为 −1。这与前文按到达状态给奖励的例子有一步计时差别，不能直接混用数值。实际数据采样器用当前样本索引与目标索引是否相同生成成功标记；轨迹结束标记另用于限制子目标采样。

当目标高于当前预测时，平方误差权重为 $\xi$；目标低于预测时，权重为 $1-\xi$。固定一批目标 $y$ 后，对标量预测 $v$ 求导，极小点满足加权残差均值为零。$\xi=1/2$ 恢复普通均方拟合；增大 $\xi$ 则更重视低估的后果。这里选择的是数据支持内的较好后果，并没有评价所有可能动作。

$$
0=\mathbb E\!\left[\bigl(\xi\mathbf1[y\geq v]+(1-\xi)\mathbf1[y<v]\bigr)(y-v)\right].
$$

例如两个等频目标为 −3 和 −1，取 $\xi=0.7$。在二者之间解 $0.3(-3-v)+0.7(-1-v)=0$，得 $v=-1.6$，高于均值 −2，但仍小于最大值 −1。这个手算只解释一次固定目标回归；带自举的整体学习还改变下一轮目标。

这也暴露了动作缺失的局限。如果两种后果来自两个可选的确定性动作，偏重好后果有控制意义；如果它们来自同一个动作的随机结果，就不能通过选动作让好运更常发生。相同的状态对数据，可能对应这两种不同控制问题。因此无动作版本在随机动力学下可能把噪声当成可控性；增加网络容量并不能消除这种信息歧义。

作者实现进一步用两个价值头。它以目标头的较小下一价值计算公共残差 $a^-$，据其符号确定权重；每个在线头则拟合各自目标头产生的 $y_i$。这不是把单头公式简单乘二，复现时应保留以下分工：

$$
\begin{aligned}y_i&=r_g+\gamma(1-d_g)\bar V_i(s',g),\qquad i=1,2,\\a^-&=r_g+\gamma(1-d_g)\min_i\bar V_i(s',g)-\tfrac12\sum_i\bar V_i(s,g),\\\kappa&=\operatorname{sg}\!\left[\xi\mathbf1[a^-\geq0]+(1-\xi)\mathbf1[a^-<0]\right],\\L_V^{\rm impl}&=\mathbb E\!\left[\sum_{i=1}^2\kappa\bigl(\operatorname{sg}[y_i]-V_i(s,g)\bigr)^2\right].\end{aligned}
$$

对应作者代码 compute_value_loss。接下来的 actor 权重使用两个在线价值头的平均，而不是这里用于确定权重的最小目标值。这样，价值如何训练、策略如何利用它，就可以分别检查。

$$
\begin{aligned}\omega_j&=\operatorname{sg}\!\left[\min\{\exp(\eta_j A_j),100\}\right],\qquad j\in\{\ell,h\},\\ z_{\rm target}&=\operatorname{sg}\!\left[\phi(s_{t+k})\right],\\ J_\ell&=\mathbb E_{\mathcal D}\left[\omega_\ell\log\pi_\ell(a_t\mid s_t,\phi(g_\ell))\right],\\ J_h&=\mathbb E_{\mathcal D}\left[\omega_h\log\pi_h(z_{\rm target}\mid s_t,g)\right].\end{aligned}
$$

两层 actor 做优势加权回归；$\eta_\ell$ 与 $\eta_h$ 分别控制权重尖锐程度。这里写出作者实现的上限 100。$\operatorname{sg}$ 保持数值、停止该通路的梯度：actor 拟合固定权重和固定高层标签，不通过改变它们来降低回归损失。

价值为两层策略提供评价，作者实现可以在同一批更新中联合计算价值、低层 actor 与高层 actor 的损失。回归权重使用价值差：低层 $A_\ell=V(s_{t+1},g_\ell)-V(s_t,g_\ell)$，高层 $A_h=V(s_{t+k},g)-V(s_t,g)$。计算这些权重及高层标签时使用更新前参数，并相对于本次 actor 求导保持固定；价值仍由自己的损失学习。目标表示还可依赖当前位置，前式用 $\phi(g)$ 简写该接口。低层策略输入中的目标编码是否接受 actor 梯度由作者代码的 policy_train_rep 配置决定，不能与停止高层回归标签梯度混为一谈。

状态序列可训练价值和高层，但低层动作策略仍需要带动作的数据。这里利用无动作数据扩大的是对“哪些状态连接到哪里”的知识；执行动作仍须通过动作数据学习。把数据来源分开，是评价这类方法与纯视频学习、完全离线控制之间关系的必要条件。

HIQL 与 HER 的关系是互补的：HER 处理目标标签和经验使用，HIQL 处理长期控制中的策略结构与离线提取。它与可变时长 Option-Critic 也不同：中间目标和训练间隔给出了分层结构，并不自动学习同样的终止梯度。用于持续 RL 时，还需研究新经验怎样更新旧价值和目标表示，而不是仅换一批测试目标。

OGBench（ICLR 2025）适合检验这类区别。它把长时域到达、轨迹拼接、像素输入和随机性等挑战放进统一的离线目标学习基准，并提供多种基线实现。起步时选一个状态输入任务，固定数据与目标抽样，比较平坦目标策略和 HIQL；再依次改变数据覆盖、子目标间隔和观测形式。先解释哪一种困难导致差距，再扩大任务规模。轨迹边界标记与 bootstrap mask 在数据接口中具有不同含义，时间截断不应自动当作真实任务终止。

<a id="lesson-language-skills"></a>

## 9. 人类如何指定技能：MaestroMotif 的语言到控制过程

深入阅读：语言提供子任务描述时，追踪描述如何成为可执行、可评价的行为规格。

有些子任务来自图结构，有些来自人的语义要求。例如“寻找食物”和“回避危险”比一个坐标更适合描述 NetHack 中的长期行为，却难以直接手写一个可靠的数值奖励。MaestroMotif（ICLR 2025）将语言模型放在技能设计环节：把语言要求转换成可学习奖励、启动与终止规则、以及训练时的技能调度，然后仍由 RL 学习执行技能的低层策略。

第一步，给定技能描述 $x_i$，让语言模型比较两段候选观察中哪个更符合要求，得到偏好标签 $y$；随后训练奖励模型 $r_i$。一个基本的成对偏好模型是令奖励差决定偏好概率。

$$
\begin{aligned}p_i(s_1\succ s_2)&=\sigma(r_i(s_1)-r_i(s_2))\\ \mathcal L_i&=-\mathbb E\left[y\log p_i+(1-y)\log(1-p_i)\right]\end{aligned}
$$

这将语言反馈蒸馏成低层训练时可反复查询的标量信号。损失只约束相对偏好，奖励平移本身不可辨识；其尺度、偏好一致性及分布外可靠性仍影响控制。

第二步，依据技能规格生成 $I_i$、$\beta_i$ 的程序和训练时的高层调度程序。第三步，在调度所产生的状态分布中，以学得的 $r_i$ 训练技能条件低层策略。第四步，面对下游任务描述，生成组合这些既有技能的程序。学习奖励、写执行规则、学原始动作和组合技能是四种不同操作。

**算法：MaestroMotif 的流程；语言反馈规定行为语义，强化学习承担行为执行**

1. 人类给出技能描述与技能组合要求
1. 语言模型比较观察对 → 训练每个技能的奖励模型
1. 语言模型生成启动、终止与训练调度程序
1. 在调度产生的交互中，用 RL 训练技能条件低层策略
1. 下游任务描述 → 组合既有技能的程序 → 执行并评估外部任务

这种方法与 STOMP 的差别在奖励来源：STOMP 保留环境沿途奖励并改变停止价值，MaestroMotif 则依据语言偏好学习技能奖励，还借助人类规格和预训练语言知识。其下游组合无需重新训练某个任务的低层策略，不意味着技能本身没有经过 RL 训练，也不意味着技能规格已经由智能体自主发现。

作者仓库可按三个目录阅读：preference 对应偏好与奖励学习，code_generation 对应规则及组合程序，rl_baseline 与 sample_factory 对应技能执行训练。辨析失败原因时，先在匹配交互与调参预算下分别改变奖励模型、训练调度，并重新训练低层技能；这检验语义评价和训练访问分布的作用。随后固定同一组技能、启动与终止规则，只改变下游组合程序，检验高层控制。若低层已冻结，且部署过程不读取训练奖励，单独替换奖励模型不会改变执行行为，不能据此检验它对技能形成的贡献。奖励模型本身还可在固定观察对上独立检验语义判断。

<a id="lesson-check"></a>

## 10. 失败诊断与自测答案

已见图上有路，冻结技能却走错：先查内部价值是否已从成功位置传到起点。目标 C 的三次练习对照正是这种情况；不能直接用图中的正确路径覆盖尚未学会的策略。技能没到目标时，模型要记录实际失败终点；只有未停止的采样前缀才不构成完整后果。

- Q 总是接近零而真实成功率不升：检查重标记比例过高、成功标签泄漏，以及评估是否错误采用事后目标。
- HER 后回报上升但原目标失败：检查训练目标覆盖；重标记只证明某些结果可学习，不保证原目标可达。
- 子任务全部立刻停止：检查 stopping value 尺度、初始化、是否错误地把 β 当作环境 done，以及是否允许零步启动就停止。
- CRL 性能忽高忽低：分别记录目标分布、环境变化和模型版本，并按这些条件分析成功率。

自测 1：为何 HER 不能随意改 s′？答：状态转移是事实证据；改变目标是在同一事实下提出新问题，改变结果则是在伪造没有实际观察过的转移。自测 2：把 stopping bonus 写进主任务模型会怎样？答：规划会把学习辅助奖励当真实收益，重复调用技能可能获得虚假的“奖励”。自测 3：目标编码共享的失败就是灾难性遗忘吗？答：不一定，也可能是目标本来不可区分、训练分布变了或价值估计尚未收敛，应通过固定目标保留集和表示可辨识性实验区分。

## 本章的实验设计

子任务成功率与主任务收益需要分别测量。设计固定目标、随机目标和学得目标三个对照。

设定：在小型地图固定低层技能和高层控制器，比较固定目标、随机目标与学习目标。研究者图结构只用于可达性诊断。

- 目标成功判定只用当时可得信号。
- 后见目标标签用于训练，不作为过去行动时已经知道的输入。
- 不可达目标有时限、失败与退出语义。

对照：数量和尺度匹配的随机目标；固定手工目标的特权诊断；固定技能只改目标；再固定目标只改技能

记录：目标到达率、到达时间与失败率；新任务收益和目标使用频率；全部目标生成和训练成本

[具体测试规程](../docs/experiment-handbook.md#handbook-modules)

## 学习与研究衔接

给定一个子目标，可以研究怎样更快达到；决定练习哪个子目标，还必须估计练习对以后主任务的作用。容易达到的目标可能从不被调用，困难目标也可能带来罕见但有用的恢复行为。研究应先固定候选与低层学习器，比较目标选择的净收益，再加入候选生成，辨认新知识究竟来自哪一步。

[分册导读](../docs/learning-route-continual-rl.md) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-goals) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=goals) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=goals)

## 从本章进入实践

[技能与规划](https://yingwen.io/zh/continual-rl/code/#practice-skills)：一个多步行为怎样成为可学习、可预测、可规划的动作？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

给定状态后可以估计价值；观测不足时，还要学习保留哪些历史。GVF 规定预测什么，RTRL 计算递归敏感度，资格迹组织时间信用。应分别检验信息是否进入状态、反馈能否教会这种保留，以及有限预测预算怎样分配，而不是把三者当作替代算法。

- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

教材可以先给定目标和技能集合；持续构造还要决定哪些行为值得练习、维护或放弃。谱结构、路径奖励、时间距离和语言先验提供不同候选偏置。先固定候选比较选择与组合，再改变生成器，才能辨认下游收益究竟来自哪一步。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [HIQL: Offline Goal-Conditioned RL with Latent States as Actions](https://yingwen.io/zh/continual-rl/research/#recent-hiql-hierarchical-goals)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)
- [Foundation Policies with Hilbert Representations](https://yingwen.io/zh/continual-rl/research/#recent-hilbert-foundation-policies)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [OGBench: Benchmarking Offline Goal-Conditioned RL](https://yingwen.io/zh/continual-rl/research/#recent-ogbench-goal-evaluation)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文规定对象与条件，架构路线提出组织方式，算法实验检验局部机制。撤掉阶段间冻结后，一个模块会改变另一个模块的学习问题；有限预算应优先维护哪条知识，成为新的决策。先检验两模块反馈和资源分配，再扩大整机，而不是由组件分别有效推断长期组合收益。

- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

### Reward-Respecting Subtasks for Model-Based Reinforcement Learning

Richard S. Sutton, Marlos C. Machado, G. Zacharias Holland, David Szepesvari, Finbarr Timbers, Brian Tanner, Adam White

Artificial Intelligence · 2023 · 支持方法与理论

#### 研究问题

学到一个能到达子目标的技能之后，为什么它仍可能不适合主任务规划？

#### 关键机制

STOMP 把子任务、option、模型和规划连起来。子任务保留原任务的路径奖励，并用带有特征偏好的终止价值表达目标；学习得到策略和终止规则后，再预测该行为的累计奖励与折扣终点。这样，技能不会因为只追求到达子目标而忽略途中代价。

#### 证据

论文用小问题展示奖励感知子任务怎样产生可用于规划的行为与后果模型。实验将各阶段依次进行，从而能够分清子任务设计、option 学习、模型学习和规划各自的作用。

#### 条件与限制

这些实验没有同时运行并更新全部阶段。特征选择、子任务淘汰和规划计算分配仍需算法；终止收益属于子任务规格，不能随意换成固定终点奖励，也不能混入真实奖励模型。

#### 阅读与实验

先在同一绕路环境比较两种子任务，并计算奖励模型、折扣终点模型和一次备份。再固定候选与容量，检验下游规划用途能否指导技能保留和模型重学；这第二步是拟议研究，不是原论文已证实的闭环。

#### 原文与相关入口

- [期刊论文](https://doi.org/10.1016/j.artint.2023.104001)：STOMP 与奖励感知子任务的正式论文。
- [作者预印本](https://arxiv.org/abs/2202.03466)：最初预印本早于期刊年份；阅读停止收益的精确定义。

### Laplacian Keyboard: Beyond the Linear Span

Siddarth Chandrasekar, Marlos C. Machado

arXiv 预印本 · 2026 · 支持方法与理论

#### 研究问题

从一组谱技能出发，能否解决超出原特征线性奖励空间的新任务？

#### 关键机制

Laplacian 特征先定义行为基，并借助后继特征预测各行为的后果。固定任务权重的价值组合受特征张成空间限制；论文进一步使用随状态变化的元策略，在不同位置组合已有行为。关键变化是组合规则从一组全局固定权重变为状态相关的行为选择。

#### 证据

论文对行为基与任务组合给出理论分析，并报告有限环境中的组合实验。它延续 eigenoptions 与 successor features 的路线，同时解释了为什么单纯线性读出会遇到表达边界。

#### 条件与限制

理论结论依赖具体的行为基、近似误差和任务条件。技能集合的长期生成、淘汰与非平稳模型维护仍是另外的问题；此处按预印本收录，不指定未经确认的会议。

#### 阅读与实验

构造一个必须在中途切换方向的奖励任务。分别比较固定权重的技能选择与状态相关切换，并解释性能差异来自哪里。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2602.07730)：阅读线性张成空间的限制及状态相关组合机制。

### HIQL: Offline Goal-Conditioned RL with Latent States as Actions

Seohong Park, Dibya Ghosh, Benjamin Eysenbach, Sergey Levine

NeurIPS 2023 · 2023 · 支持方法与理论

#### 研究问题

只拿到已有轨迹时，长距离目标为什么适合拆成高层子目标和低层动作？

#### 关键机制

HIQL 学习目标条件价值，并以潜在状态作为高层动作。高层提出中间目标，低层输出环境动作；两层利用优势加权回归学习。时间分解让低层面对较短的控制距离，而不是要求一个策略直接消化所有远距离价值误差。

#### 证据

论文在离线长时域目标任务中检验层次结构，并提供原始实现。作者后来在 OGBench 中提供更统一的实现，二者适合不同用途：原实验复现和统一基线比较。

#### 条件与限制

数据覆盖和行为分布约束仍然存在。目标采样、层级时间间隔与离线轨迹由外部流程提供，不能把效果解释为在线自主目标生成已经解决。

#### 阅读与实验

对一段轨迹明确标记最终目标、中间目标和当前动作。逐一检查价值目标、优势权重和高层标签的停止梯度边界。

#### 原文与相关入口

- [NeurIPS 2023 原文](https://papers.nips.cc/paper_files/paper/2023/file/6d7c4a0727e089ed6cdd3151cbe8d8ba-Paper-Conference.pdf)：离线目标学习和两层回归目标。
- [HIQL 原始实现](https://github.com/seohongpark/HIQL)：README 区分原始实验与 OGBench 中的新实现。

#### 作者代码

[作者仓库；更新的统一基线另见 OGBench。](https://github.com/seohongpark/HIQL)

HIQL 原论文的离线训练与评价。

### OGBench: Benchmarking Offline Goal-Conditioned RL

Seohong Park, Kevin Frans, Benjamin Eysenbach, Sergey Levine

ICLR 2025 · 2025 · 评价与实验协议

#### 研究问题

一个目标条件算法表现不好，是长时域、轨迹拼接、视觉表示还是随机性造成的？

#### 关键机制

OGBench 用不同环境类型与数据集分别施加这些困难，并提供统一的目标条件基线。固定离线数据让算法面对相同经验，从而将学习机制的差异与在线探索能力的差异暂时分离。

#### 证据

论文提供八类环境、八十五个数据集和六类算法实现。价值在于可复用的实验接口与困难分解，而不只是汇总一个排行榜。

#### 条件与限制

固定数据不检验智能体如何主动获得未来经验，也不直接检验单次生命的灾难性变化、恢复或长期资源管理。它适合 CRL 子问题实验，不是完整 CRL 的替代品。

#### 阅读与实验

先选择只改变一种困难的两个数据集，再比较 HIQL 与平坦目标策略。把观察到的差异写成可检验机制假设，而不是直接归因于“层次更好”。

#### 原文与相关入口

- [论文](https://arxiv.org/abs/2410.20092)：ICLR 2025；环境、数据与基线定义。
- [作者基准库](https://github.com/seohongpark/ogbench)：数据获取、环境与统一算法实现。

#### 作者代码

[基准作者维护的官方实现。](https://github.com/seohongpark/ogbench)

离线目标环境、数据集与标准化基线。

### MaestroMotif: Skill Design from Artificial Intelligence Feedback

Martin Klissarov, Mikael Henaff, Roberta Raileanu, Shagun Sodhani, Pascal Vincent, Amy Zhang, Pierre-Luc Bacon, Doina Precup, Marlos C. Machado, Pierluca D’Oro

ICLR 2025 · 2025 · 支持方法与理论

#### 研究问题

语言描述如何变成可训练的技能奖励，并进一步组织成一个层次策略？

#### 关键机制

设计者先给出技能描述。语言模型的偏好反馈被用于训练奖励模型，再用生成的代码规定技能启动、终止和组合方式；强化学习负责学习实际执行行为。这把语义先验、奖励学习和时间抽象串成了具体训练流程。

#### 证据

论文在 NetHack 学习环境中检验复杂技能与任务组合。作者仓库同时包含偏好、代码生成和 RL 训练模块，可以追踪自然语言到环境动作的完整依赖。

#### 条件与限制

语义知识、技能描述和语言模型来自外部设计过程。该证据并不说明智能体仅凭自身交互就能产生同样的技能体系；偏好模型也可能与真实目标不一致。

#### 阅读与实验

选择一项技能，分别列出描述、偏好标签、训练奖励、终止条件和下游用途。移除语义描述或改变奖励模型时，要单独计量额外查询与人工成本。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2dc5a0faac8102fd47363795f71126ee-Abstract-Conference.html)：技能设计、奖励学习与组合实验。
- [作者实现](https://github.com/mklissa/maestromotif)：偏好学习、代码生成和执行策略的不同模块。

#### 作者代码

[原论文作者仓库。](https://github.com/mklissa/maestromotif)

MaestroMotif 的偏好处理、技能组织与 RL 实验。

### General Agents Contain World Models

Jonathan Richens, David Abel, Alexis Bellot, Tom Everitt

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

能完成足够丰富的目标集合，是否意味着智能体内部已经包含可提取的环境预测知识？

#### 关键机制

论文在形式化条件下，将广泛多步目标上的行为能力与环境模型的可提取性联系起来。通过查询智能体对不同目标的行为，可以恢复关于环境后果的信息；目标集合和性能要求越强，所要求的预测知识也越强。

#### 证据

主要证据是给定假设下的理论结果，而不是某个世界模型架构在所有任务上击败无模型算法的实验。

#### 条件与限制

可提取模型不等于智能体显式保存一个 RSSM，也不意味着所有实用任务都需要重建全部环境。必要知识的结论不能代替如何高效学到它的算法。

#### 阅读与实验

列出定理要求的目标丰富性和查询能力，再尝试构造一个只会单一任务的反例。由此区分任务专门知识与支持广泛目标的预测模型。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2506.01622)：形式化设定、模型可提取性与证明。
- [David Abel 论文目录](https://david-abel.github.io/papers.html)：作者提供的 ICML 2025 发表信息及相关研究。

### The OaK Architecture: A Vision of SuperIntelligence from Experience

Richard S. Sutton

RLC 2025 讲座 / Oak Lab · 2025 · 定义与架构观点

#### 研究问题

持续学习是否只是在一个现成 actor–critic 上加入抗遗忘机制，还是需要重新安排知识构造与使用？

#### 关键机制

OaK 提出从经验持续形成状态、预测知识、子任务、时间抽象与模型，并让这些知识服务规划的架构方向。这里的重点是模块之间怎样产生可复用知识，而不只是保留某个固定策略网络的参数。

#### 证据

官方页面提供 Richard Sutton 的架构讲座与相关研究入口。STOMP、预测学习和在线特征学习等论文可以检验其中具体组件，但不能自动验证整体架构。

#### 条件与限制

这是研究愿景与架构讲解，不是一套已公布完整训练配方、统一基准结果和可复现端到端代码的系统。资源分配、问题生成、知识替换与模块相互干扰仍需明确算法。

#### 阅读与实验

为每个模块写出输入、输出、更新频率和资源上限。再选择一个双模块接口做可证伪实验，例如技能模型改善是否真的减少规划误差。

#### 原文与相关入口

- [Oak Lab 官方讲座页面](https://oaklab.ai/posts/the-oak-architecture)：讲座入口与架构研究方向。
- [Oak Lab 研究主页](https://oaklab.ai/)：区分已发表研究、技术文章和仍在预告中的项目。

### Reset-free Reinforcement Learning with World Models

Zhao Yang, Thomas M. Moerland, Mike Preuss, Aske Plaat, Edward S. Hu

TMLR 2025 · 2025 · 支持方法与理论

#### 研究问题

不能靠外部重置回到起点时，怎样兼顾探索新状态与持续获得对任务有用的经验？

#### 关键机制

MoReFree 在 goal-conditioned world-model 系统中交替练习评测目标、返回初始分布与探索目标；模型内的策略训练也偏向任务相关目标。返回行为通过真实动作实现，调度块结束不会将物理世界 reset。

#### 证据

作者在八个 reset-free 任务中与模型自由及模型式基线比较；公开环境、探索调度和 imagination training 实现。

#### 条件与限制

训练无 reset，但主要评价仍使用可重置的 episodic 测试。已给定初始与目标状态分布、世界模型和 replay 都是资源；这不是任意非平稳 CRL 或真实安全的完整保证。

#### 阅读与实验

把返回成本计入总步数，分别消融数据获取目标与模型内训练目标；检查外部 reward-free 是否仍依赖设计者提供目标示例。

#### 原文与相关入口

- [作者论文 v3](https://arxiv.org/html/2408.09807v3)：训练与评价协议、back-and-forth exploration 与目标分布。
- [TMLR 作者项目页](https://yangzhao-666.github.io/morefree/)：正式发表状态与作者代码链接。

#### 作者代码

[TMLR 作者项目页明确链接的官方实现。](https://github.com/yangzhao-666/MoReFree)

resetfree/env.py、goal_picker_wrapper.py、Dreamer/PEG 与目标条件实验。

### Does Zero-Shot Reinforcement Learning Exist?

Ahmed Touati, Jérémy Rapin, Yann Ollivier

ICLR 2023 · 2023 · 支持方法与理论

#### 研究问题

没有事先指定奖励时，怎样学一套预测表示，日后接收新奖励就能选行为？

#### 关键机制

Forward–Backward 表示联合学习行为条件的未来占用与奖励读出，而非先固定任意编码器再学习 successor features。新奖励被映射到任务向量，策略根据这个向量直接行动；该论文系统比较 FB 与多种 SF 基础特征。

#### 证据

原文在固定离线 replay buffers 上比较零样本任务迁移，借此把表示学习与探索数据的质量分开。不同特征与数据覆盖产生显著差异，不能仅靠“所有奖励”的理论目标预测实际效果。

#### 条件与限制

假定共享动力学与可用经验覆盖。无下游梯度更新不等于无预训练成本；有限秩、近似训练和奖励估计都有误差。新动力学、历史混叠和严格一次使用经验均须另测。

#### 阅读与实验

同一 buffer 对比随机特征、谱特征与联合 FB，再独立换 buffer。奖励读出误差、占用误差与新任务回报分别报告，避免把数据覆盖优势记成表示优势。

#### 原文与相关入口

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。
- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

#### 作者代码

[论文作者团队仓库；归档工程，依赖和旧环境需单独核验。](https://github.com/facebookresearch/controllable_agent)

FB 与 SF 的训练、固定数据实验及奖励查询示例。

### Foundation Policies with Hilbert Representations

Seohong Park, Tobias Kreiman, Sergey Levine

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

如何从无任务标签的离线轨迹形成既能按方向调用、又能用于目标任务的策略接口？

#### 关键机制

HILP 先学习近似保存时间距离的 Hilbert 表示，再以潜在位移与方向的内积训练方向条件策略。新任务通过奖励回归、目标方向或分层调用选择策略条件，结构表示也支持测试时规划。

#### 证据

ICML 原文与作者项目包含零样本 RL、离线目标条件 RL 及规划实验；官方仓库将 zero-shot 与 goal-conditioned 两套实现分开。

#### 条件与限制

精确时间距离不总能无损嵌入有限维对称欧氏距离，尤其有向不可逆行为；理论充分条件与近似神经实验需区分。方向条件策略没有自动获得任意停止条件或完整技能后果模型。

#### 阅读与实验

固定离线数据分别测距离误差、方向执行误差、奖励可表达误差与高层收益。让同一视觉观测对应不同历史，检查仅观测编码是否足够，之后再讨论 CRL 状态维护。

#### 原文与相关入口

- [ICML 2024 原文](https://proceedings.mlr.press/v235/park24g.html)：Hilbert 距离、策略提示和定理前提；不是 ICLR 论文。
- [作者项目与公式](https://seohong.me/projects/hilp/)：时间距离与方向奖励接口。
- [官方实现](https://github.com/seohongpark/HILP)：hilp_zsrl 与 hilp_gcrl 对应不同实验。

#### 作者代码

[作者项目直接链接并标为 official implementation。](https://github.com/seohongpark/HILP)

离线预训练、零样本奖励适配及目标条件实验。

### Constructing an Optimal Behavior Basis for the Option Keyboard

Lucas N. Alegre, Ana L. C. Bazzan, André Barreto, Bruno C. da Silva

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

状态相关的技能组合足够强时，是否仍须为每个新奖励保存一条完整策略？

#### 关键机制

OKB 联合扩充基础策略与 Option Keyboard 的元策略。线性支持方法选择需要补齐的任务权重，先训练现有基础上的组合，再检查无法表达的动作并新增基础，移除冗余项。优化的是可组合的行为基，而非仅增加技能数量。

#### 证据

正式原文分析基础数量与 convex coverage set 的关系，并在多任务领域检验规模与表现。附录提供元策略、新基础训练和角点枚举等实现细节；论文声明实验代码在 Supplemental Material。

#### 条件与限制

保证假定 NewPolicy 返回最优策略，且 TrainOK 能达到可表达的最优组合。近似 critic、有限训练、变化动力学不直接继承保证；非线性任务结论仅覆盖最优行为可由相应子策略组合的类别。

#### 阅读与实验

分别比较“增加基础”“只训练组合”和“去除冗余”。保留一组未用于基构造的奖励权重，记录基础规模、元策略成本、SF 误差与迁移回报。持续淘汰仍需未来任务效用检验。

#### 原文与相关入口

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。
- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。


<a id="chapter-code"></a>

## 下载与运行

目标重标记、子任务 TD 与有限目标课程的表格实验；深度方法另附原论文和作者工程。

[下载 knowledge_algorithms_lab.py](../examples/knowledge_algorithms_lab.py)

```sh
python3 examples/knowledge_algorithms_lab.py goals
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto — Reinforcement Learning: An Introduction · §17.1–17.3](http://incompleteideas.net/book/the-book-2nd.html)：辅助预测及其使用、option 模型与规划、从历史构造状态的关系。原书以 continuation 表达停止，本页以 β 表达停止概率。

- [Machado, Bellemare & Bowling — A Laplacian Framework for Option Discovery · ICML 2017](https://proceedings.mlr.press/v70/machado17a.html)：§3 定义 eigenpurpose、内部控制问题与 terminate；方向生成与学习技能分别阅读。

- [Machado — Representation-driven Option Discovery · AAAI 2025 讲座概要](https://ojs.aaai.org/index.php/AAAI/article/download/35100/37255)：经验、表示、子任务、option、再采样的循环；一页讲座概要不替代具体算法论文。

- [Eigenoptions 作者代码 — mcmachado/options](https://github.com/mcmachado/options)：main.py::discoverOptions 与 Learning.py。该表格入口读取完整图及模型；不是本节 frontier 基线的实现或在线样本效率证据。

- [SUN: Reaching for Novelty in Reinforcement Learning · 2026 预印本](https://arxiv.org/html/2609.08642v1)：首次到达价值、乘性目标选择及候选覆盖条件。

- [Viteri et al. — When Do Intrinsic Rewards Lead to Exploration? · 2026 预印本](https://arxiv.org/html/2610.02159v1)：以反事实历史信息检验内在目标；正文使用定义与闹钟例。

- [Schaul et al. — Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)：目标作为价值函数输入，状态与目标通过共享表示实现泛化；比较原文表示分解与直接拼接输入的实现。

- [Andrychowicz et al. — Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)：原文。对照 future 策略、off-policy 学习器与稀疏奖励实验；重标记不等于改变真实成功率。

- [OpenAI Baselines — her_sampler.py](https://github.com/openai/baselines/blob/master/baselines/her/her_sampler.py)：作者团队原工程入口。逐项追未来 achieved goal、目标替换和 reward_fun；依赖与 done 约定需同时检查。

- [Florensa et al. — Automatic Goal Generation for RL Agents](https://proceedings.mlr.press/v80/florensa18a.html)：原文与项目入口。Goal GAN 学的是适中难度目标分布；它不是 HER，也不是低层控制器。

- [Goal generation — 作者 rllab-curriculum](https://github.com/florensacc/rllab-curriculum)：课程层与策略层的实现：目标生成、成功率标签、策略训练各自改变什么。旧依赖适合放在独立环境。

- [Sutton et al. — Reward-Respecting Subtasks](https://arxiv.org/html/2202.03466v3)：§2–5 与 §8：return、内部策略和停止、外部后果模型及规划；顺序学习的实验证据与同时更新、反馈选择特征的未实现扩展分开阅读。

- [Park et al. — HIQL · NeurIPS 2023](https://arxiv.org/abs/2307.11949)：从同一个目标价值提取高层子目标与低层动作，分析长期价值噪声对平坦和分层策略的影响。

- [HIQL 作者实现](https://github.com/seohongpark/HIQL)：比较目标采样、价值学习、低层优势和高层跨步优势；way_steps 对应中间目标间隔。

- [HIQL 目标采样 — src/gc_dataset.py](https://github.com/seohongpark/HIQL/blob/master/src/gc_dataset.py)：GCSDataset.sample 的成功标签来自当前样本索引与目标索引相等；终止索引限制轨迹内子目标抽样。价值更新再把成功标签转成奖励和 bootstrap mask。

- [HIQL 核心文件 — src/agents/hiql.py](https://github.com/seohongpark/HIQL/blob/master/src/agents/hiql.py)：compute_actor_loss 与 compute_high_actor_loss 分别构造一步和跨步价值差；compute_value_loss 做 expectile 更新；pretrain_update 组合三项损失。

- [OGBench · ICLR 2025](https://arxiv.org/abs/2410.20092)：离线目标条件 RL 的任务和数据设计；把长时域、拼接、观测和随机性等困难分开。

- [OGBench 作者基准与基线](https://github.com/seohongpark/ogbench)：环境、数据接口与算法实现；尤其注意 masks 与 terminals 分别表示什么。

- [Klissarov et al. — MaestroMotif · ICLR 2025](https://arxiv.org/abs/2412.08542)：技能描述、偏好奖励、程序化规则及低层 RL 的完整分工；实验重点是 NetHack 中技能的训练与组合。

- [MaestroMotif 作者实现](https://github.com/mklissa/maestromotif)：preference、code_generation、rl_baseline 和 sample_factory 对应奖励、组合规则与执行策略学习。
