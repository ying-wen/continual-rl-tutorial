# 学习规则的适应：在线元梯度与跨任务元学习

一次更新减小了当前误差，但它是否让下一次学习更容易？元学习把这个问题变成可计算的评价：先按某条规则学习，再用后续表现改进这条规则。

## 本章内容

- 明确内层状态、元参数、外层目标与数据权限，区分在线元梯度、跨任务元学习和预设尺度控制。
- 从多步参数更新推导精确敏感度；说明 IDBD/TIDBD、截断元梯度和一阶 MAML 丢掉了哪些项。
- 完整理解 MAML、RL²、PEARL、meta-gradient RL 与 learned update rules 的训练/测试循环及 reset 边界。
- 用可运行的有限差分、手算与机制反例验证代码，并设计适用于持续交互而非仅任务重置的评测。

## 阅读这条论证

普通学习改变预测或策略，元学习改变产生这些更新的规则。先明确由哪一次更新后的表现评价规则，再沿连续更新应用链式法则。IDBD、TIDBD 和 Metatrace 逐步加入自举与时间信用；学习目标和更新规则的发现则扩大可学习的对象。

[区分内层与外层目标](meta.md#lesson-setting) → [传播学习过程的敏感度](meta.md#lesson-derive) → [确定评价时刻](meta.md#lesson-evaluation-timing) → [学习逐特征步长](meta.md#lesson-idbd) → [处理自举目标](meta.md#lesson-tidbd) → [处理时间信用](meta.md#lesson-metatrace) → [学习目标的参数](meta.md#lesson-meta-rl) → [发现更新规则](meta.md#lesson-learned-rules) → [评价对整个智能体的影响](meta.md#lesson-agent-coupling) → [对应实现与近似](meta.md#lesson-code)

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [时间信用分配：从资格迹到深度梯度学习](credit.md)：区分对预测参数的信用与对更新规则的敏感度。
- [流式强化学习：交互协议与更新稳定性](streaming.md)：明确更新器状态及其内存和计算约束。


### 两种时间尺度

内层变量随当前经验适应；外层变量决定内层怎样适应，并由后续表现训练。“外层”不必在另一台机器，也不必慢到离线。

### 梯度与 Hessian

梯度描述损失随参数的一阶变化；Hessian 描述梯度本身如何改变。对一次梯度下降再求导，自然会出现 Hessian，而不是额外加的技巧。

$$
\frac{\partial (w-\alpha\nabla L(w))}{\partial w}=I-\alpha\nabla^2L(w)
$$

### 固定数据条件下求导

可微优化循环通常先把采样轨迹当常量。RL 的策略变化还会改变轨迹分布；是否估计这条路径必须另行说明。

<a id="problem-definition"></a>

## 本章的问题定义

更新规则或初始化也需由经验改善。单一流的在线元梯度与跨任务元训练具有不同数据、重置和评价单位。

### 给定条件与符号

- 内层学习状态、可微更新映射和可选元参数。
- 固定外部评价、适应长度、后续评价数据、元训练/测试任务与重置权限。

### 需要求解的对象

使指定外层评价改善的初始化、步长、目标或更新规则；不能通过改写评价标准降低外层损失。

### 信息与数据权限

$W_t$ 表示内层完整持久状态，包含权重 $w_t$ 及影响下一次更新的优化器、资格迹和统计量；$\xi_t$ 是已到达经验。在线元参数及其敏感度还需另行保存。跨任务训练只使用声明的训练任务，测试任务未来不得参与外层选择。

$$
W_{t+1}=F_\eta(W_t,\xi_t),\qquad \min_\eta\mathcal J(\eta)=\mathbb E[J(W_{t+K},\xi_{t+K:t+K+M};\eta_{\rm eval})]
$$

$\eta$ 为元参数，$F_\eta$ 更新完整内层状态 $W$，$K$ 为适应长度，$M$ 为后续评价长度；$J$ 是外层损失（例如负回报），$\eta_{\rm eval}$ 固定评价约定。正文链式法则用小写 $w$ 简记参与求导的内层状态；具体算法只对权重求导时，需声明其他状态的导数处理。期望覆盖的任务、采样与生命周期分别规定。

### 成立条件与解的含义

- 固定数据导数需可微更新；完整RL期望导数还包括采样分布依赖，不能由固定轨迹链式法则自动得到。
- 对角、截断和一阶近似明确丢弃哪些敏感度；任务边界、参数/context reset和元训练成本计入协议。

判断准则：小问题多步元敏感度与有限差分一致；在独立后续数据或未见任务上、匹配适应与计算预算评价固定外层目标；在线协议无未来回流。

### 适用边界

- 在完整测试寿命上挑超参数不是智能体在线元学习。
- 跨任务快速适应不自动证明单一无重置生命期持续学习。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [时间信用分配与资格迹](credit.md)：资格分配当前误差对过去预测的更新，元敏感度追踪规则怎样影响后续学习；Metatrace还用另一条迹累计过去规则的作用。

- 改变信息或数据协议 · [流式更新与稳定性](streaming.md)：前向敏感度消耗持久状态，反向展开保存或重算学习窗口；使用在线经验不自动满足严格流式预算。

- 组合不同学习问题 · [通用价值函数与预测知识](gvf.md)：问题参数可通过预测更新、共享表示或答案读出影响后续控制；若消费者没有这条依赖，低预测误差不提供问题用途的信用。

- 组合不同学习问题 · [奖励假设与奖励设计](reward-design.md)：可学习训练信号、discount或目标生成器，同时固定外层评价；改变代理任务与改变最终目标是不同选择。

- 组合不同学习问题 · [平均奖励与差分价值](average.md)：持续控制的外层应保持原始时间上的奖励率或声明的代理；内层奖励率估计也是可能影响元导数的学习状态。

- 组合不同学习问题 · [可塑性与特征更新](plasticity.md)：评价学习能力需要让副本实际学习后再测结果；元步长能调整已有方向，但特征替换还改变表示与相关敏感度。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

一次规则改变会通过许多后续更新影响表现；改变策略还改变未来样本。

### 本章的核心思路

把内层全部状态纳入更新映射，先求完整敏感度，再声明哪些路径为降低成本而近似。

1. [传播规则对未来状态的影响](meta.md#lesson-derive)：对完整更新应用链式法则，分别保留元参数的直接作用与经旧状态传播的作用，再分析窗口截断删掉的路径。

2. [明确后续评价发生的时刻](meta.md#lesson-evaluation-timing)：训练后复用同一样本与用新样本评价会产生不同元方向；固定数据导数还应与规则改变采样分布的导数分开。

3. [在在线预算内学习步长](meta.md#lesson-idbd)：IDBD将完整敏感度压成逐特征统计，TIDBD再处理自举与资格；需要逐项说明对角、半梯度和稳定化近似。

4. [为学习规则分配时间信用](meta.md#lesson-metatrace)：Metatrace累计每个过去时刻的输出梯度与当时敏感度，而非把当前敏感度乘上整条资格迹。

5. [从调参数走向发现更新规则](meta.md#lesson-learned-rules)：区分内层参数、目标生成器和固定外层表现；规则表示、发现预算及部署时继续适应的状态各自声明。

6. [沿消费者评价规则的用途](meta.md#lesson-agent-coupling)：问题、表示、技能与控制形成不同依赖路径；保持外部目标和资源预算后，才能判断规则改善了哪段学习过程。

结论与条件：固定数据和元参数下完整敏感度递推是链式法则；截断/一阶/对角更新改变导数。没有相应采样路径估计时不能称为完整RL无偏元梯度。

### 相关方法改变了什么

- IDBD/TIDBD/Metatrace：在线适应学习参数，以结构近似减少敏感度成本。

- MAML：跨任务学习适合少量梯度更新的初始化。

- RL²/PEARL与learned rules：前者主要适应活动/context，后者学习更新信号，参数和reset边界不同。


<a id="lesson-setting"></a>

## 1. 问题设定：哪些变量在适应，适应后的表现如何评价

先看一个只有一个数的预测器。它预测下一次奖励，当前预测为 0。收到奖励 1 后，步长 0.2 把预测改成 0.2；步长 0.4 则改成 0.4。两种更新都更接近刚收到的奖励。要判断哪种步长更好，还要看后来发生了什么。下一次仍收到 1 时，较大的更新更有用；下一次收到 0 时，较小的更新误差更低。

![预测在数轴上从0移到0.2；后续目标为1或0时，当前步长对应的损失曲线斜率方向相反。](https://yingwen.io/crl-figures/concept-meta-one-update.svg)

先看上方预测移动了多远，再看下方圆点处的损失斜率。绿色曲线指向更大的步长，橙色曲线指向更小的步长。曲线、圆点和导数均由正文的一步模型计算。

$$
w'=w+\alpha(1-w),\quad J_y(\alpha)=\tfrac12(w'-y)^2,\quad \frac{dJ_y}{d\alpha}=(w'-y)(1-w).
$$

取 $w=0,\alpha=0.2$。后续目标 $y=1$ 给出导数 $-0.8$，梯度下降会增大步长；$y=0$ 给出 $+0.2$，会减小步长。这里固定两个样本，只检查一次更新的局部影响。

普通学习改变“我预测多少”；这一层学习改变“下一次应怎样修改预测”。外层也需要目标、数据和学习规则。它没有取消设计，而是将某些原先由设计者选定的量，交给经验来调整。这个简单例子接下来会扩展到多步影响、任务内适应和完整更新规则的发现。

本章有两种主要数据设定。在线元梯度可在一个持续运行的问题上工作：权重不断学习，步长或学习目标也根据后续误差调整，不要求出现一组可重置的任务。跨任务 meta-RL 则先规定任务分布与任务内适应过程，用训练任务中的经验训练初始化、推断器或更新规则，再检验新任务适应。两者都涉及学习过程，但评价单位与信息权限不同。

| 设定 | 在线元梯度 | 跨任务 meta-RL |
| --- | --- | --- |
| 数据组织 | 同一经验流中的先前更新与后续评价 | 训练任务中的适应数据和适应后评价数据 |
| 持久状态 | 权重与元敏感度通常跨更新保留；按真实边界处理资格迹 | 按任务或 trial 协议复制初值、清理 context 或循环活动 |
| 部署时变化 | 权重与元参数均可继续更新 | 外层参数常固定，任务内权重或活动状态继续适应 |
| 主要检验 | 长寿命中的累计误差、控制表现与变化后适应 | 未见任务上，在规定数据与 reset 预算内的适应 |
| 额外条件 | 元梯度展开、数据复用与每步预算 | 任务分布、元训练成本、适应次数与测试任务隔离 |

在线元梯度也可能缓存 rollout 或展开计算图，因此不自动满足严格流式协议。跨任务训练得到的规则也可以用于持续交互，但跨任务适应表现不能替代单次长生命中的持续学习证据。资格迹解决的是时间信用分配；它既可服务普通 TD，也可服务元目标，不以元学习为前提。

| 路线 | 内层适应什么 | 外层学什么 | 典型监督/边界 |
| --- | --- | --- | --- |
| 在线超参数学习：IDBD/TIDBD、meta-gradient RL | 预测器或策略权重 $w$ | 步长 $\alpha$、discount、$\lambda$、更新系数等 $\eta$ | 后续误差或回报；可在单一流上工作，不必有任务 ID。 |
| 优化型 Meta-RL：MAML | 从共同初值出发的任务内权重 $w_i$ | 适合少量梯度适应的初始化 $\theta$ | 任务分布、适应数据与适应后数据；通常有明确 trial/reset。 |
| 上下文型 Meta-RL：RL²、PEARL | 循环活动 $h$ 或任务潜变量 $z$ | 推断与控制网络参数 $\theta,\phi$ | 训练任务分布；测试时可仅更新上下文，不更新网络参数。 |
| 学习更新规则：LPG、DiscoRL 等 | 被训练的 agent 参数与学习器状态 | 产生目标/损失/更新信号的规则 $\eta$ | 跨环境、跨训练轨迹的外层表现；元训练预算必须单列。 |

“学习器状态”可以从保存并恢复一次训练来理解。只保存网络权重，重启后未必能得到相同的下一次更新：动量、资格迹和归一化统计也可能已经积累了历史。若算法使用循环网络，还需保存当前活动。它们不是都在描述外部世界，而是在共同决定学习器下一步如何运行。

| 保存的量 | 在一个具体实现中是什么 | 丢掉它会改变什么 |
| --- | --- | --- |
| 预测或策略参数 | 刚才从 0 改成 0.2 的预测权重 | 已经学到的答案和当前动作分布 |
| 优化器动量 | 例如 SGD 动量保存过去梯度的加权和 | 下一步沿哪个方向、走多远 |
| 资格迹 | 过去访问或输出梯度的衰减和 | 当前误差还更新哪些过去的预测 |
| 归一化统计 | 奖励、观测或梯度的运行均值与尺度 | 同一新样本进入算法时的数值尺度 |
| 活动或任务后验 | 走廊提示的记忆、当前任务的概率 | 此刻决策所使用的历史信息 |
| 元参数与敏感度 | 可学习步长，以及过去步长如何影响当前权重 | 怎样评价和改变学习规则本身 |

最初的标量例子只需要保存预测 $w$ 和步长 $\alpha$。一般地，将内层需要保存的量合成 $W_t$，更新写成 $W_{t+1}=F_\eta(W_t,\xi_t)$；$\xi_t$ 是本次经验，$\eta$ 是规则参数。下文链式法则用小写 $w$ 代表参与求导的内层状态向量。具体算法若只追踪权重而固定其他统计量，会在该处说明近似。在线元学习还要保存不断变化的 $\eta$ 及其敏感度。

$$
w_{t+1}=F_\eta(w_t,\xi_t),\qquad \mathcal J(\eta)=\mathbb E\bigl[J(w_{t+K},\xi_{t+K:t+K+M};\eta_{\rm eval})\bigr]
$$

$K$ 是适应长度，$M$ 是后续评价长度；$\eta_{\rm eval}$ 表示固定的评价约定。训练代理目标可以学习，最终评价标准仍需要事先定义。

- 先说明数据协议：独立任务、同一任务内多 episode，还是一条未分段的生命流。
- 再说明评价目标：预测误差、适应后回报、训练全过程回报、变化后恢复速度，不能彼此代替。
- 最后说明测试权限：能否重置参数/上下文/环境，是否拿到任务边界，是否继续更新外层参数。

<a id="lesson-design-space"></a>

## 学习规则的设计空间：经验可以改变什么

步长是最容易看见的一项设计。还可以改变开始学习的初值、用来训练的目标、经验怎样进入记忆，以及把有限计算用于哪些样本。这些选择处在同一学习过程的不同位置；一个系统可以同时学习其中几项。

![环境、经验、学习器与动作组成循环，下方用起点、步长箭头、靶心、样本选择和循环网络表示可学习的设计。](https://yingwen.io/crl-figures/concept-meta-design-space.svg)

沿上方循环读一次交互，再看下方五个可改变的位置。起点对应初始化，长短箭头对应更新幅度，靶心对应训练目标，放大镜对应计算分配，回环对应内部记忆。最下方是寻找这些设计的方法。

例如，MAML 学习一组适合随后改变的初始权重；IDBD 学习每个特征的步长；MGSC 学习从哪些状态查询模型；DiscoRL 学习怎样为策略和辅助预测生成目标。给它们统一加上“自适应”一词会丢失最有用的信息：经验究竟改变了哪一个决定。

还有一条独立的问题是规则怎样得到。可微程序允许沿更新求元梯度；不可微程序可以由黑盒搜索产生；一个已经学好的复杂规则又可以蒸馏成较小网络或符号表达。规则表示、搜索算法和部署协议需要分别选择。用许多可重置生命发现一条规则，得到的是跨生命的训练经验；把该规则放入一个不能重来的生命，才开始面对这个生命中的信息与资源限制。

<a id="lesson-derive"></a>

## 2. 多步更新的敏感度与链式法则

先固定 $\eta$ 和采样数据，定义 $H_t=\partial w_t/\partial\eta$。对更新 $F$ 求导，要包含 $F$ 对 $\eta$ 的直接依赖，以及 $\eta$ 通过旧参数 $w_t$ 产生的间接依赖。若 $\eta$ 就是初始化，则 $H_0=I$；若 $\eta$ 是独立步长而初始 $w_0$ 与它无关，则 $H_0=0$。

$$
H_{t+1}=\frac{\partial F_\eta}{\partial w_t}H_t+\frac{\partial F_\eta}{\partial\eta}
$$

这是固定元参数、固定经验序列下的精确前向敏感度。若 $F=w+f$，传播矩阵为 $I+\partial f/\partial w$，旧敏感度通常会被当前更新的 Jacobian 改变。

为什么必须包含整个学习器状态？若更新使用动量 $v_{t+1}=\rho v_t+\nabla L_t(w_t)$，同一个 $w_t$ 配上不同 $v_t$，下一次权重便不同。此时应令 $W_t=(w_t,v_t)$，在联合状态上应用上式；固定 $v_t$ 只求权重导数，会漏掉步长经早期权重、后续梯度、动量再回到权重的路径。奖励率估计、目标网络和循环活动也要按同样原则处理。某个统计量若被停止梯度，仍须保留其前向数值，只是导数图把它当作给定输入。

展开递推可以看见每次更新的贡献。记 $A_j=\partial F_\eta/\partial w_j$、$B_j=\partial F_\eta/\partial\eta$，暂令 $H_t=0$。第一次更新产生 $B_t$，经过第二次更新后变成 $A_{t+1}B_t$；第三次还会左乘 $A_{t+2}$。因此：

$$
H_{t+K}=\sum_{j=t}^{t+K-1}\left(A_{t+K-1}\cdots A_{j+1}\right)B_j.
$$

当 $j=t+K-1$ 时矩阵乘积取单位矩阵。保留窗口起点的敏感度还需加上 $(A_{t+K-1}\cdots A_t)H_t$。乘积决定早期规则选择会被后续学习放大、抵消还是转向；只累计 $B_j$ 相当于把所有 $A_j$ 当作单位矩阵。

反向微分计算同一组依赖：从末端评价梯度 $a_{t+K}=\nabla_wJ$ 出发，逐步令 $a_j=A_j^\top a_{j+1}$，累加 $B_j^\top a_{j+1}$。前向敏感度需要随元参数维数增长的持久状态；反向展开通常保存或重算窗口内的学习器状态。截断在某处设 $H=0$，删掉的是该处以前的导数贡献，学习器的 $w$ 和动量可以继续保留。计算图的截断并不要求环境终止或遗忘已经学到的参数。

$$
\nabla_\eta J=H_{t+K}^{\top}\nabla_w J+\left.\nabla_\eta J\right|_w
$$

若评价损失只通过适应后的 $w$ 依赖 $\eta$，右端第二项为零。若评价目标也含可学习 discount，第二项可能通过缩短评价时间尺度降低损失，而没有改善原定的长期目标。

$$
\begin{aligned}F_\beta(w)&=w-\alpha\nabla_w L(w),\qquad \alpha=e^\beta,\\H_{t+1}&=(I-\alpha\nabla_w^2L_t)H_t-\alpha\nabla_wL_t.\end{aligned}
$$

对数步长 $\beta$ 保证 $\alpha>0$。第一项传播已有敏感度，第二项表示步长对本次更新的直接影响。敏感度矩阵 $H$ 的行数是内层状态维度，列数是元参数维度；二者都很大时需要结构或近似。

只看一次内层更新，且 $H_t=0$ 时，外层对 $\beta$ 的导数为 $-\alpha g_{\rm train}^{\top}g_{\rm eval}$。两个梯度方向一致，适当增大步长会局部改善后续损失；方向相反，则倾向缩小步长。这是局部导数的解释，不是任意大步长下的单调性保证。样本噪声和梯度相关性还会影响该信号的可信度。

$$
\frac{\partial J(w-\alpha g_{\rm train})}{\partial\beta}=-\alpha\,g_{\rm eval}^{\top}g_{\rm train}
$$

$g_{\rm eval}$ 在更新后的参数上计算，$g_{\rm train}$ 在更新前计算。把两者都在旧参数上计算，得到的是另一种近似。

RL 还有采样路径：$\eta$ 改变策略，策略改变动作及后续数据。完整期望回报的导数一般包含轨迹分布的 score-function 项。固定数据的解析实验能够隔离优化器微分；采样路径是否无偏，则需要在随机过程的层面另外检验。

两步及多步学习的完整标量敏感度；旧 w 和旧 H 一致使用，可由有限差分独立核对。

```python
def fixed_beta_training(beta, train_targets, validation_target, initial=0.0):
    """Exact unrolled log-step-size derivative for FIXED beta and fixed data.

    w_next = w + exp(beta)*(target-w). H=dw/dbeta, H0=0.
    Includes every inner update Jacobian; it is not an online changing-beta
    total derivative and does not differentiate the data collection policy.
    """
    alpha = math.exp(beta)
    w, h = initial, 0.0
    trace = []
    for target in train_targets:
        error = target - w
        h = (1.0 - alpha) * h + alpha * error  # use old w and H
        w = w + alpha * error
        trace.append((w, h))
    loss = 0.5 * (w - validation_target) ** 2
    hypergradient = (w - validation_target) * h
    return loss, hypergradient, trace
```

<a id="lesson-evaluation-timing"></a>

## 当前误差评价了哪一次更新

一步元梯度还留下一个问题：训练样本和评价样本取自哪里？设要预测常数均值 $b$，观察为 $Y=b+\epsilon$，噪声均值为零、方差为 $\sigma^2$。暂把当前权重固定在正确均值 $w=b$，用一个样本执行 $w^+=w+\alpha(Y-w)=b+\alpha\epsilon$。此时任何移动都在拟合这次噪声。

$$
\begin{aligned}\mathbb E\left[\tfrac12(w^+-Y)^2\right]&=\tfrac12(1-\alpha)^2\sigma^2,\\\mathbb E\left[\tfrac12(w^+-Y')^2\right]&=\tfrac12(1+\alpha^2)\sigma^2.\end{aligned}
$$

$Y'$ 是均值相同、噪声独立的下一样本。第一行在原训练样本上评价更新后误差，第二行在新样本上评价；展开平方后，第二行的交叉项因独立零均值而消失。

对 $\beta=\log\alpha$ 求导，两行分别给出 $\alpha(\alpha-1)\sigma^2$ 和 $\alpha^2\sigma^2$。在 $0<\alpha<1$ 时，复用样本鼓励增大步长，新样本鼓励减小步长。若当前还存在偏差 $d=w-b$，新样本的期望损失为 $[(1-\alpha)^2d^2+(1+\alpha^2)\sigma^2]/2$，最优一步步长为 $d^2/(d^2+\sigma^2)$（分母非零）。适应必须同时考虑尚未学会的信号和无法通过追踪消除的噪声。

这并不要求在线算法把一半样本永久留作验证。IDBD 在第 $t$ 个样本到达时，先用尚未看过该样本的 $w_t$ 计算误差；旧 $h_t$ 把这项误差归因于过去的步长。随后同一样本仍可更新 $w_{t+1}$。评价过去更新与执行当前更新共用一个到达样本，和用当前样本先拟合、再评价自身，是两个时序。

在 RL 中，相邻新样本可能相关；动作还决定以后能看到什么。若完整经验记录 $D$ 的分布为 $p_\eta(D)$，把更新后的评价写成 $\widehat J(\eta,D)$，则在可交换求导与期望的条件下有：

$$
\nabla_\eta\mathbb E_{D\sim p_\eta}[\widehat J(\eta,D)]=\mathbb E_D\left[\left.\nabla_\eta\widehat J(\eta,D)\right|_D+\widehat J(\eta,D)\nabla_\eta\log p_\eta(D)\right].
$$

第一项包括穿过学习更新的导数；第二项包括规则通过策略改变适应数据或评价数据的概率。环境转移规律若与 $\eta$ 无关，score 来自 agent 的随机选择。固定轨迹有限差分只检查第一项。

因此要分别说明评价发生在更新前还是更新后、是否复用同一记录、是否重新采样，以及估计器是否包括采样分布路径。后文 MAML 的 Bernoulli 枚举将直接检验这两项；在线元梯度的近似则先在给定经验下讨论。

<a id="lesson-trace-sensitivity"></a>

## 沿同一条经验流，区分资格与步长敏感度

深入阅读：沿同一条轨迹逐项核对资格迹与步长敏感度，观察两种历史为何不同。

把链式法则用到一个已经熟悉的学习器。信用分配章给定 $A\xrightarrow{1}B\xrightarrow{0}A\xrightarrow{2}\bot$，特征依次为 $(1,0),(1,1),(1,0),0$，初值为零，$\gamma=1$、$\lambda=1/2$。本节选择传统 accumulating TD(λ) 作为内层程序，三次更新共用一个固定步长 $\alpha=1/2$。先算清这个程序的导数，再讨论在线改变步长；不在这三步之中执行元更新。

令 $H_t=\partial w_t/\partial\alpha$，它记录步长怎样影响更新前的权重。固定特征、$\gamma$ 与 $\lambda$ 时，累积资格 $z_t$ 与 $\alpha$ 无关。对 $w_{t+1}=w_t+\alpha\delta_tz_t$ 求导，当前步长直接改变本次增量，过去步长还会通过 $w_t$ 改变本次误差。

$$
\begin{aligned}H_0&=0,\qquad \delta_t=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\\H_{t+1}&=H_t+\delta_tz_t+\alpha z_t(\gamma x_{t+1}-x_t)^\top H_t.\end{aligned}
$$

中间项是本次更新对 $\alpha$ 的直接导数；末项传播旧权重造成的误差变化。若改学对数步长 $\beta=\log\alpha$，在共同固定参数的条件下有 $\partial w_t/\partial\beta=\alpha H_t$。

| 刚处理的转移 $t$ | 用于更新的资格 $z_t$ | 更新后的权重 $w_{t+1}$ | 更新后的敏感度 $H_{t+1}$ |
| --- | --- | --- | --- |
| 0：A → B | (1, 0) | (1/2, 0) | (1, 0) |
| 1：B → A | (3/2, 1) | (1/2, 0) | (1, 0) |
| 2：A → 终止 | (7/4, 1/2) | (29/16, 3/8) | (11/4, 1/2) |

第二步的资格已经包含 B 的两个特征，敏感度却仍是 $(1,0)$：当前误差为零，且 $(x(A)-x(B))^\top H_1=0$。最后一步有 $\delta_2=3/2$，而 $\partial\delta_2/\partial\alpha=-1$。因此 $H_3=(1,0)+(3/2)(7/4,1/2)-(1/2)(7/4,1/2)=(11/4,1/2)$。资格记住了过去哪些特征应接收当前误差；敏感度还经过每次学习对后续误差的影响。

现在另外给定一个评价样本：假设独立 episode 从 B 出发得到返回 2，使用它评价三次内层更新后的预测，且不再用它更新 $w_3$。这条样本与内层记录一同固定用于手算。评价规则为 $J=(w_3^\top x(B)-2)^2/2$，不随步长变化。预测是 $35/16$，残差为 $3/16$，于是：

$$
J=\frac9{512},\qquad \frac{dJ}{d\alpha}=\underbrace{\frac3{16}}_{\text{评价残差}}\underbrace{(1,1)H_3}_{13/4}=\frac{39}{64}.
$$

正导数表示在 $\alpha=1/2$ 附近，减小这个共同步长会降低所指定的后续损失。它没有比较长期控制收益，也没有执行外层训练。

还可以把三次步长暂时写成独立的 $\alpha_0,\alpha_1,\alpha_2$，都在 $1/2$ 处求导。第一次步长的影响经过后续更新，最终对 B 预测的导数是 $-1/8$；第二次是 0；第三次是 $27/8$。把三者重新绑定为同一个 $\alpha$，总导数为 $13/4$。因此只保留末次直接路径给出 $(3/16)(27/8)=81/128$，完整元梯度则还包括早期路径的 $-3/128$，得到 $39/64$。这里较早更新的影响与最后一次更新方向相反。

停止梯度的位置会再改变这个答案。内层 TD 使用半梯度来规定权重更新，并不意味着对已经写出的整个更新程序求导时，bootstrap 的数值没有依赖旧权重。若在外层也停止 bootstrap 路径，误差导数就从 $(\gamma x_{t+1}-x_t)^\top H_t$ 变成 $-x_t^\top H_t$，本例得到 $273/512$。这一结果可由另一份有限差分核对：扰动步长并重跑时，把每步 bootstrap 标签固定为中心运行保存的数值。直接重新计算所有 bootstrap 的有限差分则得到 $39/64$。两种检查对应不同的求导约定。

![上图比较累积资格与步长敏感度沿B特征方向的分量和，下图比较独立B评价与复用末次A训练样本的损失曲线；步长二分之一处损失相同而斜率反号。](../assets/crl-figures/credit-meta-sensitivity.svg)

上图横轴是完成更新数，$h>0$ 时橙线显示刚用过的 $z_{h-1}$，紫线显示 $H_h$。下图每个点都从相同初值重跑三次更新，再按所标样本计算损失；这不是元学习训练曲线。独立 B 与复用 A 在 $\alpha=1/2$ 的损失均为 $9/512$，斜率却分别为 $39/64$ 与 $-33/64$。[精确数据](https://yingwen.io/crl-figures/credit-meta-data.json)与[Python 手算脚本](../tutorials/credit_meta_walkthrough.py)可重算两条曲线；学习更新的链式法则与固定评价约定参见 [Xu 等 §1，式 (1)–(4)](https://arxiv.org/html/1805.09801#S1)。

图中第二条曲线复用了内层最后一次 A 的样本，评价 $J_A=(w_{3,1}-2)^2/2$。它的残差为 $-3/16$，导数为 $(-3/16)(11/4)=-33/64$，会局部鼓励增大步长。损失恰好相等并没有让两个目标相同。B 样本考察共享更新后的另一项预测，A 样本继续考察刚训练过的输出；在真实 RL 中，后续相邻数据还可能相关，策略改变也会改变采样分布，需要另行规定这些路径。

最后区分参数版本。这里求的是把三次步长一起扰动时的导数。若每步已经在线更新了 $\alpha_t$，对独立当前变量 $\alpha_2$ 的导数只有末次路径；把旧 $H$ 接着累积，通常是在追踪一种共同扰动或局部近似。需要先选定这个对象，再解释 IDBD、TIDBD 或 Metatrace 保存的元状态。将内层换为 true-online 时，荷兰迹本身也依赖步长，必须连同旧预测一起微分，不能直接套用本节把 $z$ 视为常量的递推。

下载 [credit_meta_walkthrough.py](../tutorials/credit_meta_walkthrough.py) 后运行 python3 tutorials/credit_meta_walkthrough.py，可打印全部前缀、目标、敏感度与三种外层导数。Python 使用有理数和前向敏感度；[JavaScript 对照](https://yingwen.io/crl-code/figures/credit-meta-walkthrough.mjs)使用反向伴随量独立计算各次步长路径。

<a id="lesson-idbd"></a>

## 3. IDBD：从多步链式法则到每个特征一个步长

IDBD 考虑线性监督预测：输入 $x_t$、标签 $y_t$、预测 $w_t^\top x_t$，误差为 $\delta_t=y_t-w_t^\top x_t$。每个特征 $i$ 有自己的步长 $\alpha_i=\exp(\beta_i)$。一个方向的更新持续同向，可能表示跟踪速度不足；频繁反向，则可能是更新过冲。IDBD 通过权重对过去步长的敏感度 $h_i$ 估计这种关系，而不只统计梯度幅度。

$$
\delta_t=y_t-w_t^\top x_t,\qquad w_{i,t+1}=w_{i,t}+\alpha_{i,t}\delta_t x_{i,t}
$$

这是内层 LMS/线性 SGD 更新。所有特征共享误差 $\delta$，但各坐标的 $\alpha_i$ 不同，因此改变的不只是总更新长度，还包括学习的几何方向。

精确敏感度是矩阵 $\partial w_j/\partial\beta_i$：改变一个权重会改变公共误差，再影响其他权重。IDBD 只保存 $h_i\approx\partial w_i/\partial\beta_i$，忽略 $i\ne j$ 的交叉影响。在固定 $\beta$ 的对角近似下，对权重更新求导得到保留因子 $1-\alpha_i x_i^2$，以及直接作用项 $\alpha_i\delta x_i$。

$$
\begin{aligned}D_\alpha&=\operatorname{diag}(\alpha_1,\ldots,\alpha_n),\qquad H_{ji}=\frac{\partial w_j}{\partial\beta_i},\\H^+&=(I-D_\alpha xx^\top)H+\operatorname{diag}(\alpha\odot\delta x),\\\nabla_\beta\tfrac12\delta^2&=-\delta H^\top x.\end{aligned}
$$

这里固定整段共用的 $\beta$，尚未作截断。因为 $\partial\delta/\partial w=-x^\top$，旧敏感度左乘 $I-D_\alpha xx^\top$；指数步长的直接导数只出现在相应对角位置。即使初始 $H=0$，共享误差也会在后续步骤产生非对角项。

只保留矩阵对角项，外层方向才从 $\delta H^\top x$ 变成逐坐标的 $\delta x_i h_i$，递推才从矩阵乘法变成 $h_i(1-\alpha_i x_i^2)+\alpha_i\delta x_i$。再加入正截断，并用本次元更新后的步长执行权重更新，就得到下面的实际 IDBD。它把二次规模的敏感度存储缩为线性规模，也舍弃了一个坐标的步长经其他权重改变预测的影响。

$$
\begin{aligned}\beta_i^+&=\beta_i+\mu\delta x_i h_i,\qquad \alpha_i=e^{\beta_i^+},\\w_i^+&=w_i+\alpha_i\delta x_i,\\h_i^+&=h_i\max(0,1-\alpha_i x_i^2)+\alpha_i\delta x_i.\end{aligned}
$$

$\delta$ 来自更新前的权重，$\beta$ 使用旧 $h$ 更新，之后才计算新 $\alpha$、$w$ 与 $h$。正截断避免保留因子为负，是稳定化操作而非精确链式法则。$\mu$ 是另需指定的元步长。

**算法：算法伪代码**

1. 初始化 $w=0$、$h=0$、$\beta_i=\log\alpha_{i,0}$。
1. 每个样本 $(x,y)$：
  1. $\delta\leftarrow y-w^\top x$              # 更新前的预测误差
  1. $\beta_i\leftarrow\beta_i+\mu\delta x_i h_i$ # 使用旧敏感度
  1. $\alpha_i\leftarrow\exp\beta_i$
  1. $w_i\leftarrow w_i+\alpha_i\delta x_i$
  1. $h_i\leftarrow h_i\max(0,1-\alpha_i x_i^2)+\alpha_i\delta x_i$
1. 用更新前误差评价当前样本，下一样本再评价学习后的预测。

初始 $h=0$，第一步元更新为零，因为尚无过去步长影响当前权重的历史。普通权重更新产生非零 $h$ 后，后续误差才能评价这种影响。特征尺度、强相关性、噪声与过大的元步长 $\mu$ 都可能使适应不稳定，因此仍需监测步长及敏感度。

在线 $\beta_t$ 不断改变时，这份 $h$ 也不是“只扰动今天的 $\beta_t$”的偏导数。原始推导考虑跨时间的共同微扰，并作坐标局部近似；若要微分整个元更新程序，则连 $\beta,h$ 的递推及其对过去元更新的依赖也要放进状态。这里的后续误差是在评价近似保存的过去学习影响。

线性监督预测的 IDBD：保存旧误差与旧敏感度，依次更新对数步长、权重和敏感度。

```python
def idbd_step(weights, beta, history, features, target, meta_rate=0.01):
    """Linear supervised IDBD, with diagonal sensitivity and positive clipping."""
    error = target - sum(w * x for w, x in zip(weights, features))
    new_beta = [b + meta_rate * error * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_weights = [w + a * error * x
                   for w, a, x in zip(weights, alpha, features)]
    new_history = [h * max(0.0, 1.0 - a * x * x) + a * error * x
                   for h, a, x in zip(history, alpha, features)]
    return new_weights, new_beta, new_history, error
```

<a id="experiment-idbd"></a>

### 实验：实验 · 一次权重反转能否激发逐特征步长适应？

在完全相同的观测流上，改变学习规则的历史敏感度能否影响反转后的追踪速度？

**环境与可用信息。** 输入为 [1,u]，u 均匀取自 [−1,1]。真权重前 600 步为 [0.2,0.7]，之后变成 [−0.2,−0.7]。标签另加标准差 0.03 的高斯噪声。算法看不到真权重，也不接收任务边界通知。

**设置。** 五种子各 1200 个样本。初始权重和敏感度全零，对数步长均为 log(0.03)，IDBD 元步长 0.01。对照是固定步长 0.03 的 LMS。每样本更新一次，无回放；两者使用相同种子的输入和噪声。

**检验的机制。** IDBD 先由当前误差与旧敏感度更新对数步长，再更新权重和敏感度。元状态记录过去步长如何影响当前权重，不是把过去奖励向前传播的资格迹。

**测量。** 纵轴为当前真函数的解析无噪声 MSE，不含不可约标签噪声。第 601 步目标才反转；反转前后应分别读图，不把目标变化造成的跳升当成数值发散。

```bash
python3 implementations/continual/idbd.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/idbd/curves.svg)

横轴：environment_steps。纵轴：当前任务无噪声预测 MSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 反转后第 750 步，IDBD 的五种子平均 MSE 为 0.0288，固定步长为 0.0391；第 900 步为 0.000865 对 0.00211。末尾两者均在数万分之一量级。这里显示的是该配置下的追踪差异，不是对所有任务的显著优势。

**结论边界。** 只有两个相关特征、一次变化，也没有 TD 自举。当前日志没有保存逐坐标步长，因而不能仅凭误差曲线声称已经直接观察到特征选择。

**继续实验。** 先补记每个坐标的步长和敏感度。再加入无关特征，并让相关特征的变化速度不同。预先固定任务族和调参预算，检验“区分特征”与“统一增大学习率”两种解释。

[源码](../implementations/continual/idbd.py) · [逐种子记录](https://yingwen.io/crl-code/results/idbd/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/idbd/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/idbd/curves.json)

<a id="rlss-idbd-salience"></a>

## 步长塑造泛化，但步长不等于特征价值

IDBD 的学习对象是“这类参数变化今后应当多快”，不是“这个输入在语义上是否重要”。区分二者，才能把逐特征步长接到持续表示学习，而不把稳定知识误当成无用知识删除。先从监督线性预测开始：输入与目标外生给定，预测为 wᵀx，误差为 y−wᵀx。

$$
\Delta w=D_\alpha\delta x,\qquad \Delta\widehat y(u)=\delta\,x(u)^\top D_\alpha x,\qquad D_\alpha=\operatorname{diag}(\alpha_1,\ldots,\alpha_d).
$$

逐坐标步长改变一次更新怎样传播到其他输入。它是一种学习偏置。若特征坐标乘 c、权重除 c，为保持相同预测变化，相应步长还需除以 c²。跨不同尺度直接比较 α 的大小，没有不变的效用含义。

IDBD 用 βᵢ=log αᵢ 保证步长为正，并根据后来的预测误差调整 β。推导中的主要近似不是指数变换，而是怎样保存步长对已有权重的历史影响。令 Hⱼᵢ=∂wⱼ/∂βᵢ。即使每个 β 直接只控制一个权重，误差共享也会让它间接影响其他权重。

$$
H^+=(I-D_\alpha xx^\top)H+\operatorname{diag}(\alpha\odot\delta x).
$$

这条精确敏感度递推针对整个数据前缀使用固定 β 的 LMS，输入和目标不依赖 β。它需要保存 d×d 矩阵。第一项中非对角元素通常不为零，所以只跟踪 Hᵢᵢ 是近似，而不是线性预测自然消除了所有交叉路径。

$$
-\frac{\partial(\delta^2/2)}{\partial\beta_i}=\delta\sum_jx_jH_{ji}\ \approx\ \delta x_i h_i.
$$

保留对角敏感度 hᵢ 后，得到逐参数元更新。β 本身在线变化时，还使用了局部、缓变的敏感度近似。不能将这一 O(d) 规则称为完整跨全部学习历史的精确元梯度。

$$
\begin{aligned}\beta_i^+&=\beta_i+\theta\delta x_i h_i,\quad \alpha_i^+=e^{\beta_i^+},\\ w_i^+&=w_i+\alpha_i^+\delta x_i,\\ h_i^+&=h_i\max(1-\alpha_i^+x_i^2,0)+\alpha_i^+\delta x_i.\end{aligned}
$$

这是原始 IDBD 的更新顺序：δ 和 h 来自旧参数，先更新 β，再用新 α 更新 w 与 h。正部截断是稳定化设计，不是未修改的敏感度链式法则。

正的 δxᵢhᵢ 表示近期沿这一坐标的变化，与当前误差给出的更新方向一致；它提供“先前可能应该更快”的证据。负值可能意味着过度修正，也可能来自噪声或相关特征的相互补偿。它不是一个逐样本正确的“相关／无关输入”分类器。

| 观察 | 至少两种可能原因 | 不能立即采取的行动 |
| --- | --- | --- |
| α 很小 | 无关噪声；有用但已稳定的预测；输入很少出现 | 直接删除该特征 |
| 输出权重很大 | 贡献显著；输入单位很小；与其他特征抵消 | 把权重绝对值当作通用效用 |
| 新特征降低本步误差 | 捕捉可重复结构；拟合不可预测噪声 | 立即宣称获得了可迁移表示 |
| 元梯度几乎为零 | 误差已小；特征不激活；敏感度衰减 | 宣称永远无需再学习 |

原始实验使用 20 个独立标准正态输入，只有 5 个参与目标，每个相关系数为 ±1；定期翻转一个相关系数，使这些输入始终需要追踪。这个设计有意把“相关”与“需要快速更新”联系起来。它不证明在任意环境中 α 都等于重要性。把系数固定后，一个已经学准的关键输入可能需要很小步长，却仍不应被删除。

逐步 IDBD 教学实现：先对照更新时序，再做敏感度有限差分

```python
def idbd_step(weights, beta, history, features, target, meta_rate=0.01):
    """Linear supervised IDBD, with diagonal sensitivity and positive clipping."""
    error = target - sum(w * x for w, x in zip(weights, features))
    new_beta = [b + meta_rate * error * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_weights = [w + a * error * x
                   for w, a, x in zip(weights, alpha, features)]
    new_history = [h * max(0.0, 1.0 - a * x * x) + a * error * x
                   for h, a, x in zip(history, alpha, features)]
    return new_weights, new_beta, new_history, error
```

**算法：将数学正确性与特征效用实验分开**

1. 固定一段二维、相关输入的监督数据，并冻结 β。
1. 分别用完整 H 递推与 β 的中心有限差分计算最终权重敏感度。
1. 先验证两者一致，再对照只保留对角的近似。
1. 恢复在线 β 后，单独记录更新前误差、步长、敏感度与数值范围。
1. 先复现相关输入持续漂移，再加入“有用但稳定”和“稀有但关键”的输入。
1. 只有比较删除前后的未来损失及下游影响，才讨论特征淘汰。

Expand-and-Add 进一步把当前观测、动作和旧状态送入非线性特征生成器，再用快速线性读出来学习预测。递归可以使线性读出依赖较长历史，但没有免费保存全部历史：状态容量、更新成本、时间信用与特征维护仍是约束。错误较大时 imprint 新特征，也可能只记住噪声。

生成与检验至少需要成熟期、真实未来样本上的预测检验，以及特征依赖关系。一个特征对当前读出贡献不大，却可能是另一个有用特征的输入。沿依赖链传播效用并加衰减是一种架构设计选择；不是 IDBD 自动提供的性质。共享 GVF、option 与模型以后，删除特征还可能改变问题函数，须维护这些对象的语义。

到 TD 或 actor–critic，误差里的后继预测和实际数据流也依赖学习过程。TIDBD、Metatrace 与不同元梯度方法保留的路径不相同。先分清所求导的目标、丢弃的依赖与固定的数据假设，再比较其数值和实验结果，不能仅因都有一个 h 或 trace 就互换公式。

| 机制 | 所表达的问题 | 改变后须重新检查什么 |
| --- | --- | --- |
| Interest i | 哪些预测处境值得分配近似资源 | 评价目标是否也被改动 |
| Emphasis M | 这些预测经自举又依赖哪些处境 | 强调分布、重要性比方差、稳定性条件 |
| 资格迹 e | 当前误差应更新哪些过去参与的参数 | 目标策略、λ、旧梯度与当前表示的一致性 |
| 逐参数步长 α | 沿各参数方向多快改变 | 更新几何、元目标、输入尺度和交叉敏感度 |

这四个量都能放大或缩小一次更新，却不能因此互换。给当前 TD 更新乘较大的 α，并不等于正确传播先前状态的 interest。反过来，emphasis 的大值可能来自罕见的重要性比乘积，不意味着该特征应永久拥有较大学习率。联合自适应时，应先冻结三项只改变一项，再检验耦合；不能把每个模块各自的固定条件定理拼成整个自适应系统的保证。

<a id="lesson-tidbd"></a>

## 4. TIDBD：bootstrap 与资格迹改变了什么？

TD 的目标不是外部固定标签 $y$，而是奖励加下一状态的当前预测。因此，对 TD 误差求导时必须指定是否对 bootstrap 目标求导。以下采用 2018 TIDBD Algorithm 1：线性、on-policy、常数 $\gamma$、累积资格迹的半梯度版本。后续 TIDBD 和 AutoTIDBD 的完整误差导数、步长归一化等扩展，需要分别识别。

$$
\begin{aligned}\delta_t&=R_{t+1}+\gamma w_t^\top x_{t+1}-w_t^\top x_t,\\z_t&=\gamma\lambda z_{t-1}+x_t,\\w_{i,t+1}&=w_{i,t}+\alpha_{i,t}\delta_t z_{i,t}.\end{aligned}
$$

$z$ 是内层资格迹，$h$ 是步长敏感度；$\gamma$ 定义预测时间尺度，$\lambda$ 控制信用回溯。资格迹在真实 episode 间如何清零由协议决定，未通知的分布变化不提供额外清零信号。

$$
\begin{aligned}\beta_i^+&=\beta_i+\mu\delta_t x_{i,t}h_{i,t},\qquad\alpha_i=e^{\beta_i^+},\\h_i^+&=h_i\max(0,1-\alpha_i x_{i,t}z_{i,t})+\alpha_i\delta_t z_{i,t}.\end{aligned}
$$

外层局部误差的半梯度使用当前 $x$，内层更新通过资格迹 $z$ 分配信用。因此 $\beta$ 更新中是 $x$，敏感度传播和权重更新中则同时出现 $z$。

$$
\left.\frac{\partial\delta_t}{\partial w_i}\right|_{\rm semi}=-x_{i,t},\qquad\left.\frac{\partial\delta_t}{\partial w_i}\right|_{\rm full}=\gamma x_{i,t+1}-x_{i,t}
$$

若改为完整 TD-error 导数，外层方向涉及 $x_t-\gamma x_{t+1}$，$h$ 的传播因子也相应变化。目标导数约定、敏感度递推与步长归一化应成套指定。

半梯度敏感度可以用一个明确的局部程序理解：在中心运行中先保存每一步的 bootstrap 数值 $b_t=\gamma w_t^\top x_{t+1}$，扰动步长重跑时仍使用这份数值，只重新计算 $R_{t+1}+b_t-w^\top x_t$。在单坐标、固定步长且正截断未触发时，TIDBD 的 $h$ 就是这个固定标签程序的导数。若扰动后也重算 $b_t$，检验对象已经换成完整 TD 更新，通常得到另一个导数。多坐标的 $h$ 还要加上对角近似的误差。

当 $\gamma=0$、旧资格迹不再贡献且 $z=x$ 时，更新退化为 IDBD。真正 terminal 的 bootstrap 为零，可以将下一状态特征置零，在完成本次更新后清空 $z$；采样预算造成的 timeout 不自动等于 terminal。自适应步长本身也不改变 off-policy TD 的稳定性条件。

TIDBD 的全部一转移更新。调用者负责真实 episode 的资格迹边界；脚本不自行猜任务变化。

```python
def tidbd_step(weights, beta, history, eligibility, features, next_features,
               reward, gamma=0.9, lam=0.8, meta_rate=0.01):
    """2018 TIDBD(lambda) semi-gradient version, on-policy, constant gamma.

    This is NOT the later full TD-error-gradient/normalization variants.
    When the protocol declares a genuine new episode, the CALLER clears z.
    """
    value = sum(w * x for w, x in zip(weights, features))
    next_value = sum(w * x for w, x in zip(weights, next_features))
    delta = reward + gamma * next_value - value
    new_beta = [b + meta_rate * delta * x * h
                for b, x, h in zip(beta, features, history)]
    alpha = [math.exp(b) for b in new_beta]
    new_z = [gamma * lam * z + x for z, x in zip(eligibility, features)]
    new_weights = [w + a * delta * z
                   for w, a, z in zip(weights, alpha, new_z)]
    new_history = [h * max(0.0, 1.0 - a * x * z) + a * delta * z
                   for h, a, x, z in zip(history, alpha, features, new_z)]
    return new_weights, new_beta, new_history, new_z, delta
```

实际交互入口 [continual/tidbd.py](../implementations/continual/tidbd.py) 的 update 保留了上述一般 γ、λ 更新，但 run 使用 γ=0 的两维回归流，因而其曲线检验的是 IDBD 退化情形。带 bootstrap 的导数由配套机制代码检查；不能把回归曲线读成带迹 TD 控制实验。

<a id="lesson-metatrace"></a>

## 5. Metatrace：过去的哪些步长影响了今天的控制误差

TIDBD 主要从一步预测误差出发。带资格迹的 actor–critic 则把一个控制目标的信用分散到许多时刻：今天的 TD 误差会更新过去状态的价值与动作概率。Metatrace（Young、Wang、Taylor，IJCAI 2019）相应地为步长建立一条元资格迹。区别不只是把 TIDBD 接上 actor，而是外层目标包含各时刻实际使用的不同权重 $w_k$，因此要保留各自的敏感度 $h_k$。

$$
\mathcal J=\tfrac12\sum_k\left[\bigl(\operatorname{sg}(G_k^\lambda)-V_{w_k}(S_k)\bigr)^2-\log\pi_{w_k}(A_k\mid S_k)\operatorname{sg}\bigl(G_k^\lambda-V_{w_k}(S_k)\bigr)\right]
$$

这是用于推导的时变权重半梯度目标：critic 拟合多步 return，actor 使用固定的优势评价。在线 return 写成 $G_k^\lambda=V_{w_k}(S_k)+\sum_{t\ge k}(\gamma\lambda)^{t-k}\delta_t$；求导时按 $\operatorname{sg}$ 约定停止目标路径。权重梯度因此组合为下面的 $\nabla U$。

$$
\begin{aligned}U_w(s,a)&=V_w(s)+\tfrac12\log\pi_w(a\mid s),\\g_t&=\nabla_w U_{w_t}(S_t,A_t),\qquad q=\gamma\lambda,\\z_t&=qz_{t-1}+g_t,\qquad \delta_t=R_{t+1}+\gamma V_{w_t}(S_{t+1})-V_{w_t}(S_t),\\w_{t+1}&=w_t+\alpha_t\delta_tz_t.\end{aligned}
$$

$U$ 把原文选择的 critic 与 actor 权重合并；$1/2$ 来自其联合目标的系数，并非任意 actor–critic 都必须采用的常数。$z$ 是权重更新的资格迹。目标策略随权重变化，以下元导数仍在给定采样路径下计算。

设 $h_k\approx\partial w_k/\partial\beta$，$\alpha=\exp\beta$。对时变权重的多步半梯度目标应用链式法则，再交换时间求和，当前 TD 误差应乘过去各时刻 $g_k^\top h_k$ 的衰减和。这导出第二条迹 $z_{\beta,t}$。如果错误地使用 $z_t^\top h_t$，就把今天的敏感度替换到了过去所有时刻，改变了目标的时间结构。

$$
\begin{aligned}z_{\beta,t}&=\sum_{k=0}^{t}q^{t-k}g_k^\top h_k=qz_{\beta,t-1}+g_t^\top h_t,\\\widetilde\nabla_\beta\mathcal J&=-\sum_t\delta_t z_{\beta,t},\qquad \beta^+=\beta+\mu\delta_tz_{\beta,t},\\h_{t+1}&\approx h_t+e^{\beta^+}z_t\left[\delta_t+\left(\gamma\nabla V_{w_t}(S_{t+1})-\nabla V_{w_t}(S_t)\right)^\top h_t\right].\end{aligned}
$$

外层目标对 return 采取半梯度约定；敏感度递推却使用 $\delta$ 对权重的完整导数，因为它要追踪实际更新怎样依赖步长。递推省略了 $\partial z_t/\partial w$，也未计算动作分布改变的采样路径。即使 critic 线性，策略对数概率一般仍非线性，因而整体不是精确超梯度。

把省略项写出来能看清近似的位置。先固定共同 $\beta$、$q$ 与采样路径，并省去归一化和熵项，定义 $Z_t=\partial z_t/\partial\beta$。由 $z_t=qz_{t-1}+\nabla U_{w_t}$ 得：

$$
\begin{aligned}Z_t&=qZ_{t-1}+\nabla_w^2U_{w_t}\,h_t,\\h_{t+1}&=h_t+\alpha\left[\delta_tz_t+z_t(\nabla_w\delta_t)^\top h_t+\delta_t Z_t\right].\end{aligned}
$$

Metatrace 的局部敏感度忽略末项 $\alpha\delta_tZ_t$。这里 $Z_t$ 是“权重资格迹对步长”的导数；前面的 $z_{\beta,t}$ 则累计各时刻 $g_k^\top h_k$ 来分配元误差。两者有不同定义，不能互相替换。

线性 critic 单独学习时 $\nabla^2U=0$，初始 $Z=0$ 会一直为零，所以固定步长下的 $h$ 可以精确匹配该程序的有限差分。加入 softmax actor 后，即使 logits 对权重线性，$\log\pi$ 仍有非零 Hessian。进一步恢复在线元更新和归一化时，若要对整个实际算法求精确导数，还必须传播 $\beta,u,v$ 的变化；论文的局部敏感度并未微分这些适应操作。

**算法：无熵项、无归一化的标量步长 Metatrace 核心**

1. 初始化 $w$、$\beta$；$z=0$，$h=0$，$z_\beta=0$。
1. 每个转移，用旧 $w$ 计算 $\delta$、$g=\nabla U$ 和 $d=\gamma\nabla V'-\nabla V$。
  1. $z\leftarrow qz+g$
  1. $z_\beta\leftarrow qz_\beta+g^\top h$        # 此时 h 仍是旧敏感度
  1. $\beta\leftarrow\beta+\mu\delta z_\beta$
  1. $\alpha\leftarrow\exp(\beta)$
  1. $h_{\rm new}\leftarrow h+\alpha z(\delta+d^\top h)$
  1. $w\leftarrow w+\alpha\delta z$；$h\leftarrow h_{\rm new}$
1. 真正 episode 结束后清空 $z,z_\beta$；保留 $w,\beta,h$。

手算两步：令 $q=0.5$、$g_0=g_1=1$、$h_0=0$、$h_1=0.1$。第二步 $z_1=1.5$，但 $z_{\beta,1}=0.5\times0+1\times0.1=0.1$，不是 $z_1h_1=0.15$。差别在于第一步发生时敏感度仍为零。延长资格迹不会创造过去并不存在的步长影响。

原文还给出逐参数版本和混合版本：逐参数版本保存对角敏感度，混合版本以 $\alpha_i=\exp(\widehat\beta+\beta_i)$ 结合全局与局部适应。实用算法先把元增量除以运行幅度估计，再限制有效步长；若有熵正则 $\psi\mathcal H(\pi)$，元增量中还包括 $\psi\nabla\mathcal H^\top h$。这些状态必须与权重资格迹分开保存。

$$
\begin{aligned}\Delta_\beta&=\delta z_\beta+\psi\nabla\mathcal H^\top h,\\v&\leftarrow\max\{|\Delta_\beta|,\ v+\mu(|\Delta_\beta|-v)\},\\\beta&\leftarrow\beta+\mu\Delta_\beta/\widetilde v,\qquad \widetilde v=\begin{cases}v&v>0\\1&v=0,\end{cases}\\u&\leftarrow\max\{e^\beta\|g\|^2,\ u+(1-q)(e^\beta\|g\|^2-u)\},\\\beta&\leftarrow\beta-\log\max(u,1).\end{aligned}
$$

这是原文标量归一化的结构。末行只有在有效步长估计超过 1 时才缩小 $\alpha$，不是对任意非线性网络误差的全局收缩定理。加入熵正则时，权重和敏感度更新也需要对应熵项。

$$
\begin{aligned}w^+&=w+\alpha(z\delta+\psi\nabla\mathcal H),\\h^+&\approx h+\alpha\left[z(\delta+\nabla\delta^\top h)+\psi\nabla\mathcal H\right].\end{aligned}
$$

这里 $\alpha$ 使用归一化后的新对数步长。敏感度保留熵更新对步长的直接依赖，但与资格迹的高阶导数一样，省略熵梯度对权重的 Hessian 项。

各状态的生命周期不同。完整标量算法在训练开始时初始化 $h=0$、$v=0$ 和给定的 $\beta_0$；每个 episode 开始时将 $z,z_\beta,u$ 清零，保留权重 $w$、步长参数 $\beta$、敏感度 $h$ 和元增量幅度估计 $v$。其中 $u$ 约束当前资格迹涉及的状态，episode 结束后这些状态不再进入新轨迹；$h$ 则追踪仍被保留的权重如何依赖步长，不应随资格迹一起清空。下面的无归一化代码没有 $u,v$，但保持相同的 $h$ 生命周期。

线性 critic 特例的完整核心；固定步长时敏感度可与有限差分逐项比较，在线更新步长时使用局部近似。

```python
@dataclass
class LinearMetatrace:
    """Scalar, critic-only, unnormalized Metatrace core (no entropy term).

    U(w,x)=V(w,x)=w*x. For mu=0 and fixed data, h is exact dw/d(log alpha).
    With online beta updates, h is the local sensitivity approximation.
    """
    w: float = 0.0
    beta: float = math.log(0.1)
    mu: float = 0.01
    gamma: float = 0.9
    lam: float = 0.8
    z: float = 0.0
    h: float = 0.0
    z_beta: float = 0.0

    def step(self, x, reward, next_x, terminal=False):
        gamma = 0.0 if terminal else self.gamma
        delta = reward + gamma * self.w * next_x - self.w * x
        # Old h belongs to the weights that generated the current prediction.
        old_h = self.h
        self.z = self.gamma * self.lam * self.z + x
        self.z_beta = self.gamma * self.lam * self.z_beta + x * old_h
        self.beta += self.mu * delta * self.z_beta
        alpha = math.exp(self.beta)
        delta_derivative = gamma * next_x - x
        self.h = old_h + alpha * self.z * (delta + delta_derivative * old_h)
        self.w += alpha * delta * self.z
        result = dict(w=self.w, h=self.h, alpha=alpha, delta=delta,
                      z=self.z, z_beta=self.z_beta)
        if terminal:
            # Genuine episodic boundary; parameters and step-size persist.
            self.z = self.z_beta = 0.0
        return result
```

对应的交互实现是 [extended_adaptation/metatrace.py](../implementations/extended_adaptation/metatrace.py)。它在可观察剩余时间的短链上执行线性 critic 与 softmax actor：ac_terms 在旧权重上同时算出 δ、∇U 与 ∇δ；step 先用旧 h 形成元资格，再调整并归一化步长，最后同步生成新 h 和新权重。真实终止后清空 z、zβ、u，下一回合继承 w、β、h、v。其独立测量是冻结贪心策略的回报，训练动作仍按随机策略抽样。

<a id="experiment-extended-metatrace"></a>

### 实验：实验 · 元参数确实变化，也可能没有带来可见的行为收益

Metatrace 学出的步长与固定步长不同，是否就意味着最终策略更好？

**环境与可用信息。** 五位置、12 步有限时域链；输入为位置 one-hot 加剩余步数，终点奖励 1，其余 −0.02。价值函数线性，策略为 softmax，18 个参数全零初始化。

**设置。** 种子 0–4，各 1200 个环境转移。归一化 scalar Metatrace 初始 α=0.03、元步长 0.01、γ=1、λ=0.8，使用 U=V+0.5 log π，熵系数为零。固定 AC 采用相同 U、迹与初始 α，但不做元更新。

**检验的机制。** 权重迹负责延迟信用；元资格迹与敏感度负责步长的历史影响。真实终止后权重迹与元迹清零，步长及参数敏感度跨回合保留。这里实现的是 scalar 归一化变体，不是每个参数一个独立步长。

**测量。** 主图是冻结贪心回报。原始日志另记 step_size 和 meta_sensitivity_norm。先确认元状态实际变化，再检查行为指标是否真的改善；不要用元参数在变化替代性能证据。

```bash
python3 implementations/extended_adaptation/metatrace.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/extended-metatrace/curves.svg)

横轴：environment_steps。纵轴：frozen_greedy_return。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 从第 60 步到末尾，两种方法五种子的贪心回报均为 0.94。Metatrace 末尾步长约为 0.0847–0.1082，明显不同于固定 0.03，但当前行为指标已经饱和，不能分辨优势。

**结论边界。** 短链、线性 critic、单个全局元步长；没有 vector/mixed 版本，也没有长时非平稳控制。局部敏感度近似不等于完整非线性策略训练历史的精确梯度。

**继续实验。** 先增加随机策略价值与样本累计奖励指标，再考虑更长延迟或奖励尺度变化。若只换一个更困难任务而不保留同骨干对照，仍无法判断元历史是不是收益来源。

[源码](../implementations/extended_adaptation/metatrace.py) · [逐种子记录](https://yingwen.io/crl-code/results/extended-metatrace/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/extended-metatrace/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/extended-metatrace/curves.json)

<a id="research-openmind-metaoptimize"></a>

## MetaOptimize：今天的误差如何评价过去的步长

调节学习率需要一个评价标准。梯度幅度可以用于归一化，却不能单独说明过去的步长是否改善了后续学习。Sharifnassab、Salehkaleybar 与 Sutton 的 MetaOptimize（ICML 2025）将这个问题写成未来损失的折扣和，再构造只使用已到达数据的元更新。

$$
F_t^{\gamma_m}=(1-\gamma_m)\sum_{\tau>t}\gamma_m^{\tau-t-1}f_\tau(w_\tau),\qquad x_{t+1}=\operatorname{Alg}_{\rm base}(x_t,\nabla f_t(w_t),\beta_t)
$$

这里将原文的元目标折扣 $\gamma$ 改记 $\gamma_m\in[0,1)$，与环境回报折扣分开。$x$ 包含权重及优化器内部状态；$\beta$ 可通过 $\alpha_i=\exp\beta_i$ 决定各参数块的步长。

直接在时刻 t 对未来目标求导需要尚未发生的数据。后向视角让每个损失在真正出现时，评价此前各次元参数选择的影响。交换项的到达时间解决因果可计算性，但元参数在此期间也会变化，所以有限元步长下不能声称与预知未来的算法轨迹完全相同。

$$
\mathcal H_\tau=(1-\gamma_m)\sum_{t<\tau}\gamma_m^{\tau-t-1}\frac{\mathrm d w_\tau}{\mathrm d\beta_t},\qquad \widehat\nabla_\beta F_\tau=\mathcal H_\tau^\top\nabla f_\tau(w_\tau)
$$

这是原文式 (6)–(7) 的因果代理。它汇集不同历史时刻的敏感度，不等于把所有 $\beta_t$ 当成同一个固定参数。前后向更新的接近性需小元步长等条件。

手算一个独立的局部例子：$w_0=0$，训练损失为 $(w-1)^2/2$，$\alpha=e^\beta=0.1$，于是 $w_1=0.1$、$\partial w_1/\partial\beta=0.1$。后来仍预测目标 1 时，评价梯度为 $-0.9$，对旧 $\beta$ 的导数为 $-0.09$；若后来目标变为 $-1$，导数则为 $0.11$。同样的旧梯度幅度，因后续目标不同而获得相反的步长评价。这个单次链式法则示例不是完整 MetaOptimize 轨迹。

| 原文中的计算选择 | 省略或保留的路径 |
| --- | --- |
| 完整框架 | 联合追踪内层状态、元优化器状态和元敏感度的动态。 |
| 2×2 近似 | 删除敏感度状态自身相关的 Jacobian 行列块。 |
| L 近似 | 进一步删除内层状态对元更新的部分反馈。 |
| Hessian-free 变体 | 对指定优化器及导数路径作简化；不能把所有变体都称作精确二阶元梯度。 |

论文包含非平稳 CIFAR100：十个顺序任务、每样本一次、batch size 为一、切换不通知优化器且不重置权重；同时含图像与语言建模实验。它为在线步长适应提供证据，但不是 actor–critic 控制回报证据。ImageNet 中分块步长未优于标量版，也应保留。作者实现见 sabersalehk/MetaOptimize；读取具体 optimizer 类与配置，不将框架名当成唯一算法。

与本章 IDBD、Metatrace 对照时，先比较所评价的损失、敏感度路径与更新时序；与 Intentional Updates 对照时，再比较“学习全局尺度”与“按本次输出变化反解尺度”。二者可组合，但组合后的收益需要单独验证。

- 思考：若元折扣变小，哪些过去步长的影响更快退出评价？这与缩短环境回报目标有什么不同？
- 研究练习：固定数据和内层更新，分别消融敏感度反馈、分块尺度和元目标时域；记录累计损失、元状态内存、每步耗时与任务变化后的恢复。

<a id="lesson-meta-rl"></a>

## 6. Meta-gradient RL：根据后续表现选择学习目标

discount 和 $\lambda$ 影响学习目标的时间尺度与偏差—方差权衡。Meta-gradient RL 将这些选择作为内层学习的元参数 $\eta$：先用一段经验更新 agent，再用后续经验和固定评价约定计算元损失。外层并不知道一个监督标签意义上的“正确 $\gamma$”，而是评价由该 $\gamma$ 产生的参数更新。

$$
G_t^\lambda=R_{t+1}+\gamma\left[(1-\lambda)v_w(S_{t+1})+\lambda G_{t+1}^\lambda\right]
$$

在有限 rollout 内从末端向前计算，末端 $G$ 由 bootstrap 值或真正 terminal 的零值初始化。固定 $w$、轨迹与末端值，可以单独推导 return 对超参数的直接导数。

$$
\begin{aligned}\frac{\partial G_t^\lambda}{\partial\gamma}&=(1-\lambda)v_{t+1}+\lambda G_{t+1}^\lambda+\gamma\lambda\frac{\partial G_{t+1}^\lambda}{\partial\gamma},\\\frac{\partial G_t^\lambda}{\partial\lambda}&=\gamma\left(G_{t+1}^\lambda-v_{t+1}+\lambda\frac{\partial G_{t+1}^\lambda}{\partial\lambda}\right).\end{aligned}
$$

改变 $\gamma$ 或 $\lambda$，既影响当前组合，也影响后续 return。若用 sigmoid 将它们限制在 $(0,1)$，链式法则还需乘 sigmoid 导数。若末端值依赖 $\eta$，边界导数也必须保留。

将 return 导数代入 actor/critic 的更新函数 $f$，即可得到直接导数 $\partial f/\partial\eta$，再经敏感度递推影响后续参数。原 Meta-gradient RL 使用衰减 trace 降低成本：把完整的 Jacobian 传播替换为一个标量衰减。

$$
\begin{aligned}H^+&=\left(I+\frac{\partial f}{\partial w}\right)H+\frac{\partial f}{\partial\eta},\\z^+&=\mu z+\frac{\partial f}{\partial\eta},\qquad\eta^+=\eta-\beta\,z^{+\top}\nabla_{w^+}J_{\rm eval}.\end{aligned}
$$

第二行以标量衰减近似完整 Jacobian 传播；$\mu=0$ 只考虑当前更新。这一节的 $z$ 是元敏感度近似，不是前文的权重资格迹。在线变化的 $\eta$ 又引入局部近似，因此它不等于固定元参数下的全历史总导数。

**算法：算法伪代码**

1. 保存 agent 参数 $w$、元参数 $\eta$ 和元敏感度 $z$。
1. 内层 rollout：
  1. 用 $\eta$ 构造 return，计算更新 $f$。
  1. 计算 $\partial f/\partial\eta$，推进完整 $H$ 或近似 $z$。
  1. $w^+\leftarrow w+f$
1. 后续评价 rollout：
  1. 用固定 $\eta_{\rm eval}$ 计算 $J_{\rm eval}$。
  1. 沿 $w^+$ 对 $\eta$ 的影响更新元参数。
1. 继续交互；元梯度截断与状态重置按预先声明的协议执行。

相邻 rollout 通常不是真正独立样本，验证经验也受策略变化影响。研究时需要报告数据复用、是否用新策略采样、元梯度截断长度，以及是否将策略采样路径计入估计。训练误差降低本身不足以说明学习规则更适合长期控制。

$\lambda$-return 对 $\gamma$ 和 $\lambda$ 的直接导数：固定边界值，执行后向递推，并由有限差分检验。

```python
def lambda_return_sensitivity(rewards, next_values, bootstrap, gamma=0.9, lam=0.8):
    """Lambda return and exact direct derivatives for fixed values/boundary."""
    if len(rewards) != len(next_values):
        raise ValueError("one next-state value is required per reward")
    g, dg, dl = bootstrap, 0.0, 0.0
    for reward, value in reversed(list(zip(rewards, next_values))):
        mixture = (1.0 - lam) * value + lam * g
        new_dg = mixture + gamma * lam * dg
        new_dl = gamma * (g - value + lam * dl)
        g = reward + gamma * mixture
        dg, dl = new_dg, new_dl
    return g, dg, dl
```

实际 [meta_gradient.py](../implementations/extended_adaptation/meta_gradient.py) 将这条链落在五状态随机游走预测上。它固定策略、令 $\lambda=\sigma(\beta)$，用一条 train 轨迹计算 return；旧价值仅作为固定 bootstrap 数值，$\lambda$ 路径仍保留。内层对均方误差做一次步长 0.1 的 SGD，随后在另一条新 validation 轨迹上评价 $w^+$，外层固定 $\lambda_{\rm eval}=1$。反传到 $\beta$ 后，程序把新权重与元参数从计算图中分离，下一批从这些数值继续。

这里的截断窗口恰好是一批内层更新。每批两条轨迹都耗费真实转移；每条最多 24 步，超时的尾值继续 bootstrap，真实端点尾值为零。若 validation 被截断，其尾值来自 $w^+$ 但在外层停止梯度。核对元梯度时，要固定中心运行的这份尾值；重新计算扰动后的尾值会加入实现有意删去的路径。这段代码验证固定策略的目标适应，策略依赖的采样项在这个环境协议中没有出现。

<a id="experiment-extended-meta_gradient"></a>

### 实验：实验 · 对验证轨迹求元梯度，能否学出有用的 λ？

元梯度能通过一次价值更新传回 λ，但它在当前预算内是否改变了实际学习过程？

**环境与可用信息。** 五个非终止状态的一维对称随机游走，从中间状态开始。到右端奖励 1，左端奖励 0，γ=1。真值依次为 1/6 到 5/6。每条轨迹最多 24 个转移；预算截断继续自举，真实终点不自举。

**设置。** 五种子各运行 1200 个训练批。每批独立采集训练与验证两条轨迹。价值初始为零，内层 SGD 步长 0.1；λ=sigmoid(β) 初始为 0.8，元步长 0.02，β 限于 [−4,4]。对照保留同样采样与元梯度计算，只将元更新率设为零。

**检验的机制。** 训练轨迹构造 λ-return 并产生一次可微价值更新。验证轨迹用固定 λ=1 的目标评价新参数，再将导数传回 β。旧价值自举停止梯度，但 λ 对训练回报的依赖保留。

**测量。** 主图使用相对解析真值的冻结 MSE；横轴是训练批，不是环境步。五次运行实际分别使用约 2.06–2.14 万个转移。原始日志的 learned_lambda 与 meta_gradient 是判断元学习是否真正起作用的必要补充。

```bash
python3 implementations/extended_adaptation/meta_gradient.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/extended-meta_gradient/curves.svg)

横轴：training_batches。纵轴：frozen_value_mse。每种方法 1200 training_batches；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 第 1200 批，元学习 MSE 为 0.00714235，固定 λ 为 0.00714212，几乎重合。学出的 λ 仅为 0.80008–0.80042。这证明实现产生了非零元更新，但不能证明当前设置学出了有用的新目标。

**结论边界。** 只反传一批内的更新，没有 actor、跨批长元历史或 IMPALA 系统。两条轨迹意味着额外环境查询。微小带符号的最终差值不是效能证据。

**继续实验。** 先检查 β 的有限差分导数，再预先设计不同奖励噪声与延迟的任务族。扫描元步长时同时报告固定 λ 网格基线，检验自适应是否胜过直接选一个常数。

[源码](../implementations/extended_adaptation/meta_gradient.py) · [逐种子记录](https://yingwen.io/crl-code/results/extended-meta_gradient/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/extended-meta_gradient/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/extended-meta_gradient/curves.json)

<a id="lesson-output-steps"></a>

## 7. 在线元梯度与预设更新规则的边界

步长随数据变化，不足以判断一个方法是否在学习更新规则。IDBD、TIDBD 和 Metatrace 保存参数对过去步长的敏感度，用后续误差评价这种影响，再改变步长参数。Stream-X 与 Intentional Updates 则按预先规定的归一化或输出变化公式计算本次更新，不估计这条元敏感度。它们属于流式更新稳定化方法，可以作为元学习的对照或与之组合。

| 机制 | 保留的历史 | 本次更新回答的问题 |
| --- | --- | --- |
| 资格迹 | 过去输出梯度的衰减统计 | 当前误差应分给哪些过去预测或决策？ |
| Stream-X / Intentional 的尺度控制 | 梯度、误差或完整增量的幅度统计 | 在给定方向上，本次参数或输出允许改变多少？ |
| 在线元梯度 | 内层参数对学习规则参数的敏感度 | 过去的规则选择怎样影响了后续误差，应如何调整？ |

流式学习章完整推导 ObGD、StreamingOptimizer 与 Intentional-AC，并保留实现、手算及边界检查。时间信用分配章讨论误差在时间上的分配。这里继续研究学习规则本身的适应；这三个问题可以在同一个 agent 中共同出现，但不是同一种算法分类。

<a id="lesson-maml"></a>

## 8. MAML：学习一个适合少量更新的初始化

深入阅读：换到允许跨任务训练和重置的设定，推导适合少量更新的初始化。

MAML 换了问题设定：从任务分布 $p(T)$ 采样任务，每个任务允许收集少量适应经验，再评价适应后策略。外层变量是共享初始化 $\theta$。每个任务从同一初值复制出 $w_i$，执行若干梯度更新；外层评价这些更新后的表现。直接平均各任务未适应的损失，则是不同的多任务学习目标。

$$
\begin{aligned}w_i&=\theta-\alpha\nabla_\theta L_i^{\rm train}(\theta),\\\mathcal J(\theta)&=\mathbb E_{T_i\sim p(T)}\left[L_i^{\rm test}(w_i)\right],\\\nabla_\theta\mathcal J_i&=\left(I-\alpha\nabla^2_\theta L_i^{\rm train}\right)^\top\nabla_{w_i}L_i^{\rm test}.\end{aligned}
$$

train/test 在这里指同一任务内的适应与评价数据，不是把最终测试任务反复拿来调外层参数。两阶段数据区分能避免只优化对同一批样本的过拟合。

多步版本重复 $w^{k+1}=w^k-\alpha\nabla L^k$，元梯度经过每一步 Jacobian 的乘积。FOMAML 将曲率修正近似为单位矩阵，仍然使用内层适应后的评价梯度。Reptile 则令 $\theta\leftarrow\theta+\varepsilon(w_i-\theta)$，将初始化向任务内训练后的参数移动；两者都避免显式二阶导数，但并非同一更新规则。

**算法：算法伪代码**

1. 每轮外层更新：
  1. 从训练任务分布采样任务 $T_i$。
  1. 对每个任务复制初始化：$w_i\leftarrow\theta$。
    1. 收集适应数据 $D_{\rm train}$，估计策略梯度损失。
    1. 执行 $K$ 次内层更新，保留计算图。
    1. 用适应后的策略收集 $D_{\rm test}$，计算外层损失。
  1. 合并外层梯度，沿内层更新反传到 $\theta$。
1. 测试新任务：从 $\theta$ 出发，按相同适应预算更新 $w$，不更新 $\theta$。

在 RL 中，对策略梯度 surrogate 再做自动微分，并不自动得到期望元回报的正确高阶导数。适应轨迹的分布依赖 $\theta$，评价轨迹的分布依赖 $w_i$；数值相同的 surrogate 可能具有不同的高阶导数。因而需要明确 likelihood-ratio、baseline 和 stop-gradient 的位置。二次函数实验可以隔离优化器链式法则，下面的随机例子再分析采样路径。

解析二次任务：完整 MAML 与 FO 近似的数值差别可直接检查。

```python
def scalar_maml(initialization, train_target, test_target, curvature=2.0, alpha=0.1):
    """One task, deterministic quadratic losses, exact MAML and FO approximation.

    L_train = curvature/2*(w-train_target)^2; L_test = (w-test_target)^2/2.
    This illustrates optimizer differentiation, not policy-gradient sampling.
    """
    adapted = initialization - alpha * curvature * (initialization - train_target)
    outer_gradient = adapted - test_target
    exact = (1.0 - alpha * curvature) * outer_gradient
    first_order = outer_gradient
    return 0.5 * outer_gradient ** 2, exact, first_order, adapted
```

$$
\nabla_\theta\mathbb E_{D\sim p_\theta}[J(U_\theta(D))]=\mathbb E_D\left[\left.\nabla_\theta J(U_\theta(D))\right|_D+J(U_\theta(D))\nabla_\theta\log p_\theta(D)\right]
$$

第一项在给定适应数据下求导，第二项反映初值改变了适应数据的分布。若 $J$ 还通过新策略采样估计，它自身也需要正确的策略梯度；sampled reward 的数值通常没有可直接反传的动作导数。

下面穷举一个 Bernoulli 动作，比较只保留更新路径与同时保留采样 score 项。由于动作只有两种，期望可以精确求和，有限差分不含 Monte Carlo 噪声。DiCE 等工作进一步处理随机计算图的高阶梯度；它们解决的是估计器问题，而不是替代 MAML 的内外层目标。

固定数据微分遗漏了什么：用一个能精确穷举的随机适应过程检查两条梯度路径。

```python
def stochastic_adaptation(theta, alpha=0.2, target=1.0):
    """Enumerate a Bernoulli sample's TWO gradient paths exactly.

    a~Bernoulli(sigmoid(theta)); w=(1-alpha)*theta+alpha*a.
    The toy outer loss is (w-target)^2/2. No Monte Carlo noise is present.
    """
    probability = 1.0 / (1.0 + math.exp(-theta))
    expected_loss = path_gradient = score_gradient = 0.0
    for action, mass in [(0.0, 1.0 - probability), (1.0, probability)]:
        adapted = (1.0 - alpha) * theta + alpha * action
        loss = 0.5 * (adapted - target) ** 2
        expected_loss += mass * loss
        path_gradient += mass * (adapted - target) * (1.0 - alpha)
        score_gradient += mass * loss * (action - probability)
    return expected_loss, path_gradient, score_gradient
```

<a id="lesson-context"></a>

## 9. RL² 与 PEARL：在活动状态中进行任务推断

深入阅读：比较把适应保存在递归活动或任务后验中的方法，以及它们与参数更新的关系。

另一种办法是把“学到了哪个任务”放进活动状态或潜变量，而把推断规则留在固定网络权重里。RL² 将前一动作、奖励和 episode 结束信号与当前观测输入 RNN；内层适应发生在 hidden state 的变化，外层用 RL 训练整条 trial 的累计表现。若一个 trial 包含同一任务的多个 episode，hidden state 在 episode 间保留，在新的任务 trial 开始才重置；这是任务协议的一部分。

$$
h_{t+1}=f_\theta(h_t,O_{t+1},A_t,R_{t+1},d_{t+1}),\qquad A_{t+1}\sim\pi_\theta(\cdot\mid h_{t+1})
$$

测试时可以冻结 $\theta$，而 $h$ 仍随经验改变。活动变化支持探索、记忆和适应，也可以与参数学习组合；两种变化应分别记录。

**算法：算法伪代码**

1. RL² 元训练：
  1. 采样训练任务，初始化 trial 活动状态 $h$。
  1. 在同一任务运行多个 episode；环境重置时保留跨 episode 的 $h$。
  1. 将已到达的动作、奖励和 done 输入 RNN，累计 trial 回报。
  1. 用序列梯度更新共享参数 $\theta$。
1. 测试：冻结 $\theta$；新任务 trial 才初始化 $h$，适应协议与训练一致。

PEARL 引入任务潜变量 $z$，从转移集合 $c$ 推断 $q_\phi(z\mid c)$，策略和价值函数均以 $z$ 为条件。其 context encoder 将各条转移编码成高斯因子后相乘，因此不依赖 context 内转移的排列。这适合任务在 context 范围内固定的设定；混合变化前后的证据，可能推断出并不存在的折中任务。

$$
q_\phi(z\mid c)\propto\prod_{\xi\in c}\Psi_\phi(z\mid\xi),\qquad \sigma^{-2}=\sum_j\sigma_j^{-2},\qquad \mu=\sigma^2\sum_j\frac{\mu_j}{\sigma_j^2}
$$

右式是单坐标高斯因子乘积，其他坐标可逐项计算；作者实现还通过 KL 正则将编码分布约束到先验附近。不要在已有因子定义之外又无说明地重复乘同一个先验。

$$
\begin{aligned}L_Q&=\mathbb E\left[\left(Q_\omega(s,a,z)-\operatorname{sg}(y)\right)^2\right],\\y&=r+\gamma\,\bar V(s',z),\\L_{\rm enc}&=L_Q+\beta_{\rm KL}\,D_{\rm KL}\bigl(q_\phi(z\mid c)\Vert p(z)\bigr).\end{aligned}
$$

原实现使用 SAC 的 $V$ 网络版本：慢更新网络提供目标价值，terminal 时停止 bootstrap。编码器由 critic 误差与 KL 训练；原作者代码在 policy 通路对 $z$ detach，使该通路不直接更新 encoder。

**算法：算法伪代码**

1. PEARL 元训练：
  1. 为训练任务维护 replay；从 context 推断 $q_\phi(z\mid c)$ 并采样 $z$。
  1. 用 $\pi(a\mid s,z)$ 交互，将经验加入该任务 replay/context。
  1. 采样 critic 训练转移和 context，生成可重参数化的 $z$。
  1. 更新 $Q$ 与 encoder（包含 KL）；更新条件策略和价值/target 网络。
1. 元测试：冻结网络参数，从先验采样 $z$，交互收集 context。
  1. 重新推断 posterior 并采样 $z$；不输入真实任务参数。

解析例子采用 $y=z+\epsilon$，先验 $z\sim\mathcal N(0,1)$，独立噪声方差为 1。没有证据时后验仍为先验；一条 $y=1$ 后为 $\mathcal N(0.5,0.5)$；三条相同观测后为 $\mathcal N(0.75,0.25)$。即使模型参数不变，条件推断仍改变预测。这说明 context adaptation 的性质；PEARL 另外需要学习推断网络及控制策略。

已知高斯任务的可运行 context 推断；明确展示先验与观测因子的不同角色。

```python
def gaussian_context(observations, noise_variance=1.0, prior_mean=0.0,
                     prior_variance=1.0):
    """Exact Bayesian latent-task inference for y_i=z+Gaussian noise.

    It demonstrates context-driven adaptation with NO parameter gradient.
    It is not PEARL's learned encoder or SAC training loop.
    """
    if noise_variance <= 0 or prior_variance <= 0:
        raise ValueError("variances must be positive")
    precision = 1.0 / prior_variance + len(observations) / noise_variance
    natural_mean = prior_mean / prior_variance + sum(observations) / noise_variance
    return natural_mean / precision, 1.0 / precision
```

<a id="lesson-learned-rules"></a>

## 10. 学习完整更新规则：LPG、DiscoRL 与外层优化

TD 的目标由人写成“奖励加上下一个价值”。策略梯度的更新方向由人写成“动作对数概率梯度乘以优势”。规则发现进一步问：能否让另一个网络根据经验，生成指导这些输出学习的目标？外部奖励仍用来评价行为，但 agent 内部可以同时学习由这些目标定义的预测。

![多个环境与智能体的轨迹进入共同规则网络，规则产生目标以更新智能体，后续回报回到规则。部署时锁住规则参数，新智能体继续更新。](https://yingwen.io/crl-figures/concept-meta-discovery.svg)

紫色网络生成训练目标，蓝色网络按目标学习；橙色回路用后续回报改进规则。锁表示规则参数固定，其活动状态仍可变化。右下曲线标记评价环节。流程依据 DiscoRL 原文与作者接口独立绘制。

单一标量步长在给定方向下缩放更新，逐参数步长还会改变更新的几何方向。更一般的 learned update rule 让元网络读取奖励、动作、agent 输出、预测误差或训练统计量，产生目标、损失系数或更新信号。agent 参数 $w$ 按内层流程变化，元网络 $\eta$ 根据训练轨迹的外层表现优化。状态构造、预测目标与学习规则可以联合学习，但分别具有自己的状态和梯度路径。

$$
\begin{aligned}(y_t^{\rm target},m_{t+1})&=U_\eta(\xi_{t:t+k},p_{w_t},m_t),\\w_{t+1}&=w_t-\alpha\nabla_w\ell\bigl(p_w,y_t^{\rm target}\bigr),\\\eta^+&=\eta-\beta\nabla_\eta\,\mathcal J(w_{t+K}).\end{aligned}
$$

这是一个通用接口，具体方法的损失仍有差异。$m$ 为元学习器活动状态，$p_w$ 为 agent 的策略或辅助预测。target 对内层 $w$ 和外层 $\eta$ 的导数可以采用不同约定；整块 detach 会同时删除外层希望学习的通路。

目标生成器为什么能从行为回报学到东西？先在一个更新内把 $w$、轨迹和规则活动固定，记生成目标为 $\widehat y_\eta$。内层对 $w$ 求偏导时保持目标不动，外层再对这一偏导的结果求导：

$$
\frac{\partial w^+}{\partial\eta}=-\alpha\frac{\partial^2\ell(p_w,\widehat y)}{\partial w\,\partial\widehat y}\frac{\partial\widehat y_\eta}{\partial\eta},\qquad\nabla_\eta J=\left(\frac{\partial w^+}{\partial\eta}\right)^\top\nabla_{w^+}J.
$$

这是一次更新的直接路径；多步时仍需传播旧 $H$、规则活动及允许通过的其他状态导数。外层若是期望回报，$\nabla_{w^+}J$ 需由相应的策略梯度估计，而不是对环境给出的奖励数值直接反传。

以分类输出的 $D_{\rm KL}(\widehat p\Vert p_w)$ 为例，若直接把 logits 记为 $a$，内层梯度为 $p_a-\widehat p$，于是 $a^+=a-\alpha(p_a-\widehat p_\eta)$，固定旧 $a$ 时有 $\partial a^+/\partial\eta=\alpha\,\partial\widehat p_\eta/\partial\eta$。后续评价因此能告诉生成器，怎样移动目标分布有助于改善更新后的策略。最小化内层 KL 只说明 agent 更接近所给目标；目标是否有用要由这条后续评价路径决定。

[LPG 原文 §3](https://arxiv.org/html/2007.08794#S3)采用更特定的接口：反向 LSTM 读取一段已到达轨迹，输出策略 score 的标量权重 $\widehat\pi$ 与预测分布目标 $\widehat y$。其内层方向是 $\widehat\pi\nabla_w\log\pi_w-\alpha_y\nabla_wD_{\rm KL}(y_w\Vert\widehat y)$。因此 LPG 的 $\widehat\pi$ 不是目标策略分布，辅助 KL 的方向也与下面 DiscoRL 的目标匹配式不同。外层用更新后策略的回报改进规则，实际在有限更新窗口内反传；一个 agent 的生命可跨越多个窗口。

**算法：算法伪代码**

1. 元训练：
  1. 采样训练环境与 agent 初值，初始化 learner state。
  1. 重复内层更新：
    1. 采集交互数据，运行 $U_\eta$，得到目标或损失。
    1. 更新 $w$，推进 optimizer 与 meta state。
  1. 用外部回报目标评价训练过程，按指定估计器更新 $\eta$。
1. 部署固定规则：冻结 $\eta$，新 agent 仍按规则持续更新 $w$。
1. 若部署时也更新 $\eta$，另行定义其数据、目标和预算。

DiscoRL（Nature 2025）的 agent 除了策略 $\pi$ 和动作价值 $q$，还输出观测条件向量 $y$ 与动作条件向量 $z$。后两者的语义由学习过程形成。作者实现的元网络读取轨迹、奖励、终止信号、agent 输出及价值统计，产生 $\widehat\pi,\widehat y,\widehat z$。例如同一画面下，不同动作的 $z$ 可以不同；这为规则提供了比单个状态价值更丰富的输入。

| 程序中的对象 | 谁改变它 | 下次更新怎样使用 |
| --- | --- | --- |
| agent 参数 w | 内层优化器 | 产生动作分布、价值和辅助预测 |
| 规则参数 η | 发现阶段的外层优化 | 规定从轨迹生成什么目标 |
| meta_state 中的 rnn_state | 规则前向运行 | 保留规则网络使用的活动状态 |
| adv_ema_state / td_ema_state | 运行均值更新 | 调整输入统计的尺度 |
| target_params | agent 参数的移动平均 | 提供较慢变化的输出副本 |

源码把 $\pi,y,z$ 的 logits 转成分类分布，并让 agent 输出接近规则给出的目标。省略时间平均与权重后，可将这部分损失写成下面的形式。辅助的下一步策略预测和固定语义的价值学习另有分支；因此“发现规则”仍在设计好的接口与损失族内进行。

$$
\ell_{\rm rule}=c_\pi D_{\rm KL}(\widehat\pi\Vert\pi_w)+c_yD_{\rm KL}(\widehat y\Vert y_w)+c_zD_{\rm KL}(\widehat z\Vert z_w(\cdot,A_t)).
$$

这里用同一字母表示 logits 对应的分布，$c_\pi,c_y,c_z$ 是损失系数。$z$ 的损失取实际动作 $A_t$ 对应的输出。作者代码用 backprop 标志控制是否切断目标分支；做外层微分时要沿实现逐项跟踪，不能把目标在所有层级一律视为常数。

具体看 agent.py 的 learner_step：程序先生成 meta_out，再把它作为独立参数传给内层损失函数。因此计算本次 $\nabla_w\ell$ 时，已生成目标保持固定；外层对整个更新程序求导时，仍可通过目标生成路径影响 $\eta$。在非元训练模式，backprop=False 再显式停止这些目标的梯度。区分这两个求导层级，才能理解为什么“内层固定目标”与“外层学习目标生成器”可以同时成立。

这一点也说明它与 GVF 的关系。GVF 先规定累积信号、延续规则和目标策略，再学习答案；DiscoRL 的辅助坐标先由有利于学习的目标塑造，之后再分析它们与环境事件的关系。要把某一坐标称为可检验的预测知识，还需给出它所预测的量与行为条件。两种路线可以互相启发：一条重视明确的问题语义，一条扩大可发现的内部学习信号。

作者发布的仓库提供发现和评价两个 notebook，以及 Disco103 的规则参数。读代码可以依次查看 update_rules/disco.py 的输入与状态、agent_loss 的目标匹配、agent_loss_no_meta 的价值分支，再看 agent.py 怎样实际执行更新。先运行 Catch 的小规模示例，可观察规则固定时 agent 的学习；完整论文结果还涉及更大的环境和计算配置。

比较规则时，先固定新 agent 可用的交互步数、架构与计算，再分别报告规则发现的成本。跨环境迁移检验规则能否用于没参与发现的环境；长生命试验则检验同一 agent 遇到变化后还能否学习。前者成功后，仍可以继续问：奖励尺度改变、过去情境回归、训练长度远超发现时长度，是否改变了规则的作用？

<a id="lesson-agent-coupling"></a>

## 从调步长到发现问题：完整 agent 怎样评价学习规则

同一个持续运行的 agent 可以学习状态、许多预测、技能及其模型，再用这些知识选择行动。元学习接收的评价取决于其中哪个消费者。一个预测器的误差下降，回答的是这个问题更容易被当前表示拟合；只有当它改变了后续状态、规划或控制，才能沿相应路径评价它对行为的作用。这也使“学习预测问题”和“学习预测问题的答案”成为两个不同层次。

在 [GVF 章](gvf.md) 中，问题由累积信号、延续规则和目标策略规定。若问题本身写成 $q_\eta=(C_\eta,\Gamma_\eta,\pi^q_\eta)$，答案参数为 $\nu$，共享表示参数为 $\theta$，则元参数可以经“问题 → 预测更新 → 表示 → 后续控制”获得评价。只最小化当前预测误差可能鼓励选择恒为零的信号；固定问题的可预测性与选择问题的用途需要不同的评价。

$$
\begin{aligned}(\theta^+,\nu^+)&=F_\eta(\theta,\nu,D_{\rm pred}),\\\nabla_\eta J_{\rm control}&=\left(\frac{\partial\theta^+}{\partial\eta}\right)^\top\nabla_{\theta^+}J_{\rm control}+\left(\frac{\partial\nu^+}{\partial\eta}\right)^\top\nabla_{\nu^+}J_{\rm control}.\end{aligned}
$$

此式先固定数据并假设外层无其他直接依赖。若预测头既不供控制器读取，也不改变共享表示，右侧对应路径为零；“多学一个预测”不会仅因名字而获得控制信用。把策略采样重新开放，还需前述分布导数。

改变 $C,\Gamma$ 或目标策略会改变 GVF 的真值语义，因此旧答案、资格迹与模型的版本也要一起考虑。若保留问题不变，只学习其步长或损失权重，改变的则是求答案的规则。发现新问题时可以先固定消费者，比较候选启用前后的后续评价；把“问题更有用”与“分配了更多计算”分开。这里给出的是可检验的组合设计，不是 IDBD 或 DiscoRL 已经完成问题发现的结论。

[Option 章](options.md)还有类似区分。直接用行为回报更新技能内部策略或终止参数，是扩展策略的普通控制学习；学习“技能内部应优化什么子目标”或“以什么步长训练技能”，再经训练后的技能表现评价这些选择，才出现本章的外层。终止规则改变会同时改变技能数据、持续时间和后果模型，固定旧轨迹求导只能覆盖其中允许通过的路径。

如果完整 agent 采用 [平均奖励准则](average.md)，外层应继续按原始时间衡量奖励率，或明确有限窗口累计奖励作为其代理。学习一个辅助 discount 不应顺便改变外层控制目标。对随机持续时间 $\tau$ 的技能，在相应的稳态决策分布及有限正平均时长下，奖励率为 $\mathbb E[R_o]/\mathbb E[\tau]$；按技能次数平均奖励会给不同的偏好。内层奖励率估计也属于学习器状态，对差分误差 $\delta=R-\bar r+v'-v$ 求元导数时，是否保留 $-\partial\bar r/\partial\eta$ 必须声明。

[知识保留](retention.md)与[可塑性](plasticity.md)进一步决定外层该看到多长的学习过程。只读旧知识的损失可以评价保持，却不能测量以后再学习的速度；要评价可塑性，需要让副本执行规定预算的新学习，再评价其结果。若更新规则把受损的梯度通路缩放得再大，也可能没有有效方向可用。元步长、特征替换和旧知识约束因而可以组合，但各自改变的状态、后续表现和计算成本要分别测量。

从这些消费者回看方法，比较应沿四个维度进行：可改变的是步长、目标、初始化还是活动；后续评价来自预测、控制还是再次学习；导数保留了哪些状态、采样和时间路径；部署时哪些量继续变化。MAML 的初值、RL² 的活动、IDBD 的步长和 learned rule 的目标生成器可以放进同一框架，同时保留各自的数据权限与研究问题。[完整架构](architectures.md#lesson-cycle-meta)把局部预测步长放进一条可运行的交互循环，再沿链式法则说明：若改用下游控制评价，还必须保留经过模型与规划的哪些导数。

<a id="lesson-search-control"></a>

## 把规划计算用于哪里：元学习也可以选择经验

深入阅读：把可学习对象扩展为模型查询分布，分析规划计算如何服务未来学习。

模型已经能够生成经验时，下一项选择是从哪个状态开始模拟。把所有状态平均采样，可能反复更新已经准确的值，也可能过多使用有偏模型。Meta-Gradient Search Control（Burega 等，2024）学习这个查询分布：先用模型经验更新价值，再用真实经验给出继续学习的方向。

$$
\begin{aligned}d_\eta(s)&=\frac{\exp\eta_s}{\sum_j\exp\eta_j},\\\bar w(\eta)&=w+\alpha\,\mathbb E_{\tilde s\sim d_\eta,\,\tilde a\sim\pi,\,(\tilde r,\tilde s')\sim\hat P}[\Delta(\tilde s,\tilde a,\tilde r,\tilde s';w)],\\\widehat w&=\operatorname{sg}\!\left(\bar w+\alpha\Delta(s,a,r,s';\bar w)\right),\\L(\eta)&=\|\widehat w-\bar w(\eta)\|^2.\end{aligned}
$$

$\Delta$ 是不含步长的半梯度 Q-learning 增量，$\hat P$ 是学习模型；最后一行目标中的转移来自真实环境。对本次元更新，$\widehat w$ 固定。它给出一个可获得的学习方向，代替未知的最优参数。

用两个坐标手算这一选择。设两种模拟更新方向分别为 $(1,0)$ 和 $(0,1)$，当前权重为零，步长 0.2；查询第一种的概率为 $p$，期望更新得到 $\bar w=(0.2p,0.2(1-p))$。在 $p=0.5$ 时，真实经验把目标推进到 $(0.3,0.1)$。固定此目标后，$dL/dp=-0.08$；若 $p=\sigma(\eta)$，则 $dL/d\eta=-0.02$。下降元损失会把更多查询分给第一种更新。

这个例子解释了“学习计算分配”的含义：Q-learning 公式可以保持不变，经验选择本身已经可学。原工作在会切换奖励的迷宫中考察这种方法。真实经验稀少或模型偏差改变时，元目标提供的方向也会变化；需要同时观察模型误差、查询分布与实际累计收益。

<a id="lesson-frontier"></a>

## 11. 前沿问题：元目标短视、规则表示与发现预算

深入阅读：进一步比较元目标的时间范围、规则表示和发现成本。

除了环境中的奖励延迟，还存在学习过程自身的时间范围。一次更新可能当前收益很小，却改变了以后能够学到什么。两种时间不能混用：一个是行动以后多久得到奖励，另一个是这次更新以后还会执行多少学习。

第一条问题是元目标的时间范围。只沿 $K$ 步内层更新求导，容易偏好立即降低误差的更新，而忽视更久以后的表示学习或探索。Bootstrapped Meta-Learning（ICLR 2022）在可微展开得到 $w_K$ 后，执行额外内层更新以产生未来目标 $\widehat w$，再让 $w_K$ 在选定的距离下接近这个目标。计算图仍只穿过前 $K$ 步，目标分支停止梯度；额外未来学习需要计算与数据，但不必保存同等长度的反传图。

$$
w_K=F_\eta^{(K)}(w_0),\qquad \widehat w=\operatorname{sg}\bigl(T^{(L)}(w_K)\bigr),\qquad J_{\rm BMG}=D\bigl(\widehat w,w_K\bigr)
$$

$T^{(L)}$ 表示构造目标的额外更新，可与 $F_\eta$ 不同；$D$ 可以是参数距离，也可以是策略分布距离。这样改变的是外层目标的几何与监督范围；梯度仍只穿过前 $K$ 步更新，不穿过未来目标分支。目标如果自身很差，匹配它也未必有益。

第二条问题是外层应该怎样搜索规则。RLC 2025 的 How Should We Meta-Learn Reinforcement Learning Algorithms? 比较黑盒进化学习、神经蒸馏、符号蒸馏和 LLM 代码发现等方法，涉及更新规则、优化器和策略损失等不同对象。其意义在于把“规则的表示形式”与“搜索规则的方法”拆开：可解释公式不一定必须由人工写出，神经更新规则也不一定必须靠元梯度训练。

Osband 的 Planning to Learn（2026 年预印本）从分类中的损失选择讨论学习预算。若按 softmax 采样标签，正确类别的概率为 $p=\sigma(z)$，提高这个期望准确率的梯度权重是 $p(1-p)$，交叉熵的权重是 $1-p$；这里评价的是采样正确率，与取最大概率标签的 top-1 指标不同。两种损失对困难样本分配的更新量不同。论文用有限学习范围内的未来误差积分构造损失，研究先投入学习、以后才获得收益的取舍。这里的实验是分类；对无预定终点的 CRL，一个值得继续研究的问题是如何评价长期有用、短期收益很低的更新。

| 研究条线 | 最小实验设计 | 应回答的机制问题 |
| --- | --- | --- |
| 在线元梯度：IDBD → TIDBD → Metatrace | 固定特征/小网络，比较完整低维敏感度、对角近似和元资格迹；再引入未通知漂移。 | 收益来自步长幅度、跨时间信用，还是额外归一化？同等内存下能否保留？ |
| 短视与目标几何：meta-gradient → BMG | 固定内层算法与总交互，分别增加反传窗口和停止梯度的目标生成窗口。 | 更多未来信息与更多反传计算哪一个更重要？目标滞后造成什么偏差？ |
| 规则发现：LPG → DiscoRL；RLC 2025 方法比较 | 把元训练环境、规则参数量、种子和外层预算独立控制，保留未见任务与长生命测试。 | 学到的是可迁移更新机制，还是某组训练环境/奖励尺度/训练长度的特化？ |

这些实验不应只比较最后一个 checkpoint 的分数。需要同时记录全过程回报、漂移后恢复曲线、参数与步长范数、资格迹统计、每步延迟、持久内存和外层总预算。对算法发现尤其要区分：发现一次规则的成本、训练一个新 agent 的成本，以及部署期间是否继续改规则。原始代码能帮助恢复这些训练协议，不能用单个优化器函数替代整个实验定义。

<a id="lesson-example"></a>

## 12. 两个可手算的元学习过程

算例 A：$w_0=0$，两个训练标签都是 2，$\alpha=0.1$，$\beta=\log(0.1)$，外层标签为 3。内层得到 $w_1=0.2$、$w_2=0.38$。由 $H_0=0$，先有 $H_1=0.2$，再有 $H_2=0.9\times0.2+0.1\times1.8=0.36$。外层损失为 $(0.38-3)^2/2=3.4322$，元梯度为 $(0.38-3)\times0.36=-0.9432$。梯度下降因而增大 $\beta$，使后续学习更快地向正目标移动。

| 两步超梯度方案 | 第二步敏感度 $H_2$ | 外层对 $\beta$ 的梯度 | 改变了什么 |
| --- | --- | --- | --- |
| 精确固定 $\beta$ 多步递推 | 0.36 | −0.9432 | 传播了第一次更新经过第二次更新的影响。 |
| 只保留当前更新，H 从零开始 | 0.18 | −0.4716 | 完全删掉过去影响。 |
| 忽略曲率、直接累计 | 0.38 | −0.9956 | 保留过去，但漏掉第二次更新的 0.9 收缩。 |

算例 B：MAML 初值 $\theta=0$，内层损失 $(w-2)^2$，步长 0.1，外层损失 $(w-3)^2/2$。内层曲率为 2，适应后 $w=0.4$。完整元梯度为 $(1-0.1\times2)(0.4-3)=-2.08$；FOMAML 为 $-2.6$。二者在这个例子中同向但不等；曲率更大时连符号也可能不同。有限差分应扰动初值 $\theta$，再重新执行内层更新。

若评价目标自身可以通过缩短时间尺度改变，元学习器有可能把 $\gamma$ 推向零，以回避长时误差。这说明评价目标应先于训练代理目标定义；降低一个可变目标的数值，不直接意味着改善了原定任务。

<a id="lesson-code"></a>

## 13. 可运行代码与控制变量实验

开篇步长算例、规划分配的导数与三张示意图共用一份计算源。下载 /crl-code/figures/meta-learning.mjs 后，可以直接导入 oneStep 与 planningAllocation 修改数字；仓库测试用有限差分独立检查两个导数。

在下载脚本的目录运行；只计算确定性教学问题。

```bash
node --input-type=module <<'JS'
import {oneStep, planningAllocation} from './meta-learning.mjs';
console.log(oneStep(0.2, 1)); // 后续目标为 1：步长导数为 -0.8
console.log(oneStep(0.2, 0)); // 后续目标为 0：步长导数为 +0.2
console.log(planningAllocation(0)); // 均匀查询时的元梯度为 -0.02
JS
```

脚本只依赖 Python 3.10+ 标准库；直接运行即可打印全部手算对应值。

```sh
python3 examples/state_meta_lab.py meta
python3 examples/state_meta_lab.py test
```

基础实验依次给出两步超梯度、IDBD/TIDBD、MAML 与 FO、随机采样的额外梯度项、$\lambda$-return 导数和 context 后验。固定数据的实验用有限差分检验导数；随机例子穷举全部动作以检验期望的两条求导路径。TD 退化关系与边界测试检查更新时序。这里采用低维解析问题，便于观察每一个内部量。

Metatrace 机制实验与配套回归测试；标准库即可运行

```sh
python3 examples/meta_frontier_lab.py metatrace
python3 examples/meta_frontier_lab.py test
```

进阶脚本中的 Metatrace 是无熵正则、无归一化的线性 critic 特例，用来检查元信用的时间索引和敏感度传播。它不包含完整神经网络与环境。复现原论文回报，需要使用作者训练入口、网络、环境版本与统计协议。配套脚本中的 IntentionalStep 属于流式章的输出尺度实验，不作为本章的元学习方法。

完整机制实验驱动代码：每一种“适应”都有独立可观察的对象。

```python
def meta_demo():
    loss, grad, trace = fixed_beta_training(math.log(0.1), [2.0, 2.0], 3.0)
    print("two inner steps (w,H):", trace)
    print("validation loss, d/dbeta:", loss, grad)
    w, b, h = [0.0], [math.log(0.1)], [0.0]
    for target in [2.0, 2.0, -2.0]:
        w, b, h, delta = idbd_step(w, b, h, [1.0], target)
        print("IDBD: target,w,alpha,H:", target, w[0], math.exp(b[0]), h[0])
    result = tidbd_step([0.0, 0.0], [math.log(0.1)] * 2, [0.0] * 2,
                       [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], 1.0)
    print("TIDBD: weights, traces:", result[0], result[3])
    print("MAML: loss, exact, FO, adapted:", scalar_maml(0.0, 2.0, 3.0))
    print("stochastic adaptation loss,path,score:", stochastic_adaptation(0.3))
    print("lambda return, d/dgamma, d/dlambda:",
          lambda_return_sensitivity([1, 2], [0.3, 0.4], 0.4))
    for context in [[], [1.0], [1.0, 1.0, 1.0]]:
        print("context posterior (mean,var):", context, gaussian_context(context))
    print("scope: analytic mechanism tests, not meta-RL benchmark performance")
```

- 修改一：把两个训练标签改为 (2,−2)，保留外层标签 3；记录精确、多步截断与直接累计的符号和幅度，检查“连续梯度同向”的直觉何时失效。
- 修改二：改变 MAML 的内层曲率与步长，让 α×曲率大于 1；解释完整元梯度的符号为什么可能与 FO 相反。
- 修改三：将 context 后半部分的目标从 +1 改为 −1；累积全部证据与只使用近期证据的估计为何不同？这需要推断协议的改变，不是简单调大 actor 步长。
- 修改四：先用完全相同的数据核验导数，再开放策略影响采样。若性能改变，分别检查估计器偏差、探索差异与计算预算，不将它们混为元梯度精度。

<a id="lesson-branches"></a>

## 14. 算法条线与 CRL 的适用条件

| 路线 | 可继续推进的 CRL 问题 | 不能默认继承的假设 |
| --- | --- | --- |
| IDBD/TIDBD | 不同特征、预测问题和环境阶段需要不同适应速度。 | 线性、近似对角敏感度、采样与半梯度稳定条件；仍有元步长。 |
| Metatrace | 带资格迹的在线控制，显式追踪历史步长影响。 | 半梯度外层目标、忽略资格迹高阶导数和策略采样路径；标量/向量/混合变体不同。 |
| Meta-gradient RL | 在线适应回报尺度、λ、学习目标或更新系数。 | 外层评价可信、元梯度时序正确、不会靠改变目标投机。 |
| MAML/FOMAML | 过去任务是否能提供更易适应的参数起点？ | 独立任务采样、可回到共同初值、有清晰适应预算。 |
| RL²/PEARL | 上下文推断能否在固定参数下识别变化？ | 训练任务分布、context 内任务稳定、task/trial reset 权限。 |
| Learned update rules | 能否学到跨环境、跨生命长度的通用学习机制？ | 外层预算、分布外泛化、时间尺度与持久状态稳定性。 |
| 持续元学习 | 外层规则本身也要一生更新，且不能遗忘旧适应能力。 | 不能用无限保留过去所有训练轨迹作为默认实现。 |

这些方法处理不同的适应对象，没有单一的替代关系。上下文推断器可以使用在线自适应步长训练，预测网络的 GVF 问题参数可以接受元梯度，结构化 RNN 也可以成为 learned update rule 的记忆。组合方法时，需要重新定义内层与外层状态、数据来源和导数截断边界，以明确各模块之间传递哪些信息和梯度。

原始实现中，MAML-RL 的 maml_vpg.py 连接适应前后 surrogate 与内层计算图；PEARL 的 agent.py 实现 context、posterior 与潜变量采样，sac.py 组织 critic/encoder 更新；DiscoRL 的 disco.py 定义更新规则接口与持久元状态。检查这些接口后，再恢复配置中的环境、采样量和外层预算，才能比较完整算法。

<a id="research-trac-scale-adaptation"></a>

## TRAC：参考位移与多时间尺度的在线适应

IDBD 追踪步长如何影响后续误差；TRAC 的适应对象是参数相对参考点的位移尺度。它将基础优化器与一维在线 tuner 组合，让近期证据决定离初始化多远。参考点是正则化锚，不是已证明安全的策略。

$$
\theta_{t+1}=\theta_{\rm ref}+s_{t+1}(\theta^{\rm Base}_{t+1}-\theta_{\rm ref}),\quad s_{t+1}=\sum_j s_{t+1,j}
$$

论文 Algorithm 1 的参数化。只有 s∈[0,1] 才是凸组合，范围之外会外推；典型实测范围不能被写成普适约束。

$$
z_t=\langle g_t,\theta_t-\theta_{\rm ref}\rangle,\quad v_{t,j}=\beta_j^2v_{t-1,j}+z_t^2,\quad u_{t,j}=\beta_j u_{t-1,j}-z_t
$$

采用损失下降梯度约定。不同 β 保留不同时间范围；正内积表示增大位移的局部损失代价，负内积支持增大位移。回报上升梯度需改符号。

$$
s_{t+1,j}=\frac{\epsilon}{\operatorname{erfi}(1/\sqrt2)}\operatorname{erfi}\!\left(\frac{u_{t,j}}{\sqrt{2v_{t,j}}+\epsilon}\right)
$$

论文 tuner 的决策形式。稳定 erfi、初期尺度与裁剪都是实现条件；任意 sigmoid 不能替代后仍称原算法。

若参考点为 0，基础点为 2，尺度 0.2 时部署点为 0.4，尺度 1 时为 2。同一方向产生不同偏移。此时 $g=+1$ 表示继续增大位移局部有害，$g=-1$ 则相反。这个例子解释反馈符号，并不证明任意 RL 梯度下都选择最优正则。

原作者 trac.py 从已缩放的部署点重建未缩放位移，加入基础 optimizer 增量，再重新缩放；inner product 使用重建方向并含非负尺度处理。它与抽象 Algorithm 1 的参数化需分别核对。基础动量、参考参数和 tuner 统计都需跨日志分段保留，复现应固定代码版本。

- 先冻结梯度序列，打印基础/部署位移及各 u/v/s；检查零梯度、符号和状态保存恢复。
- 同一 Adam 与学习率下比较固定尺度、固定 L2、单 tuner 与多个 tuner；匹配 warm-start 和调参预算。
- 未通知变化后同时测适应、旧功能与初期代价。凸在线损失遗憾不等于非凸、策略依赖采样的深度 RL 终生回报保证。

<a id="research-rule-discovery-resources"></a>

## 算法表示、规则搜索与部署成本

RLC 2025 的 How Should We Meta-Learn Reinforcement Learning Algorithms? 将学习的组件与发现它的方法分开比较。更新规则可以是神经函数、符号公式或代码；它可以通过进化搜索、蒸馏、代码提案等过程产生。表示的可解释性不能代替发现预算与部署效果。

$$
\eta^*=\operatorname*{arg\,max}_{\eta\in\mathcal U}\mathbb E_{E\sim\mathcal D_{\rm train}}[J_H(\mathcal A_\eta,E)],\qquad J_H=\mathbb E[\sum_{t=0}^{H-1}R_{t+1}]
$$

统一比较接口，不是论文全部方法共用的具体损失。U 是规则类，H 是外层生命长度；输入权限、状态与规则容量改变都会改变 U。

| 层次 | 应固定或记录 | 混淆 |
| --- | --- | --- |
| 发现 | 训练环境、模拟步、搜索次数、设备时间 | 用更多搜索发现的规则却不报告搜索资源。 |
| 部署 | 每步延迟、规则网络与持久 meta-state | 只计 agent 权重，不计学习规则本身。 |
| 泛化 | 未见任务、奖励尺度、漂移与生命长度 | 看完整测试未来再选规则，称作在线适应。 |

最小研究可只学习 critic 更新中的一个小函数，固定 actor、网络、数据权限及内层预算，比较不同发现方式。封存规则后，测试十倍生命长度和未通知变化。这样能识别收益来自规则结构、搜索方法还是外部资源。作者 AlexGoldie/learn-rl-algorithms 按发现方法与评价流程组织实现。

**算法：算法发现比较流程**

1. 注册被替换组件、可读变量、训练环境与发现预算
1. 独立记录每种搜索方法的全部提案和失败
1. 封存规则及外层选择，不读取测试未来轨迹
1. 新 agent 按相同初始化运行，报告全程收益和部署成本

若规则本身也要在单生命内持续学习，还必须定义在线外层目标、旧/新经验分配、外层状态与内层 optimizer 的相容性。跨任务规则迁移是一个研究轴，部署智能体终生修改规则是另一个闭环，不能从前者直接宣称后者已解决。

<a id="lesson-check"></a>

## 15. 诊断、自测与研究起点

| 看到的症状 | 要检查的具体量 |
| --- | --- |
| 元参数不动 | H 是否被清零/截断；target 是否把 η 路径一起 detach；第一步 H=0 是否被误认为 bug。 |
| 步长突然爆炸 | β、α、特征尺度、元梯度范数、clip 触发频率；不能只画平滑后的 return。 |
| 测试适应特别快 | 是否拿到 task ID/边界、重置先验、测试轨迹参与外层更新；与基线权限是否相同。 |
| context 很自信却一直错 | 环境已变化、经验高度相关、后验模型失配；小方差不等于正确推断。 |
| 外层损失下降但控制无改善 | 评价目标是否与实际回报错位；是否只改善同批样本或代理目标。 |
| 精确梯度测试过了但 RL 效果差 | 采样路径、方差、截断偏差、外层过拟合、额外算力，逐项隔离。 |

- 问：Adam 是不是就等于本页的元学习？答：Adam 的统计量改变预设更新规则，但没有在这里所定义的外层后续目标上学习这些规则参数；“有自适应状态”不是充分定义。
- 问：测试时冻结权重就不可能 Meta-RL 适应吗？答：RL² 的活动和 PEARL 的 posterior 仍能变；必须指出变化发生在何处。
- 问：把 IDBD 的 y 换成 TD target 就已经完成 TIDBD 推导吗？答：没有，还要处理 target 导数约定与资格迹，特别是 β 用 x、内层更新用 z 的区别。
- 问：有限差分能证明 RL 元梯度无偏吗？答：只能验证给定可微程序的导数；轨迹分布的导数是否被正确估计，是另一个统计问题。

可开始的研究题：固定预测问题与 feature，先比较常数步长、IDBD/TIDBD 及精确低维敏感度在未通知变化下的终生预测误差，再扩展到 actor-critic。预算应同时限制环境交互和每步计算；变化检测只用于离线统计，不给智能体任务标签。记录变化前误差、恢复积分误差、长期稳定性、步长与 trace 轨迹，并保留元学习无益或有害的情形。这样得到的是可解释的机制研究，而不只是多一个可调超参数的学习曲线。

## 本章的实验设计

外层更新、超参数搜索和基线调参都消耗预算。先锁定开发程序，再用独立运行比较表现。

设定：先用两步标量学习核元敏感度，再选择在线步长学习或跨任务适应协议。两种设定分别声明重置、任务分布与测试权限。

- 完整元梯度与固定随机路径的有限差分一致。
- 第一步零历史敏感度不被误判为实现故障。
- 测试经验不得通过未声明外层更新泄漏到方法选择。

对照：调好的固定率、简单日程与元率；完整导数、截断和停止梯度；相同任务分布及额外计算对照

记录：元目标、敏感度、步长及实际参数写入；适应前后收益和所需数据；元训练、搜索与部署开销

[具体测试规程](../docs/experiment-handbook.md#handbook-selection)

## 学习与研究衔接

不同特征可能需要不同更新速度：稳定信号应保留，正在漂移的信号需要追踪，噪声又不宜追得过快。元学习用后续表现调整这些学习参数，但短时间窗可能无法评价较早更新的长期作用。状态与技能同时改变时，还应分别在共同数据和真实交互中比较，判断学到的规则是否改善更新，以及这种改善怎样改变采样。

[分册导读](../docs/learning-route-continual-rl.md) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-meta) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=meta) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=meta)

## 从本章进入实践

[估计与适应](https://yingwen.io/zh/continual-rl/code/#practice-tracking)：旧经验越来越多时，怎样仍对新变化作出反应？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](https://yingwen.io/zh/continual-rl/research/#recent-reset-and-distill)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Step-size Optimization for Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-step-size-optimization)
- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)
- [Discovering state-of-the-art reinforcement learning algorithms](https://yingwen.io/zh/continual-rl/research/#recent-disco-rl)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Intentional Updates for Streaming Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-intentional-updates)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)
- [A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning](https://yingwen.io/zh/continual-rl/research/#recent-lambda-greedy)
- [MetaOptimize: A Framework for Optimizing Step Sizes and Other Meta-parameters](https://yingwen.io/zh/continual-rl/research/#recent-openmind-metaoptimize)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Position: Lifetime tuning is incompatible with continual reinforcement learning](https://yingwen.io/zh/continual-rl/research/#recent-lifetime-tuning)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

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

### Discovering state-of-the-art reinforcement learning algorithms

Junhyuk Oh, Gregory Farquhar, Iurii Kemaev, Dan A. Calian, Matteo Hessel, Luisa Zintgraf, Satinder Singh, Hado van Hasselt, David Silver

Nature · 2025 · 支持方法与理论

#### 研究问题

除了学习策略，能否从大量学习过程里学出更有效的 RL 更新规则？

#### 关键机制

DiscoRL 用外层优化评价执行若干内层更新后的行为表现，学习价值、策略与辅助预测之间的更新方式。被训练的对象是学习算法本身，而不仅是某个任务的策略参数。内外两层有各自的数据、时间尺度与计算预算。

#### 证据

论文报告跨环境发现更新规则与迁移到未见环境的结果，并公开配套算法实现。它展示了自动算法发现的可能性，但依赖大规模外层训练。

#### 条件与限制

外层在大量环境和设备上的搜索属于设计者侧资源，不能记作测试智能体单次生命内的自主学习。公开规则的执行成本与发现该规则的成本应分别报告。

#### 阅读与实验

画出内层参数和外层参数的更新依赖，再列出测试时哪些量被冻结。与在线 IDBD 比较时，先区分跨任务算法发现和单流步长追踪。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-09761-x)：算法发现过程、外层资源与泛化实验。
- [作者实现](https://github.com/google-deepmind/disco_rl)：配套代码与发现的更新规则。

#### 作者代码

[Google DeepMind 的论文配套仓库。](https://github.com/google-deepmind/disco_rl)

DiscoRL 配套实现与学习到的更新规则；具体训练资源以仓库说明为准。

### Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning

Ke Sun, Hongming Zhang, Jun Jin, Chao Gao, Xi Chen, Wulong Liu, Linglong Kong

ICLR 2026 · 2026 · 直接研究持续学习

#### 研究问题

快速学习新任务和整合旧知识，能否由不同学习器承担并以明确目标连接？

#### 关键机制

FAME 的快速学习器适应当前任务，元学习器整合此前知识。论文按旧策略的重要访问分布度量价值或策略变化，再据此构造减少遗忘的整合目标。自适应预热决定如何利用旧知识初始化或约束早期行为，以减少负迁移。

#### 证据

论文分析价值型和策略型版本，并在像素与连续控制任务序列中比较。作者提供官方实现，可追踪快速适应与知识整合两个阶段。

#### 条件与限制

设定要求相同状态与动作空间、已知任务边界以及额外整合计算。这里的 meta learner 主要是知识整合模块，不应因名称就当作通过长期回报反向求导的在线元梯度算法。脑机制类比也不是神经科学实验证据。

#### 阅读与实验

分别报告新任务前向迁移、旧任务保留和两个学习阶段的计算量。改变任务相似性，检验自适应预热是否确实避免有害旧知识。

#### 原文与相关入口

- [ICLR 2026 原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2230ffcd5da10015ce0c6ce588fc2936-Abstract-Conference.html)：任务边界假设、遗忘度量与快慢知识机制。
- [FAME 官方实现](https://github.com/datake/FAME)：论文链接的快速学习与知识整合代码。

#### 作者代码

[论文与仓库均注明为官方实现。](https://github.com/datake/FAME)

FAME 的价值型、策略型持续学习实验。

### Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline

Hongjoon Ahn, Jinu Hyeon, Youngmin Oh, Bosun Hwang, Taesup Moon

ICLR 2025 · 2025 · 直接研究持续学习

#### 研究问题

一个网络还能拟合新目标，为什么先前训练仍可能让它在新任务上学得更慢？

#### 关键机制

论文把任务之间的负迁移与一般可塑性损失区分开。Reset & Distill 在新任务开始时重置在线 actor 和 critic，避免旧初始化阻碍学习；随后离线蒸馏当前策略与旧专家的动作分布以整合知识。适应和保留通过不同过程实现。

#### 证据

作者在控制与游戏任务中分析负迁移，并在长 MetaWorld 序列上检验该基线。原文直接提供实现地址。

#### 条件与限制

任务边界、在线网络重置、旧专家和离线蒸馏都需要资源。它不能直接当作无边界、不能重置、禁止回放的单次生命方案。

#### 阅读与实验

除了与连续微调比较，还要与同等预算的从头训练比较。若新任务表现低于从头训练，先检查负迁移，再判断是否属于单纯容量损失。

#### 原文与相关入口

- [ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba9e3d60610f3525717665966d86e0cd-Abstract-Conference.html)：负迁移诊断、Reset & Distill 机制与边界。
- [原文代码入口](https://github.com/hongjoon0805/Reset-Distill)：论文首页提供的作者实现。

#### 作者代码

[ICLR 正式论文首页明确链接的代码。](https://github.com/hongjoon0805/Reset-Distill)

Reset & Distill 以及任务序列实验。

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

### Position: Lifetime tuning is incompatible with continual reinforcement learning

Golnaz Mesbahi, Parham Mohammad Panahi, Olya Mastikhina, Steven Tang, Martha White, Adam White

ICML 2025 Position Paper · 2025 · 评价与实验协议

#### 研究问题

如果设计者用完整未来生命反复调参，实验还在测智能体面对未知变化的能力吗？

#### 关键机制

论文限制调参可访问的生命阶段，并比较这种选择方式与利用完整生命回报挑选配置的差异。外部设计者掌握未来变化信息，可能使一个并不自适应的固定算法显得适应良好。核心改变发生在评价协议，而不是 TD 更新公式。

#### 证据

作者用持续、非平稳设置中的深度 RL 实验说明超参数选择可以改变方法比较。该工作属于立场论文，论证与示例用于推动更符合问题目标的评价。

#### 条件与限制

允许多少开发阶段经验需要按应用规定，不存在由该论文推出的普适固定比例。仅限制时间前缀也不能替代独立测试种子和计算预算控制。

#### 阅读与实验

对同一配置集合分别按开发前缀和完整生命选择超参数，再在独立测试生命上比较。报告两种选择使用了哪些未来信息。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/mesbahi25a.html)：调参协议、论证与示例实验。

### Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning

Aneesh Muppidi, Zhiyu Zhang, Heng Yang

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

未知环境变化时间和速度时，怎样在线决定参数应离参考初始化多远？

#### 关键机制

TRAC 在基础优化器外维护一组具有不同遗忘时间尺度的一维 tuner，根据梯度与参考方向的内积调整参数位移尺度。它通过数据驱动的缩放联系到正则化，而不是对未来任务回报进行长窗口元梯度反传。

#### 证据

作者在 Procgen、Atari 与 Gym Control 变化序列中比较适应与可塑性，并分析在线凸优化对该设计的启发。

#### 条件与限制

凸在线优化中的遗憾理论不等于非凸、策略依赖采样的深度 RL 收敛定理。“parameter-free”不表示没有基础学习率、初始化、时间尺度网格、warm-start 或协议选择。

#### 阅读与实验

记录 tuner 尺度、距参考点的位移、旧分布干扰与变化后适应。用相同基础优化器比较固定尺度、单时间尺度和多时间尺度。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。
- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

#### 作者代码

[作者项目页与仓库均明确标为官方实现。](https://github.com/ComputationalRobotics/TRAC)

trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。

### How Should We Meta-Learn Reinforcement Learning Algorithms?

Alexander David Goldie, Zilin Wang, Jaron Cohen, Jakob Foerster, Shimon Whiteson

RLC 2025 / RLJ · 2025 · 评价与实验协议

#### 研究问题

算法表示、发现算法的方法和测试智能体的学习成本，应该如何独立比较？

#### 关键机制

对 RL 流程的不同组件进行算法发现，比较黑盒学习、神经/符号蒸馏与 LLM 代码提案。学习器的表示形式和搜索过程分开定义，才能识别泛化、可解释性和成本之间的取舍。

#### 证据

论文直接比较元训练、元测试、样本成本、训练时间与可解释性，作者代码按发现方法和评价入口组织。

#### 条件与限制

跨环境发现规则主要发生在设计者侧，不等于运行智能体已能终生修改规则。论文的训练任务和预算范围不支持所有算法发现方法的普适排序。

#### 阅读与实验

固定被学习组件、输入权限、元训练数据和发现预算；封存规则后再检验未见环境、长生命与未通知漂移。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)：比较对象、元训练/测试与多维成本。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：作者与正式会议收录。

#### 作者代码

[论文提供、仓库标为官方的作者实现。](https://github.com/AlexGoldie/learn-rl-algorithms)

learning_algorithms 中各发现方法与独立 evaluation 流程。

### A Greedy Approach to Adapting the Trace Parameter for Temporal Difference Learning

Martha White, Adam White

arXiv预印本 · 2016 · 支持方法与理论

#### 研究问题

不同状态的预测可靠性不同，固定λ是否浪费了多步信用？

#### 关键机制

将下一处bootstrap选择写成局部偏差平方与回报方差的折中，得到$λ=b^2/(b^2+\operatorname{Var}(G))$。完整λ-greedy还用在线预测器估计回报均值和二阶矩。

#### 证据

原文给出状态相关λ的目标、增量算法和多个预测设置的实验。信用章仅核对已知统计量下的局部最优与变量λ恒等式。

#### 条件与限制

局部贪心目标不是整条轨迹的联合最优。逼近误差、统计滞后和非平稳性会影响λ估计；辅助资源需要计入比较。

#### 阅读与实验

先让噪声方差变化，再让bootstrap可靠性变化。比较固定λ、已知统计参照和在线估计，分别观察目标偏差与适应速度。

#### 原文与相关入口

- [作者原文](https://arxiv.org/html/1607.00446)：局部目标、状态λ、均值／二阶矩预测与完整算法。

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


<a id="chapter-code"></a>

## 下载与运行

多步超梯度、IDBD/TIDBD、标量 MAML 与解析 context；meta_frontier_lab.py 提供 Metatrace 线性特例。

[下载 state_meta_lab.py](../examples/state_meta_lab.py)

```sh
python3 examples/state_meta_lab.py meta
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · Reinforcement Learning: An Introduction, 2nd edition](http://incompleteideas.net/book/the-book-2nd.html)：第 9 章的表示与泛化、第 11 章的离策略目标，为逐坐标步长与 TD 元学习提供基础。

- [配套章节：时间信用分配](credit.md)：多步回报、资格迹与序列信用；区分权重更新的信用与元参数影响的信用。

- [配套章节：流式学习](streaming.md#lesson-output-steps)：数据与计算协议、Stream-X 和 Intentional 的完整尺度推导、实现与对照。

- [Sutton · Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta](https://cdn.aaai.org/AAAI/1992/AAAI92-027.pdf)：原始 IDBD、逐参数学习率和漂移目标实验。其对角敏感度近似与现代完整元梯度需要区分。

- [Kearney 等：TIDBD 2018 版本](https://arxiv.org/abs/1804.03334)：本页代码具体对应 Algorithm 1 的 semi-gradient、累积迹版本。

- [Kearney 等：Learning Feature Relevance Through Step Size Adaptation](https://arxiv.org/abs/1903.03252)：后续 TIDBD 文献；比较时必须逐式区分版本与归一化/梯度选择。

- [Vivek Veeriah 的 IDBD 实验实现](https://github.com/vivekveeriah/IncrementalDeltaBarDelta)：TIDBD 合作者提供的 IDBD 实验代码；不是 Sutton 1992 官方发布，也不是 TIDBD 完整复现仓库。

- [Young、Wang、Taylor：Metatrace Actor-Critic（IJCAI 2019）](https://www.ijcai.org/proceedings/2019/0581.pdf)：时变权重的元目标、scalar/vector/mixed Metatrace；重点对照元资格迹与敏感度中的完整 TD-error 导数。

- [Metatrace 扩展原文](https://arxiv.org/abs/1805.04514)：含归一化与熵项细节；配套脚本为独立实现的线性特例。

- [Xu、van Hasselt、Silver：Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：内层更新、后续评价、精确敏感度与衰减 trace 近似的原文。

- [Finn、Abbeel、Levine：MAML](https://proceedings.mlr.press/v70/finn17a.html)：明确任务分布、适应前后目标与可微内层更新。

- [Foerster 等：DiCE](https://proceedings.mlr.press/v80/foerster18a.html)：随机计算图的高阶梯度：普通一阶 surrogate 反复微分可能遗漏或产生错误项。

- [MAML-RL 作者源码](https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/sandbox/rocky/tf/algos/maml_vpg.py)：固定版本的策略梯度适应实现；旧 TensorFlow/rllab 栈，不能假设现代依赖直接兼容。

- [Duan 等：RL²](https://arxiv.org/abs/1611.02779)：循环活动承载快速学习，外层 RL 学习整个 trial 内的适应行为。

- [Rakelly 等：PEARL](https://proceedings.mlr.press/v97/rakelly19a.html)：概率 context、off-policy 数据与任务推断；注意任务稳定范围与 reset 权限。

- [PEARL 作者源码：agent.py](https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/torch/sac/agent.py)：高斯因子乘积、reparameterized sample、clear_z 与 policy 输入的 detach 实现。

- [PEARL 作者源码：sac.py](https://github.com/katerakelly/oyster/blob/44e20fddf181d8ca3852bdf9b6927d6b8c6f48fc/rlkit/torch/sac/sac.py)：原实现的 Q、V、策略及 encoder/KL 更新时序。

- [Oh 等：Discovering Reinforcement Learning Algorithms](https://arxiv.org/abs/2007.08794)：将学习目标与更新机制本身作为可学习对象的原始工作。

- [Flennerhag 等：Bootstrapped Meta-Learning（ICLR 2022）](https://arxiv.org/abs/2109.04504)：用未来学习产生停止梯度的目标，区分元目标的有效时间范围与反传展开长度。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2025/papers/Paper218.html)：作者与正式会议收录。

- [How Should We Meta-Learn Reinforcement Learning Algorithms? · 作者实现](https://github.com/AlexGoldie/learn-rl-algorithms)：learning_algorithms 中各发现方法与独立 evaluation 流程。 论文提供、仓库标为官方的作者实现。

- [Discovering state-of-the-art reinforcement learning algorithms（Nature 2025）](https://www.nature.com/articles/s41586-025-09761-x)：DiscoRL 的原论文；区分环境间规则迁移和单一 agent 的持续学习。

- [DiscoRL 作者仓库](https://github.com/google-deepmind/disco_rl)：元训练、评价入口与配置；区分规则发现预算和新 agent 的训练预算。

- [DiscoRL 固定版本更新规则源码](https://github.com/google-deepmind/disco_rl/blob/9059a29f7121d60948f25ef165e08e050e9399c8/disco_rl/update_rules/disco.py)：重点读 agent 输出契约、规则输入、辅助预测目标与持久 meta-state。

- [DiscoRL 固定版本内层训练源码](https://github.com/google-deepmind/disco_rl/blob/9059a29f7121d60948f25ef165e08e050e9399c8/disco_rl/agent.py)：learner_step 先生成 meta_out，再对以它为独立输入的损失求权重偏导；区别内层固定目标与外层目标路径。

- [元学习图解的可运行计算源](https://yingwen.io/crl-code/figures/meta-learning.mjs)：一步步长元梯度、固定目标的规划分配导数和图示生成；与正文数值逐项对应。

- [Burega、Martin、Kapeluck、Bowling：Meta-Gradient Search Control（2024）](https://arxiv.org/html/2406.19561v1)：第 3 节从模型查询分布推到停止梯度的元目标；第 4 节给出奖励变化的迷宫评价。

- [Osband：Planning to Learn（2026，预印本）](https://arxiv.org/html/2610.03667v1)：第 2–3 节区分当前梯度收益与剩余学习预算；实验对象是分类，不是持续控制。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。

- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning · 作者实现](https://github.com/ComputationalRobotics/TRAC)：trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。 作者项目页与仓库均明确标为官方实现。

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_218.pdf)：比较对象、元训练/测试与多维成本。

- [Sharifnassab, Salehkaleybar & Sutton · MetaOptimize · ICML 2025](https://proceedings.mlr.press/v267/sharifnassab25a.html)：正式会议记录；原文第 2–6 节定义因果元目标、联合敏感度与计算近似，第 7 节区分静态及非平稳监督实验。

- [MetaOptimize · 2025 定稿原文](https://arxiv.org/html/2402.02342v6)：采用 v6 而非仍只含早期实验的 2024 版本；元目标折扣不能与环境折扣混用。

- [MetaOptimize · 作者实现](https://github.com/sabersalehk/MetaOptimize)：各 base/meta optimizer 组合对应不同实现；仍有元步长和配置选择。
