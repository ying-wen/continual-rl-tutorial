# 平均奖励：奖励率、差分价值与持续控制

为什么长期奖励率与折扣回报可能选择不同策略？去掉目标中的折扣后，预测、控制与规划各需要多解决什么问题？

## 本章内容

- 在同一小型 MDP 中手算折扣最优与平均奖励最优策略的反转；区分固定起点评价、稳态评价和完整学习过程。
- 从稳态分布与 Poisson 方程推导奖励率、差分价值、策略改善及平均奖励策略梯度。
- 独立对应同策略与异策略 Differential TD、多状态 Differential Q、已知模型 RVI 和学习模型 Differential Dyna 的更新与诊断。
- 严格区分目标折扣、中心化折扣 critic 和资格迹衰减；识别有限折扣带来的价值及 actor 方向偏差。
- 解释非线性逼近、流式更新、慢混合与 option 随机时长增加的困难，不将表格保证外推到深网。

## 阅读这条论证

先比较同一个世界中的折扣总量与长期奖励率，再用首尾价值的相消说明差分方程为什么成立。从这条方程构造样本更新，分别处理预测、控制与规划。技能改变决策间隔，资格迹改变反馈传播；它们都必须保留原始时间上的奖励目标。

[比较两种目标](average.md#lesson-discount-choice) → [定义奖励率与差分价值](average.md#lesson-existence) → [从方程到预测更新](average.md#lesson-derive) → [改善长期收益](average.md#lesson-gain-improvement) → [由模型求解](average.md#lesson-rvi) → [计入技能时长](average.md#lesson-duration) → [核对代码与测量](average.md#lesson-code) → [分析共享表示](average.md#lesson-nonlinear)

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [持续控制：比较策略与学习智能体](control.md)：明确在相同资源和交互条件下比较什么。
- [持续控制与平均奖励](../foundations/approximation/average-control.md)：掌握奖励率、差分价值和基本预测/控制更新。


### 状态与马尔可夫性

给定当前状态和动作，下一步奖励与状态的条件分布不再依赖更早历史；观测不充分时，需要先构造带记忆的 agent state。

### 策略评估与控制

策略评估求固定策略 $\pi$ 的价值；控制则寻找收益更高的策略。控制样本可由探索策略 $b$ 产生，因此样本奖励均值未必是最优策略的奖励率。

### TD 与半梯度

用当前估计的下一状态价值补齐未知未来，把它视作本步固定目标再更新当前估计；这不是对整个 Bellman residual 求全梯度。

<a id="problem-definition"></a>

## 本章的问题定义

持续运行没有自然终点，关注每个原始时间步的长期收益；预测与控制分别求指定策略奖励率或最优奖励率。

### 给定条件与符号

- 固定MDP、原始步计时、有界奖励和可用动作。
- 预测时给定目标策略；控制时给定探索、访问与更新预算。

### 需要求解的对象

指定策略的奖励率与差分价值，或使奖励率尽可能大的策略及相应相对动作价值。

### 信息与数据权限

数据为 $(S_t,A_t,R_{t+1},S_{t+1})$；离策略预测另需实际行为 $b(A_t\mid S_t)$，目标策略为 $\pi$。技能更新另记录原始持续时间 $\tau$。

$$
g_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right],\qquad g_* =\sup_{\pi\in\Pi}g_\pi
$$

$T$ 是原始环境步数，$\Pi$ 是允许的策略集合。预测只估计固定 $\pi$ 的 $g_\pi$；控制才比较 $g_*$。差分价值 $h_\pi$ 描述去掉奖励率后的相对收益，只在加常数意义下确定。

### 成立条件与解的含义

- 本章基础预测先假设奖励率不依赖初始状态，例如有限不可约策略链；控制需对应算法的通信和访问条件。
- 差分方程与相对值的锚定需要明确；深网、非平稳世界和随机技能不能直接继承表格收敛结论。

判断准则：小MDP上核对奖励率和Poisson/Bellman方程残差；比较相对价值差而非任意偏移；技能按原始时间计收益并检验时长扣除。

### 适用边界

- 有限窗口平均不等于已证明存在的无限时域奖励率。
- 奖励中心化参照量不自动等于精确平均奖励。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 改变评价目标 · [价值预测与资格迹](value.md)：将折扣累计量改成长期奖励率，预测对象随之变为奖励率与差分价值。

- 改变评价目标 · [通用价值函数与预测知识](gvf.md)：差分预测需要同时识别信号的长期率和相对值；将累积信号中心化并令延续为1，还须处理解的存在性和常数自由度，不能直接继承折扣GVF的唯一解条件。

- 改变评价目标 · [Options 与技能发现](options.md)：相对options章的折扣控制，本章采用每个原始步的长期奖励率；同样的随机时长技能需扣除奖励率乘时长，而不使用折扣尾值。

- 组合不同学习问题 · [时间信用分配与资格迹](credit.md)：资格迹混合无折扣的多步差分目标；它改变反馈的传播，不把奖励率目标变成折扣回报。

- 组合不同学习问题 · [持续控制与学习智能体比较](control.md)：固定策略奖励率是局部控制工具；完整学习器仍需计入有限寿命的适应与探索成本。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

无折扣总奖励随时间增长，普通价值无法直接作为有限相对量；技能还改变了决策间隔。

### 本章的核心思路

估计共同增长的奖励率与剩余相对价值，并保持原始时间单位；联合TD更新与参考函数锚定是两种实现。

1. [从长期率得到可采样的方程](average.md#lesson-derive)：沿轨迹累加Poisson方程时，中间价值相消，剩下奖励率和有限首尾差；再以单条转移估计方程残差，用同一旧误差更新率与价值。

2. [由策略改善推到控制更新](average.md#lesson-gain-improvement)：新策略的平均优势给出奖励率差；最优方程中的max因此对应奖励率控制，Differential Q用样本逐步求解它。

3. [从模型计算同一后果](average.md#lesson-rvi)：RVI用参考状态固定价值原点，Dyna以学到的转移和奖励替代实际样本；模型误差与规划计算需另外记录。

4. [将机会成本按技能时长计算](average.md#lesson-duration)：因为一个option消耗多个原始步，理想半Markov方程扣除奖励率乘实际时长；本章更新变体先用旧期望长度估计扣除并归一化，再更新长度，不能任意换成随机时长分母。

结论与条件：表格差分TD/Q与RVI有各自的链结构、步长及覆盖条件；本章深度骨架不提供普遍收敛或重置免费保证。

### 相关方法改变了什么

- Differential TD/Q：由同一TD误差联合更新奖励率与价值。

- RVI：用参考函数提供相对价值的锚定。

- 奖励中心化：处理共同奖励偏移；折扣联合更新的中心不应预先当成奖励率。


<a id="lesson-setting"></a>

## 1 · 每单位时间的收益

前面的折扣控制用一串加权奖励比较策略。若智能体长期运行，我们也可以问：采用这条策略，每个原始时间步平均获得多少奖励？这把评价对象从某个起点的折扣总量改为长期收益率。下面先在同一个有限、平稳的 MDP 中比较两种目标，再研究怎样从连续经验中学习这个率，以及怎样据此改善策略。

$$
g_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

$g_\pi$ 是每个原始环境步的平均收益。本章假设它与初始状态无关；例如有限不可约策略链满足这一点。多个互不连通的常返类可能具有不同奖励率，因而需要更一般的、依赖状态的表述。

未折扣的无限奖励和通常发散；平均奖励目标使用时间平均，不能仅由折扣回报中代入 $\gamma=1$ 得到。与此同时，没有自然终点的环境仍可采用折扣目标。目标函数、是否重放数据、是否允许外部重置，是三个需要分别指定的选择。

| 量 | 含义 | 在算法中何时改变 |
| --- | --- | --- |
| $g_\pi$ / $g_*$ | 固定策略 / 最优策略的真实奖励率 | 问题的性质，未知 |
| $\bar g$ | 奖励率估计 | 按指定样本或模型备份更新 |
| $h(s)$ / $q(s,a)$ | 差分价值，又称 bias | 根据 TD 误差更新 |
| $\alpha$、$\eta$ | 价值步长、奖励率相对步长 | 本页固定；非平稳场景可研究自适应 |

奖励率衡量长期单位时间的表现。差分价值通过扣除这一共同增长率，描述起点对累计收益的相对影响；它并不假定过程会在某个有限时刻精确进入稳态。两条策略可能奖励率相同，却在前几百步收益相差很大。差分价值在跨策略比较时还要统一规范，有限预算研究则应直接报告累计收益和恢复成本。

<a id="lesson-discount-choice"></a>

## 1.1 · 同一个世界，两个目标为什么选择相反的动作

折扣回报给距当前时刻第 $k$ 步的奖励权重 $\gamma^k$。平均奖励则给长度为 $T$ 的统计窗口内每一步相同权重 $1/T$，再研究 $T\to\infty$。两者都考虑未来，但对时间的偏好不同。若奖励本来就有经济贴现、随机生存期限等含义，折扣是任务的一部分；若只是为了让无限和有限，则应把这种计算便利与最终评价目标分开。

$$
V^\pi_\gamma(s)=\mathbb E_\pi\!\left[\sum_{k=0}^{\infty}\gamma^kR_{k+1}\mid S_0=s\right],\qquad g_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{k=0}^{T-1}R_{k+1}\right].
$$

$V^\pi_\gamma$ 是以当前时刻为原点的加权奖励总量。$g_\pi$ 是每个原始时间步的奖励率。$h_\pi$ 后面将表示累计奖励相对 $Tg_\pi$ 的有限差额，单位仍是奖励，而不是奖励率。若每步时长为 $\Delta t$ 秒，物理奖励率是 $g_\pi/\Delta t$。

| 状态与动作 | 本步奖励 | 下一状态 |
| --- | --- | --- |
| 0：选择 A | 4 | 一定进入 A |
| 0：选择 B | 0 | 一定进入 B |
| A：只有等待动作 | 0 | 以 0.1 回到 0；以 0.9 留在 A |
| B：只有等待动作 | 1 | 以 0.1 回到 0；以 0.9 留在 B |

所有转移都花一个原始步。没有终止，也没有评价器提供的外部重置。回到 0 是世界自身的转移。策略 $\pi_A$ 每次在 0 选 A；$\pi_B$ 每次选 B。前者先得一笔收益，随后等待；后者先没有收益，随后持续获得收益。两条策略均只有一个常返类，且该类非周期。另一分支的状态是暂态，而不是第二个吸收类。因此两条策略各自的长期奖励率都不依赖起点。

$$
\mathbb E[\text{一次回到 0 的周期长度}]=1+\frac1{0.1}=11,\qquad g_{\pi_A}=\frac4{11},\qquad g_{\pi_B}=\frac{10}{11}.
$$

从 0 出发的一步也计时。进入 B 后，返回 0 的最后一步同样获得奖励 1，所以一个周期的期望奖励是 10。这里用的是期望总奖励除以期望总时间，不是逐条随机周期奖励率的简单平均。

折扣评价可直接联立 Bellman 方程。A 分支满足 $V_A(0)=4+\gamma V_A(A)$ 与 $V_A(A)=\gamma[0.1V_A(0)+0.9V_A(A)]$。B 分支将首步 4 换成 0，并在等待状态的方程中加入奖励 1。解得：

$$
V_A(0)=\frac{4(1-0.9\gamma)}{(1-\gamma)(1+0.1\gamma)},\qquad V_B(0)=\frac{\gamma}{(1-\gamma)(1+0.1\gamma)}.
$$

分母在 $0\leq\gamma<1$ 为正。因此 $V_A(0)>V_B(0)$ 当且仅当 $\gamma<20/23$。这个 MDP 只有在 0 需要选动作，所以两条确定性策略的比较也给出了折扣最优动作；在阈值处它们并列。

| 评价规则 | 策略 A | 策略 B | 最优选择 |
| --- | --- | --- | --- |
| 固定起点 0，$\gamma=0.5$ | $88/21\approx4.19048$ | $20/21\approx0.95238$ | A |
| 固定起点 0，$\gamma=0.9$ | $760/109\approx6.97248$ | $900/109\approx8.25688$ | B |
| 长期每步奖励率 | $4/11\approx0.36364$ | $10/11\approx0.90909$ | B |

![同一三状态环境的两条策略从共同起点出发，期望累计奖励曲线有不同斜率；下方比较0.5和0.9折扣对各时刻奖励的权重。](../assets/crl-figures/average-time-objectives.svg)

先看上图：A 先取得收益，B 后来的累计收益增长更快。下图保留同一世界，只改变每个时间位置的权重。曲线由转移概率精确传播，不是训练样本曲线。

这个反转不需要函数逼近、估计噪声或优化失败。它仅由目标不同造成。把短期权重较大的折扣算法训练得更充分，也不能让它自动优化另一个目标。反过来，平均奖励较高不意味着任意有限时长内都更好：只运行第一步时，A 的收益就是更高。

直接复制运行：纯 Python 标准库，用分数核对目标反转

```python
from fractions import Fraction as F

def discounted_values(gamma):
    denominator = (1 - gamma) * (1 + gamma / 10)
    a = 4 * (1 - 9 * gamma / 10) / denominator
    b = gamma / denominator
    return a, b

gain_a, gain_b = F(4, 11), F(10, 11)
threshold = F(20, 23)
print("gain A/B:", gain_a, gain_b)
print("discount threshold:", threshold)
for gamma in (F(1, 2), F(9, 10)):
    a, b = discounted_values(gamma)
    winner = "A" if a > b else "B"
    print("gamma =", gamma, "V(A/B) =", a, b, "winner =", winner)
assert discounted_values(threshold)[0] == discounted_values(threshold)[1]
assert gain_b > gain_a
# 这是闭式精确评价，不是 Monte Carlo 学习曲线。
```

预期输出：gain 为 4/11、10/11；折扣阈值为 20/23；两组折扣价值分别为 88/21、20/21 与 760/109、900/109。[独立线性方程和有限差分测试](../tests/test_reading_derivations.py)不依赖这两条价值闭式；下载[完整代码包](https://yingwen.io/crl-code/learning-code.zip)后，在根目录运行下列命令。它是公式验证，不是新增的采样性能实验。

用独立线性求解与有限差分核对本页解析结果

```bash
python -m unittest discover -s tests -p test_reading_derivations.py -v
```

<a id="lesson-policy-comparison"></a>

## 1.2 · 先说明比较的起点、策略与学习过程

上一节固定从状态 0 比较折扣回报。另一种定义是按各策略自身的稳态分布评价。令 $r_\pi(s)$ 为期望即时奖励。由 $V_\gamma=r_\pi+\gamma P_\pi V_\gamma$，左乘 $d_\pi^\top$ 得：

$$
d_\pi^\top V^\pi_\gamma=\frac{d_\pi^\top r_\pi}{1-\gamma}=\frac{g_\pi}{1-\gamma}.
$$

因此按每条策略自己的稳态分布平均折扣价值，排序与奖励率一致。这是 Sutton 与 Barto §10.4 讨论的重要区别。它不推翻固定起点反例，因为比较时的状态分布也随策略变了。

| 比较对象 | 需要明确的量 | 不能混用的结论 |
| --- | --- | --- |
| 固定起点或固定初始分布 $\nu$ | $J_\gamma(\pi;\nu)=\nu^\top V^\pi_\gamma$ | 可以有上述折扣/平均排序反转 |
| 每条策略自身的稳态分布 $d_\pi$ | $d_\pi^\top V^\pi_\gamma=g_\pi/(1-\gamma)$ | 求梯度时不能忽略分布随策略改变 |
| 冻结策略的长期性能 | $g_\pi$；必要时再比较规范化的 bias | 不能替代学习期间实际交互的收益 |
| 具有记忆与参数更新的完整学习器 | 给定历史 $H_t$、预算 $T$ 与权限，比较实际后续累计收益 | 未必存在固定策略稳态分布，不能直接套用本章固定策略定理 |

$$
J_T(\mathcal A\mid H_t)=\mathbb E_{\mathcal A}\!\left[\sum_{k=0}^{T-1}R_{t+k+1}\mid H_t\right].
$$

$\mathcal A$ 包含行为规则、内部状态和参数更新。比较在线学习与冻结参数时，应从同一历史和同一参数快照开始，继续承受各自行动的后果。停止参数更新不等于清空记忆。这个有限生命期量不是一般偏离遗憾的估计。

本网站的持续控制走廊实验采用这一做法：两组先共同学习 400 步，再让一组冻结参数；环境在预先固定的第 600 步改变，但不向智能体提供变化标签。观测、记忆更新和真实时钟保持一致。它检验适应过程，不以一个最终冻结策略的奖励率代替完整表现。

平均奖励还可能忽略只发生有限次的损失。两条策略都最终每步得 1，第一条却先损失 100，它们的 gain 仍相同。bias、有限时域收益和风险约束可以进一步区分它们。若行动永久改变可达常返类、造成不可逆损害或终结生命，还必须重新说明初始状态依赖、比较器与安全目标；标量 gain 不是全部决策准则。

<a id="lesson-problems"></a>

## 2 · 同一个目标下，预测、控制、规划分别求什么

先不要选神经网络。先写明待求的对象。给定有限状态集、动作集和未知平稳转移，固定一个策略是预测问题；允许改变策略是控制问题；允许查询转移模型是求解时可用的信息。规划不是第三种奖励目标，它可以服务于预测，也可以服务于控制。

| 问题 | 给定什么 | 求什么 | 允许读到什么 |
| --- | --- | --- | --- |
| 同策略预测 | 固定目标策略 π；实际也按 π 行动 | gπ 与满足参考规范的 hπ | 单条连续经验流 |
| 异策略预测 | 固定目标 π；行为 b 提供覆盖 | 同一个 gπ 与 hπ，不是 gb | 样本及 π(a\|s)、b(a\|s) |
| 控制 | 动作集、真实奖励和持续交互 | 提高实际可执行策略的奖励率；理想极限 g* | 探索行为产生的样本 |
| 已知模型规划 | 完整 P(s′\|s,a) 与 r(s,a) | 策略评价或最优性方程的解 | 可主动查询任意状态动作 |
| 学习模型规划 | 只有交互数据；模型也需估计 | 先解经验模型，再接受真实过程检验 | 仅已见到的信息，额外计算不是新数据 |

$$
d_\pi^\top P_\pi=d_\pi^\top,\qquad d_\pi^\top\mathbf1=1,\qquad g_\pi=d_\pi^\top r_\pi.
$$

$d_\pi(s)$ 是策略长期访问状态 s 的比例。把 Poisson 方程左乘 $d_\pi^\top$，价值差的平均为零，留下这条奖励率等式。它同时解释为什么行为奖励均值一般是 $d_b^\top r_b$，不是目标策略的量。

本章多状态实验把精确模型放在评价器中，而不是学习器中。评价器冻结当前策略，解线性方程得到其真实奖励率。学习器只能看到交互样本。这样可以把“估计器认为自己很好”与“策略实际很好”分开，也不需要额外重置在线学习轨迹。

预测需同时检查奖励率误差与参考对齐后的 bias 误差。控制需同时检查冻结策略奖励率、真实探索轨迹累计奖励和最优性残差。规划还需单列模型误差、真实环境步、模型备份数。单一 TD loss 不能替代这些量。

<a id="research-average-map"></a>

## 平均奖励并不是一种统一的训练协议

深入阅读：进一步比较训练协议与评价目标，不把平均奖励等同于一种数据收集方式。

在选择算法前，先分开四个问题：优化什么、从哪里取得数据、用什么表示、允许怎样重置。平均奖励规定长期目标。它不自动规定严格流式训练、非平稳环境、神经网络或单次生命。近期研究往往只改变其中一个轴；下面的地图先标明研究位置，正文随后推导各个求解器。

| 研究问题 | 主要对象 | 不能直接推出的结论 |
| --- | --- | --- |
| 深度连续控制 | ARO-DDPG、RVI-SAC、ASAC | 神经网络实测有效，不等于一般非线性收敛。 |
| 离策略预测的数值稳定 | Differential GQ；相邻的 CTDC、RETD | 折扣误差中心化，不等于平均奖励预测。 |
| 环境变化后如何改善策略 | NS-NAC、BORL | 可重新采样初始状态的遗憾界，不等于单次生命保证。 |
| 动作持续时间不同 | SMDP 的 RVI Q-learning | 固定 option 的收敛，不等于 option 内部同时学习时收敛。 |
| 用多少样本学到平均奖励 | 逐渐增大折扣的 Q-learning | 生成模型下的样本界，不等于真实轨迹的探索代价。 |

后文在预测、控制、中心化、时间抽象和样本权限的相关位置分别展开这些方法，Reward Centering、RVI-SAC 与重置协议也在对应问题处讨论。每次复现先写清样本权限，再记录真实交互、模型查询、梯度更新和重置。只有这些条件一致，算法比较才有解释力。

<a id="lesson-existence"></a>

## 2.1 · 奖励率与差分价值在什么条件下存在

本章首先假定奖励均值有界、有限平稳 MDP、固定平稳 Markov 策略。平稳指环境条件不随时间改变，也指策略在同一状态使用相同动作分布；它不表示训练参数永远不更新，而是每次分析策略评价时先冻结这些参数。以下三个链结构需要分开。

| 策略诱导的链 | 长期奖励率 | 差分价值与极限 |
| --- | --- | --- |
| 有限、不可约、非周期 | 唯一稳态分布；gain 不依赖初态 | Poisson 解只差常数；期望中心化回报级数收敛 |
| 单常返类，允许暂态 | 仍有唯一稳态分布；gain 不依赖初态 | Poisson 解只差常数；周期性仍可能使逐步极限振荡 |
| 多个常返类 | 可能因初态或最终进入的类而异 | 一般需要状态相关 gain；本章标量方程可能无解 |

最小反例是两个互不相通的自环状态，奖励分别恒为 0 和 1。两个起点的奖励率分别为 0 和 1；若强行共用一个标量 $g$，Poisson 方程会同时要求 $g=0$ 和 $g=1$。所以“有限、平稳、持续运行”不足以保证本章的标量设定。相反，周期性本身不阻止时间平均存在：奖励依次为 0、2 的二状态环具有 $g=1$，但逐时刻期望奖励不收敛。

$$
(I-P_\pi)h_\pi=r_\pi-g_\pi\mathbf1,\qquad d_\pi^\top h_\pi=0.
$$

第一式有 $|\mathcal S|+1$ 个未知量：每个状态的 $h_\pi(s)$ 和一个 $g_\pi$。由于 $I-P_\pi$ 的常数方向为零，第二式提供一个规范，选出唯一代表。也可改用 $h_\pi(s_{\rm ref})=0$，但两个规范下的数值不能直接混合比较。

先左乘 $d_\pi^\top$，利用 $d_\pi^\top P_\pi=d_\pi^\top$，得到 $g_\pi=d_\pi^\top r_\pi$。再将此率代回线性方程，并加一个规范，就能解出 bias。这是评价器可直接验证的精确计算。学习器不知道 $P_\pi$ 与 $r_\pi$，因而需要通过样本同时估计率与价值差。

$$
h_\pi(s)=\lim_{T\to\infty}\mathbb E_\pi\!\left[\sum_{k=0}^{T-1}(R_{k+1}-g_\pi)\mid S_0=s\right],\qquad d_\pi^\top h_\pi=0.
$$

在上述非周期、单常返类有限设定下，这个期望极限给出稳态均值为零的规范。这里是“先取有限和的期望，再取极限”。随机奖励的中心化样本和通常仍有不断增长的方差，不能宣称单条无限样本和几乎处处收敛。周期链可改由 Poisson 方程定义 bias，不能强套这条普通级数。

在 A/B 分支的三状态解析反例中，以状态 0 为参照，$\pi_A$ 的解按 $(0,A,B)$ 排列为 $h_A=(0,-40/11,70/11)$，$\pi_B$ 为 $h_B=(0,-100/11,10/11)$。它们各自满足三个状态的方程，包括暂态状态。用这一参照得到的 $h(0)=0$ 不表示从状态 0 出发的瞬态损失消失了，只是改变了共同原点。

这是去掉折扣后的第一个额外困难：不能只学一个“回报总量”网络。还要识别奖励率，并处理价值的公共常数自由度。第二个困难来自链本身：混合速度、周期和可达性现在直接影响评价与规划。

<a id="lesson-derive"></a>

## 3 · 从奖励率方程到 Differential TD

固定策略的有限时域价值记为 $V_T(s)$。先将第一步与剩下 $T-1$ 步分开。在有限、不可约且非周期的链上，采用 $d_\pi^\top h_\pi=0$ 的规范，可得到 $V_T(s)=Tg_\pi+h_\pi(s)+o(1)$。代入递推并消去线性增长项，便得到下面的方程。若改用参考状态为零的规范，渐近式还要补回被移走的常数。周期链不一定有这个逐点渐近形式，但仍可以直接研究相应的 Poisson 方程。

$$
\begin{aligned}V_T(s)&=\mathbb E_\pi[R_{t+1}+V_{T-1}(S_{t+1})\mid S_t=s],\\g_\pi+h_\pi(s)&=\mathbb E_\pi[R_{t+1}+h_\pi(S_{t+1})\mid S_t=s].\end{aligned}
$$

这是 Poisson 方程的状态形式。令 $P_\pi$ 为策略转移矩阵、$r_\pi$ 为期望即时奖励向量，矩阵形式为 $(I-P_\pi)h=r_\pi-g_\pi\mathbf1$。由于 $(I-P_\pi)\mathbf1=0$，$h$ 与 $h+c\mathbf1$ 满足同一方程。

为什么解这个一步方程就能求长期奖励率？反过来，设一对 $(g,h)$ 满足它。将每一步的条件期望累加，中间状态的 h 前后抵消，只剩首尾两项：

$$
\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}R_{t+1}\mid S_0=s\right]=Tg+h(s)-\mathbb E_\pi[h(S_T)\mid S_0=s].
$$

若 h 有界，右边后两项除以 T 后趋于零。因此该方程中的 g 正是长期奖励率。差分价值承担的是首尾位置造成的有限差额；这条有限 T 恒等式不要求每条随机中心化奖励和收敛，也不要求非周期。

![策略B的期望累计奖励与共同增长直线Tg之间保留有限差额；下方用0到B的一次转移展示中心化奖励和差分价值差相消。](../assets/crl-figures/average-bias-geometry.svg)

仍看前面的 B 策略，并取 h(0)=0。虚线给出共同增长 Tg，橙线保留起点影响；两者的间隙正是首尾 bias 项。沿 0→B，中心化奖励 −10/11 与后继价值增加 +10/11 相抵。

差分价值 $h$ 的绝对数值依赖常数规范。可以指定某个参考状态的价值为零，也可以允许算法得到由初始化决定的偏移；跨运行比较关注差值、策略与 Bellman 残差。Differential TD 的基本版本不固定某个参考状态。

现在遮住模型，只保留已经收到的转移。固定本步更新前的估计 $(\bar g,h)$，定义状态 s 的方程残差 $F_s(\bar g,h)=r_\pi(s)-\bar g+(P_\pi h)(s)-h(s)$。模型已知时可直接算 F；模型未知时，一条同策略转移中的 $R_{t+1}-\bar g+h(S_{t+1})-h(S_t)$ 就是它的样本。若行为来自 b，则在动作有覆盖的条件下乘 $\rho_t=\pi(A_t\mid S_t)/b(A_t\mid S_t)$，恢复同一状态条件下的期望。

$$
\begin{aligned}\mathbb E_b[\rho_t\delta_t\mid S_t=s,\bar g,h\text{ 固定}]&=F_s(\bar g,h),\\\sum_s d_\pi(s)F_s(\bar g,h)&=g_\pi-\bar g.\end{aligned}
$$

第一行把待解方程变成可观察的误差信号。第二行用 $d_\pi^\top P_\pi=d_\pi^\top$ 消去价值差，说明率偏低时，目标稳态下的平均残差为正。实际采样仍来自行为过程；动作重要性比没有提供这第二行所需的状态权重。

Differential TD 用同一个误差推进两个未知量：提高或降低当前状态价值，同时调整共同的奖励率估计。这样得到一项求解方程的随机逼近方法，并不是对 $\sum_s F_s^2$ 直接做梯度下降；后者还会对后继 h 求导，得到另一种更新。两者是否收敛必须分别分析。

$$
\begin{aligned}\delta_t&=R_{t+1}-\bar g_t+h_t(S_{t+1})-h_t(S_t),\\h_{t+1}(S_t)&=h_t(S_t)+\alpha_t\rho_t\delta_t,\\\bar g_{t+1}&=\bar g_t+\eta\alpha_t\rho_t\delta_t,\qquad \rho_t=\frac{\pi(A_t\mid S_t)}{b(A_t\mid S_t)}.\end{aligned}
$$

On-policy 时 $\rho=1$。Off-policy 预测要求 $\pi$ 可能选择的动作在 $b$ 下也有正概率。两个更新共用旧参数下的 TD 误差，即计算一次 $\delta_t$ 后同时用于价值与奖励率。

只平均真实奖励是另一种率估计。同策略时，它可以估计 $g_\pi$；异策略时，它通常估计 $g_b$，与本节价值头的预测问题不一致。两种率更新的可执行对照见实现部分。

**算法：Differential TD：一个转移、一次误差、两项写入**

1. 给定目标策略 $\pi$、提供覆盖的行为策略 b；初始化 h 与 $\bar g$
1. 从 b 执行动作，保存其实际采样概率；观察 $(s,a,r,s')$
1. 按同一份旧 h 与 $\bar g$ 计算 $\delta=r-\bar g+h(s')-h(s)$
1. 计算 $\rho=\pi(a\mid s)/b(a\mid s)$ 与单次增量 $u=\alpha\rho\delta$
1. 写入 $h(s)\leftarrow h(s)+u$，$\bar g\leftarrow\bar g+\eta u$
1. 令 $s\leftarrow s'$；下条经验使用更新后的估计

这正对应 [differential_td_offpolicy.py](../implementations/average_systems/differential_td_offpolicy.py) 的 `update`：先形成 `delta`，复制旧表格后只写当前状态，再用同一 `delta` 返回新 `rate`。`run` 才负责采样动作、计算比率和推进真实状态。即使 s 与 s′ 相同，也必须先完成误差计算，再改写表格。

在理论证明里常使用满足随机逼近条件的递减步长并要求覆盖；在持续漂移中通常保留常数步长追踪新目标，代价是稳态方差和不同的保证。将线性特征换成深网也不是无条件继承表格收敛性。

<a id="lesson-gain-improvement"></a>

## 3.1 · 控制不是让任意差分价值变大，而是改善奖励率

先考虑固定平稳 MDP 中的两条平稳策略。用旧策略 $\pi$ 的 Poisson 解定义动作价值和优势：

$$
q_\pi(s,a)=r(s,a)-g_\pi+\sum_{s'}P(s'\mid s,a)h_\pi(s'),\qquad A_\pi(s,a)=q_\pi(s,a)-h_\pi(s).
$$

$q_\pi$ 问：现在选动作 a，再恢复旧策略，相对每步都获得 $g_\pi$ 的基准多出多少价值？$h_\pi$ 与 $q_\pi$ 共同平移时，优势不变；不同策略未经规范化的 q 数值则不能直接比大小。

$$
\begin{aligned}g_{\pi'}-g_\pi&=\sum_s d_{\pi'}(s)\sum_a\pi'(a\mid s)A_\pi(s,a),\\\sum_s d_{\pi'}(s)\left[(P_{\pi'}h_\pi)(s)-h_\pi(s)\right]&=0.\end{aligned}
$$

第一式由第二式消去稳态下价值的净变化得到。注意权重是新策略的 $d_{\pi'}$。新策略不仅改变当前动作，也改变长期到达哪些状态。这个等式是平均奖励的策略性能差异公式。

若对每个状态都选择使 $q_\pi(s,a)$ 最大的动作，则新策略在该状态的平均优势不小于零。若新旧策略都满足相应单常返类条件，上式保证 $g_{\pi'}\geq g_\pi$。这就是精确策略迭代的依据。它不表示一次带噪神经梯度更新具有同样保证，也不表示 gain 相同的所有策略都有同样的暂态损失。

若把反复评价、再改善的过程写成一个最优性方程，便用最大的后继动作价值替代给定策略的平均。令 $h_*(s)=\max_a q_*(s,a)$；假定存在有界解，方程使任意动作都满足 $\mathbb E[R_{t+1}+h_*(S_{t+1})\mid s,a]\leq g_*+h_*(s)$，贪心动作取等号。沿真实轨迹累加这个不等式，首尾 h 又一次抵消：任意策略的长期率不超过 $g_*$，而贪心策略达到它。这给出了 max 与奖励率最优之间的联系。

$$
\begin{aligned}q_*(s,a)&=\mathbb E[R_{t+1}-g_*+\max_{a'}q_*(S_{t+1},a')\mid s,a],\\\delta_t&=R_{t+1}-\bar g_t+\max_{a'}Q_t(S_{t+1},a')-Q_t(S_t,A_t),\\Q_{t+1}(S_t,A_t)&=Q_t(S_t,A_t)+\alpha_t\delta_t,\\\bar g_{t+1}&=\bar g_t+\eta\alpha_t\delta_t.\end{aligned}
$$

Differential Q 用 max 定义控制目标；采到的 (s,a) 在自身转移分布下更新，因此表格单步控制式不再乘评估式的 π/b。仍需要充分访问状态动作与适当步长。

**算法：Differential Q-learning；统计窗口不改变学习器状态**

1. 初始化 $Q(s,a)=0$、$\bar g=0$，选择 $\alpha>0$、$\eta>0$
1. 在每个环境步：
  1. 按 $\varepsilon$-greedy 策略选择动作，观察 $(s,a,r,s')$
  1. 用更新前参数计算 $\delta=r-\bar g+\max_{a'}Q(s',a')-Q(s,a)$
  1. 更新 $Q(s,a)\leftarrow Q(s,a)+\alpha\delta$
  1. 更新 $\bar g\leftarrow\bar g+\eta\alpha\delta$
  1. 令 $s\leftarrow s'$，继续交互

对应 [differential_q_multistate.py](../implementations/average_systems/differential_q_multistate.py) 的 `update`，本步被更新的对象是已经采到的状态—动作对。下一动作的 max 是学习目标；`run` 中的 ε-greedy 才决定真实下一步做什么。增加 max 所选动作的估计，不等于评价器已经观察到该动作的长期收益。

Differential Q 直接逼近最优方程，所以其率参照量瞄准 $g_*$，不是探索行为的真实平均奖励。策略评价型 critic 则瞄准当前目标策略的 $g_\pi$。两者不能因都写成“reward-rate”就合并。实际轨迹的 $\sum R/T$ 是第三个量；它包括探索和学习失误，应原样保留。

| 控制途径 | 要求哪种 critic / 率 | 主要额外困难 |
| --- | --- | --- |
| 精确策略迭代 | 固定策略的 $h_\pi,q_\pi,g_\pi$ | 评价器要解稳态与 Poisson 方程；改善后重新评价 |
| Differential Q-learning | 最优差分动作价值与 $g_*$ | 行为覆盖、max 目标、率/价值联合更新 |
| 平均奖励 actor–critic | 当前策略的差分 critic 与 $g_\pi$ | 当前稳态权重、critic 误差、策略变化的时间尺度 |
| 严格持续学习器比较 | 真实历史下的整段奖励与资源成本 | 学习与探索也改变未来；不能只比较冻结策略 |

<a id="lesson-average-policy-gradient"></a>

## 3.2 · 平均奖励策略梯度从哪里来

设策略 $\pi_\theta(a\mid s)$ 可微，环境奖励与转移不直接依赖参数 $\theta$。先在有限问题中推导，要求参数邻域内的策略链具有唯一、可微的稳态分布，并且相关导数存在。记 $P_\theta,r_\theta,d_\theta,h_\theta$ 为当前策略对应的量。直接对 $g_\theta=d_\theta^\top r_\theta$ 求导时，不能丢掉稳态分布的导数。

$$
\begin{aligned}\nabla g_\theta&=(\nabla d_\theta^\top)r_\theta+d_\theta^\top\nabla r_\theta,\\(\nabla d_\theta^\top)(I-P_\theta)&=d_\theta^\top\nabla P_\theta,\qquad(\nabla d_\theta^\top)\mathbf1=0,\\r_\theta&=g_\theta\mathbf1+(I-P_\theta)h_\theta,\\\nabla g_\theta&=d_\theta^\top\left[\nabla r_\theta+(\nabla P_\theta)h_\theta\right].\end{aligned}
$$

第二行来自对稳态方程与概率归一化求导。将第三行代入第一行，再用第二行，就消去了难以直接估计的 $\nabla d_\theta$。这里每个参数分量分别满足同样的等式。

$$
\nabla_\theta g_\theta=\sum_s d_\theta(s)\sum_a\nabla_\theta\pi_\theta(a\mid s)q_\theta(s,a)=\mathbb E_{d_\theta,\pi_\theta}\!\left[\nabla_\theta\log\pi_\theta(A\mid S)A_\theta(S,A)\right].
$$

减去只依赖状态的 baseline 不改变期望，因为 $\sum_a\nabla_\theta\pi_\theta(a\mid s)=0$。这个公式使用当前策略的差分价值和稳态分布。它不是把折扣策略梯度中的 $\gamma$ 删除后凭形式猜出的更新。

$$
\delta_t=R_{t+1}-g_\theta+h_\theta(S_{t+1})-h_\theta(S_t),\qquad\mathbb E[\delta_t\mid S_t=s,A_t=a]=A_\theta(s,a).
$$

因此，在当前策略的稳态样本下，用精确 critic 的单步 TD 误差乘 score，可以估计上面的梯度。实际算法用 $\bar g_t,h_{w_t}$ 替代真值，再更新 $\theta\leftarrow\theta+\beta_t\delta_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)$。估计 critic、尚未混合和参数同时改变带来的误差，需要另外分析。

**算法：平均奖励 actor–critic 的信息时序；不是深网收敛声明**

1. 固定本步的 actor、critic 与率参数快照
1. 由当前策略执行一个真实动作，获得奖励和后继状态
1. 用旧参数计算 $\delta=r-\bar g+h_w(s')-h_w(s)$
1. 用 $\delta$ 更新 critic；按所选率估计算法更新 $\bar g$
1. 用同一个旧参数下的 $\delta$ 和 score 更新 actor
1. 分别记录真实奖励率、critic 残差、参数变化及更新次数

若样本来自行为稳态 $d_b$，只乘动作比 $\pi/b$ 仍不会把状态权重变成 $d_\pi$。若将 actor 更新得过快，critic 追踪的目标也在移动。即使单独的预测算法收敛，也不能据此证明这个耦合控制系统收敛。后面的 ARO-DDPG、RVI-SAC 与 ASAC 正是在不同约束下处理其中部分困难。

<a id="lesson-discount-estimator"></a>

## 3.3 · 在学习器里保留折扣：得到的是哪个 critic

希望降低长时标估计难度，不等于必须改变最终评价指标。可以把平均奖励作为目标，同时研究带折扣的辅助估计器。但必须写清估计器的固定点，以及将它用于 actor 后产生的误差。先冻结策略，并假定其真实 $g_\pi$ 已知。定义中心化的折扣价值：

$$
\begin{aligned}u^\pi_\gamma(s)&=\mathbb E_\pi\!\left[\sum_{k=0}^{\infty}\gamma^k(R_{k+1}-g_\pi)\mid S_0=s\right]\\&=V^\pi_\gamma(s)-\frac{g_\pi}{1-\gamma},\\(I-\gamma P_\pi)u^\pi_\gamma&=r_\pi-g_\pi\mathbf1,\qquad d_\pi^\top u^\pi_\gamma=0.\end{aligned}
$$

这是折扣的中心化方程。真正差分价值满足的是 $(I-P_\pi)h_\pi=r_\pi-g_\pi\mathbf1$。两个矩阵不同，所以有限 $\gamma$ 时一般有 $u^\pi_\gamma\ne h_\pi$。奖励减去了准确 gain，并不能消除这种差别。

$$
\delta_t^{(\gamma)}=R_{t+1}-\bar g_t+\gamma u_w(S_{t+1})-u_w(S_t),\qquad w_{t+1}=w_t+\alpha_t\delta_t^{(\gamma)}\nabla_wu_w(S_t).
$$

这是该辅助 critic 的候选半梯度更新，不是 Differential TD 的同一个更新。即使 $\bar g_t=g_\pi$，它也拟合中心化折扣值。若 $\bar g_t$ 有误，在表格精确评价中还会产生 $(g_\pi-\bar g_t)/(1-\gamma)$ 的公共偏移。联合率学习和非线性逼近须单独验证。

A/B 分支的三状态解析反例给出一个直接检查：$\pi_A$ 在 $\gamma=0.5$ 时，$u_\gamma(A)-u_\gamma(0)=-80/21$；真正差分价值的差是 $-40/11$。这两个量连减去参考状态后都不相等。另一方面，$u^A_\gamma(0)=400/[11(10+\gamma)]$，当 $\gamma\uparrow1$ 时趋向 $400/121$，恰好是稳态均值为零规范下的 $h_A(0)$，不是参考状态为零规范中的 0。

在这里的有限单常返类条件下，$u^\pi_\gamma\to h_\pi$，其中 $d_\pi^\top h_\pi=0$。固定策略的这个 Abel 极限，不等于任意有限折扣的精确等价，也不自动给出同时改变策略、模型和表示时的统一误差界。慢混合会使需要的折扣时标更长。

$$
B_\gamma(\theta)=\mathbb E_{d_\theta,\pi_\theta}\!\left[\nabla_\theta\log\pi_\theta(A\mid S)\left(R-g_\theta+\gamma u^\theta_\gamma(S')-u^\theta_\gamma(S)\right)\right].
$$

一般 $B_\gamma(\theta)\ne\nabla g_\theta$。在精确值下，括号其实等于普通折扣 TD 误差，而状态权重仍是稳态分布。它既不是自动正确的平均奖励梯度，也不能不说明起点分布就称为标准折扣目标的梯度。

甚至方向也会相反。令 $\pi_p$ 在状态 0 以概率 p 选择 A，其余时候选择 B，$0<p<1$。其稳态分布为 $(1,10p,10(1-p))/11$，所以 $g(p)=(10-6p)/11$、$dg/dp=-6/11$。但 $\gamma=0.5$ 时，精确折扣动作价值差是 $4-\gamma/(1-0.9\gamma)=34/11$。稳态 score 更新方向因此是 $B_\gamma(p)=34/121>0$，与真实梯度相反。这里 critic 没有训练误差，gain 也完全正确。

若用 logit 参数 $p=\sigma(\theta)$，两个方向都再乘 $p(1-p)$；在 $p=1/2$ 时分别为 $-3/22$ 和 $17/242$。因此“去掉目标中的折扣、在学习算法中加回”是一项需要证明或量化偏差的设计，不是一条不带条件的等价变换。可以研究随数据增加的折扣调度、对平均梯度的显式校正，或不改变 Poisson 目标的信用分配衰减；三者应分别命名。

<a id="course-average-slow-modes"></a>

## 把折扣作为求解参数：它先压掉了哪些长期差异？

深入阅读：用转移矩阵的慢变化方向分析折扣估计器的偏差。

中心化折扣 critic 在有限折扣下不同于差分价值。差异有多大，取决于环境的时间尺度，不只取决于 γ 看上去是否接近 1。冻结一条有限、不可约的策略链，可以用一个最小例子把这层误差算出来。

$$
P=\begin{pmatrix}1-\varepsilon&\varepsilon\\\varepsilon&1-\varepsilon\end{pmatrix},\qquad r=\begin{pmatrix}1\\-1\end{pmatrix},\qquad d=\begin{pmatrix}1/2\\1/2\end{pmatrix},\qquad g=d^\top r=0.
$$

$0<\varepsilon<1/2$。两个状态都有较大概率停留，长期访问各半；奖励率为零，但初始处于高奖励状态仍比处于低奖励状态有利。

$$
Pr=(1-2\varepsilon)r,\quad h=\frac{r}{2\varepsilon},\quad u_\gamma=\frac{r}{1-\gamma(1-2\varepsilon)},\quad \frac{u_\gamma(s)}{h(s)}=\frac{2\varepsilon}{1-\gamma+2\gamma\varepsilon}.
$$

这里 $r$ 恰是非恒定特征方向，且其稳态均值为零。分别代入 $(I-P)h=r$ 与 $(I-\gamma P)u_\gamma=r$ 即得。比值同时适用于两个非零状态值，不涉及常数规范的混淆。

取 ε=0.01，差分价值是 (50,−50)。即使 γ=0.99，中心化折扣值也只有约 (33.557,−33.557)。准确奖励率已经被减去，仍有约 32.9% 的状态差异被压小。若 ε=0.0001 而 γ 仍为 0.99，该比值仅约 0.0196。对慢得多的过程，同一个折扣几乎抹掉了差分价值中的长期结构。

若希望这项相对缩小不超过 $\eta\in(0,1)$，令 $q=1-\gamma$，由上式可直接得到 $q\leq 2\varepsilon\eta/[(1-\eta)(1-2\varepsilon)]$。这个界只属于本例，但它说明“足够接近 1”必须相对于混合时间解释。增大 γ 让估计更接近 bias，也增加需要从经验分辨的时长。

课程提出过从较小折扣逐步趋向 1 的思路：目标保持平均奖励，折扣仅作为中间估计器参数。它是一条求解思路，不是仅凭固定策略极限就得到的在线控制定理。每次调整折扣，critic 的目标会移动；同时学习策略、奖励率和表示时，还有其他移动目标。

$$
u_{\gamma_{t+1}}-u_{\gamma_t}\approx(\gamma_{t+1}-\gamma_t)(I-\gamma_tP)^{-1}P\,u_{\gamma_t}.
$$

这是冻结 P、奖励与正确 g 后，对折扣求导得到的一阶变化。靠近慢模态时逆矩阵的作用较强。固定折扣增量未必造成固定大小的目标移动；critic 尚未跟上时再次增大折扣，会增加跟踪误差。

这个例子没有动作，所以尚不证明策略选择受损。要检验控制影响，应加入一个可以改变到达概率或转换成本的动作，再与精确 Poisson 解比较。记录有限折扣偏差、采样误差与策略改善方向，不能只因学习曲线变平就判定折扣调度已正确逼近平均奖励控制。

一个有区分力的研究问题是：能否依据已有经验对关键时间尺度的分辨程度调整折扣，而不是按训练步数预设日程？在严格流式条件下，旧数据不能重新计算新时域标签；适应折扣、保留多尺度 GVF 和直接差分预测各需要不同的记忆与计算。应在同一预算下比较，而非将辅助预测数量视为免费。

<a id="lesson-trace-decay"></a>

## 3.4 · 资格迹衰减可以保留平均奖励目标，但仍有近似误差

目标折扣回答“较晚的奖励是否本来就更不重要”。资格迹参数 $\lambda$ 回答“这次误差把多少更新分配给较早的预测”。在平均奖励 TD 中可以使用 $\lambda<1$，而不在奖励率定义中加入时间偏好。先固定策略、特征与正确奖励率，写出有限步的差分预测目标：

$$
G_t^{(n)}=\sum_{k=0}^{n-1}(R_{t+k+1}-g_\pi)+h(S_{t+n}),\qquad G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)},\quad0\leq\lambda<1.
$$

每个 n 步目标中都没有奖励折扣。$\lambda$ 只混合不同长度的 bootstrap 目标。精确的 $h_\pi$ 对每个 n 步目标都满足同一个条件期望方程，因此也是这个混合目标的解。

$$
e_t=\lambda e_{t-1}+\nabla_wh_w(S_t),\qquad\delta_t=R_{t+1}-\bar g_t+h_w(S_{t+1})-h_w(S_t),\qquad w_{t+1}=w_t+\alpha_t\delta_te_t.
$$

这是累积迹与差分 TD 的基本形式。它展示信用分配机制，不指定全部奖励率估计与异策略校正。完整算法还必须规定率更新、迹初值、步长及采样假设；在线参数移动时，朴素前向/后向实现也不自动精确等价。

沿这条推导进入[在线信用分配](credit.md#lesson-state-lambda)：先冻结 h 与 g，展开有限多步目标，逐项消去中间的 h，再交换“预测时刻”与“误差到达时刻”的求和，就得到资格迹。若每一步重估奖励率，过去各个目标用了哪一版率也要说明；仅将折扣式中的 $\gamma$ 改为 1，并没有完成这个时序问题。

| 衰减出现的位置 | 改变的数学对象 | 必须保留的边界 |
| --- | --- | --- |
| 目标中 $\gamma^kR_{t+k+1}$ | 折扣回报与可能不同的最优策略 | 若它表示真实任务偏好，不应随意删除 |
| 辅助 critic 中 $\gamma u(S')$ | 中心化折扣 Poisson 方程 | 有限折扣下不是精确 bias；actor 可能有方向偏差 |
| 资格迹中 $\lambda e$ | 不同信用时标的更新混合 | 标称 gain 目标可不变，但投影解、方差和稳定性可能变化 |
| 优化器的移动平均或目标网络 | 参数估计器的时间尺度 | 不直接定义奖励目标，也不自动保证收敛 |

表格精确表示与共享函数逼近必须分开。线性近似只能满足投影后的方程，改变迹参数可能改变投影固定点。神经表示更新后，旧迹还携带旧特征方向。异策略下还要处理重要性比和状态分布；不能给已有折扣深网代码去掉一个 gamma，就宣称得到正确且稳定的平均奖励资格迹算法。

Bellman Error Centering 与 RETD 研究的是相邻的折扣预测稳定性问题。它们对中心化与迹的联合反馈有启发，但不构成上述平均奖励公式在深度流式控制中的收敛证明。对应原文与作者工程位于本章相关研究段，保持各自的问题设定。

<a id="lesson-offpolicy"></a>

## 4 · 异策略预测：修正了动作，还会错在哪里

$$
\mathbb E_b[\rho_t\delta_t\mid S_t=s]=r_\pi(s)-\bar g+(P_\pi h)(s)-h(s).
$$

给定 s，$b(a|s)$ 与 $\rho=\pi(a|s)/b(a|s)$ 相消。每个状态的条件期望为零时，便得到目标策略 Poisson 方程。动作覆盖是必要条件；未访问的状态仍无法凭这条式子学会。

这不是“重要性采样能修正一切”。在表格中，每个状态可以独立满足自己的方程。共享特征只能满足投影后的条件。投影权重通常来自行为状态分布。动作比率并未把它变成目标稳态分布，也没有保证投影后的迭代稳定。非线性特征还会随参数更新改变投影空间。

下面学习曲线使用的是另一个三状态模型：每行转移含 0.2 的均匀混合成分，不是前文 A/B 分支的解析反例。在此学习实验中，目标策略的 $g_\pi\approx0.4806503$，行为策略的 $g_b=1/3$。如果价值头使用 $\rho\delta$，奖励率却直接平均行为奖励，那么即使已经得到真实 $h_\pi$，每个状态的条件 TD 残差仍是 $g_\pi-g_b\approx0.147317$。这不是“训练还不够久”，而是方程写错了。

错误奖励率有时主要表现为价值共同偏移不断增长。减掉参考状态后，bias 曲线甚至可能暂时很好看。因此实验保留 raw_bias_offset、gain_abs_error 与差分价值 RMSE。不能只看归一化后的漂亮曲线。

$$
\bar g_t-\eta\sum_s h_t(s)=\bar g_0-\eta\sum_s h_0(s).
$$

表格 Differential TD 的两条更新共享同一增量，因此满足这条不变量。它选定一个由初始化决定的常数规范。控制版本把求和对象改成全部 Q(s,a)。这不是“奖励率必须在更慢时间尺度更新”的证明；此算法使用固定比例 η。

<a id="experiment-differential_td_offpolicy"></a>

### 实验：实验 · 行为奖励率为什么不能替代目标奖励率

价值更新已经乘了重要性比，奖励率仍直接平均行为奖励，会怎样？

**环境与可用信息。** 三状态、双动作、完全可观测的持续 MDP。状态 0/1/2 的两动作奖励依次为 (0.1,−0.05)、(0.5,0.05)、(0.2,1.2)。优先后继依次为 (1,2)、(0,2)、(0,1)：转移以 0.8 概率取该后继，以 0.2 概率在三状态中均匀选择。无终止、无外部重置。行为策略在每个状态都各以 0.5 概率选两动作；目标策略选动作 1 的概率依次为 0.85、0.75、0.8。

**设置。** 1200 个真实步；从状态 0 开始，差分价值和奖励率估计全零。价值步长 0.08，正确方法的奖励率步长为 0.2×0.08。两种方法以相同 seed 产生相同的行为轨迹。错误负对照仅把奖励率更新改成行为奖励的样本均值。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** differential_td_offpolicy.py 先计算奖励减率加后继差分值的 TD 误差，再把动作重要性比同时用于价值和率更新。wrong_behavior_mean_td.py 是有意错误的负对照，不是另一个推荐算法。

**测量。** 先看参考状态对齐后的差分价值 RMSE，再看目标奖励率绝对误差和未对齐公共偏移。目标 gain 的精确值约 0.480650，行为 gain 为 1/3；真模型只用于评价。

```bash
python3 implementations/average_systems/differential_td_offpolicy.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](../assets/crl-figures/result-differential_td_offpolicy.svg)

横轴：environment_steps。纵轴：参考状态对齐后的差分价值RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 先从三个当前差分价值中统一减去状态0的估计，再与目标策略的差分价值比较，取三状态均匀 RMSE。目标策略的真值由真实模型仅作评价求得，不提供给学习更新。

**step：怎样计时。** step 是行为策略产生的真实转移数；每次转移更新差分价值和奖励率，无回合终止或重置。评价解方程不增加环境步。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 主图消除了公共平移，需同时看 gain_abs_error（目标奖励率误差）和 raw_bias_offset（未经对齐的状态0值）；小的对齐误差不能代替正确的奖励率。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 保存 bias_0、bias_1、bias_2，可结合已知模型独立重算 RMSE。gain 与 true_gain 可重算奖励率误差；experienced_reward_rate × step 可恢复行为累计奖励，但行为奖励率不等于目标策略奖励率。

计算位置：[average_systems/differential_td_offpolicy.py](../implementations/average_systems/differential_td_offpolicy.py) · [average_systems/wrong_behavior_mean_td.py](../implementations/average_systems/wrong_behavior_mean_td.py) · [average_systems/_common.py](../implementations/average_systems/_common.py)

</details>

**结果分析。** 1200 步时，对齐 RMSE 均值为 0.04395，错误负对照为 0.05323，差别不大。但奖励率绝对误差分别为 0.01666 和 0.14124；原始参考值偏移分别约 0.5793 和 4.5044。只展示对齐 RMSE 会掩盖错误率估计。

**结论边界。** 这是固定策略、表格、充分覆盖的平稳链。常数步长仍有噪声；5 个 seed 和短预算不建立非线性稳定性或普遍性能排序。共同偏移需要与错误率结合判断，不能单凭非零偏移宣称算法有错。

**继续实验。** 保留完整相同的行为轨迹，逐步核对重要性比出现的位置。把目标策略改成行为策略后，错误负对照的率语义还错吗？若减少行为支持集，程序应怎样拒绝不可识别的问题？

[源码](../implementations/average_systems/differential_td_offpolicy.py) · [逐种子记录](https://yingwen.io/crl-code/results/differential_td_offpolicy/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/differential_td_offpolicy/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/differential_td_offpolicy/curves.json)

<a id="lesson-rvi"></a>

## 5 · 相对价值迭代与 RVI Q-learning

转移分布与奖励均值已知时，可以直接计算 Bellman 更新。它与折扣算子不同：给所有 $h$ 加同一常数，更新结果也整体平移，不会由 $\gamma<1$ 压缩这个方向。RVI 每轮减去参考状态更新后的值，保持一个约定的原点。

$$
\begin{aligned}(Th)(s)&=\max_a\sum_{s',r}p(s',r\mid s,a)[r+h(s')],\\h_{k+1}(s)&=(Th_k)(s)-(Th_k)(s_{\rm ref}).\end{aligned}
$$

初始化 h(sref)=0；在适当条件下稳定后，(Th)(sref) 给出 g*。不要把第一轮任意偏移下的参考值立即当作真实奖励率；周期性也可能导致普通同步迭代振荡。

$$
Q_{t+1}(S_t,A_t)=Q_t(S_t,A_t)+\alpha_t\big[R_{t+1}-f(Q_t)+\max_a Q_t(S_{t+1},a)-Q_t(S_t,A_t)\big]
$$

RVI Q-learning 是采样形式：用满足所选理论条件的参考函数 f(Q) 代替另学一个 ḡ。一个常见选项是参考状态动作的 Q 值。参考函数影响数值锚定与瞬态行为，不能随意换成任何神经网络统计量后仍引用原证明。

| 方法 | 奖励率 / 原点从何而来 | 样本或模型 |
| --- | --- | --- |
| Differential TD/Q | 由同一 TD error 更新 ḡ；h/Q 可留常数偏移 | 单步实际样本 |
| RVI | 每轮减去参考状态的 backup | 完整已知模型 |
| RVI Q | f(Q) 提供参照；理论限制 f 的性质 | 访问到的状态动作样本 |
| Differential planning | 模型提供真实转移分布的模拟样本或期望 backup | 需另外检查模型误差和更新调度 |

<a id="experiment-rvi_multistate"></a>

### 实验：实验 · 相对价值归一化改变的是公共尺度

不减参考值也能学到正确的价值差，为什么还需要 RVI？

**环境与可用信息。** 三状态、双动作、完全可观测的持续 MDP。状态 0/1/2 的两动作奖励依次为 (0.1,−0.05)、(0.5,0.05)、(0.2,1.2)。优先后继依次为 (1,2)、(0,2)、(0,1)：转移以 0.8 概率取该后继，以 0.2 概率在三状态中均匀选择。无终止、无外部重置。本实验明确把真实转移与奖励模型交给规划器，不采集真实轨迹。

**设置。** 从全零状态价值开始，做 1200 次同步模型扫描；每扫描包含 6 个状态动作期望备份。RVI 每次减去备份后的状态 0 值；对照不减。无采样、无随机初始化，因此五个 seed 的输出完全相同。

**检验的机制。** rvi_multistate.py 的 update 先用旧价值计算全部 Bellman 最优备份，再统一减参照。unnormalized_vi_multistate.py 保留共同增长量。两者的规范化后价值差应一致。

**测量。** 主图是对最优 bias 做相同参考对齐后的 RMSE；横轴是模型扫描，不是真实环境步。还应读运行日志中的原始公共偏移和率估计。零 seed 方差来自确定性，不是五次独立统计证据。

```bash
python3 implementations/average_systems/rvi_multistate.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](../assets/crl-figures/result-rvi_multistate.svg)

横轴：model_sweeps。纵轴：最优偏差参考对齐RMSE。每种方法 1200 model_sweeps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 把三个规划值对齐到状态0后，与已知三状态模型的最优差分价值作均匀 RMSE。主图比较相对值，不比较共同增长的绝对值；未规范化 VI 也可得到正确的相对值。

**step：怎样计时。** step 是一轮同步扫描：3状态 × 2动作，共6次状态动作期望备份，每个备份再对3个后继求和。model_backups = 6 × step，environment_updates 始终为0；该图没有真实采样。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 本计算完全确定，种子不参与更新；重复曲线及零标准差不构成多次随机实验的证据。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 的三个 bias 字段可独立重算对齐 RMSE，raw_bias_offset 保留公共偏移。RVI 的 gain 是该轮减去的参考值；未规范化 VI 的 gain 是参考值的单轮增量，不是随扫描累积的绝对值。

计算位置：[average_systems/rvi_multistate.py](../implementations/average_systems/rvi_multistate.py) · [average_systems/unnormalized_vi_multistate.py](../implementations/average_systems/unnormalized_vi_multistate.py) · [average_systems/_common.py](../implementations/average_systems/_common.py)

</details>

**结果分析。** 初始误差约 0.43049；60 次扫描后两者均约 $6.76\times10^{-8}$；120 次后均约 $10^{-13}$。最终两者都接近浮点精度。图中没有 RVI 在价值差上的性能优势，它隔离的是公共尺度的控制。

**结论边界。** 仅验证这个非周期遍历模型。不能从本图推断任意周期链中的普通 RVI 都收敛。已知模型的 1200 次扫描不等于只花 1200 次真实交互学得模型。

**继续实验。** 在每次扫描后比较两算法任意两状态之差，再观察未归一化参考值的增量。把模型改成确定性二周期链，检查价值差是否仍平稳收敛。

[源码](../implementations/average_systems/rvi_multistate.py) · [逐种子记录](https://yingwen.io/crl-code/results/rvi_multistate/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/rvi_multistate/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/rvi_multistate/curves.json)

<a id="lesson-dyna"></a>

## 6 · 学习模型后规划：同一方程，不同误差来源

Differential Q 的采样更新不要求样本一定来自真实世界。它也能处理模型生成的转移。但在真实模型与错误模型上收敛，是两件事。先记录每个已访问状态动作的次数、奖励总和和终点次数。它们构成经验模型。未见状态动作不应被默认为“零奖励、返回自己”。

$$
\widehat r(s,a)=\frac{\sum_{i:(S_i,A_i)=(s,a)}R_{i+1}}{N(s,a)},\qquad\widehat P(s'|s,a)=\frac{N(s,a,s')}{N(s,a)}.
$$

分母只在 N(s,a)>0 时有效。本页模型是有限状态频数模型；不把对已访问样本的拟合误差等同于覆盖完整世界。

$$
\delta^{\rm model}=\widehat r(s,a)-\bar g+\sum_{s'}\widehat P(s'|s,a)\max_{a'}Q(s',a')-Q(s,a).
$$

这是一次状态动作的期望备份。Q(s,a) 与奖励率仍使用同一个旧参数 TD 误差；随后才轮到下一次规划备份。采样模型变体用一个模拟终点替代求和。

**算法：Differential Dyna 教学实现；本页 K=5，不表示最优调度**

1. 初始化 Q、奖励率 g、空模型与规划随机数流
1. 每次真实交互：
  1. 1. epsilon-greedy 选动作；执行一次真实转移
  1. 2. 用真实样本对 Q 与 g 做一次 Differential Q 更新
  1. 3. 累积该状态动作的奖励和终点频数
  1. 4. 重复 K 次：从已观察状态动作中选一个；做一次经验模型备份
  1. 5. 记录真实环境步 t、模型备份 Kt、真实累计奖励
1. 日志窗口结束时不重置状态、Q、g 或模型

数据少时，多规划可能反复强化错误转移。数据多时，多规划可以让远处的奖励变化更快影响当前动作。两者是同一种复用机制。必须同时报告样本效率与计算代价。本页 1200 个真实转移加 6000 个模型备份，不能称为与只做 1200 次更新的基线“相同计算预算”。

$$
\left|\widehat r-r+(\widehat P-P)h\right|\leq|\widehat r-r|+\frac12\|\widehat P-P\|_1\operatorname{sp}(h).
$$

这是固定状态动作、固定 bias 下的一步期望目标误差界；两行转移概率都归一化，故共同常数相消。它不是最终策略损失界。若长期后果使 $\operatorname{sp}(h)$ 很大，很小的转移误差也可能产生明显备份误差。更多模型更新只能反复利用现有模型，不能凭空减少其统计误差。

因此平均奖励规划要同时处理三层问题：先保证模型与真实时间单位一致；再处理未知奖励率和价值原点；最后检查规划算子的收敛条件。没有折扣收缩时，减去参考值只解决第二层中的公共平移，不能代替第三层。经验模型若混合不同环境阶段或不同 option 版本，第一层也已经失效。

固定正确模型的规划理论，不自动覆盖一边改变的经验模型、不断变化的策略、常数步长和非平稳环境的耦合系统。本页将这些组合明确标为教学实验。负面结果同样保留。

<a id="experiment-differential_dyna"></a>

### 实验：实验 · 学得模型的五次规划值得多少真实经验

相同真实交互下，经验模型备份能否更早形成好的策略？

**环境与可用信息。** 三状态、双动作、完全可观测的持续 MDP。状态 0/1/2 的两动作奖励依次为 (0.1,−0.05)、(0.5,0.05)、(0.2,1.2)。优先后继依次为 (1,2)、(0,2)、(0,1)：转移以 0.8 概率取该后继，以 0.2 概率在三状态中均匀选择。无终止、无外部重置。控制器只能看到实际转移；模型由各状态动作的奖励均值和后继频数学习，不能访问真实转移矩阵。

**设置。** 1200 个真实步；状态 0、Q=0、率=0 起步。ε=0.25，Q 步长 0.08，率步长 0.008。每个真实步后做 5 次已访问状态动作的模型期望备份；模型自由对照没有这些备份。规划 RNG 与环境 RNG 分开。已运行种子为 0、1、2、3、4；图中离散程度是种子间样本标准差，不是置信区间。

**检验的机制。** 先做真实差分 Q 更新，再写入经验模型，再用 planning_update 更新 Q 和率。模型不能从未访问状态动作生成知识；模拟奖励不计入真实生命期奖励。

**测量。** 主图是冻结当前贪心策略后，用真实模型精确解得的 gain。另看真实交互奖励率和总备份数：最终策略能力、学习过程所得收益和计算开销是三个量。

```bash
python3 implementations/average_systems/differential_dyna.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](../assets/crl-figures/result-differential_dyna.svg)

横轴：environment_steps。纵轴：冻结贪心策略的精确平均奖励。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 固定当前 Q 所选的贪心策略，在真实三状态模型上求其长期平均奖励。真实模型只用于评价。value 不是算法内部的 gain_estimate，也不是带探索的行为所收到的平均奖励。

**step：怎样计时。** step 是真实交互数。两方法每步都直接更新 Q 和率；Differential Dyna 另做5次经验模型期望备份。environment_updates、model_backups、total_backups 分别记录这两类更新及其和，尚未计模型维护与求和的全部运行时间。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 全程行为奖励率另看 experienced_reward_rate：它已把从第1步开始的探索与学习成本纳入平均，不应再次平均各稀疏检查点。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算奖励率差 optimal_gain − value 和各更新计数。experienced_reward_rate × step 给出累计真实奖励；模型奖励不加入该总量。日志未保存 Q 表与每步奖励，不能仅凭冻结策略的 value 恢复策略或逐步行为轨迹。

计算位置：[average_systems/differential_dyna.py](../implementations/average_systems/differential_dyna.py) · [average_systems/differential_q_multistate.py](../implementations/average_systems/differential_q_multistate.py) · [average_systems/_common.py](../implementations/average_systems/_common.py)

</details>

**结果分析。** 第 300 步，Dyna 的冻结策略 gain 均值约 0.58763，对照约 0.42844；第 1200 步两者均为最优 0.59704。真实全程奖励率分别约 0.50207、0.45040，但总备份数分别为 7200 和 1200。相同交互不等于相同计算。

**结论边界。** 这是有限经验模型的组件实验，不是非平稳或神经 Dyna 的收敛证明。两者最后达到同一策略，不能仅凭终点图说明规划无用；也不能忽略六倍备份成本。

**继续实验。** 比较真实步预算相同和总备份预算相同两种口径。另冻结模型学习但保留规划，说明旧模型何时无法提供新世界的证据。

[源码](../implementations/average_systems/differential_dyna.py) · [逐种子记录](https://yingwen.io/crl-code/results/differential_dyna/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/differential_dyna/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/differential_dyna/curves.json)

<a id="lesson-duration"></a>

## 7 · 随机持续时间与半马尔可夫更新

执行一次导航技能，可能持续 2 步，也可能持续 200 步。若每次 option 结束只平均奖励，慢技能可能因为单次收益高而被偏爱。单位原始步奖励应为总奖励除以总耗时。在独立重复执行、均值有限且平均时长为正的例子中，它是 $\mathbb E[R]/\mathbb E[\tau]$，一般不同于 $\mathbb E[R/\tau]$。

![两个回到同一状态的环，A每秒得2，B每三秒得3；下方把两者放到同一段十二秒时间，奖励总量分别为24和12。](../assets/crl-figures/concept-depth-average-clock.svg)

沿环走一圈才发生下一次高层决策。A 每圈得 2，B 每圈得 3，所以按决策次数平均会偏爱 B。下方每枚圆点表示 1 奖励，横轴使用同一个物理时钟；此时 A 的奖励率是 B 的两倍。回到 S 是环境自身的转移。原创确定性计算；[计算与绘图代码](https://yingwen.io/crl-code/figures/continual-visual-depth.mjs)。

把 A、B 交替执行，一个两次决策的周期花 $1+3=4$ 秒，获得 $2+3=5$ 奖励，奖励率为 $5/4$。若先算每次执行的率，再对两次决策等权平均，却得到 $(2/1+3/3)/2=3/2$。差异来自 B 占用了四分之三的真实时间，却只占一半的决策次数。需要平均的是按时长加权的收益率。

在这个单状态控制问题里，最优奖励率为 $g_*=2$。以 $\max_o q(S,o)=0$ 固定价值原点，半马尔可夫方程给出 $q(S,A)=2-2\times1=0$、$q(S,B)=3-2\times3=-3$。这里的 $-3$ 表示一次选择 B 相对于最优奖励率损失了多少奖励，不是 B 的每秒收益。

$$
q_*(s,o)=\mathbb E\!\left[R_{t:t+\tau}-g_*\tau+\max_{o'}q_*(S_{t+\tau},o')\mid s,o\right]
$$

$R_{t:t+\tau}$ 是 option 内未折扣的外部奖励总和，$\tau$ 是原始步数，最大值只对终点可用的 options 取。时间扣除项为 $g_*\tau$，反映技能执行期间的机会成本。

从方程得到一个直接学习规则：用本次累计奖励和实际时长形成 $\delta_{\rm sample}=R-\bar g\tau+\max Q(s',\cdot)-Q(s,o)$，再据此更新价值和奖励率。问题在于，如果率估计偏高 $\varepsilon$，仅这个误差就让本次目标偏低 $\varepsilon\tau$。同一个步长对两步技能与两百步技能的作用因此可能相差很大。

一种处理办法是估计各状态—技能对的期望时长 $L(s,o)$，并按它缩放更新。先固定本次更新前的 $Q,\bar g,L$，记 $L_*=\mathbb E[\tau\mid s,o]$、$\Delta Q=\max Q(S',\cdot)-Q(s,o)$，便能看清这种缩放改变了什么：

$$
\begin{aligned}\mathbb E[\delta_{\rm sample}\mid s,o]&=\mathbb E[R+\Delta Q\mid s,o]-\bar g L_*,\\\mathbb E\!\left[\frac{R-\bar g L+\Delta Q}{L}\middle|s,o\right]&=\frac{\mathbb E[R+\Delta Q\mid s,o]-\bar g L}{L}.\end{aligned}
$$

$L$ 是当前保存的正数，不是本次随机持续时间。当 $L=L_*$ 时，第二行等于第一行除以正数 $L_*$：该状态—技能对的零点不变，更新尺度改变。学习到准确期望时长及整个迭代的收敛仍需相应条件，不能只由这一步代数推出。

下面是这一期望时长归一化算法。先缓存旧 $Q,\bar g,L$，形成同一个误差；价值与率都使用该误差及旧 $L$，最后才根据本次 $\tau$ 更新长度估计。技能必须停止且期望时长有限，长度估计始终为正。

$$
\begin{aligned}\delta&=R-\bar g L_{\rm old}+\max_{o'}Q(s',o')-Q(s,o),\\Q(s,o)&\leftarrow Q(s,o)+\alpha\delta/L_{\rm old},\\\bar g&\leftarrow\bar g+\eta\alpha\delta/L_{\rm old},\\L(s,o)&\leftarrow L_{\rm old}+\alpha_L(\tau-L_{\rm old}).\end{aligned}
$$

若改为采样 τ 直接扣除、同时又用随机 τ 作分母，噪声与期望可能改变；必须追踪相应算法及其假设，不能把 L 和 τ 任意互换。

$$
\mathbb E\!\left[\frac{R-\bar g\tau+\Delta Q}{\tau}\middle|s,o\right]=\mathbb E\!\left[\frac{R+\Delta Q}{\tau}\middle|s,o\right]-\bar g
$$

随机 $\tau$ 进入分母后，一般不能移到期望之外，零点也可能改变。即使每次回到同一状态、使 $\Delta Q=0$，右侧也在比较 $\mathbb E[R/\tau]$，而不是本节的 $\mathbb E[R]/\mathbb E[\tau]$。因此“消除长技能的大更新”与“保持原目标的零点”必须同时满足。

可以在配套 `lifelong_algorithms_lab.py` 的 `option_rate_step` 中逐行核对：`error` 对应 δ，`scaled` 保存 αδ/L，返回前两个分量更新价值和率，第三个分量才改变 L。运行 `python3 examples/lifelong_algorithms_lab.py average` 会同时输出一次技能更新及原始动作下的预测、控制结果；源码在本章后面的实现部分。真实时长用于学习 L，未被丢弃或改记为一次决策。

规划一个固定 option 时，模型至少要保存条件期望累计奖励、条件期望时长和结束状态分布。期望目标为 $\widehat R-\bar g\widehat\tau+\widehat P_o\max Q$。option 内部策略改变，这三个量都可能改变；只更新高层 Q 而保留旧时长和结束模型，求解的就是混合了不同时期语义的方程。未执行完的 option 也不能在日志窗口末尾被伪造为已终止。

这些后果将接入[完整架构的逐步实现](architectures.md#lesson-cycle-completion)。读到那里时要保留一个区别：本节用期望时长估计作扣除与归一化，逐步实现用本次采样时长直接扣除、更新不除以时长。这是不同的学习规则，即使它们都以同一个固定技能的差分方程为出发点。该实现先固定技能，分别追踪真实样本更新与模型备份；技能也学习的原型另列为深入阅读。固定策略的长期奖励率与有限运行的实测全程收益也应区分，后者受采集行为、学习过程和随机后果影响。

<a id="research-average-smdp-rvi"></a>

## 时间抽象与规划 · SMDP 的 RVI 需要学平均时长，而非平均每次比率

深入阅读：回到随机时长下的相对价值求解条件与模型更新。

Yu、Wan 与 Sutton 的 SMDP RVI 工作（2025 预印本）把动作持续时间纳入相对价值迭代。固定 option 后，一次高层决策产生累计奖励 R、持续时长 τ 和结束状态。目标仍是每单位原始时间的奖励，不是每次 option 的奖励。

$$
Q(s,o)=\bar R(s,o)-g\,\bar\tau(s,o)+\sum_{s^+}P_o(s^+\mid s)\max_{o^+}Q(s^+,o^+)
$$

固定 option 所诱导 SMDP 的最优性方程。R̄ 与 τ̄ 是条件期望。它使用期望累计奖励和期望时长，不要求 R 与 τ 独立，也不把 E[R/τ] 当作奖励率。

$$
Q^+(s,o)=Q(s,o)+\alpha\!\left[\frac{R+\max_{o^+}Q(S^+,o^+)-Q(s,o)}{T(s,o)\vee\varepsilon}-f(Q)\right],\quad T^+(s,o)=T(s,o)+\beta\bigl(\tau-T(s,o)\bigr)
$$

原文模型无关 RVI 的主要更新结构。T 估计期望时长，ε 为正下界，∨ 表示取最大。当前 Q 更新使用更新前的 T。f 是满足论文条件的全局参照函数，不是随意选取的行为奖励均值。

分母的意义可由最优性方程看清：先将累计奖励与相对价值差除以平均时长，再减去统一的率参照。若直接用每个样本的 τ 除，通常改变期望更新。期望时长估计、访问频率、参照函数与步长条件因此必须一起分析。

收敛讨论要求固定有限 SMDP、适当的弱连通或最优性方程结构、覆盖、时长与奖励的矩条件及非爆炸条件。参照函数可以非线性，不代表 Q 可以任意采用深度神经网络。option 内部策略持续变化，会同时改变其奖励、时长和结束分布；固定 SMDP 的证明不再直接适用。

**算法：接入整体智能体时的接口要求；不是该论文已经证明的非平稳 option 算法**

1. 记录每个 option 样本的策略版本、外部奖励总和、原始步数和结束状态
1. 对同一版本分别学习奖励、时长和结束分布模型
1. 只用语义相同的模型计算高层规划 target，并记录模型备份预算
1. option 策略改变时，使旧模型失效或明确建模版本差异
1. 把“模块各自收敛”与“所有模块共同变化时稳定”分成两个研究问题

配套的异步随机逼近研究进一步澄清异步更新与参照函数条件。尚未确认 SMDP 论文独立发布的作者代码。较早的 average-reward-methods 及 average-reward options 工程可以作为前置材料，但不能冒充这一版本的新算法实现。

<a id="research-average-harmonic-audit"></a>

## 目标一致性检查 · 不能把算术奖励率直接替换为调和平均

深入阅读：通过目标反例检验不同平均方式是否仍比较同一长期收益。

一项 2026 预印本提出以 harmonic aggregation 改写平均奖励。阅读这类新提议时，先检查它是否仍在优化原来的物理量。下面只审查目标一致性，不因论文使用“非平稳”或“相关性”就接受新的聚合规则。

$$
\frac{\sum_{i=1}^{n}R_i}{\sum_{i=1}^{n}\tau_i}=\sum_{i=1}^{n}\frac{\tau_i}{\sum_j\tau_j}\,\frac{R_i}{\tau_i},\qquad\tau_i>0
$$

有限样本的恒等式：总奖励除以总时间，等于各段奖励率的时间加权算术平均。此恒等式不需要独立性，也不需要平稳性。随机极限是否存在，是另一个问题。

取两段交互记录：(奖励, 时长) 分别为 (1,1) 和 (4,2)。总奖励率为 5/3。两个局部率为 1 和 2，其简单调和平均为 4/3。二者在没有噪声的例子中已经不同。这个反例不评价某种聚合在其他目标上的用途，只说明不能未经附加条件便宣称两者等价。

固定路程下求平均速度时，调和平均有适用条件：每段路程相同。在奖励问题中，对应的是每段奖励满足特定约束，而不是任意 option。若另加正负奖励分组、平移或截断，也必须重新证明目标与策略排序；数值更平稳不是目标等价的证明。

**算法：教材层面的可复查反例；该提议不作为默认训练算法**

1. 对任意聚合提议，先写出优化对象及其单位
1. 检查常数奖励偏移、时间拆分、负奖励和不同持续时长
1. 构造两段确定性交互记录，比较目标值与策略排序
1. 若排序改变，把它命名为新的偏好目标；不要仍称同一平均奖励问题

尚未确认该预印本对应的作者代码。这里保留原文入口，目的在于学习如何审查一个目标主张。当前平均奖励与整体智能体实现继续使用实际外部奖励总和除以实际时间。

<a id="average-rate-task"></a>

### 7.1 · 贯穿算例：让预测与控制共用一个时钟

深入阅读：用一个共同物理时钟的有限任务，逐步核算下面的预测、改善和规划。

前面的单状态例子隔离了时间分母。现在给服务系统加入启动过程和可预测的中间状态，使同一个世界能够用于策略评价、一次学习更新、策略改善与模型规划。第一册的[服务站算例](../foundations/approximation/average-control.md#average-rate-cycles)从预测讲起；这里完整给出环境，再追踪这些量怎样被控制器使用。

| 当前位置与动作 | 后继状态 | 动作总奖励 R | 耗时 τ（秒） |
| --- | --- | --- | --- |
| E：启动 | H | −3 | 2 |
| H：选快环 | A | 0 | 1 |
| A：返回 | H | 4 | 1 |
| H：选慢环 | B | 0 | 1 |
| B：返回 | H | 9 | 3 |

E 只在生命开始时访问一次；H 是反复返回的服务站。所有奖励在对应动作完成时到账，执行期间没有额外奖励或新选择。以 $n$ 数决策间的转移，以 $T_n$ 数真实秒，$\tau_n=T_{n+1}-T_n$。快环 $\pi_F$ 固定在 H 选 A，慢环 $\pi_L$ 固定选 B。每条策略都只有一个常返类，允许另一分支与 E 为暂态；两条确定性循环都有周期。

$$
g_F=\frac{4}{1+1}=2,\qquad g_L=\frac{9}{1+3}=\frac94.
$$

两条策略的 gain 单位均为奖励/秒。这里是每圈总奖励除以总时长，启动成本不改变长期斜率。真实生命中启动的 2 秒与损失 3 仍然保留。

看图前预测：如果日志把每个动作记成一行，然后把每行的“奖励/耗时”等权平均，哪条路线会被误判？

![四状态服务站：启动两秒成本3；快环两秒得4，慢环四秒得9，所有边与奖励到账时刻共享物理秒数坐标。](../assets/crl-figures/average-rate-walkthrough-cycles.svg)

每条环的水平长度按真实秒数，实心小圆点每枚代表 1 奖励。快环每次回站要 2 秒，慢环要 4 秒；H 的再次出现表示自然返回。原创确定性算例；[解析核](https://yingwen.io/crl-code/figures/average-rate-walkthrough.mjs)与[标准库教程](../tutorials/average_rate_walkthrough.py)可重算。

$$
\frac{0/1+4/1}{2}=2,\qquad\frac{0/1+9/3}{2}=\frac32<2.
$$

逐次率平均把慢环的 3 秒返回动作与 1 秒出发动作赋予相同权重，因而反转了真实排序。每次转移平均奖励则是 2 与 9/2，单位为奖励/决策，同样不等于奖励/秒。

令 $\bar d_\pi$ 为“只在决策时刻观察”得到的嵌入链平稳分布，$r_\pi(s)$ 和 $\ell_\pi(s)$ 分别为一次动作的期望总奖励和期望时长。将半马尔可夫 Poisson 方程左乘 $\bar d_\pi^\top$，后继价值相消，得到：

$$
h_\pi=r_\pi-g_\pi\ell_\pi+P_\pi h_\pi,\qquad g_\pi=\frac{\bar d_\pi^\top r_\pi}{\bar d_\pi^\top\ell_\pi}.
$$

分母是每次决策之间经过的平均真实时间。慢环决策分布在 H、B 各占 1/2，因此 $g_L=(9/2)/[(1+3)/2]=9/4$。按决策次数的平稳分布不是按每秒观察的占用分布。

若每次到 H 以概率 p 选慢环，则期望一圈奖励为 $4(1-p)+9p$，时间为 $2(1-p)+4p$，所以 $g(p)=(4+5p)/(2+2p)$。$p=1/2$ 时为 $13/6$，不同于两条固定路线率的均值 $17/8$。这个区别只来自时长，没有估计噪声。

<a id="average-rate-prediction"></a>

### 7.2 · 先学哪一个量：同一规范下的 gain 与 bias

深入阅读：沿同一任务显式解出奖励率和参考对齐的差分价值。

固定策略后，取 $h_\pi(H)=0$。从每个状态到下一个决策时刻，扣除这段时间本可按 $g_\pi$ 获得的收益。每条边满足 $h_\pi(s)=R-g_\pi\tau+h_\pi(s')$；在 H 对策略选择取期望。这里通过 Poisson 方程固定差分价值，周期链的普通无限中心化和不必收敛。

$$
h_\pi(E)=-3-2g_\pi,\quad h_\pi(A)=4-g_\pi,\quad h_\pi(B)=9-3g_\pi,\quad h_\pi(H)=0.
$$

第一条同时计入启动成本与启动期间的机会成本。其余两条保留到站奖励与等待秒数。

| 策略 | g（奖励/秒） | h(E) | h(H) | h(A) | h(B) |
| --- | --- | --- | --- | --- | --- |
| 快环 | 2 | −7 | 0 | 2 | 3 |
| 慢环 | 9/4 | −15/2 | 0 | 7/4 | 9/4 |

检查慢环的两条常返边：H 的残差为 $0-9/4+9/4=0$，B 的残差为 $9-3\times9/4-9/4=0$。即使只跟随慢环，模型也能算出暂态 A 和 E 的值；真实在线流却不会反复访问 E。这里算出所有状态是模型评价，不声称单条生命能无限次采样启动阶段的经验。

把 N 次高层转移的方程相加，可得到精确的有限交互序列关系：

$$
\mathbb E_\pi\!\left[\sum_{n=0}^{N-1}R_n\right]-g_\pi\mathbb E_\pi[T_N]=h_\pi(S_0)-\mathbb E_\pi[h_\pi(S_N)],\qquad T_0=0.
$$

$N$ 是固定转移次数，$T_N$ 是完成它们后的真实时刻。若窗口恰在长动作中途截止，还需保留未完成动作的状态；不能把它伪造成已返回 H。

未知模型时，一条实际返回样本也能形成残差。固定慢环策略，旧估计为 $\bar g=2,\hat h(B)=2,\hat h(H)=0$，观察 B→H 得 9、花 3 秒。用已知确定性期望时长 $L=3$，采用前节 inter-option 预测更新的特例：

$$
\delta=9-2\times3+0-2=1,\quad\Delta=\frac{\alpha\delta}{L}=\frac1{10},\quad\hat h^+(B)=2.1,\quad\bar g^+=2.05.
$$

取 $\alpha=0.3$ 秒、$\eta=0.5/\text{秒}$，率增量为 $\eta\Delta$。一秒为时间单位时直接使用数值 0.3、0.5。固定确定性策略下每个访问状态只有一个选定动作，$\hat h(s)$ 对应其动作价值估计。

这个单步核采用 Wan、Naik 与 Sutton 的 options 论文 §3，式 (6)–(10) 的时长归一化与旧参数时序。价值与率必须共用同一个 δ：先改率再重算，会把 δ 从 1 变为 0.85。随机时长下，论文使用期望长度估计 L；本例 L 恰好等于每次实际时长，不把两者在一般情形中混用。

<a id="average-rate-control"></a>

### 7.3 · 从预测到控制：一次策略改善能增加多少率

深入阅读：核算一次策略改善的奖励率差，而不是只比较未经对齐的价值。

预测回答“总选快环会怎样”。控制再问“只在当前 H 改选一次慢环，随后恢复快环，会增加多少收益”。使用刚算出的快环 $g_F=2$ 和 $h_F$，定义带时间成本的动作价值：

$$
q_F(H,F)=0-2\times1+h_F(A)=0,\qquad q_F(H,L)=0-2\times1+h_F(B)=1.
$$

相对 $h_F(H)=0$，选择一次慢环多出 1 奖励。它是差分动作价值，单位为奖励；数值 1 不是“每秒提高 1”。

将这种选择每次重复，就得到慢环策略。慢环每圈用 4 秒，多出的中心化奖励恰为 $9-2\times4=1$，所以真正的率改善是 $1/4$。更一般地，令 $A_\pi(s,a)=r(s,a)-g_\pi\ell(s,a)+(P_a h_\pi)(s)-h_\pi(s)$，将它按新策略的嵌入链平稳分布求平均：

$$
g_{\pi'}-g_\pi=\frac{\sum_s\bar d_{\pi'}(s)\sum_a\pi'(a\mid s)A_\pi(s,a)}{\sum_s\bar d_{\pi'}(s)\sum_a\pi'(a\mid s)\ell(s,a)}.
$$

后继 bias 项仍因平稳性相消，分母把每次决策的改善换回每秒。本例新策略在 H、B 各占 1/2，平均优势为 1/2，平均时长为 2 秒，故 $g_L-g_F=(1/2)/2=1/4$。当所有动作耗时为 1，这条式子退化为前面的 MDP 性能差异公式。

在这个给定模型上，规划可先精确评价快环，再按 q 选择慢环，最后重新解 Poisson 方程。这就是一次完整的策略迭代。新评价得到 $g_L=9/4$，此时 $q_L(H,F)=4-2\times9/4=-1/2$、$q_L(H,L)=9-4\times9/4=0$，动作不再改变。由于 $g(p)$ 在 $[0,1]$ 上递增，慢环也优于任何固定随机混合策略。

精确模型允许查阅 B 的后果。若只从一直执行快环的经验学习，B 从未被访问，改善所需的值就没有数据支持。模型规划、实际探索与一次样本更新使用不同的信息，不能把解析策略迭代的答案当作学习器已经获得的知识。

再预测一个有限时间问题：两个相同服务系统都从 E 启动，一个固定快环，一个固定慢环。只运行 12 秒时，奖励率较高的慢环一定已经赚得更多吗？

![真实累计奖励的阶梯曲线：第12秒快环17高于慢环15，第18秒快环29低于慢环33，启动成本保留。](../assets/crl-figures/average-rate-walkthrough-lifetime.svg)

蓝实线为快环，橙虚线为慢环；横轴从同一 E 启动时计秒，奖励只在动作完成时到账。慢环有更高长期斜率，却在第 12 秒落后，因为下一笔奖励尚未到达。原创逐秒确定性展开，图中点由[教程](../tutorials/average_rate_walkthrough.py)独立重算。

第 12 秒：快环完成 5 圈，收益 $-3+5\times4=17$；慢环完成 2 圈，收益 $-3+2\times9=15$。第 18 秒：两者分别完成 8 圈和 4 圈，收益为 29 与 33。这个比较无需引入学习误差。研究完整 CRL 学习器时，真实收益还会包含探索、模型更新与适应过程，因此应保留生命期累计收益，不能只用最终冻结策略的 gain 替代。

<a id="average-rate-planning"></a>

### 7.4 · 时长模型改变，旧规划答案如何失效

深入阅读：改变技能时长模型，分析何时会改变规划的动作选择。

仍在这个服务站，只让 B 返回 H 的耗时从 3 秒变成 4 秒；奖励 9 与终点 H 都不变，其他转移也不变。旧模型会继续报告慢环率 9/4，真实慢环却已经变为 9/5，低于快环的 2。这给出了一个只改变时长、即可使动作选择反转的控制问题。

看图前预测：若模型的奖励均方误差与终点预测误差都为零，只保留过时的时长，它能否继续正确规划？

![旧模型把完整慢环记为4秒得9，新世界需要5秒得9；率从2.25降到1.8，越过快环率2。](../assets/crl-figures/average-rate-walkthrough-model.svg)

上方两条路线共用秒数坐标，紫色虚线表示旧模型，橙实线表示改变后的真实转移。下方同一奖励/秒轴上的两根条形跨过快环率 2 的参考线。只改变 B→H 的时长；这是给定模型的精确比较。

$$
h_F^{\rm new}(B)=9-2\times4=1,\qquad q_F^{\rm new}(H,L)=0-2+1=-1.
$$

更新时长后，先按快环基准重新评价 B，再把该值传回 H，就会选快环。若只更新 B 的值却还未传播到 H，暂时的动作错误来自未完成的备份；若时长模型本身仍错，重复同一个旧备份不会修正答案。

因此一条 option 模型记录至少要绑定期望总奖励、期望时长、终点分布及对应的内部策略版本。预测精度在这里由控制用途检验：只预测终点与奖励仍不足以比较每秒收益。环境何时改变、怎样发现时长变化、需要多少真实样本以及多久完成传播，是下一层学习问题；本算例只固定改变前后的模型，精确显示它们要求不同的动作。

下载下方单文件教程，在仓库根目录重算预测、控制、规划与真实累计收益

```bash
python3 tutorials/average_rate_walkthrough.py
```

完整[标准库教程](../tutorials/average_rate_walkthrough.py)使用 Fraction 解四条策略方程及 h(H)=0 的规范，逐秒展开真实过程；[图形计算核](https://yingwen.io/crl-code/figures/average-rate-walkthrough.mjs)另由再生周期求解。两种计算交叉检查所有状态，包括暂态 E。脚本不运行随机训练，也不把单步备份通过当作收敛或效率证据。

<a id="lesson-chain-geometry"></a>

## 8 · 为什么去掉折扣后，问题会更难

$$
\operatorname{sp}(x)=\max_s x(s)-\min_s x(s),\qquad T(h+c\mathbf1)=Th+c\mathbf1.
$$

span 是半范数。相差常数的两个向量距离为零，恰好适合差分价值。平均奖励 Bellman 算子保留常数方向；它通常不像折扣算子那样在最大范数下严格压缩。参考归一化只去掉一个无关方向，不能单独保证收敛。

后文学习曲线的三状态实验，每行转移都含同一均匀混合成分 0.2；这不是 §1.1 的 A/B 分支解析反例。任意两个状态动作的下一状态分布至少共享这个质量，因此该学习实验的 Bellman 算子在 span 下至多以 0.8 缩小差异。这是该模型可验证的附加结构，不是所有持续任务的性质。把共同混合质量减小，会使远期状态分布更难由短轨迹估计。

周期反例：0→1 奖励 0，1→0 奖励 2。奖励率为 1，Poisson 方程也有解。但从 h=[0,0] 开始，同步 RVI 的 h(1) 在 2 和 0 之间振荡。存在价值函数，不等于某个迭代器逐点收敛。修改规划算子、加入惰性自环或采用合适的平均处理，需要重新说明是否改变了问题与时间单位。

多常返类反例：两个互不相通的吸收状态分别每步奖励 0 和 2。奖励率依赖起始状态，没有一个标量 g 可以满足两个状态的 Poisson 方程。弱连通控制问题允许更一般的结构，但不能据此断言所有固定策略都只有一个常返类。

慢混合意味着一次长期轨迹仍可能长时间留在某一区域。短窗口均值、参考状态更新和策略比较都因此高方差。若新策略要在很久之后才到达高收益区域，当前评价窗口可能无法看到它的优势。这首先是信息与时间尺度问题，不是把学习率再调小就能消除的噪声。

$$
h_\pi=(I-P_\pi+\mathbf1d_\pi^\top)^{-1}(r_\pi-g_\pi\mathbf1),\qquad d_\pi^\top h_\pi=0.
$$

有限单常返类链中，这个增广矩阵可逆。它将常数方向用稳态规范补齐。与折扣情形的 $(I-\gamma P_\pi)^{-1}$ 相比，平均奖励需要利用链的结构控制剩余方向；可逆、数值条件好、某个迭代收敛，是三个不同命题。

$$
\lim_{\gamma\uparrow1}(1-\gamma)V^\pi_\gamma(s)=g_\pi.
$$

在这里的有限遍历固定策略设定下，该极限成立。但一个固定的 $\gamma=0.99$ 并不等于平均奖励。短期瞬态、混合时间、动作间差异仍可能决定折扣策略排序；也没有一个固定折扣能统一解决任意长时标的问题。

还要区分极限关系与 Blackwell 最优性。在固定的有限状态、有限动作、有界奖励 MDP 中，存在一条确定性平稳策略，对所有足够接近 1 的折扣都最优。这类策略也是 gain 最优，但任意 gain 最优策略未必都是 Blackwell 最优。阈值依赖具体 MDP；本章 A/B 反例的阈值是 $20/23$，不是通用常数。

$$
V^\pi_\gamma(s)=\frac{g_\pi}{1-\gamma}+h_\pi(s)+O(1-\gamma),\qquad d_\pi^\top h_\pi=0.
$$

在这里的有限单常返类设定下，折扣 Laurent 展开给出这个规范。gain 并列时可以继续比较该展开的常数项，必要时比较更高阶系数。不能把每条策略分别设 $h_\pi(s_{\rm ref})=0$ 后的数值直接当作跨策略的 Laurent 系数：这会把参考起点处真正不同的暂态项都强制清零。Blackwell 最优还要求所有足够大折扣下的策略最优性，不能只凭任意 bias 数值排序宣布。

2023 年关于 Blackwell 折扣阈值的工作研究怎样把平均奖励/Blackwell 最优求解归约为特定折扣问题。这是带问题依赖阈值的理论归约，不是给任意深度 RL 算法设置 0.99 就获得平均奖励保证。学习一个未知 MDP 时，还要支付估计这些长时标差异的样本成本。

<a id="research-average-oracle-complexity"></a>

## 平均奖励的样本复杂度 · 逐渐增大折扣与固定大折扣并不相同

深入阅读：进一步区分已知模型、可查询生成模型与真实在线交互中的样本复杂度。

Jiao 等人的 2026 工作分析从单智能体到联邦设置的平均奖励 Q-learning 样本复杂度。其一个重要机制是配合调度逐渐增大折扣，而不是宣布任何足够接近 1 的固定折扣都与平均奖励等价。样本来自能查询指定状态动作的生成模型。

$$
Q_{k,t}(s,a)=(1-\eta_{k,t})Q_{k,t-1}(s,a)+\eta_{k,t}\!\left[(1-\gamma_k)r(s,a)+\gamma_k\max_{a^+}Q_{\iota(k,t)}(S^+,a^+)\right]
$$

k 为外层轮次，t 为轮内更新；ι 指定使用哪一次历史 Q。两种调度分别使用上一步 Q 和上一轮最终 Q，不能忽略这个区别。即时奖励乘以 1−γ，因此这里的 Q 既不是未缩放的折扣 Q，也不是已规范化的平均奖励 bias。

为什么缩放？在固定遍历策略下，折扣价值通常含随 1/(1−γ) 增大的共同率分量。乘以 1−γ 后，主要尺度变成 gain。控制问题仍需处理 bias span、有限样本误差和策略提取；仅有固定策略的极限恒等式，不足以证明任意实际算法正确。

原文在表格、弱连通等条件下给出样本复杂度，并研究多个客户端的独立采样与通信。每轮为所有状态动作采样的权限不同于真实生命里逐步选择动作。生成模型结果没有支付到达稀有状态的探索成本，也不等于机器人可以无代价复制自己的交互历史。

- 先在已知模型上画固定策略的缩放折扣值与精确 gain，分别画参考对齐后的 bias。
- 再用可查询任意状态动作的模拟器核对同步采样算法，统计每次模型查询。
- 最后换成单条轨迹，单独研究覆盖与混合；不要直接沿用生成模型复杂度。

尚未确认论文对应的作者代码仓库。独立教学实现应保留奖励缩放和全部调度，并明确模型查询预算。删去这些设计、只把现有 DQN 的 γ 调大，不是这一理论算法的实现。

<a id="lesson-example"></a>

## 9 · 手算：奖励率、控制目标与更新时序

每两步都重复获得 0、2，为什么还需要两个不同的状态价值？先把共同的每步收益从奖励流中拿走，再看起点之间留下什么差别。

![两状态循环产生零和二交替的奖励脉冲，减去每步基准一后，中心化奖励与相对值变化互相抵消。](https://yingwen.io/crl-figures/concept-crl-mechanisms-average.svg)

每步等长的持续环，没有终止或外部重置。上方每两步产生两枚奖励；中间虚线是每步基准，减去它后得到 −1、+1。下方取 $h(0)=0$、$h(1)=1$：沿每条边，中心化奖励与后继价值差恰好相消。此周期例按 Poisson 方程定义差分值；交替的中心化样本和本身不收敛。数值来自本页确定性算例。

例一，确定性两状态环：$0\to1$ 的奖励为 0，$1\to0$ 的奖励为 2。每两步收益为 2，因此 $g=1$。取 $h(0)=0$，由 $1+h(0)=h(1)$ 得 $h(1)=1$，另一个方程 $1+1=2+0$ 也成立。另取一条用于单步计算的样本：$h=[1,3]$、$\bar g=0.5$，转移 $0\to1$、奖励为 2，则 $\delta=2-0.5+3-1=3.5$。当 $\alpha=\eta=0.1$ 时，更新后 $h(0)=1.35$、$\bar g=0.535$。这条单步样本的奖励与前述固定环不同。

例二，单状态双动作：动作 0 给奖励 0，动作 1 给奖励 1，然后返回原状态。探索行为各做一半，实际奖励率为 $0.5$；最优奖励率为 $1$，最优方程要求 $Q(1)-Q(0)=1$。Differential Q 的 $\bar g$ 估计的是最优目标，因此应接近 1。这个例子把控制目标与行为奖励滑动平均明确区分开。

例三，当前 $Q=1$、$\bar g=0.5$、旧 $L=2$；option 累计奖励为 5，实际长度为 4，终点最大价值为 3。由 $\delta=5-0.5\times2+3-1=6$，取 $\alpha=\eta=0.1$ 得更新后 $Q=1.3$、$\bar g=0.53$；最后取 $\alpha_L=0.1$ 得 $L=2.2$。三个结果均按旧长度构造，更新顺序因此是算法定义的一部分。

<a id="lesson-code"></a>

## 10 · 从单步核到完整三状态实验

这里的三状态学习实验与前面的 A/B 解析反例不同。学习实验中每个状态都有两个动作，所有状态动作都有正概率转移到全部状态；转移含 0.2 的均匀混合成分。每个方法都有自己的完整循环。环境无 episode，起点只在整次训练开始时设置一次。精确线性方程只供评价器使用。前面的 A/B 例子用于分离数学目标，这里的样本实验用于检查学习过程。

| 实验问题 | 独立实现与具体函数 | 应检查什么 |
| --- | --- | --- |
| 同策略预测 | [differential_td_onpolicy.py](../implementations/average_systems/differential_td_onpolicy.py) 的 update 与 run；[sample_mean_td.py](../implementations/average_systems/sample_mean_td.py) 是奖励样本均值对照 | 奖励率和对齐 bias 都需正确；同策略均值对照不是错误算法 |
| 异策略预测 | [differential_td_offpolicy.py](../implementations/average_systems/differential_td_offpolicy.py) 的 update；[wrong_behavior_mean_td.py](../implementations/average_systems/wrong_behavior_mean_td.py) 是故意保留的错误率负对照 | $g_\pi$ 与 $g_b$ 分开；归一化 bias 好看不能证明奖励率正确 |
| 多状态控制 | [differential_q_multistate.py](../implementations/average_systems/differential_q_multistate.py) 的 max 目标与同误差率更新 | 冻结贪心策略真实 gain、探索累计奖励与 Q 残差 |
| 学习模型规划 | [differential_dyna.py](../implementations/average_systems/differential_dyna.py) 的 model_update 与 planning_update | 模型误差、未见状态动作、额外备份预算；每次备份同时更新 Q 与率 |
| 已知模型规划 | [rvi_multistate.py](../implementations/average_systems/rvi_multistate.py) 的同步 update；[unnormalized_vi_multistate.py](../implementations/average_systems/unnormalized_vi_multistate.py) 不减参考值 | 相对价值应一致；不锚定值的公共偏移增长；一次扫描为六个状态动作备份 |
| 持续适应过程 | [online_differential_q.py](../implementations/learner_control/online_differential_q.py) 与 [frozen_parameters.py](../implementations/learner_control/frozen_parameters.py) | 相同前 400 步参数/记忆快照；随后比较真实生命期收益，不以冻结策略 gain 代替 |

三状态学习环境与精确评价器在 [_common.py](../implementations/average_systems/_common.py)。evaluate 解 Poisson 线性系统，optimal 枚举八条确定性策略。它们只用于核对与记录，不把真模型、最优率或 bias 答案传入模型无关更新。平均奖励策略梯度的解析验证在本页前半和阅读推导测试中；现有这些表格实验不冒充 ARO-DDPG 或 ASAC 的完整复现。

在 continual-rl-tutorial 仓库根目录运行；仅 Python 标准库；输出目录必须尚不存在

```sh
python -m implementations.average_systems.differential_td_offpolicy --steps 1200 --out results/average-prediction
python -m implementations.average_systems.differential_dyna --steps 1200 --out results/average-dyna
python -m unittest discover -s tests -p test_average_systems.py -v
```

精确目标：固定目标策略奖励率约 0.4806503；均匀行为策略奖励率 1/3；最优奖励率约 0.5970370。它们来自本例模型的独立线性方程与八条确定性策略枚举。不是对其他任务的理论常数，也不是训练器直接读取的答案。

以下原有 lab 保留为手算回归测试。它验证单步公式，不能替代上面的多状态学习曲线。

可执行源码：函数返回新表格，避免原地修改使后续目标读到半更新参数。

```python
def differential_td(values, rate, state, reward, next_state,
                    alpha=0.1, eta=0.1, rho=1.0):
    """Tabular fixed-policy prediction. Compute BOTH updates from old values."""
    if alpha < 0 or eta < 0 or rho < 0:
        raise ValueError("nonnegative steps and importance ratio required")
    error = reward - rate + values[next_state] - values[state]
    result = list(values)
    result[state] += alpha * rho * error
    return result, rate + eta * alpha * rho * error, error


def differential_q(q, rate, state, action, reward, next_state,
                   alpha=0.1, eta=0.1):
    """Off-policy control: max target; no behavior-reward EMA."""
    error = reward - rate + max(q[next_state]) - q[state][action]
    result = [row[:] for row in q]
    result[state][action] += alpha * error
    return result, rate + eta * alpha * error, error


def rvi_sweep(h, transitions, reference=0):
    """Known model: transitions[s][a] contains (probability,reward,next_state)."""
    backed = [max(sum(prob * (reward + h[sp]) for prob, reward, sp in action)
                  for action in state) for state in transitions]
    reference_value = backed[reference]
    return [v - reference_value for v in backed], reference_value


def option_rate_step(q, rate, mean_duration, total_reward, duration,
                     next_best, alpha=0.1, eta=0.1, duration_alpha=0.1):
    """Duration-normalized expected-length variant; use OLD positive length."""
    if mean_duration <= 0 or duration <= 0:
        raise ValueError("option durations must be positive")
    error = total_reward - rate * mean_duration + next_best - q
    scaled = alpha * error / mean_duration
    return (q + scaled, rate + eta * scaled,
            mean_duration + duration_alpha * (duration - mean_duration))


def average_demo():
    values, rate = [0.0, 0.0], 0.0
    for t in range(20000):
        state = t % 2
        values, rate, _ = differential_td(
            values, rate, state, 2.0 * state, 1 - state, 0.02, 0.1)
    q, optimal_rate = [[0.0, 0.0]], 0.0
    for t in range(10000):
        action = t % 2  # behavior reward rate is exactly 0.5
        q, optimal_rate, _ = differential_q(q, optimal_rate, 0, action,
                                            float(action), 0, 0.02, 0.1)
    print("average", {"prediction_rate": round(rate, 6),
          "h1_minus_h0": round(values[1] - values[0], 6),
          "control_rate": round(optimal_rate, 6), "behavior_rate": 0.5,
          "option_step": option_rate_step(1, 0.5, 2, 5, 4, 3)})
    q_centered, reference = 0.0, 0.0
    for _ in range(1000):
        q_centered, reference, _ = centered_single_state_step(
            q_centered, reference, 1.0, gamma=0.9, eta=0.1, alpha=0.3)
    print("discounted_centering", {"q": round(q_centered, 6),
          "reference_c": round(reference, 6), "true_reward_rate": 1.0,
          "invariant_c_minus_eta_q": round(reference - 0.1*q_centered, 12)})
```

下载本页实验文件后运行；Python 3.10+，仅标准库

```sh
python examples/lifelong_algorithms_lab.py average
python examples/lifelong_algorithms_lab.py test
```

预期：两状态环 prediction_rate≈1、h1_minus_h0≈1；单状态控制 control_rate≈1，而 behavior_rate=0.5；option_step=(1.3,0.53,2.2)。测试额外验证值函数整体平移不改变 TD error、ρ=0 时不更新、零时长被拒绝、随机时长的“比值均值”反例。

作者仓库 average-reward-methods 将预测与控制分别放在 prediction_agents.py 和 control_agents.py。阅读时先对应奖励率与价值更新，再检查参考函数、探索策略、步长与环境重置规则。本页脚本用于表格机制实验；作者的统计实验还需要运行相应环境、多个随机种子及完整训练配置。

<a id="lesson-centering"></a>

## 11 · 奖励中心化：连接折扣价值与平均奖励

为什么 $\gamma$ 接近 $1$ 时，折扣价值往往很大，而动作之间的差异相对很小？设每步奖励减去固定常数 $c$。由于几何级数之和为 $1/(1-\gamma)$，所有策略、状态和动作的折扣价值都平移同一个量：

$$
Q_{r-c,\gamma}^{\pi}(s,a)=Q_{r,\gamma}^{\pi}(s,a)-\frac{c}{1-\gamma}
$$

此式假设无限持续交互、固定折扣且每一步均减去 $c$。因此固定 $c$ 不改变动作排序。若只在长度可变的 episode 内减去常数，累计平移依赖终止时间，这个结论就不再直接成立。

Naik 等人的 Reward Centering（RLC 2024）据此将共同的奖励偏移从价值学习中分离。它与 Differential TD/Q 共用一个重要思想：同时学习价值差异与标量参照量。但保留 $\gamma<1$ 时，学习的仍是中心化的折扣价值，而不是自动改成平均奖励控制。

$$
\delta_t=R_{t+1}-c_t+\gamma\max_{a'}Q_t(S_{t+1},a')-Q_t(S_t,A_t),\qquad c_{t+1}=c_t+\eta\alpha_t\delta_t
$$

这是 TD 驱动的控制中心化形式，$Q$ 同时按 $\alpha_t\delta_t$ 更新。$c$ 是联合学习的参照量；当 $\gamma<1$ 时，不应预先把它等同于精确的平均奖励率。固定常数的平移恒等式与这个联合学习过程是不同结论。

反例：单状态、单动作、每步奖励 1，取 $\gamma=0.9$、$\eta=0.1$、$Q_0=c_0=0$。两条更新使 $c_t-\eta Q_t$ 恒为零；稳定固定点为 $Q_*=5,c_*=0.5$，真实奖励率却为 1。此时 Q 正确表示中心化奖励 0.5 的折扣价值。详细推导见本章的 TD 中心化固定点研究节。on-policy 奖励均值、TD 参照与平均奖励 Differential TD 需分别命名和评价。

例如 $c=2$、$\gamma=0.99$ 时，共同价值偏移是 $200$；移除它可让网络更多容量用于区分状态和动作。Wan、Korenkevych 与 Zhu 的 continuing-task 研究（2025）进一步比较了无重置、预设重置和智能体控制重置的环境，并发现中心化不能完全消除大折扣带来的性能下降。它解决部分数值与估计问题，并不消除恢复困难或探索不足。

- 对照实验一：给所有真实奖励加同一个常数，比较中心化前后的学习曲线和价值尺度。
- 对照实验二：保留奖励但改变重置规则，区分数值问题与恢复能力。
- 对照实验三：分别采用行为奖励滑动平均、TD 驱动参照量和指定参考状态，检查 off-policy 目标是否一致。

<a id="research-average-bec"></a>

## 相邻预测方法 · Bellman Error Centering 不等于估计目标奖励率

深入阅读：比较 Bellman 误差中心化与目标策略奖励率估计。

Bellman Error Centering（2025 预印本，2026 年发表于 Neural Networks）讨论折扣预测中的误差中心化。它与本章相关，是因为“从 TD 误差减一个标量”具有不同语义。本文的 CTD/CTDC 使用折扣价值和固定线性特征，不应被归类为已经解决非线性平均奖励预测的方法。

$$
\delta_t=R_{t+1}+\gamma x_{t+1}^{\top}w_t-x_t^{\top}w_t,\quad \omega_{t+1}=\omega_t+\beta_t\rho_t(\delta_t-\omega_t),\quad w_{t+1}=w_t+\alpha_t\rho_t(\delta_t-\omega_t)x_t
$$

离策略 CTD 的中心化更新，0≤γ<1，ρ 为动作重要性比。ω 跟踪的是重加权 TD 误差的平均，不先验等于目标策略的平均奖励率。论文也明确指出，仅此更新不能保证一般离策略稳定。

$$
A_c=\operatorname{Cov}\bigl(x_t,x_t-\gamma x_{t+1}\bigr),\qquad b_c=\operatorname{Cov}(R_{t+1},x_t),\qquad A_cw=b_c
$$

这里展示 on-policy 的协方差形式。离策略时还必须明确重加权后的联合采样分布，不能直接代入行为数据的原始协方差。常数特征可能消失，因此可辨识性需单独检查。该方程不是平均奖励 Poisson 方程的同义写法。

CTDC 还引入辅助向量，形成梯度 TD 的校正。只实现上面三个式子却称作 CTDC，会漏掉关键反馈通道。原文收敛条件包含固定特征、矩阵条件与步长时间尺度；不能由两状态线性实验外推到共同训练表示的深网。

作者 GameAI-NJUPT/BEC 的反例目录包含 CTD、CTDC 相关实现，适合逐项比较主参数、辅助向量和中心化标量。真正的平均奖励离策略预测可对照此前 Differential GQ 的作者实现：ShangtongZhang/DeepRL 中的 LinearOPEAgent.py 与 NeuralOPEAgent.py。线性保证和神经版本实测仍需分开。

**算法：中心化语义与稳定性的对照实验**

1. 固定线性特征与目标策略，先计算理论固定点
1. 用相同经验分别运行 TD、CTD、CTDC 和适用的平均奖励预测方法
1. 同时画 value 误差、共同偏移、ω、辅助向量范数
1. 只在各自目标下比较；不把折扣值误差与 bias 误差混成一条指标

<a id="research-average-retd"></a>

## 相邻稳定性研究 · RETD 说明两种稳定化机制不能直接相加

深入阅读：进一步研究中心化与资格迹的联合反馈。

RETD（2026 预印本）研究一个具体失败：把 Bellman 误差中心化加入 emphatic TD，可能破坏原有联合更新的稳定性。问题仍是折扣、固定线性特征下的离策略预测。它提示平均奖励深网研究者：增加一个参照量，也增加一条反馈通道；不能只观察单个 TD 误差是否变小。

$$
F_t=1+\gamma\rho_{t-1}F_{t-1},\quad w_{t+1}=w_t+\alpha_t(F_t\rho_t\delta_t-\omega_t)x_t,\quad\omega_{t+1}=\omega_t+\alpha_t\bigl(F_t\rho_t\delta_t-(1+c)\omega_t\bigr)
$$

论文的递推骨架，δ 为普通折扣 TD 误差，$F_0=1$，c≥0。注意中心化发生在 emphatic 重加权之后，不是把 ω 也乘以 Fρ。c 只增加辅助标量的阻尼。

$$
G_c=\begin{bmatrix}0.3812&0.62\\0.62&1+c\end{bmatrix},\qquad\det G_c=0.3812(1+c)-0.3844
$$

原文两状态反例的期望更新矩阵。c=0 时行列式为负，正的两个对角项仍不足以稳定联合系统。在此例中 c>0.0084 才恢复正定；这是具体反例的阈值，不是通用调参值。

取很大的 c 会压低辅助标量，使方法逐渐接近未中心化的 ETD，而不是无限增强“中心化”。原文的一般充分条件更保守，并需要有限状态、行为链正则性、固定特征和随机逼近步长条件。不能令 γ=1 后直接宣布获得平均奖励收敛；非线性特征与不断变化的目标策略也不在该结论内。

- 先重算两状态矩阵的特征值，再核对期望更新与随机样本更新。
- 同时报告发散比例、误差、emphatic 迹分布与辅助量，不只挑成功曲线。
- 固定采样序列比较 c=0、小阻尼与大阻尼，分辨反馈稳定和方差变化。

原文提供线性诊断实验，但尚未确认对应的作者代码仓库。BEC 的作者仓库不能自动充当 RETD 的官方实现。上述矩阵检查是可独立完成的教学验证；完整复现仍需核对论文附录的全部采样、步长与统计协议。

<a id="research-openmind-episodic-centering"></a>

## 回合会终止，中心化怎样保持原来的目标

深入阅读：回合终止后重新检查中心化的边界项与目标保持条件。

在无限折扣流中，每步减去常数只平移价值；在长短不一的回合中，这会变成按持续时间收费。De Asis、Elsayed 与 He（RLC 2026）从终止边界出发推广 Differential TD，让中心化也能服务原有的 episodic 目标。此处的“differential”描述参数化与更新，不能据名字把评价目标改为平均奖励。

反例：$\gamma=0.9$。动作 A 立即给 1 并终止；B 先给 0，再给 1.2 并终止。原回报分别为 1 和 1.08，B 更好。若仅在真实回合内每步减去 $c=0.2$，回报变成 0.8 和 0.7，排序翻转。

$$
\Phi(s)=\frac{c}{1-\gamma}\ (s\ne S_T),\quad \Phi(S_T)=0,\qquad F(s,s')=\gamma\Phi(s')-\Phi(s)=\begin{cases}-c/(1-\gamma),&s'=S_T,\\-c,&s'\ne S_T,\end{cases}\quad(s\ne S_T).
$$

固定 $c$、$\gamma<1$ 时，这是满足终止势函数为零的 potential shaping。分段式针对从非终止状态出发的转移；吸收终点自身的转移有 $F(S_T,S_T)=0$。普通转移仍减 $c$，进入终点则减 $c/(1-\gamma)$。这样整段回报统一减去起点势函数，而非按回合长度扣费。

在上例中，修正后的 A 为 −1，B 为 −0.92，二者都恰好减去 2，原排序恢复。终止边界补偿并不是让机器人真的在终点继续运行；它只是将假想吸收状态上的折扣尾部写入价值定义。

$$
V(s;w,b)=V^\Delta(s;w)+b,\qquad V^\Delta(S_T;w)=-b,\qquad \delta_t=\begin{cases}R_{t+1}-b_t-V^\Delta(S_t;w_t),&S_{t+1}=S_T,\\R_{t+1}-(1-\gamma)b_t+\gamma V^\Delta(S_{t+1};w_t)-V^\Delta(S_t;w_t),&\text{otherwise}.\end{cases}
$$

改用共享的价值偏置 $b$ 可避免除以 $1-\gamma$。在折扣情形 $c=(1-\gamma)b$；这两个符号属于不同单位，更新比例也需对应重参数化。该价值偏置形式还能用于最终终止、总回报有限的 $\gamma=1$ 回合。

$$
w_{t+1}=w_t+\alpha_t\delta_t\nabla_wV^\Delta(S_t;w_t),\qquad b_{t+1}=b_t+\eta\alpha_t\delta_t
$$

两条更新使用同一个旧误差。在线变化的 $b$ 不能只靠“固定势函数保持策略排序”的结论论证；论文进一步通过共享偏置的线性 TD 表示讨论其保证。

共享偏置是所有非终止状态共用一个标量，多动作网络中也须跨输出共享；它不等同于为每个动作随意增加一个独立 bias。论文的线性预测分析依赖固定表示、访问与终止等条件；流式深度实验是另一层证据，不能推广为任意深网控制收敛定理。

这与本章的奖励率 g、折扣 TD 参照 c 各有含义。把终点视为无限零奖励吸收状态时，任何最终终止策略的长期平均奖励均为零；因此仍需评价原来的回合总回报或折扣回报。

- 思考：如果只是采样窗口用完，能否采用终止分支？不能；真实后继价值仍属于目标。
- 比较固定中心化、遗漏终止补偿、正确补偿与学习共享偏置四种情况；先核对策略排序，再研究学习速度。
- 将常数奖励偏移、回合时长和任务 reset 权限分别控制，避免把目标变化误认为样本效率提升。

<a id="lesson-nonlinear"></a>

## 12 · 换成神经网络：哪些结构不再成立

表格中，更新一个状态不会改变另一个状态的预测。共享网络中，一次更新会同时移动许多状态的价值。表示本身变化时，旧参数中的“相同特征”也不再代表相同对象。连续采样、bootstrap 和目标策略变化会把这些移动相互反馈。

先只改变一项条件：把表格 h 换成固定线性特征 $h=Xw$，其他条件保持不变。令 $\nu$ 为行为过程的稳态状态分布，$D_\nu=\operatorname{diag}(\nu)$。冻结当前参数，在该分布下平均一步动作重要性加权的更新，得到：

$$
\begin{aligned}F(\bar g,Xw)&=r_\pi-\bar g\mathbf1+(P_\pi-I)Xw,\\\mathbb E[\Delta w]/\alpha&=X^\top D_\nu F(\bar g,Xw),\\\mathbb E[\Delta\bar g]/(\eta\alpha)&=\nu^\top F(\bar g,Xw).\end{aligned}
$$

这是固定参数与指定采样分布下的平均更新，不把实际相关的在线参数与转移当作独立。$X=I$ 且每个状态有覆盖时，价值更新为零迫使各状态残差都为零。特征共享后，只要求残差与特征加权正交；它们仍可能在不同状态相互抵消。

若 $\nu=d_\pi$，最后一行化为 $g_\pi-\bar g$，因为稳态下价值净变化为零。若 $\nu=d_b\ne d_\pi$，还留下 $d_b^\top(P_\pi-I)Xw$，率估计与近似价值误差便相互耦合。这说明离策略动作纠偏、状态加权和表示误差为何是三个不同问题，也解释了为什么换成共享表示后需要重新研究联合更新。

$$
\delta_t=R_{t+1}-\bar g_t+h_{w_t}(S_{t+1})-h_{w_t}(S_t),\qquad w_{t+1}=w_t+\alpha_t\delta_t\nabla_w h_{w_t}(S_t).
$$

这是一条半梯度候选更新。目标端被当成本步常数，不代表该算法在整个训练过程中优化一个固定监督目标。把 ρ 乘上去可以修正动作条件分布，但不能自动恢复线性投影或表格收敛证明。

$$
\Delta\sum_s h_w(s)\;\approx\;\alpha_t\delta_t\sum_s\nabla h_w(s)^\top\nabla h_w(S_t).
$$

右边一般不等于 αδ。因此表格算法的“奖励率增量等于 η 倍全部价值增量”不变量通常消失。即使沿用同名 Differential TD 更新，表格证明中的关键结构也不能直接搬来。

| 常见工程措施 | 直接改变什么 | 不能由此得到什么 |
| --- | --- | --- |
| 目标网络 | 暂缓目标端参数变化；引入滞后 | 不是恢复严格收缩的证明；环境变动时旧目标会过时 |
| 经验重放 | 改变采样相关性与状态权重 | 不是获得新信息；旧数据可能对应旧动力学或旧表示 |
| 价值/奖励中心化 | 移除公共偏移或控制量级 | 不恢复丢失的状态信息；不保证策略覆盖 |
| 更慢奖励率或 actor 步长 | 降低某些耦合瞬态 | 固定小比例不等于双时间尺度定理；非平稳时也可能跟不上 |
| 梯度裁剪或归一化 | 限制一次数值更新幅度 | 不纠正目标错误，也不确保最终策略最优 |

有些 actor–critic 分析要求 actor 步长 $\beta_t$ 相对 critic 步长 $\alpha_t$ 满足 $\beta_t/\alpha_t\to0$，并配合其他假设。不能把“代码里 actor 学习率较小”当成全部条件。表格 Differential TD/Q 的固定 η 联合更新则属于另一条证明路线，两者不能混称。

流式实验应先固定表示与策略，检查预测；再允许策略改善；最后才允许表示学习与模型规划同时发生。固定数据的预测比较可以使用同一条转移流。控制比较则匹配初始条件、外生随机性和资源预算，让各算法的动作产生各自的后续经验；强行给它们相同状态轨迹会改变闭环控制问题。每增加一个可学习对象，都保留关闭该机制的对照。至少记录奖励率误差、参考漂移、梯度范数、不同状态的预测变化与真实策略收益，并计入学习和规划的成本。

<a id="research-average-aro-ddpg"></a>

## 深度连续控制 · ARO-DDPG 如何连接奖励率、critic 与确定性策略

深入阅读：把差分动作价值用于确定性策略改善，核对其假设与实际训练。

差分 critic 已经能比较“先做这个动作，再执行当前策略”的长期后果。连续动作控制还需要知道，稍微改变这个动作会怎样改变奖励率。确定性策略 μθ 直接输出动作；下面从它的差分方程出发，寻找可以交给 actor 的方向。先考虑有限状态、连续动作，策略附近的诱导链不可约，所需导数存在且可交换求和与微分。

$$
q^\mu(s,a)=r(s,a)-g_\mu+\sum_{s'}P(s'\mid s,a)h_\mu(s'),\qquad h_\mu(s)=q^\mu(s,\mu_\theta(s)).
$$

g 是每步奖励率，h 是按固定规范确定的差分状态价值。给全部 h 和 q 加同一个常数，不改变动作间的差别。动作导数先固定当前策略及其后续 h，只改变最初的动作。

$$
\nabla_\theta g_\mu+\nabla_\theta h_\mu(s)=u_\theta(s)+\sum_{s'}P_\mu(s,s')\nabla_\theta h_\mu(s'),\qquad u_\theta(s)=\nabla_\theta\mu_\theta(s)\nabla_aq^\mu(s,a)\big|_{a=\mu_\theta(s)}.
$$

对差分方程求导。u 收集当前动作改变奖励与转移的影响；最后一项保留后续策略变化造成的差分价值变化。下一步将它消去，而不是把它假装成常数。

$$
\nabla_\theta g_{\mu_\theta}=\mathbb E_{S\sim d_{\mu_\theta}}\!\left[\nabla_\theta\mu_\theta(S)\,\nabla_a q^{\mu_\theta}(S,a)\big|_{a=\mu_\theta(S)}\right]
$$

用当前策略的平稳分布 d 对前式求平均。由 $d^\top P_\mu=d^\top$，两边的差分价值导数相消，剩下奖励率梯度。连续状态需以积分替代求和，并满足相应平稳、可积与微分条件。

这个消去步骤解释了为什么状态分布不能任换成 replay 频率：其他分布一般不满足同一平稳等式。Saxena 等人的 ARO-DDPG（ICML 2023）研究怎样在离策略数据上近似这项更新，再将其用于深度连续控制。目标网络减慢 critic 变化，但不完成状态分布校正。

原论文定理 3.4 明确把行为稳态权重下的方向称为近似梯度，并在一致遍历与其他正则条件下约束其误差。第 4 节的随机逼近分析使用线性 critic、步长/时间尺度和投影等条件；有限时间结论控制梯度范数，并保留逼近与采样误差项。它不是深网找到全局最优策略的保证。

$$
y=r+(1-d_{\rm end})\min_i Q^-_i(s^+,\mu^-(s^+)),\qquad L_Q=\frac12\sum_i\bigl(Q_i(s,a)+\rho-y\bigr)^2
$$

作者工程 cheetah_run/ddpg_model.py 中的主要 critic 损失。ρ 是可训练的奖励率标量；同一损失同时更新 critic 与 ρ。$d_{\rm end}$ 是代码的边界标记，与上面的稳态分布 d 不同；是否代表真正终止，须由环境协议确定。

作者工程包含双 critic、目标网络、replay 和训练段落。train.py 会重新初始化环境段落。因此该工程可以帮助研究深度平均奖励控制，却不能仅凭名称当作不重置的单次生命算法。论文理论的函数类、采样与步长条件，也不能替代完整深网联合训练的证明。

**算法：教学到研究工程的递进实验；不是作者完整实验的复现声明**

1. 第一步：固定策略，在三状态 continuing 链上核对真实 gain 与 critic 的差分值
1. 第二步：只开放 actor 更新，记录 critic 误差和冻结策略的真实 gain
1. 第三步：改用作者网络、replay 与目标网络；单独记录 ρ、Q 的共同偏移和动作导数
1. 第四步：区分真实终止、时间截断和真实 reset，检查 target 使用实际后继状态

建议先做一个常数偏移测试：所有即时奖励加 3，固定策略的奖励率应加 3，参考对齐后的差分值应不变。若 actor 改变，先检查训练未收敛、共同偏移、优化器尺度与终止协议，而不是直接解释成平均奖励目标本身发生了变化。作者代码入口与固定版本见本章来源。

<a id="research-average-asac"></a>

## 平均奖励最大熵控制 · ASAC 的策略改善比较什么

深入阅读：在长期目标中加入熵正则后，重新推导评价和策略改善。

差分价值允许共同加一个常数，所以不能通过“新 critic 的数值更大”判断策略是否变好。若还希望策略保留一定随机性，应先把这种偏好写进长期目标，再推导改善。固定参考策略 $\pi_0$ 和温度 α，目标取长期外部奖励率减去相对参考策略的信息代价。这是 ASAC（RLC 2025）研究的平均奖励熵正则问题。

$$
g^\alpha_\pi=\lim_{T\to\infty}\frac1T\mathbb E_\pi\!\left[\sum_{t=0}^{T-1}\left(R_{t+1}-\alpha\log\frac{\pi(A_t\mid S_t)}{\pi_0(A_t\mid S_t)}\right)\right],\quad\alpha>0
$$

使用 α 表示温度，相当于论文的逆温度 β 的倒数。要求策略在参考策略支持集内，并满足相应长期平均存在条件。外部奖励率与含正则项的奖励率需分开报告。

先在有限状态动作上推导，假定参考动作概率为正，所比较策略诱导不可约链，奖励有界。固定当前策略 π，令 q 为它的准确 soft 差分动作价值：本步先计外部奖励，动作的信息代价通过状态价值计入。这样可把一次动作改善和长期率的变化接起来。

$$
q(s,a)=r(s,a)-g_\pi^\alpha+\sum_{s'}P(s'\mid s,a)h(s'),\qquad h(s)=\sum_a\pi(a\mid s)\left[q(s,a)-\alpha\log\frac{\pi(a\mid s)}{\pi_0(a\mid s)}\right].
$$

q 和 h 对应同一固定策略与同一熵约定，不是任意两个网络输出。h 是在当前状态按 π 选择动作时的正则价值。

$$
\mathcal I_q(\kappa;s)=\sum_a\kappa(a\mid s)\left[q(s,a)-\alpha\log\frac{\kappa(a\mid s)}{\pi_0(a\mid s)}\right],\qquad\pi^+(a\mid s)=\frac{\pi_0(a\mid s)e^{q(s,a)/\alpha}}{Z_q(s)}.
$$

κ 是候选动作分布。对概率求导并满足总和为一，得到右边的指数倾斜分布，Z 是其归一化常数。连续动作可改为密度与积分，但须存在有限的 Z 及相关期望。

$$
\mathcal I_q(\kappa;s)=\alpha\log Z_q(s)-\alpha D_{\rm KL}(\kappa(\cdot\mid s)\Vert\pi^+(\cdot\mid s)),\qquad\mathcal I_q(\pi^+;s)\ge\mathcal I_q(\pi;s)=h(s).
$$

KL 非负，因此 π⁺ 对每个状态至少与旧策略一样好。这时比较的是固定旧 q 下的局部动作选择，还不是两个新旧 critic 的任意数值。

$$
g_\kappa^\alpha-g_\pi^\alpha=\mathbb E_{S\sim d_\kappa}[\mathcal I_q(\kappa;S)-h(S)].
$$

代入 q 的差分方程，右边成为新策略的正则奖励率减旧率，再加 dκ 加权的 Pκh−h；最后一项由平稳性消去。取 κ=π⁺，上面的逐状态不等式便给出率不下降。

一个单状态例子是 $q=(0,1)$、参考概率各半、$\alpha=1$。指数倾斜给出高值动作概率 $e/(1+e)\approx.7311$。此时 $\mathcal I_q(\pi^+)=\log((1+e)/2)\approx.6201$，高于均匀选择的 .5。给两个 q 同加十，动作概率不变，两个局部值都加十，改善差仍为 .1201。

$$
L_\pi(\theta)=\mathbb E_{S\sim\nu,\ A\sim\pi_\theta(\cdot\mid S)}\left[\alpha\log\frac{\pi_\theta(A\mid S)}{\pi_0(A\mid S)}-Q_\phi(S,A)\right].
$$

工程中以经验状态分布 ν 和近似 critic 替代上述逐状态准确改善。这一步固定 critic 参数 φ 和参考策略；重参数化 actor 时仍需对其采样动作求 Q 的动作导数。共享策略网络只能近似指数倾斜，减小此经验损失不等于已经证明真实率改善。

$$
\operatorname{KL}\!\left(\pi\,\middle\|\,\frac{\pi_0 e^{\beta q}}{Z}\right)=\mathbb E_{\pi}\!\left[\log\frac{\pi}{\pi_0}-\beta q\right]+\log Z,\qquad\alpha=1/\beta
$$

温度换元须从 KL 投影展开核对。将整个目标乘以正数 α，才得到前文 α·log−q 的形式。不能只把 q 前的 β 换成 1/β 而保持其他项不变；那会改变温度与策略。

$$
y=r-\widehat g^\alpha+\mathbb E_{A^+\sim\pi}\!\left[Q^-(s^+,A^+)-\alpha\log\frac{\pi(A^+\mid s^+)}{\pi_0(A^+\mid s^+)}\right]
$$

soft 差分 Bellman 目标的结构。所有项须采用同一种熵约定。若自动调整温度，目标奖励率也在变化；不能把外部 reward 的简单均值直接视为固定 soft gain。

作者 JacobHA 的 BARL 仓库有 ASAC.py，可读到奖励中心化、critic target 与温度更新。它是作者的探索性实现入口，但仓库没有充分确定其与正式论文全部实验配置的对应关系。该代码存在 done 掩码，因此必须核对 reset 后后继状态与 continuing 目标。此入口不应标成已经完成的论文复现。

论文实现还指定参考状态动作的价值锚定、双 critic 和自适应重置成本。这些机制分别控制公共偏移、估计偏差和任务中的重置后果，不能仅以“把 SAC 的折扣改成 1”概括。正式 benchmark 的评估回合、种子与误差带也不能替代严格单次生命期奖励记录。

- 离散小 MDP：精确求 soft gain 与相对价值，再核对指数倾斜改善。
- 比较三组：不含熵的平均奖励、固定温度的平均奖励、自动温度。分别报告外部 gain、soft gain 与熵。
- 让所有 q 同加常数，验证策略不变；让 replay 分布改变，检查奖励率估计是否仍对应目标策略。

<a id="lesson-branches"></a>

## 13 · 从差分预测到深度平均奖励控制

平均奖励 off-policy 预测加函数逼近后，数据分布与目标占用分布不一致会破坏简单 TD 的稳定性。Differential GQ 等方法引入辅助变量与投影目标；它们解决的是策略评估，不等于完成了未知目标策略的深度控制。本页先把表格目标和时序固定，避免将数学对象、求解器和网络结构混为一谈。

$$
\begin{aligned}\widetilde V_Q(s')&=\mathbb E_{a'\sim\pi_\theta}[Q(s',a')-\tau_{\!H}\log\pi_\theta(a'\mid s')],\\y&=r-f(Q)+\widetilde V_Q(s'),\qquad L_Q=\mathbb E[(Q(s,a)-\operatorname{stopgrad}(y))^2].\end{aligned}
$$

这是 soft average-reward critic 的概念骨架，不是 RVI-SAC 全部实现。熵温度 τH 与 option 时长 τ 不同；具体算法还需要目标网络、双 critic、参考项估计及 reset 成本等。

RVI-SAC 将参考项估计与最大熵控制结合。训练收益可包含熵和重置成本，评测的外部奖励率则可能不含这些项。若失败后的重置是持续过程中的真实转移，价值目标需要计入其后续收益；若它是问题定义中的吸收终止，边界条件便不同。重置的动作、代价与时间决定了目标的形式。

$$
\xi_{t+1}=\xi_t+\beta_t\left(f(Q_t;\mathcal B_t)-\xi_t\right),\qquad y=r-\xi_t+\widetilde V_{Q^-}(s').
$$

RVI-SAC 用单独变量 $\xi$ 跟踪批次估计的参照量，减轻参照噪声直接进入所有 critic target 的反馈。原论文的定理 3.1 分析的是带裁剪参照、覆盖及随机逼近条件的表格版本；其时间尺度要求 $\alpha_t/\beta_t\to0$，参照相对 Q 更快，而不是泛指“所有奖励率都应更慢”。深网中的固定步长、目标网络与 replay 属于另一个实测层次。

因此三个近期深度分支应分别理解：ARO-DDPG 用确定性策略的差分动作导数；RVI-SAC 用相对价值参照与 soft 改善；ASAC 用平均奖励的熵正则评价与策略投影。它们都需要处理率、价值偏移和持续转移，但并没有共享一条可直接覆盖所有神经网络、非平稳世界和单次生命的收敛定理。

- 研究目标改变：比较平均奖励、折扣目标和 reward centering。保留 γ<1 的 centering 主要改变数值与 baseline，不自动变成平均奖励。
- 研究可追踪性：在不向 agent 提供切换时间的情况下改变奖励或转移，报告变化后累计损失与估计延迟。
- 研究函数逼近：固定特征、线性辅助变量、深网三层递进；先确定失效来自覆盖、投影还是表示漂移。
- 研究 SMDP：控制时长分布、奖励与时长相关性，并统一按原始环境步计算分母。

<a id="research-average-nsnac"></a>

## 非平稳控制 · NS-NAC 与 BORL 的遗憾界依赖哪些数据权限

深入阅读：比较非平稳研究中的数据权限、变化约束与遗憾对象。

NS-NAC 研究奖励与转移随时间变化的平均奖励控制。BORL 用上层选择机制处理未知变化预算。这里要区分“环境在变化”和“智能体只有一条不能重来的生命”：原文算法会分块重新初始化，并具有比沿单条自然轨迹更强的状态采样权限。

$$
\operatorname{DRegret}(T)=\mathbb E\!\left[\sum_{t=1}^{T}(g_t^*-R_t)\right],\qquad\pi^+(a\mid s)\propto\pi(a\mid s)\exp\!\bigl(\beta Q(s,a)\bigr)
$$

$g_t^*$ 是冻结在时刻 t 的 MDP 的最优平均奖励率，不是一个真实非平稳 oracle 沿自己的轨迹获得的即时收益。右式是自然策略改善的指数更新形式；它仍依赖 critic 的估计质量。

为什么要分块？旧数据能降低方差，却可能来自已经失效的动力学。短块更快遗忘，但重新学习的代价更大。变化预算控制这种取舍。比较器、变化度量和状态采样方式都属于问题定义，不是实现时可以随意删去的细节。

arXiv 版本 Algorithm 1 在块起点重设策略与估计，并在内层均匀采样状态。critic 采用投影，奖励率另有跟踪。理论使用有限问题和相应混合条件。因此不能把该版本直接标注为严格单次生命 actor–critic，也不能省略重新采样的成本后与无重置算法比较。TMLR 版本应按其最终算法逐项复核，不能只沿用预印本标签。

**算法：研究权限差异的实验设计；当前教学代码不构成 NS-NAC 完整实现**

1. 构造两个三状态 MDP，事先固定改变时刻；总预算变动不改变任务
1. A：按论文所需的状态采样与分块权限运行
1. B：沿真实状态连续运行；不重置环境与学习状态
1. 报告全程奖励、分段冻结策略 gain、重新采样次数和恢复代价
1. 若 B 改动了论文算法，明确命名为消融，不沿用 A 的理论保证

尚未确认与该论文对应的作者开源工程。可先用本章多状态环境说明变化后的率估计滞后，再研究分块与状态采样的影响。教学环境的数值稳定，不能替代动态遗憾定理的全部假设。

<a id="research-centering-fixed-point"></a>

## TD 中心化标量为何不总是奖励率？

深入阅读：从联合更新的不变量求出 TD 中心化参照，不预先把它当作真实奖励率。

固定 c 时，中心化折扣价值只发生共同平移。TD 中心化却同时更新 q 与 c，二者相互影响。下面用常奖励的单状态系统求解其固定点，检查 c 是否等于真实平均奖励率。

$$
\delta_t=r-c_t-(1-\gamma)q_t,\quad q_{t+1}=q_t+\alpha\delta_t,\quad c_{t+1}=c_t+\eta\alpha\delta_t
$$

单状态、单动作、常奖励 r，γ<1。所有右侧使用旧参数；c 是 TD 参照量，不预先称作真实 g。

$$
c_t-\eta q_t=c_0-\eta q_0=:k,\quad q_*={r-k\over\eta+1-\gamma},\quad c_*={\eta r+(1-\gamma)k\over\eta+1-\gamma}
$$

两种增量成比例，所以 c−ηq 不变；联立 δ=0 得固定点。确定性误差倍率为 1−α(η+1−γ)，还需满足其绝对值小于 1 的稳定条件。

取 $r=1$、$\gamma=0.9$、$\eta=0.1$、初值全零，得 $q_*=5,c_*=0.5$，实际奖励率却是 1。q 正是中心化奖励 0.5 的折扣价值 $0.5/(1-0.9)=5$。若令 $\gamma=1$，这个单状态方程才要求 $c_*=r$。

Reward Centering 的 TD 驱动参照、on-policy 行为奖励均值与 Differential TD 是不同对象。保留 γ<1 可改善共同数值尺度，却不能证明有限折扣与平均奖励在任意策略上排序相同。off-policy 场景也不能把真实行为均值直接当作目标策略的奖励率。

| 量 | 含义 | 检查 |
| --- | --- | --- |
| c | TD 学习参照 | γ、初始化约束及联合固定点。 |
| reward/time | 实际行为外部奖励率 | 完整奖励和真实时间。 |
| g | 平均奖励方法的目标奖励率 | 策略对象、覆盖与收敛条件。 |

**算法：参照语义的验证方案**

1. 常奖励单状态：逐步更新 q,c，核对 c−ηq 不变量
1. 改变 γ、奖励常数偏移和初值，核对解析固定点
1. 多动作控制：保持同一 reset 协议，比较排序和外部率
1. 深度比较：同预算分别加入行为均值与 TD 中心化

可运行的单状态更新与独立固定点参考；本章 average 命令打印反例，test 命令核对不变量、固定点与差分极限。

```python
def centered_single_state_step(q, reference, reward, gamma=0.9,
                               eta=0.1, alpha=0.1):
    """One-state diagnostic, not a complete reward-centering implementation.

    Discounted TD reference c need not equal the actual reward rate.
    Both writes use the SAME old-parameter error. gamma=1 is the
    differential limiting comparison, not discounted policy equivalence.
    """
    if not 0 <= gamma <= 1 or eta < 0 or alpha < 0:
        raise ValueError("invalid discount or nonnegative update scale")
    delta = reward - reference - (1 - gamma)*q
    return q + alpha*delta, reference + eta*alpha*delta, delta


def centered_single_state_fixed_point(reward, gamma, eta, q0=0.0, c0=0.0):
    """Solve c-eta*q invariant and zero TD error; not a stability claim."""
    if not 0 <= gamma <= 1 or eta < 0 or eta + 1 - gamma <= 0:
        raise ValueError("a positive fixed-point denominator is required")
    invariant = c0 - eta*q0
    q = (reward - invariant)/(eta + 1 - gamma)
    return q, invariant + eta*q
```

原始 Reward Centering 论文与 DeepRL-continuing-tasks 的 rc 配置可追踪具体变体。函数逼近、随机采样及不同更新时间尺度带来额外误差；单状态不变量用于发现语义混淆，不能直接推广成深网守恒律。

<a id="research-rvi-sac-reset-cost"></a>

## RVI-SAC：平均奖励、soft 参照与重置成本

深入阅读：把相对价值参照、soft 控制和重置成本放到同一算法中检查。

RVI-SAC（ICML 2024）直接面向平均奖励最大熵控制。在 soft 后继价值中减去参考项，不使用小于一的环境折扣；另用 reset critic 和成本控制重置频率。完整系统不是仅把 SAC 的 γ 改成 1。

$$
\bar v(s')=\mathbb E_{a'\sim\pi_\theta}[\min_k\bar Q_k(s',a')-\tau_H\log\pi_\theta(a'\mid s')],\quad y=r-cd-f+\bar v(s')
$$

target critics、策略与温度构造 y 后停止梯度；d 是本次 reset 指示，s′ 是真实后继。τH 是熵温度，不是任务耗时。

$$
L_Q=\sum_k\mathbb E[(Q_k(s,a)-\operatorname{sg}(y))^2],\quad f^+=(1-\kappa)f+\kappa\zeta\mathbb E_{\rm batch}[\bar v(s')],\quad L_\pi=\mathbb E[\tau_H\log\pi_\theta(a\mid s)-\min_k Q_k(s,a)]
$$

f 式对应作者当前 rvi_sac.py 的移动参考，ζ 对应 fq_gain；其他 reference 变体需逐文件区分。actor 使用可重参数化动作，温度另行训练。

$$
y_d=d-f_d+\bar Q_d(s',a'),\quad L_{Q_d}=\mathbb E[(Q_d(s,a)-\operatorname{sg}(y_d))^2],\quad c^+=\max\{0,c+\alpha_c(f_d-p_0)\}
$$

reset critic 使用 reset 指示作为信号，fd 按对应移动参考更新。c 式是普通 dual 梯度步解释；作者实际用 Adam 再投影非负，不能把此式称作完整 Adam。p0 为目标 reset 频率。

若 fd=0.03、p0=0.01，梯度增大重置成本；若 fd=0.005，则减小至非负边界。成本改变 critic 与 actor，actor 又改变真实重置频率。这是依赖准确估计的反馈，并非一次失败便固定加罚。

**算法：对应 average_reward_drl/algorithms/rvi_sac.py 与 train.py；固定版本核对**

1. 从 replay 采样，以旧 f/fd/c 和 target networks 构造两个 target
1. 更新双任务 critic 与 reset critic，再更新 reference
1. 更新 actor 和温度；更新非负 reset cost
1. 最后更新 target networks
1. 采用 reset scheme 时：训练循环先实际 reset，再保存新状态为后继

| 分别报告 | 理由 |
| --- | --- |
| 外部奖励率 | 应用收益不混入 entropy。 |
| 熵及 reset 成本修正目标 | 与原始外部奖励数值不同。 |
| 重置率、恢复时间与真实耗时 | 仿真一步 reset 不等于即时免费复位。 |

作者 yhisaki/average-reward-drl 的固定成本与 reference 变体适合机制对照。精确平均奖励 soft improvement 的条件不自动覆盖重放、目标网络、非凸逼近及成本反馈联合学习。CRL 还需检查变化后的参照滞后和过时 replay。

<a id="lesson-check"></a>

## 14 · 习题与诊断

| 问题 | 应当能给出的解释 |
| --- | --- |
| h 全体加 100，策略会变吗？ | max 的相对次序和 TD 差分不变；参考规范化会改变存储值。 |
| 为何控制实验中 ḡ≈1，但真实流只有 0.5？ | ḡ 对应最优 Bellman 目标；实际探索仍选择低奖励动作。二者必须分别报告。 |
| 只有一次不可重置生命，能否用 g 的渐近目标解释全部价值？ | 目标仍可用，但有限生命损失、不可逆风险与恢复成本不能由极限奖励率单独概括。 |
| 算法收敛证明能否覆盖突然改变的深网实验？ | 不直接覆盖；需要重新陈述平稳性、函数逼近、覆盖和步长假设。 |

动手题：把两状态环的一条奖励改为 4，保留所有学习器状态。分别比较递减步长与常数步长的追踪速度；预测新 g=2、h(1)−h(0)=2。再把日志分段，但绝不在分段边界清零参数。将“记录一个新窗口”和“重新开始学习”严格分开。

## 本章的实验设计

记录每单位原始时间的收益。技能调用次数不能代替经过的时间。分别报告暂态和长期表现。

设定：先比较两个持续时间不同的固定技能：累计奖励分别为 2、9，时长分别为 1、9；再接小型非回合式 MDP 的 Differential TD/Q。用常奖励单状态反例检查有限折扣下的 TD 中心化。

- 总奖励率为 11/10，而不是两个技能奖励率的平均数。
- 每单位原始时间的奖励加同一常数后，奖励率按相同常数平移。
- 差分价值的常数偏移不改变所比较的动作优势。
- 中心化代码保持 c−ηq 不变量；r=1、γ=0.9、η=0.1、零初值时收敛到 q=5、c=0.5，而真实奖励率为 1。

对照：固定策略的解析奖励率；匹配动作持续时间的 primitive 对照；单独列出的折扣目标基线；行为奖励均值、TD 中心化和 γ=1 差分极限分别测试

记录：总原始奖励除以总原始时长；奖励率估计误差与暂态表现；差分 Bellman 残差及各状态访问量；中心化参照、实际奖励率与联合固定点误差分别报告

[具体测试规程](../docs/experiment-handbook.md#handbook-modules)

## 学习与研究衔接

调用一次技能可能只走一步，也可能走许多步。评价奖励率要按真实持续时间累计奖励；一般不能把各次调用的奖励除以时长后再取平均。长期极限适合描述持续行为，却会忽略有限的启动损失。研究一个仍在学习的系统时，应同时规定有限评价时域，显式保留练习、模型修正和恢复所花的时间。

[分册导读](../docs/learning-route-continual-rl.md) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-average) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=average) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=average)

## 从本章进入实践

[策略梯度与控制](https://yingwen.io/zh/continual-rl/code/#practice-policy-control)：优化器确实降低了损失，为什么行动仍可能变差？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

教材可以先给定目标和技能集合；持续构造还要决定哪些行为值得练习、维护或放弃。谱结构、路径奖励、时间距离和语言先验提供不同候选偏置。先固定候选比较选择与组合，再改变生成器，才能辨认下游收益究竟来自哪一步。

- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [An Idiosyncrasy of Time-discretization in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-openmind-time-discretization)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rvi-sac-average-control)
- [Reward Centering](https://yingwen.io/zh/continual-rl/research/#recent-reward-centering-discounted)
- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)
- [Posterior Sampling for Continuing Environments](https://yingwen.io/zh/continual-rl/research/#recent-cpsrl-continuing-exploration)
- [Extending Differential Temporal Difference Methods for Episodic Problems](https://yingwen.io/zh/continual-rl/research/#recent-openmind-episodic-differential)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks](https://yingwen.io/zh/continual-rl/research/#recent-continuing-task-deep-study)

### RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning

Yukinari Hisaki, Isao Ono

ICML 2024 · 2024 · 支持方法与理论

#### 研究问题

深度连续控制若最终按单位时间收益评测，训练能否直接采用平均奖励而非有限折扣？

#### 关键机制

RVI-SAC 将相对价值参照项加入 soft critic，以平均奖励的 soft policy improvement 构造 actor，并用额外 reset critic 与可学习成本控制重置频率。完整实现包含双 critic、经验重放、目标网络和温度更新。

#### 证据

论文给出平均奖励最大熵控制推导，并在 MuJoCo 运动任务中比较；公开实现可核对重置转移是否继续 bootstrap。

#### 条件与限制

理论的表格或精确评价条件不自动覆盖所有神经网络训练。最大熵奖励率、外部奖励率与带 reset 成本的奖励率是三个量；不可将有限折扣 reward centering 当作同一算法。

#### 阅读与实验

逐项对应 critic 参照、actor 分布、reset 指示与 reset 后状态；评价保留外部原始奖励、实际时长、重置次数和训练修正目标。

#### 原文与相关入口

- [ICML 2024 正式论文](https://proceedings.mlr.press/v235/hisaki24a.html)：平均奖励 soft improvement、RVI 与自动 reset cost。

#### 作者代码

[作者仓库 README 标明 reference code 与同名原论文。](https://github.com/yhisaki/average-reward-drl)

average_reward_drl/algorithms/rvi_sac.py 及其参照项、固定 reset cost 变体。

### Reward Centering

Abhishek Naik, Yi Wan, Manan Tomar, Richard S. Sutton

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

接近一的折扣为何使共同价值偏移很大，中心化能改善什么、又不能改变什么？

#### 关键机制

从折扣价值的共同偏移与相对价值分解出发，移除奖励参照量；on-policy 可估计行为奖励均值，off-policy 提出 TD 驱动的参照更新。保留小于一的折扣时，中心化没有消除折扣对策略排序的影响。

#### 证据

原文给出理论动机与表格、线性、非线性控制实验，检验折扣及奖励常数平移。深度 continuing-task 后续研究扩大了算法与环境范围。

#### 条件与限制

TD 中心化中的标量在有限折扣下不必精确等于真实奖励率。训练期的联合参照/价值更新与固定常数下的平移恒等式需分别分析；真实终止改变平移条件。

#### 阅读与实验

用单状态常奖励问题解出联合更新固定点，再用多动作问题检查策略排序；同时记录参照量与直接观测的外部奖励率。

#### 原文与相关入口

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_261.pdf)：中心化分解、on/off-policy 区别及收敛讨论。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper261.html)：正式题名、作者与会议年份。

### An Empirical Study of Deep Reinforcement Learning in Continuing Tasks

Yi Wan, Dmytro Korenkevych, Zheqing Zhu

arXiv 预印本 · 2025 · 评价与实验协议

#### 研究问题

把环境作为持续的转移过程后，无重置、预设重置和智能体控制重置怎样改变学习难点？

#### 关键机制

构造三类 continuing 协议，将重置后的收益纳入同一条持续过程；对深度控制算法及不同 reward centering 方法进行比较。重置权限属于环境/接口设计，而不是一个可以隐藏的评测便利。

#### 证据

作者公开 MuJoCo 与 Atari testbeds、训练和评价配置。论文报告中心化在多种方法中的收益，同时保留大折扣及无重置恢复困难等限制。

#### 条件与限制

continuing 指非回合式持续交互，不自动意味着环境任意非平稳或无限容量学习。仓库 citation 中的 2024 草稿年与 arXiv 2025 发布年不同；此处按可核验预印本记录，不指定未确认的会议。

#### 阅读与实验

先固定重置转移、成本和时间，再比较目标与算法；分别评价全程学习收益、冻结策略奖励率及失败恢复。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2501.06937)：三类持续协议与深度中心化实验；2025 年 arXiv 首稿。

#### 作者代码

[论文对应 Meta 作者团队的研究仓库，README 明确区分三个 reset 协议。](https://github.com/facebookresearch/DeepRL-continuing-tasks)

testbeds、Pearl 算法、experiments 配置与评测/作图。

### Posterior Sampling for Continuing Environments

Wanqiao Xu, Shi Dong, Benjamin Van Roy

RLC 2024 / RLJ · 2024 · 支持方法与理论

#### 研究问题

没有自然回合边界，后验采样探索应在什么时候更换整条行动假设？

#### 关键机制

CPSRL 以独立随机时钟重采样模型并规划，而不等待真实 reset 或逐状态计数翻倍。几何持续时间把策略试验的未折扣收益与相应折扣规划目标联系起来；改变的是探索承诺的时间尺度。

#### 证据

论文在有限平稳 MDP 条件下分析 Bayesian regret，得到含奖励平均时间 $\tau$ 的 $\widetilde O(\tau S\sqrt{AT})$ 量级，并给出模拟。

#### 条件与限制

定理依赖正确后验、规划与平均时间条件；深网 ensemble 只是一种近似，不直接继承表格界。重采样不重置世界；平稳后验也不会自动遗忘已过时的动力学。

#### 阅读与实验

比较每步换假设、几何时钟与固定时钟，控制同一模型学习预算；在漂移实验中另外定义后验遗忘，避免误用平稳遗憾保证。

#### 原文与相关入口

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_277.pdf)：随机重采样、折扣联系与 Bayesian regret 假设。
- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper277.html)：作者、会议与理论结果。

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


<a id="chapter-code"></a>

## 下载与运行

标准库表格机制实验与单步公式测试；不是深度算法或论文 benchmark 复现。

[下载 lifelong_algorithms_lab.py](../examples/lifelong_algorithms_lab.py)

```sh
python examples/lifelong_algorithms_lab.py average
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto — Reinforcement Learning: An Introduction，第二版](http://incompleteideas.net/book/the-book-2nd.html)：§2.5 非平稳追踪、§8.4–8.5 模型与规划分布、§10.3–10.4 平均奖励、§17.3 状态与未来方向。

- [Sutton, McAllester, Singh & Mansour · Policy Gradient Methods with Function Approximation · 1999](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)：平均奖励策略梯度及相容函数逼近的经典原文；本章从稳态与 Poisson 方程逐步重推，不将相容条件省略为任意神经 critic。

- [Grand-Clément & Petrik · Blackwell Discount Factor · NeurIPS 2023](https://papers.nips.cc/paper_files/paper/2023/hash/a4e720ce31ccd8ba747d8863e1580fa8-Abstract-Conference.html)：定理 4.2、4.4 分别讨论有限 MDP 的阈值存在性及问题依赖的上界；不是任意固定大折扣或未知深网模型的保证。

- [Wan, Naik & Sutton · Learning and Planning in Average-Reward MDPs](https://proceedings.mlr.press/v139/wan21a.html)：表格 Differential TD/Q 与规划的原论文；对应本文奖励率由 TD error 驱动的核心。

- [Tsitsiklis & Van Roy · Average Cost Temporal-Difference Learning](https://web.stanford.edu/~bvr/pubs/average-td.pdf)：平均代价同策略线性 TD 的经典分析；固定特征与链结构是条件，不是任意神经网络的结论。

- [Zhang, Wan, Sutton & Whiteson · Average-Reward Off-Policy Policy Evaluation with Function Approximation](https://proceedings.mlr.press/v139/zhang21u.html)：奖励率与差分价值的异策略逼近；线性收敛分析与非线性实验必须分别理解。

- [Zhang, Zhang & Maguluri · Finite Sample Analysis of Average-Reward TD Learning and Q-Learning](https://proceedings.neurips.cc/paper/2021/hash/096ffc299200f51751b08da6d865ae95-Abstract.html)：有限样本、线性 TD(λ) 与表格 Q 学习的分析；不同于只判断渐近存在性的结论。

- [作者实验仓库 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：从 agents/prediction_agents.py 和 control_agents.py 对照更新，再读各实验配置；非本页教学代码的性能保证。

- [Wan, Naik & Sutton · Average-Reward Learning and Planning with Options](https://arxiv.org/abs/2110.13855)：不同 option 学习/规划更新与时长归一化；本页展示的是明确标记的 expected-duration 变体。

- [Wan & Sutton · Weakly Communicating MDPs](https://arxiv.org/abs/2209.15141)：扩展理解平均奖励控制的链结构和收敛条件；不可外推为任意非平稳深网定理。

- [ICML 2024 正式论文](https://proceedings.mlr.press/v235/hisaki24a.html)：平均奖励 soft improvement、RVI 与自动 reset cost。

- [RVI-SAC: Average Reward Off-Policy Deep Reinforcement Learning · 作者实现](https://github.com/yhisaki/average-reward-drl)：average_reward_drl/algorithms/rvi_sac.py 及其参照项、固定 reset cost 变体。 作者仓库 README 标明 reference code 与同名原论文。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper261.html)：正式题名、作者与会议年份。

- [作者论文](https://arxiv.org/abs/2501.06937)：三类持续协议与深度中心化实验；2025 年 arXiv 首稿。

- [An Empirical Study of Deep Reinforcement Learning in Continuing Tasks · 作者实现](https://github.com/facebookresearch/DeepRL-continuing-tasks)：testbeds、Pearl 算法、experiments 配置与评测/作图。 论文对应 Meta 作者团队的研究仓库，README 明确区分三个 reset 协议。

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_261.pdf)：中心化分解、on/off-policy 区别及收敛讨论。

- [RLC 2024 原文](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_277.pdf)：随机重采样、折扣联系与 Bayesian regret 假设。

- [RLJ 论文记录](https://rlj.cs.umass.edu/2024/papers/Paper277.html)：作者、会议与理论结果。

- [ARO-DDPG · ICML 2023 原文](https://proceedings.mlr.press/v202/saxena23a.html)：平均奖励确定性 actor–critic；需分别阅读策略梯度、理论条件和深网实验。

- [ARO-DDPG · 作者工程固定版本](https://github.com/namansaxena9/ARO-DDPG/tree/5803e81b3bc0d1ee0e45236bae688002d6f1685a)：重点 cheetah_run/ddpg_model.py、train.py 与 config.py；记录 replay、done 掩码与训练段落。

- [Average-Reward Soft Actor-Critic · RLC 2025](https://rlj.cs.umass.edu/2025/papers/Paper34.html)：平均奖励、KL 正则与 soft 策略改善；不要比较任意常数规范下的原始 Q 大小。

- [ASAC · 原论文 v2 的推导与实现细节](https://arxiv.org/html/2501.09080v2)：§4、附录 §8–9；按 KL 投影定义核对逆温度、差分 critic、参考价值、重置与完整实验协议。

- [ASAC.py · 作者探索性代码入口](https://github.com/JacobHA/BARL/blob/200ccd2c48702bc683861f49843f86446075ab16/ASAC.py)：可检查 target、reward-rate 与温度更新；尚不能确认与正式论文全部实验的逐项对应。

- [Bellman Error Centering · 原始预印本](https://arxiv.org/abs/2502.03104)：CTD 与 CTDC 的折扣、线性预测机制；不是平均奖励深网收敛证明。

- [Bellman Error Centering · Neural Networks 2026](https://doi.org/10.1016/j.neunet.2026.108896)：期刊版本入口；复现时需固定论文版本及代码配置。

- [BEC · 作者诊断实验代码](https://github.com/GameAI-NJUPT/BEC/tree/4d55a79afa733095e527924c21edf2689ea5c45d)：VM_evaluation 下的二状态、Baird 反例；核对主参数、辅助向量和中心化标量。

- [Differential GQ · 作者离策略评价工程](https://github.com/ShangtongZhang/DeepRL/tree/05e38af518972fa80f36cef69ac78428553f4671)：deep_rl/agent/LinearOPEAgent.py、NeuralOPEAgent.py 与 template_jobs.py；固定策略评价，不是完整控制智能体。

- [Regularized Centered Emphatic Temporal Difference Learning · 2026](https://arxiv.org/abs/2605.04100)：折扣线性离策略预测；中心化与 emphatic 更新联合稳定性的两状态反例。尚未确认作者代码。

- [NS-NAC / BORL · 2025 原始预印本](https://arxiv.org/abs/2504.16415v1)：检查动态遗憾比较器、块重启及 Algorithm 1 的均匀状态采样权限。

- [NS-NAC / BORL · TMLR 论文入口](https://openreview.net/forum?id=hBJYNAYtoo)：最终版本入口；本文涉及具体状态采样步骤的说明明确对应预印本。尚未确认作者工程。

- [Average-reward RL in SMDPs via relative value iteration · 2025](https://arxiv.org/abs/2512.06218)：固定有限 SMDP 的参照函数、平均时长估计、RVI 更新与收敛条件。

- [异步随机逼近的稳定性与应用 · SIAM 2026](https://doi.org/10.1137/25M1769806)：RVI 理论的配套数学条件；参照函数非线性不等于深度函数逼近收敛。

- [Sample Complexity of Average-Reward Q-Learning · 2026](https://arxiv.org/abs/2601.13642)：单智能体与联邦、生成模型采样、折扣调度和奖励缩放；尚未确认作者代码。

- [A Harmonic Mean Formulation of Average Reward RL in SMDPs · 2026](https://arxiv.org/abs/2605.04880)：作为目标一致性审查材料；本节给出与总奖励/总时间不等价的确定性反例，不作为默认算法。

- [De Asis, Elsayed & He · Extending Differential TD for Episodic Problems · RLC 2026](https://rlj.cs.umass.edu/2026/papers/Paper33.html)：RLJ 第 7 卷正式条目；固定中心化的策略不变性、在线偏置学习与深度实测分别理解。

- [Episodic Differential TD · 原文 §3–4 与附录 C](https://arxiv.org/html/2605.04368v1)：区分奖励单位的中心 c 与价值单位的共享偏置 b；终止边界和 γ=1 回合形式需成套使用。
