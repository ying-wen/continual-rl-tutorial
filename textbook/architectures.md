# 持续智能体架构：从一次经验到下一次行动

一次真实交互怎样用于预测、模型与控制的学习，这些变化又怎样决定下一次行动？

## 本章内容

- 沿同一条持续轨迹追踪行动输入、学习器记忆、资格迹和技能后果，说明每次真实反馈先改变什么。
- 从技能的外部奖励、时长和终点预测推到平均奖励备份，再追踪规划后的价值如何改变下一行动。
- 用后续真实执行评价先前预测更新，区分局部元梯度与经过模型、规划、采样的整机导数。
- 把固定技能的短计算与学习技能的旧实验分开，检查版本、资源和全程收益而不借用局部算法的收敛保证。

## 阅读这条论证

跟随同一个持续运行的智能体，先区分行动所需的状态与学习所需的记忆。沿一段实际技能执行，看资格迹怎样改变后果预测；技能停止后，再把奖励、时长和终点预测用于平均奖励规划。后来取得的经验评价先前更新的步长，更新后的价值决定新的行动。各章的方法由此进入同一条经验流。

[完整交互的问题](architectures.md#lesson-setting) → [区分状态与学习记忆](architectures.md#lesson-agent-state) → [规定预测及其用途](architectures.md#lesson-cycle-targets) → [逐步更新与信用](architectures.md#lesson-cycle-real-step) → [技能停止时更新什么](architectures.md#lesson-cycle-completion) → [从模型计算到行动](architectures.md#lesson-cycle-planning) → [用后续经验调整学习](architectures.md#lesson-cycle-meta) → [运行同一条交互](architectures.md#lesson-cycle-code) → [资源与知识变化](architectures.md#lesson-cycle-resources) → [状态与技能也从经验中形成](architectures.md#lesson-learning-components) → [评价完整学习器](architectures.md#lesson-evaluation)

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [智能体状态：从经验学习怎样记住](state.md)：理解内部状态如何随经验更新。
- [模型与后果预测：学什么，才能用于下一次决策？](models.md)：理解可学习知识的规格与版本。
- [规划：把模型中的经验转成更好的决策](planning.md)：理解模型使用、资源分配与实际行动。


### Agent state 与 learner state

$z_t$ 是从历史形成的行动输入；完整持久状态 $W_t$ 还包含技能执行进度、参数、模型、资格迹、步长和敏感度。相同 $z_t$ 不代表下一次学习相同。

### Prediction / control / model

预测回答指定策略下未来会怎样；控制选择外部收益高的行为；模型保存可用于重新评价行动的后果。使用相同经验不使三者成为同一个目标。

### 真实时间与计算时间

环境转移推进原始时间 $t$，技能完成推进调用序号 $n$，模型备份推进计算序号 $k$。平均奖励的分母采用原始时间。

<a id="problem-definition"></a>

## 本章的问题定义

状态、预测、控制、技能、模型和规划共享有限资源；各模块单独可学并不保证组合后目标与时序一致。

### 给定条件与符号

- 各模块的输入/输出、目标、状态和更新规则。
- 真实交互接口、共享参数、总内存和计算预算、设计/测试权限。

### 需要求解的对象

完整可执行智能体与可反驳的模块协同假设，明确每条真实经验被哪些学习过程如何使用。

### 信息与数据权限

$z_t$ 是行动所用的历史摘要，$W_t$ 是包括参数、技能进度、模型、资格迹和学习规则状态的全部持久信息。行为 $b(\cdot\mid W_t)$ 产生动作。真实步 $t$、技能完成序号 $n$ 与规划备份序号 $k$ 分别计数。

$$
W_{t+1}=U(W_t,A_t,R_{t+1},O_{t+1}),\qquad J_T(U,b)=\frac{1}{T}\mathbb E\!\left[\sum_{t=0}^{T-1}R_{t+1}\right]
$$

$U$ 包含真实更新与其间的模型计算，$b$ 是行为规则，$T$ 为预先给定的真实交互期限。本章按原始时间比较有限期限收益；在相应平稳性和极限存在条件下才能进一步讨论长期奖励率。预测损失和元损失各自训练内部计算，不是额外的外部奖励。

### 成立条件与解的含义

- 各模块计时、目标参数版本、行为概率和表示版本明确，更新依赖已到达数据。
- 共享表示改变时需检查旧价值、模型和技能坐标；总成本包含规划、teacher与统计。

判断准则：同一条样本的控制与固定策略预测分别符合手算target；快照、模型折扣与资源记账一致；以等预算消融检验完整寿命收益、失败和模块漂移。

### 适用边界

- Alberta/OaK研究纲领不等于已验证的通用智能体。
- 模块loss各自下降不证明组合寿命收益上升。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [智能体状态与递归学习](state.md)：行动输入与完整学习器记忆有不同用途；状态构造器改变还会影响多个预测。

- 组合不同学习问题 · [通用价值函数与预测知识](gvf.md)：给定技能下的奖励、时长和终点是三个预测问题，其答案共同组成模型。

- 组合不同学习问题 · [Options 与技能发现](options.md)：技能规定执行策略、停止与时长，完成一次技能不等于环境终止。

- 组合不同学习问题 · [平均奖励与差分价值](average.md)：外部收益按原始时间比较，模型备份保留奖励减奖励率乘时长以及终点差分价值。

- 组合不同学习问题 · [时间信用分配与资格迹](credit.md)：资格迹让新误差更新这次真实执行中的旧预测，模拟查询须有独立的轨迹与迹。

- 组合不同学习问题 · [转移模型与后果模型](models.md)：模型保存后果问题的答案，不把规划重复读取算成新证据。

- 组合不同学习问题 · [时间抽象与规划](planning.md)：有限备份把学到的后果传播到决策价值，继而影响下一行动。

- 组合不同学习问题 · [流式更新与稳定性](streaming.md)：每次反馈后的更新须在声明的时间与记忆预算内完成。

- 组合不同学习问题 · [元学习与学习规则的适应](meta.md)：后来收到的后果评价先前学习规则，局部敏感度与整机未来收益导数须区分。

- 组合不同学习问题 · [持续控制与学习智能体比较](control.md)：最终评价完整行动与学习过程，不能仅评价最后的冻结策略。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

多个学习过程共用一条真实轨迹，却回答不同问题、跨越不同时间；单独正确的更新仍可能组成错误的学习过程。

### 本章的核心思路

固定一个教学世界，沿同一条经验追踪预测、完成后果、模型计算与下一选择，再用后来的经验评价先前学习。

1. [明确模型要回答的三个问题](architectures.md#lesson-cycle-targets)：在固定技能与剩余时间下定义奖励、时长和终点预测，由其条件期望推得平均奖励规划目标。

2. [新反馈先改变沿途预测](architectures.md#lesson-cycle-real-step)：先读旧预测形成误差，活动资格迹再分配写入；技能最后一步的奖励参与更新后才清迹。

3. [完整技能后果进入高层](architectures.md#lesson-cycle-completion)：累计外部奖励与真实时长形成差分控制残差；完整终点才更新终止模型。

4. [模型计算改变下一次选择](architectures.md#lesson-cycle-planning)：规划读取更新后的模型与控制值，只在规定预算内改变决策估计，再由真实行动产生新证据。

5. [后来的执行评价先前学习](architectures.md#lesson-cycle-meta)：启动前缓存预测，完成后用新回报和旧局部敏感度调整下一次步长，并注明停止求导的路径。

6. [检查完整循环与资源](architectures.md#lesson-cycle-code)：从单文件日志逐项重算目标、参数变化与决策，随后讨论技能和表示共同学习时的版本、成本与评价。

结论与条件：教学表格组合可验证接口与时序；不将各模块的局部理论相加成任意共享深网的收敛或通用智能保证。

### 相关方法改变了什么

- 表格Dyna组合：接口可解析，适合验证真实学习和规划分工。

- STOMP式链路：从子任务到技能、后果模型与规划，需完整计入发现与维护。

- 共享深度模块：减少部分重复表示，却引入梯度冲突、表示漂移和版本一致性问题。


<a id="lesson-setting"></a>

## 同一条经验怎样改变下一次行动

一个技能还在执行时，新奖励已经到来。它可以立即修正沿途预测，也必须计入这次技能的总收益；技能停止后，这个总收益才成为高层学习的一条完整样本。规划还可以使用早先学到的后果，提前改变下一次选择。架构要解释的就是这些计算怎样共用一条经验，而不混淆它们的对象与时间。

本章主线只使用一个持续任务：七个位置组成环，向左或向右移动，以 0.1 概率原地停留。每步付出 0.02，到达位置 1 另得 1，随后继续行动。位置直接可观测，外部奖励规则不变。两个给定技能分别持续向左、朝位置 4 行进，或持续向右、朝位置 1 行进；到达各自目标或执行两步便停止。每次至少执行一步，即使启动时已在目标位置也不作零时长停止。停止只将选择权交还高层，环境没有重置。

先固定这些[技能](options.md)，使读者可以追踪同一个预测问题：当前技能的累计外部奖励由带资格迹的 GVF 学习，时长和终点由完整执行学习；三类答案共同进入平均奖励规划。以后一次执行还会评价先前奖励预测的更新，从而调整其步长。状态、目标和技能策略在此给定，学习的是预测、控制值与局部学习速率。学习技能、表示及其维护随后再逐项放开。

这是为解释相互作用而构造的有限教学学习器，不是将所列方法合并后已有的命名算法。Dyna 提供真实学习与模型规划的组织方式，GVF 与 options 给出预测和行为对象，平均奖励规定时间目标。局部元更新如何嵌入其中，则由下面的具体计算定义。[旧六状态骨架](architectures.md#lesson-code)、[变化技能的七状态实验](architectures.md#lesson-integrated-code)和[配送调度算例](architectures.md#lesson-schedule)分别保留为机制对照，不作为同一次运行的阶段。

<a id="rlss-common-model-dimensions"></a>

## Common Model：四类知识、两类改进过程与问题维度

深入阅读：比较状态、预测、控制和模型的表示维度，明确哪些知识可以共享。

Common Model 将智能体内部常见知识区分为状态、策略、价值与模型。它们是功能角色，不要求四个互不共享的神经网络，也不是已证明唯一的智能架构。状态概括经验；策略提出行动；价值评价后果；模型预测行动可能带来的后果。一个共同 encoder 可以被多个角色使用，但依赖也因此更强。

| 区分 | 前者 | 后者 | 避免混淆 |
| --- | --- | --- | --- |
| 知识与过程 | 状态、策略、价值、模型是被保存的对象 | 学习和规划是改变或使用对象的过程 | 规划不是与模型同一种组件 |
| 预测与控制 | 固定所预测行为，估计后果 | 改变行为以改善目标 | 一个预测很准确，不等于已经选择好行动 |
| 世界信息与学习权限 | 完全或部分可观测 | online、batch、replay、是否可重置 | 部分可观测并不规定必须怎样训练 |
| 表示与目标 | 表格、线性、神经或递归表示 | 折扣、平均或有限生命期目标 | 神经网络不要求折扣目标；平均奖励也不要求表格 |
| 知识内容与获取方式 | 奖励、GVF 或 option 后果 | on-policy、off-policy、直接经验或模型计算 | 多种知识可以由同一流学习，却有不同覆盖条件 |

“补全方格”是一种研究方法：既然两个设计维度不同，就检查其四种组合，而不是先把某种组合当成禁区。例如固定/变化目标与固定/变化表示构成四格；每一格再清楚规定数据权限。但逻辑上可区分的维度，在性能上完全可以相互作用。

$$
\Delta_{\rm interaction}=(J_{11}-J_{10})-(J_{01}-J_{00}).
$$

给每个设计选择编码零或一。在同一问题与预算上测四个组合；这个差分比较第一个选择的作用是否随第二个选择变化。应报告不确定性，不能凭一个合成表格给算法交互下结论。

原讲义提出从问题出发、从智能体视角出发、让知识可以用经验检验，以及分别研究正交维度。它们是研究原则，不是性能保证。例如可以测量奖励，并不表示当前传感信息足以实现高奖励；可以检验一个预测，也不表示该预测值得花资源学习。

最小反例：公平隐藏位决定正确动作，观察始终相同。任意无额外信息的动作选择，其正确概率至多二分之一；即使事后奖励完全可观察，也无法提前知道这次隐藏位。若把奖励改成“总选动作一”，学习器能达到新指标满分，却没有提高原任务正确率。可测量、可辨识、可实现和符合目标，是四种不同要求。

同样，直接预测决策所需后果可以节省容量，但不能由此否定生成模型。若未来查询不断变化，广泛建模也可能有价值。比较应固定查询集合、表示容量、经验预算与使用方式。架构的研究愿景需要分解为这些可以形式化、实现和反驳的具体问题。

<a id="lesson-agent-state"></a>

## 1. 位置相同，学习器也可能不同

一般情况下，智能体先用 $z_{t+1}=f_{\phi_t}(z_t,A_t,R_{t+1},O_{t+1})$ 更新历史摘要。本例直接令 $z_t$ 为可见环位置，把状态估计误差暂时排除。仅有位置还不足以继续执行技能：同在位置 0，刚启动的向左技能尚可走两步，而已经走过一步的同名技能只能再走一步。

$$
W_t=(z_t,\chi_t,\Theta_t,\mathcal M_t,e_t,\beta_t,h_t,\kappa_t),\qquad \chi_t=(\omega,\nu,z_{\rm start},\ell,G,d).
$$

$W_t$ 记完整持久状态；$\chi_t$ 记录当前技能 $\omega$、执行版本 $\nu$、起点、剩余步数 $\ell$、已累计外部奖励 $G$ 和时长 $d$。$\Theta_t$ 包含各预测、控制参数与奖励率估计，$\mathcal M_t$ 是模型统计，$e_t$ 是资格迹，$\beta_t$ 是对数步长，$h_t$ 是供后续评价的局部敏感度。$\kappa_t$ 汇集预测评价缓存与调度状态。本例的环境随机数由环境持有，不交给智能体；供读者检查的完整日志也不属于学习器所需记忆。

行动输入与学习器状态因用途不同而区分。控制器在技能边界读取 $Q(z,\omega)$；执行中的低层策略读取位置及当前技能。资格迹没有作为行动输入，却决定新误差改到哪些旧预测。两个副本即使当前 $Q$ 相同，若 $e_t$ 或 $\beta_t$ 不同，收到同一反馈也会得到不同参数与未来行为。因此，恢复持续学习不能只恢复权重。

设第 $n$ 次技能从原始时刻 $T_n$ 开始，完成于 $T_{n+1}=T_n+\tau_n$。$n$ 只在技能结束时增加，$t$ 每次环境转移增加；规划计数 $k$ 只在一次模型备份完成时增加。下面每个奖励率都以原始步为单位。若需要真实物理时间，还必须把动作等待和计算延迟纳入时钟；本例操作次数不等于墙钟耗时。

<a id="lesson-cycle-targets"></a>

## 2. 先规定三个预测，再看控制怎样使用它们

对固定版本技能 $\omega$，从位置 $z$、剩余执行预算 $\ell$ 开始，令 $\tau$ 为到目标或预算用尽的首次停止时间。记以下真实条件期望，再用帽号表示学习器估计。它们的目标策略都是该技能的实际执行策略，而不是高层选择技能的策略。

$$
\begin{aligned}R_\omega(\ell,z)&=\mathbb E_\omega[\textstyle\sum_{i=0}^{\tau-1}R_{t+i+1}\mid \ell,z],\\D_\omega(\ell,z)&=\mathbb E_\omega[\tau\mid\ell,z],\\P_\omega(\ell,z,j)&=\Pr_\omega(Z_{t+\tau}=j\mid\ell,z).\end{aligned}
$$

三个问题的累积信号分别是外部奖励、每原始步的 1、实际停止时的终点指示量；在技能停止后延续为零。有限两步上限使这些累计量存在。启动模型读取 $\ell=2$，而逐步奖励预测还保存 $\ell=1$ 的答案。

保留剩余步数有数学作用。如果只问“在位置 0 向左走到技能停止”，却不说明还可走几步，同一个输入就混合了不同的终点与时长。增广输入 $(\omega,\ell,z)$ 使有限执行问题可以递推。到目标或 $\ell=1$ 时，本次奖励仍然收到，只是没有下一步预测尾项。

$$
\widehat B(z,\omega)=\widehat R_\omega(2,z)-\bar g\widehat D_\omega(2,z)+\sum_j\widehat P_\omega(2,z,j)\max_{\omega'}Q(j,\omega').
$$

planner 读取三个启动预测来重算技能后果。$\bar g$ 为奖励/原始步，$Q$ 为差分动作价值估计。在一次备份中固定 $\bar g,Q$，线性期望将 $\mathbb E[G-\bar g\tau+V(Z_{\rm end})]$ 分解成这三项，并不要求奖励、时长和终点独立。

上式说明预测怎样参与选择。改变奖励预测会改变这个技能的备份目标，备份再写入高层 Q；当控制器在相应位置比较技能时，才会读到这个变化。仅被记录、没有参与 Q 计算的预测，不通过这条路径影响动作。[GVF 章](gvf.md#gvf-control-interfaces)还讨论预测直接作为行动输入，以及通过辅助学习改变表示的用法。本例所需的模型量由备份公式确定：若改为折扣目标，就需要联合折扣终点预测；若比较风险，均值之外的分布信息也可能影响选择。

外部目标始终是原始时间上的收益。技能的“到位置 4”只是行为及停止的规定，主任务并未额外奖励它到达 4。预测损失训练后果模型，后续预测误差评价局部步长；二者不是新的外部奖励。一个预测变准却不改变最优候选，当前行动可以完全不变。

<a id="lesson-cycle-real-step"></a>

## 3. 新奖励先沿真实轨迹改变预测

启动一次技能时，保存起点、版本、启动前奖励预测和本次步长 $\alpha=\exp\beta(z_{\rm start},\omega)$，将段内资格迹置零。整次执行使用这一个步长与同一技能策略。每个原子步只推进环境一次，再把真实反馈分给奖励预测与累计器。用 $i=0,\ldots,\tau-1$ 表示本次执行内的位置，$x_i$ 为当前 $(\omega,\ell_i,z_i)$ 的单位坐标向量。

$$
\begin{aligned}\delta_i^R&=r_{i+1}+c_{i+1}w_i^\top x_{i+1}-w_i^\top x_i,\\e_i&=\lambda e_{i-1}+x_i,\qquad w_{i+1}=w_i+\alpha\delta_i^Re_i,\\G&\leftarrow G+r_{i+1},\qquad d\leftarrow d+1.\end{aligned}
$$

$w$ 是奖励 GVF 的表格参数，$c_{i+1}$ 在技能继续时为 1、停止时为 0；终止时无需构造有效的 $x_{i+1}$。旧值先形成整个误差，再写权重。资格迹在本段开始时为零，结束步仍先用它更新，随后才清除。

两步执行时，第一步有 $e_0=x_0$，第二步有 $e_1=\lambda x_0+x_1$，所以启动预测总共写入 $\alpha(\delta_0^R+\lambda\delta_1^R)$。最后一步的奖励因此可以在本次停止时回到启动坐标，而不必等下一次访问。$\lambda=0$ 时它不直接回传，$\lambda=1$ 时在本例的固定初值与不重复增广坐标下，启动写入等于对完整样本回报的一次误差更新。这是信用分配，尚未发生模型规划。

$$
\delta_0^R+\lambda\delta_1^R=(1-\lambda)(r_1+w_0^\top x_1)+\lambda(r_1+r_2)-w_0^\top x_0.
$$

右边混合一步自举目标与两步完整目标。因为剩余步数递减，本例的第二步当前坐标尚未被第一步改写，才可在这里统一使用启动参数 $w_0$。一般在线共享特征或重访问题需要[信用分配章](credit.md#lesson-online)的参数版本分析。

默认记录的第 4、5 个真实步，向右技能从 6 经 0 到 1，收到 $(-0.02,0.98)$。两个相关预测起初都为零，$\alpha=0.25,\lambda=0.8$；启动预测先变为 $-0.005$，终点反馈再写入 $0.25\times0.8\times0.98=0.196$，得到 $0.191$。它尚不是完整回报 $0.96$：当前只做了小步长的混合目标更新。完整样本不会作为额外奖励模型标签再训练一次。

![七状态环中向右技能从6经0到1；第4步留下启动资格1，第5步仍保留0.8，让0.98终点奖励把启动预测从负0.005改为0.191。](../assets/crl-figures/agent-cycle-credit.svg)

先看环上的两次真实移动，再读同一启动坐标的资格与预测变化。两步共得到外部奖励0.96，TD(λ)的小步长更新得到预测0.191。数值来自单文件脚本默认 seed 0 的第4、5步；终点反馈先使用活动迹，再清除段内迹。

技能外的环境仍然继续，但这个有限预测问题已经停止，所以本段迹应清除。把另一技能或下一次调用的坐标直接续到旧迹上，会改变所学回报的边界。模型备份也不把这条真实迹推进：规划查询并非刚才行为的下一次访问；若在模拟轨迹上使用多步学习，须另建模拟轨迹及其迹。

本例只对正在真实执行的技能作这种更新，因此不需要动作重要性比。若要让同一转移训练其他技能的预测，必须检查这些技能在该动作上的概率与行为覆盖，采用相应离策略估计器。表格 TD 的当前简化不能直接替换成共享神经网络后继续沿用全部结论。

<a id="lesson-cycle-completion"></a>

## 4. 技能停止后，完整后果才进入高层

一次技能完成提供 $(z_{\rm start},\omega,G,\tau,z')$。奖励 GVF 此前已经逐步学习；时长与终点预测现在才获得完整标签 $\tau$ 与终点 one-hot。教学实现保留每个起点—技能最近至多四次完整后果，用其均值估计 $\widehat D,\widehat P$；这些是有限记忆下的样本估计，不是给学习器查阅的真实转移表。

$$
\delta_n^Q=G_n-\bar g_n\tau_n+\max_{\omega'}Q_n(z',\omega')-Q_n(z_{\rm start},\omega),\qquad Q^+=Q_n+\alpha_Q\delta_n^Qx_{z_{\rm start},\omega},\qquad\bar g^+=\bar g_n+\alpha_g\delta_n^Q.
$$

$x_{z,\omega}$ 是高层表格的单位坐标，$Q_n,\bar g_n$ 指本次完成写入之前的值。先算一个残差，再取 $\alpha_Q=0.1,\alpha_g=0.01$ 分别写入。该常步长教学更新使用真实时长，区别于[平均奖励章](average.md#lesson-duration)的期望时长归一化算法；这里不借用后者的收敛保证。

这个残差从固定技能的差分最优方程获得：完成一段行为得到 $G_n$，按消耗的真实时间扣除 $\bar g_n\tau_n$，再保留终点继续行动的价值。option 停止并非世界终止，所以尾值没有置零。奖励率更新也不是对这个残差平方直接做梯度下降；它是与价值联合求解差分方程的另一条更新。

相应平稳、单链且平均时长有限的策略中，长期率按累计奖励除以累计原始时长比较，而非每次技能奖励的平均。即使每个技能都有一次完整标签，短技能与长技能也不能各被算成同样长的一“步”。本次短运行实际报告 $\sum_{t<T}R_{t+1}/T$，另记录学习器的 $\bar g$；有限轨迹均值与内部率估计不要求相等。

若记录预算在技能中途耗尽，只把真实发生的奖励与时间计入全程表现，不将部分执行塞入完整时长/终点模型，也不执行完整技能的高层更新。逐步 GVF 已经获得的那几次更新可以保留。相反，两步上限本来就是技能的停止规则，因此达到上限会提供合法完整样本。

<a id="lesson-cycle-planning"></a>

## 5. 规划改变估计，再让真实行为改变经验

现在模型中至少有一条已观察的完整后果。调度器优先查询当前即将决策的位置，再考虑刚完成的技能行及其他已知行，按固定顺序至多做两次备份；可查询行不足时允许重复。每次查询读取真实更新后的奖励率和当前高层 Q；如果上一项备份已改了某个后继值，下一项可以立即使用这个新值。

$$
Q^{(k+1)}(z,\omega)=Q^{(k)}(z,\omega)+\alpha_{\rm plan}\left[\widehat R_\omega-\bar g^+\widehat D_\omega+\sum_j\widehat P_\omega(j)\max_{\omega'}Q^{(k)}(j,\omega')-Q^{(k)}(z,\omega)\right].
$$

省略模型中固定的启动输入 $(2,z)$，取 $\alpha_{\rm plan}=0.1$。单次备份先读取完整的第 $k$ 版 Q，再写一个表格项。这个设计只修改 Q：不增加真实时钟，不增加完整后果样本数，不推进真实资格迹，也不把模型奖励送去再次训练模型或奖励率。

下一次技能选择发生在实际到达的位置。它读取规划后的该位置 Q，而不是读取刚被更新的任意起点。更新一个远处状态可以暂时不影响当前选择，直到后续备份把影响传过来；即使查询了当前位置，数值变化也可能尚未跨过两候选的差距。应当沿实际读取关系解释一次行动变化。

沿默认短运行，在 $t=9$ 到达位置 1 后，一次规划查询“从 1 向左”的模型：$\widehat R=-0.016,\widehat D=2,\widehat P(6)=1$，读取 $\bar g=0.02709391256672$ 和 $\max_\omega Q(6,\omega)=0.1876379403872$。所以目标为 $0.11745011525376$，将该 Q 从 $0.005201092192$ 写为 $0.016425994498176$；向右 Q 仍为 0。这里规划前已经偏好向左，本次只是继续提高其估计，并未在这一刻翻转动作。

要看到完整反馈，可对照从一开始就不规划的副本。两者使用相同采集规则和随机数种子，前九次真实转移恰好相同；此时无规划副本的 $Q(1,\mathrm L)=-0.003672064$，向右 Q 为 0，所以选右。每次完成做两项规划的副本则选左；第十个真实步分别是 $1\to2$ 与 $1\to0$。这是此前多次规划及其后续真实更新共同造成的差异，不是只开关第九步某次备份的反事实。此后采样已经分开。

![前9个真实步相同的两个学习器在位置1比较左右价值；无规划选右并到2，有规划选左并到0。](../assets/crl-figures/agent-cycle-planning.svg)

左、右面板从运行开始分别使用每次完成0次或2次模型备份。第9步后的选择读取各自全部学习历史产生的Q，第10步开始得到不同真实经验；有规划分支在第9步本次备份之前已经偏好左。图示一条固定种子短轨迹的机制差异，不作为收益优势估计。

被选技能接着执行真实原子动作，环境才产生下一条样本。因而有两个反馈闭环：实际反馈训练预测与控制，学到的预测通过规划影响行为；新行为又改变后续哪些模型行会得到数据。规划重复利用的是当前模型的判断，不是获得了更多独立事实。模型有偏时，多次备份也可能反复传播同一偏差。

<a id="lesson-cycle-meta"></a>

## 6. 后来的执行怎样评价先前学习

奖励预测已经被 planner 使用，但还可先研究一个更小的元问题：上一次学习率是否让这个预测在下一次执行前更准确？对每个起点—技能保存上一段启动预测写入对对数步长的局部敏感度。固定启动参数和本段数据，剩余步数坐标保证本段中每个当前值和后继值在被读取前未受此前写入影响，故样本残差不依赖本段步长：

$$
h_n=\frac{\partial\widehat R_{\omega,n}^{+}(2,z)}{\partial\beta(z,\omega)}\Big|_{\rm local}=\alpha_n\sum_{i=0}^{\tau_n-1}\lambda^i\delta_i^R.
$$

在这个有限表格执行内，它是固定启动参数与轨迹的精确局部导数。$h_n$ 评价一次参数写入对步长的敏感性，$e_i$ 分配一次奖励误差对旧预测的信用；二者不能互换。共享特征或重复坐标时，残差会另含步长依赖，不能继续省去它。

下一次从同一起点调用同一技能时，先缓存学习前的启动预测 $\bar q_m$，随后照常执行与学习，直到获得新的完整外部奖励 $G_m$。外层误差只能在这时计算。若 $n$ 是该起点—技能上一次完整执行，使用 $L_m=\tfrac12(\bar q_m-G_m)^2$ 给出局部元方向：

$$
\widehat{\nabla_\beta L_m}=(\bar q_m-G_m)h_n,\qquad\beta\leftarrow\operatorname{clip}\!\left(\beta-\eta\widehat{\nabla_\beta L_m},\beta_{\min},\beta_{\max}\right).
$$

实现取 $\beta_{\min}=\log0.02,\beta_{\max}=\log0.8$。第一次完整执行没有此前敏感度，只能建立记录。当前执行使用的步长在启动时已经固定；新步长用于后续调用。$\bar q_m$ 必须是本次学习前的缓存，不能拿已经用 $G_m$ 学过的预测来冒充后续验证。

在第 5 步完成的 $6\to0\to1$ 段中，$h_n=0.191$。第 8、9 步又从 6 执行同一技能，启动前缓存 $\bar q_m=0.191$，新的完整奖励仍为 $0.96$，于是元方向为 $(0.191-0.96)\times0.191=-0.146879$。取 $\eta=0.05$，原先 $\alpha=0.25$ 被调整到约 $0.251842746$，但本段始终使用 0.25。下一次调用才读取新步长。

![两次从6向右执行的真实轨迹沿时间排列；前次保存敏感度0.191，后次用学习前预测0.191与新回报0.96形成负元梯度，下一次步长增加。](../assets/crl-figures/agent-cycle-meta.svg)

沿真实时间从第4、5步读到第8、9步。验证预测在第二次执行开始时缓存，完整新回报在其停止时才可用；第二次执行中的更新仍使用原步长0.25。箭头连接的是先前局部写入与后续预测评价，没有穿过planner的元梯度。

敏感度在上一段实际使用的 $\beta_n$ 处计算，却延迟应用到当前保存的 $\beta$；若其间规则已更新，两者未必相等。这仍是一项延迟局部元更新，而非对当前完整学习器重算的精确梯度。[元学习章](meta.md#lesson-evaluation-timing)展开评价时刻和导数对象的区别。

这个元信号问的是固定技能下的预测学习，不是主任务收益的完整梯度。它截断更早参数、其他模型统计和真实轨迹的导数，也没有穿过 planner。固定世界中下一次执行仍有停留随机性，因此单次元方向带有抽样噪声；它没有获得真实期望回报的标签。模型误差降低是否改善高层选择，仍需沿前一节的读取链检验。

若希望直接用下游控制评价规则，令 $U_\eta$ 表示真实反馈的全部写入，$P_\eta$ 表示一次规划写入，$Y=U_\eta(W,\xi)$。在固定数据、固定查询顺序和可微分支内，做一次规划后的链式法则是：

$$
\frac{\partial P_\eta(Y)}{\partial\eta}=\frac{\partial P_\eta}{\partial Y}\frac{\partial U_\eta}{\partial\eta}+\frac{\partial P_\eta}{\partial\eta}\Big|_Y,\qquad\frac{\partial J}{\partial\eta}=\left(\frac{\partial P_\eta(Y)}{\partial\eta}\right)^\top\nabla J.
$$

第一项包含“规则改变预测写入，再经模型备份改变 Q”的路径，第二项包含规则直接改变规划步长等作用。多次规划需逐次乘相应 Jacobian；旧状态也依赖元参数时还要加上其敏感度传播。外层若显式依赖元参数，另有直接导数。

例如规则只改变奖励模型，而控制仅通过规划读取它：若将规划结果整体停止梯度，这条控制元信号就被切断；若只将模型输入停止梯度，则丢掉模型学习的用途路径，但仍可能保留价值递推路径。冻结数值、冻结参数更新与停止求导是三种不同操作。离散技能选择、查询选择与策略版本提交还会改变真实数据分布，不能靠对一段固定记录自动求导获得它们全部的影响。

Meta-Gradient RL 的后续样本评价和 Meta-Gradient Search Control 的模型查询分布分别提供相关做法，但它们有各自的内外层目标。此处只实现前面的局部步长评价；完整规划路径的公式用于指出进一步设计还需要什么，而不把它列为已运行功能。

<a id="lesson-cycle-code"></a>

## 7. 沿运行记录走完同一次学习循环

下载[单文件实现](../tutorials/agent_cycle_walkthrough.py)，在仓库根目录运行下方命令；也可从任意目录使用文件的绝对路径。只需 Python 3.10+ 标准库。默认从位置 0 出发，先按采集协议调用右、左、右、左四次技能，然后按高层 Q 贪心选择，并列时选右。前四次是明确给定的采集，不是学习器已经作出的选择；后续没有额外探索保证。

固定短运行及机制检查；不是性能训练或完整方法复现。

```bash
python3 tutorials/agent_cycle_walkthrough.py --steps 32 --seed 0
python3 tutorials/agent_cycle_walkthrough.py --test
```

**算法：算法伪代码**

1. 启动技能：缓存预测、步长、先前敏感度；清空本段资格迹
1. 每个真实步：
  1. 执行当前技能动作，接收位置与外部奖励
  1. 读取旧预测，计算误差与资格迹，再写奖励 GVF
  1. 累加外部奖励与原始时长
  1. 若技能尚未停止，携带执行进度继续
1. 技能停止后：
  1. 用旧高层 Q 和奖励率算一个真实残差，分别更新二者
  1. 用启动前缓存预测和本次完整回报评价先前步长
  1. 保存本段局部敏感度，更新时长与终点样本；清空段内迹
  1. 查询已观察模型，逐次做至多两项高层 Q 备份
  1. 在实际到达位置读取更新后的 Q，选择下一技能

| 正文对象 | 实现位置 | 日志中怎样辨认 |
| --- | --- | --- |
| 技能开始时的版本和缓存 | Agent.start | option_start 的 prediction_before、alpha_used、previous_h |
| 原子奖励、延续和活动资格迹 | reward_td_step / Agent.observe | atomic 的 reads、trace_active、writes；最后一步迹仍参与写入 |
| 真实高层误差与率更新 | Agent.complete | option_end 的 control；先读旧值再写 |
| 后续样本评价局部步长 | Agent.complete | meta 的 validation_prediction、previous_h、alpha_next |
| 模型中的真实后果记忆 | Agent.model / complete | model_after 的 reward、duration、endpoint、evidence_in_window |
| 模型备份到下一次选择 | Agent.plan / run | planning 的 target、after；decision 的 preplan_greedy 与 postplan_greedy |

先按 events 中的真实时间读 atomic，再在同一个时间处读 option_end、planning 和 decision。planning 只增加计算序号，option_end 也不额外消耗真实步。日志保存了用于教学的完整快照，学习器只读取自己的有限内部状态；不要把离线记录大小当作学习器运行内存。

可先遮住最后一个 decision，利用前面的模型、奖励率和 Q 自行算出备份，然后预测下一项 atomic 的实际动作。再加上 `--planning-budget 0` 检查失去模型传播后什么不同；两个完整运行很快会采到不同经验，不能把所有后续差异都归为固定数据上的单个更新效果。

<a id="lesson-cycle-resources"></a>

## 8. 完整循环还必须能在预算内继续

这条循环的存储有不同职责：当前位置与技能进度让行为继续；资格迹保存未兑现的信用；模型保存可重复查询的后果；预测缓存和敏感度让后续经验评价先前学习。删除其中一项，不只是少一个数组，而是改变未来会执行的更新。

短实现每条原子经验只用于当时的预测写入，不存整条原子轨迹供反复训练；模型另保留每个起点—技能最近四次的时长与终点摘要。它是有界样本记忆的在线原型，不冒称完全不保存样本的严格流式方法。[流式学习章](streaming.md#lesson-learner-state)区分这些记忆的用途。若要求连这种摘要也不能保存，可改用固定维数递推矩估计，但估计权重及其适应性质随之改变。离线输出日志只供读者检查，学习器不读取它。

每次技能完成至多做 $B=2$ 次备份，所以模型计算上限随完成次数而不是原始时间分配。若所有技能需一或两步，则长期平均每原始步的备份数会受时长选择影响。比较规划是否有益时，至少同时记录真实步数 $T$、完成次数 $N$、实际备份数 $K$ 和模型容量；若实验改为每原始步固定预算，又是另一项调度。

当前技能版本固定，省去了变化技能模型的失配。以后开始学习内部策略时，可在一次执行完成后提交新版本，并重新处理该技能的后果预测、高层 Q、资格迹和待评价敏感度；完整执行标签属于实际执行的旧版本。表示改变又会影响所有读出坐标。[版本反例](architectures.md#lesson-version-contract)和[表示漂移](architectures.md#lesson-drift)分别展开这两类变化，知识保留与特征替换也必须沿这些依赖处理。

由此可以分层放开假设：先确认固定技能时预测确实进入备份，再学习内部策略并核对后果版本，随后考虑学习状态或发现新问题。每次新增能力都要说明它从什么反馈学习、其输出由谁使用，以及新增的计算和存储。完整架构的研究价值在于使这些相互作用可检查，而非从局部正确直接推出长期系统性能。

<a id="lesson-learning-components"></a>

## 状态与技能也从经验中形成时

前面的运行固定了状态与技能，才能逐步看清预测、信用与规划怎样相互作用。现在再去掉这两项便利。智能体不仅要估计一个已有技能的后果，还要从经验中学会记住什么、值得练习什么，以及怎样完成练习。新的困难是：用于表示问题和执行行为的对象本身也在变化。

先看状态。观测到达时，状态更新函数把上一活动与新信号合成当前活动；预测误差随后改变这个函数的参数。[状态章的写入学习](state.md#lesson-learned-memory)沿真实连续交互展示这一过程：后来取得的结果通过保存的敏感度改变写入门，新的提示因而以不同权重进入记忆，之后的选择也随之改变。参数更新不会凭空重算全部历史。除非明确保存并重放历史，当前活动仍是实际运行至今形成的活动，下一次递推才使用新的参数。

接着看子目标。候选可以由已访问状态及尚未尝试的行为产生，但候选只是一个有待解决的问题。必须先规定如何判断到达、途中用什么信号学习、什么时候停止，再用实际动作学习内部策略。[子目标章](goals.md#lesson-learning-cycle)把这一步与候选评分分开：候选的覆盖分数决定先练什么，动作价值决定练习中如何行动，外部后果模型才决定学得的技能是否值得用于主任务。三种数值回答的是三个不同问题。

设一个技能版本已经固定，从状态 $z_t$ 执行至停止时刻 $t+\tau$。它的奖励后果可作为一个 GVF：策略是该技能的实际策略，累积信号是外部奖励，延续在技能停止时变为零。终点预测和时长预测另有各自的标签。若技能还在执行中学习并改变动作规则，这条完整经验属于实际执行的变化规则，而不自动成为某个冻结新策略的无偏样本。先冻结策略、再执行并学习其模型，是一种便于逐项核对的方法；持续改变技能则需要显式处理策略版本或离策略校正。

$$
G_t^\omega=\sum_{j=0}^{\tau-1}\gamma^jR_{t+j+1},\qquad y_t^{\rm plan}=G_t^\omega+\gamma^\tau V(z_{t+\tau}).
$$

这里采用折扣目标，把一个真实完整技能后果写成规划目标的样本形式。内部到达奖励用来训练技能，不替代式中的外部奖励。平均奖励目标改用 $\sum_{j=0}^{\tau-1}R_{t+j+1}-\bar g\tau+h(z_{t+\tau})$，仍须保留同一段真实时长。有限记录截止而技能未停止时，不能把它当作完整后果。

规划把这些后果预测与当前后继价值组合，更新不同技能的价值；高层选择其中一个，才开始下一段真实执行。这段执行又同时提供几类反馈：观测序列供状态构造器保留历史，子任务信号训练内部策略，外部奖励、时长和停止位置训练后果模型。选择练习哪个技能，因而决定这些学习过程接下来能得到什么数据。[探索章](exploration.md#lesson-data-cycle)沿同一运输任务进一步分析这种选择。反复规划可以传播已学后果的影响；要检验尚未执行的行为，仍需取得相应经验。

状态与技能同时变化时，模型要跟踪的对象也随之变化。状态构造器决定用什么摘要描述起点；若模型预测停止后的活动，它还决定终点标签。技能策略与停止规则则决定从这个起点实际走过什么路径、何时停下。前一类变化可能只改变同一后果的坐标，也可能丢掉原有信息；后一类变化直接改变被预测的行为。一次整合可以先用本步已有活动形成预测和学习信号，再写入相关参数，在技能边界处理策略版本，随后用与当前状态和行为相符的模型更新高层价值。保留旧样本时需说明怎样恢复其状态，保留旧模型时需说明它预测的是哪一套状态与技能。

这种变化还会影响尚未完成的学习。资格迹保存哪些旧预测应接受后来误差，元敏感度保存先前更新怎样受学习规则影响。若某个状态坐标被替换，沿用它原来的迹，便可能把后来误差写给一个含义已经不同的预测；若技能变了，待评价的敏感度也未必仍对应当前问题。变换、清除或保留这些量，都应根据它们所记录的依赖来决定。[知识保留](retention.md#lesson-agent-memory)与[学习可塑性](plasticity.md#lesson-consumers)因此不仅处理已学答案，也要处理还会影响以后更新的记忆。

两份教学程序各自放开一项条件：[learned-memory-walkthrough.py](../tutorials/learned-memory-walkthrough.py)学习状态写入，[learned_subgoal_cycle.py](../tutorials/learned_subgoal_cycle.py)在给定地点表示上发现候选并学习技能、后果和选择。它们分别让学习对象及其行为影响可见，尚未共同训练一个状态与技能都在变化的系统。共同学习首先需要分清：模型失配来自表示换了坐标，还是技能真的改变了行为，抑或新的表示丢失了信息？[精确后果诊断](../tutorials/knowledge_transport.py)用同一六状态任务分别检查这三种变化。[共同研究问题](../docs/research-joint-learning.md#shared-learning-inquiry)继续由此推导需要的对照，以及构建和维护知识的代价。

<a id="lesson-derive"></a>

## 一次交互的参数与目标

深入阅读：换用六状态骨架，逐项推导并行控制、固定策略 GVF 与原子动作模型的不同更新。

以下六状态骨架将技能收缩为原子动作，单独检查动作概率、参数快照和并行预测。它另用一个依赖记忆提示的持续环境，预测头在该骨架中不供控制器读取。

$$
a_t\sim b_{\theta_t}(\cdot\mid z_t),\qquad z_{t+1}=f_{\phi_t}(z_t,a_t,r_{t+1},o_{t+1}),\qquad \xi_t=(z_t,a_t,r_{t+1},z_{t+1},b_{\theta_t}(a_t\mid z_t))
$$

先从已有活动状态选动作；收到奖励与新观测后，用本步构造器形成下一活动，再供学习器构造目标。这里记录的是学习接口中的经验；原观测是否另外保存由数据协议决定。行为概率在采样时保存，后续策略更新不改变它。

这里的量有三种来源。$(a_t,b_t(a_t\mid z_t))$ 在采样时保存，$(r_{t+1},o_{t+1})$ 由环境返回；构造器按声明的规则生成 $z_{t+1}$。$Q_t,v_t,\bar g_t$ 则是学习器在写入前已有的估计。任务给定外部奖励与平均奖励目标，设计者另给 GVF 的目标策略 $\pi(a=1\mid z)=1$ 和折扣 $0.8$。隐藏任务变化时刻、真实转移概率与最优奖励率均不在本条经验中。

控制误差从差分最优方程而来。在固定世界和 Markov 状态下，若最优奖励率 $g_* $ 与起点无关，则 $Q_*(z,a)=\mathbb E[R_{t+1}-g_*+\max_{a'}Q_*(Z_{t+1},a')\mid z,a]$。学习器用当前 $\bar g_t,Q_t$ 代替未知解，并用实际后果替代条件期望，得到下面第一项样本残差。GVF 则先规定“以后总选动作 1”的折扣预测，再由同一个真实后继构造自己的残差；它无需与控制器使用同一目标策略。

$$
\begin{aligned}\delta_t^{\rm control}&=r_{t+1}-\bar g_t+\max_a Q_t(z_{t+1},a)-Q_t(z_t,a_t),\\\delta_t^{\rm pred}&=r_{t+1}+0.8\,v_t(z_{t+1})-v_t(z_t),\\\rho_t&=\mathbf1[a_t=1]/b_t(a_t\mid z_t).\end{aligned}
$$

控制的 max 对应最优动作目标；GVF 的 ρ 对应固定 target policy。行为率、最优奖励率和 GVF 预测值必须分别命名与记录。

$$
\begin{aligned}Q(z_t,a_t)&\leftarrow Q_t(z_t,a_t)+\alpha\delta_t^{\rm control},\\\bar g&\leftarrow\bar g_t+\eta\alpha\delta_t^{\rm control},\\v(z_t)&\leftarrow v_t(z_t)+\alpha\rho_t\delta_t^{\rm pred}.\end{aligned}
$$

先算完两种误差，再写这三项，即使 z′=z 也读取写入前的值。Q 的一次写入可视为固定 bootstrap 目标的表格回归；奖励率沿控制误差另作跟踪更新，并不是对同一平方损失求 g 的下降梯度。

动作纠偏为什么只出现在这里的预测更新中？在给定 $z_t$ 的动作平均里，$b_t(a\mid z_t)\rho_t(a)=\pi(a\mid z_t)$，所以有支持的行为样本可以估计目标策略的一步预测目标。控制更新已条件化于执行的 $(z_t,a_t)$，再以 max 定义后续控制目标。这个等式没有修复状态信息不足或任意函数逼近的不稳定；六状态实现使用固定的充分手工状态和独立表格，才可把这些接口单独检查。真实世界改变后，程序继续跟踪新后果，不能将整条变化过程视作同一个平稳 Bellman 方程。

经验模型记录每个 (z,a) 后看到各 (r,z′) 的次数，形成经验条件分布。它不同于只记最后一次转移：后者会把真实随机性误当作不断变化的确定性结果。本例使用固定的手工状态，因此这些计数始终位于同一坐标中；状态构造器也学习时，需要进一步处理后文的表示漂移。状态、动作与奖励取值有限，使表格槽位数固定。但 Python 整数计数与规划游标的位宽仍可随运行时间增长；槽位固定不等于无限运行时固定字节预算。若奖励连续，以每个不同奖励作为字典键还会增加槽位数。

严格的资源限制需要另行规定有限精度计数、固定容量的近期统计或参数化模型，并明确溢出与淘汰规则。这些选择会改变模型估计。本例保留累计计数，以便观察旧经验如何延缓对环境变化的适应；它不是严格固定字节预算的终生实现。

$$
\widehat p_t(r,z'\mid z,a)=\frac{N_t(z,a,r,z')}{\sum_{\tilde r,\tilde z}N_t(z,a,\tilde r,\tilde z)},\quad \delta^{\rm plan}=\sum_{r,z'}\widehat p_t(r,z'\mid z,a)[r-\bar g+\max_{a'}Q(z',a')]-Q(z,a)
$$

规划选择已观察的 (z,a) 做模型期望 backup。这里是教学性的表格 Dyna 组合；奖励率只从真实数据学习，规划只改 Q，这项选择写进接口并测试。

实际次序是先将本条真实后果加到 N，再进入规划。上式的 $\widehat p_t$ 表示这次查询时已经包含新样本的模型；$Q,\bar g$ 则读取真实更新后的数值。一项备份先算完整个和式，再写对应 Q 格；下一项备份可以读取这个新值。规划期间 N、GVF 和 $\bar g$ 都不写入，因此重复查询模型不会把一条真实观察计成多条证据。将来共享神经表示时，还需规定哪些损失改变构造器、哪些目标停止梯度；当前表格实现并未替这种联合更新作出选择。

另一些 differential planning 方法也从模型更新奖励率，因此上述接口是一项具体设计选择。模型样本的分布与真实时钟不同；若扩展为规划也更新 $\bar g$，需要说明模型采样分布、时长和更新比例，不能将模拟次数直接当成实际交互的时间。

一次循环于是完整了：从活动状态选动作并保存概率，执行一次真实转移，形成下一活动，用旧参数计算控制和预测目标，写入各自参数与模型，执行给定预算的模型备份，再依据更新后的控制值选择下一动作。下面先逐项计算同一条转移，再看这个循环在持续交互中产生什么结果。

<a id="lesson-example"></a>

## 手算同一条样本的两个不同目标

深入阅读：手算六状态骨架中的一条给定转移，检查共同旧值与随后规划的参数版本。

这条转移只发生一次。沿着下面三条分支，看它究竟改变了哪个价值格、哪个预测和哪条模型边。

![一次状态转移向三个学习对象分流：控制值的一个格子、GVF预测的一根柱以及模型中新增计数的有向边。](https://yingwen.io/crl-figures/concept-crl-mechanisms-architecture.svg)

虚线以下的三个学习器共同读取更新前快照。左边只改变执行动作的 Q 格和奖励率；中间灰柱是旧预测，绿色短柱是本步后的预测；右边为这条首次记录的实际后果增加一个计数。采样时行为概率随经验保留，因此 GVF 的重要性比为 4。此图尚未加入模型规划；下一阶段的备份另计。

设当前状态 z0 的 Q=[1,2]，下一状态 z1 的 Q=[3,4]，ḡ=0.5；GVF 值为 v(z0)=2、v(z1)=5。实际采动作 1 的行为概率为 0.25，收到奖励 1。控制 error=1−0.5+4−2=2.5；预测 error=1+0.8×5−2=3；目标策略固定选动作 1，故 ρ=4。

取 α=0.1、η=0.1：Q(z0,1) 从 2 变成 2.25，ḡ 从 0.5 变成 0.525，GVF 从 2 变成 3.2。三个量共用一次 experience，却遵循不同目标。若采的是动作 0，控制依然学习该动作后果，而该固定目标策略 GVF 的重要性比为 0，本步不更新。

若这也是模型首次记录 (z0,1)，该模型行便只有 (r=1,z1) 这个后果。随后备份这一行时，读取更新后的 ḡ=0.525、Q(z0,1)=2.25 和仍为 4 的后继最大值，故残差为 1−0.525+4−2.25=2.225，Q 再写成 2.4725；ḡ 与 GVF 保持 0.525 和 3.2。若先用旧 ḡ 做规划，会得到另一个数。这个继续计算说明快照边界如何决定实际算法。

下一次真实动作在已经到达的 z1 选择，读取该处当前 Q=[3,4]；本次备份改的是 z0 的价值，要等查询或再次到达相关状态时才可能影响选择。这里为手算另设行为概率 0.25；下面脚本的 ε=0.2、双动作贪心策略实际给出 0.1 或 0.9。两处都须使用采样当时的概率，不能拿更新后的贪心动作倒算它。

<a id="lesson-code"></a>

## 不间断交互的集成实验

深入阅读：运行六状态无技能原型，考察累计模型在隐藏奖励映射变化后的适应。

明确的参数快照、保存行为概率、经验分布模型和固定预算 round-robin planning。

```python
class ModularAgent:
    """Hand-designed Markov state; tabular control/GVF/one-step model.
    The model stores empirical counts of (reward,next_state) per state/action.
    Planning changes q only. Reward-rate learning uses REAL data only here.
    """
    def __init__(self, states=6, alpha=0.05, eta=0.05, gamma=0.8):
        self.q = [[0., 0.] for _ in range(states)]
        self.gvf = [0.] * states
        self.rate = 0.
        self.model = {}
        self.planning_cursor = 0
        self.alpha, self.eta, self.gamma = alpha, eta, gamma

    def action_probabilities(self, state, epsilon=0.2):
        best = max(range(2), key=lambda a: self.q[state][a])
        prob = [epsilon / 2., epsilon / 2.]
        prob[best] += 1. - epsilon
        return prob

    def observe(self, state, action, reward, next_state, behavior_probability,
                planning_budget=0):
        if behavior_probability <= 0 or planning_budget < 0:
            raise ValueError("invalid recorded probability or planning budget")
        # All REAL targets use one parameter snapshot before any mutation.
        q_error = reward - self.rate + max(self.q[next_state]) - self.q[state][action]
        prediction_error = reward + self.gamma*self.gvf[next_state] - self.gvf[state]
        ratio = (1.0 if action == 1 else 0.0) / behavior_probability
        self.q[state][action] += self.alpha*q_error
        self.rate += self.eta*self.alpha*q_error
        self.gvf[state] += self.alpha*ratio*prediction_error
        # Finite toy outcome slots, not an unlimited-lifetime byte bound:
        # Python counts/cursor grow in bit width; new reward values add keys.
        outcomes = self.model.setdefault((state, action), {})
        outcomes[reward, next_state] = outcomes.get((reward, next_state), 0) + 1
        # A deterministic scheduler makes the simulated-update budget inspectable.
        keys = sorted(self.model)
        for _ in range(planning_budget):
            s, a = keys[self.planning_cursor % len(keys)]
            self.planning_cursor += 1
            outcomes = self.model[s, a]
            count = sum(outcomes.values())
            delta = sum(n * (r - self.rate + max(self.q[sp]) - self.q[s][a])
                        for (r, sp), n in outcomes.items()) / count
            self.q[s][a] += self.alpha*delta
        return {"control_error": q_error, "prediction_error": prediction_error,
                "ratio": ratio, "planning_updates": planning_budget}


def architecture_run(seed=7, planning_budget=0, steps=18000):
    rng, agent = random.Random(seed), ModularAgent()
    cue, phase = rng.randrange(2), 0
    rewards, trace = [], None
    for t in range(steps):
        # State is a hand-coded (phase, remembered cue), not a discovered representation.
        state = 2*phase + cue
        probabilities = agent.action_probabilities(state)
        action = 0 if rng.random() < probabilities[0] else 1
        mapping = 0 if t < steps // 2 else 1  # hidden change, never supplied to agent
        reward = float(action == (cue ^ mapping)) if phase == 1 else 0.0
        if phase == 2:
            phase, cue = 0, rng.randrange(2)
        else:
            phase += 1
        next_state = 2*phase + cue
        trace = agent.observe(state, action, reward, next_state,
                              probabilities[action], planning_budget)
        rewards.append(reward)
    block = steps // 3
    means = [sum(rewards[i*block:(i+1)*block]) / block for i in range(3)]
    return {"window_reward_rates": [round(v, 6) for v in means],
            "optimal_rate_estimate": round(agent.rate, 6),
            "model_entries": len(agent.model), "last_update": trace}


def architectures_demo():
    print("architectures", {"without_planning": architecture_run(),
          "one_simulated_update_per_step": architecture_run(planning_budget=1),
          "scope": "fixed-budget toy integration; not an OaK implementation"})
```

Python 3.10+，无外部环境、无 GPU、无网络

```sh
python examples/lifelong_algorithms_lab.py architectures
python examples/lifelong_algorithms_lab.py test
```

环境每三个原始步骤形成一个自然周期：阶段 0 显示二值 cue，阶段 1 必须依据记住的 cue 选动作并获得奖励，阶段 2 过渡到下一 cue。Agent state 是手工编码的 (phase,cue)，共六种。半程隐藏地翻转 cue→正确动作的映射；程序不把切换时间或映射传给 agent，也不清参数、记忆或模型。这个自然周期不是一个对 agent reset 的训练回合。

对比每步零次和一次模型更新。最优外部奖励率为 $1/3$；使用 $\epsilon=0.2$ 探索的实际率接近 $0.3$。固定随机种子 7、运行 18000 步，无规划的三个窗口实际率约为 $[0.299667,0.2895,0.300833]$；有规划时为 $[0.299667,0.202167,0.300833]$，变化附近反而更差。规划条件下最终 $\bar g\approx0.31847$，无规划约为 $1/3$。奖励率估计与行为实际率的对象不同，因此不能将这两个量互换。累积模型仍保留旧映射，可能解释适应迟缓；需通过改变模型记忆长度、多个种子和预算匹配进一步检验。

这段源码把时序写在两个位置：architecture_run 保存采样概率并推进阶段和 cue，ModularAgent.observe 先计算两个旧值误差，再写 Q、rate、gvf 和模型，最后轮转备份已见模型行。这里以 phase、cue 直接实现状态索引，cue 从可见提示写入后保持；它没有调用可学习的状态构造器。

GVF 在这个骨架中是独立的预测记录：action_probabilities 只读取 Q，模型也不读取 gvf 数组。因此该实验可以检查同一经验上的并行预测更新，尚未检验“学会 GVF 导致控制改善”。下面的技能模型会给出一个真正被 planner 读取的预测接口；要检验辅助 GVF 或学习记忆对行为的作用，则需另外接入并检验相应消费者。

<a id="lesson-construction"></a>

## 从 Dyna 扩展到 subtask → option → model → planning

深入阅读：从子任务出发推导技能与后果模型，比较折扣接口和平均奖励接口。

Dyna 的基础链路是“真实数据学价值和模型，模型供模拟价值更新”。时间抽象扩展还需要四个不同对象。Subtask 定义为何要学某种行为；option 定义具体如何执行以及何时停止；option model 描述实际执行产生的外部奖励、终点与时长；planner 用这些后果与当前主任务价值组合。学习技能用的人工奖励不能不加区分地写进主任务模型。

$$
\begin{aligned}r_o(z)&=\mathbb E\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}\mid z,o\right],\\p_o(z,z')&=\mathbb E[\gamma^\tau\mathbf1(Z_{t+\tau}=z')\mid z,o],\\(T_oV)(z)&=r_o(z)+\sum_{z'}p_o(z,z')V(z').\end{aligned}
$$

折扣 option model 已把 $γ^τ$ 包进终点权重，因此 planner 不能再乘一次 γ。τ 随机时也不能把 $E[γ^τV]$ 简化成 $γ^{Eτ}E[V]$。平均奖励 option model 则保留奖励、时长和未折扣转移，使用 R−gτ。

例如学习“到门口”为了形成可复用 option，终止奖励可用于驱动这个 subtask；但主任务可能是递送物品，门口本身没有外部奖励。Planner 必须知道此技能真实消耗多少步、沿路得到什么外部收益、到达哪里，才能判断其价值。STOMP 的研究价值在于把这个链路作为可连接的学习问题，而不是在图中简单把 option 画成一条箭头。

Successor features 提供另一种可复用接口：若外部奖励近似 $r=φ(s,a,s^{\prime})^Tw$，则策略的 successor feature 预测折扣特征累积，价值近似 $ψ_π^Tw$。它方便奖励改变时快速重算价值，但已知特征线性分解、固定策略及动力学变化都是重要边界；并非等价于任意世界模型。

<a id="rlss-oak-acquisition-maintenance"></a>

## OaK 的八个过程：一个子问题从哪里来，何时值得保留

深入阅读：区分知识获取和已有知识的维护，再讨论二者的计算分配。

OaK 提出从特征产生子问题，从子问题学习 option，再学习其后果模型，用模型改善主任务行为。每条连接都需要可观察的输入、被优化的量与成本。OaK 是一项研究架构；特征生成和长期效用维护尤其不是已经确定的通用算法。

| 过程 | 本次产生或改变什么 | 必须追踪的后果 |
| --- | --- | --- |
| 主任务学习 | 策略、价值与奖励率估计 | 改变访问分布，也改变其他模块收到的数据 |
| 生成特征 | 候选状态分量 | 表示容量、计算成本与现有知识坐标 |
| 排列特征 | 当前与未来效用的估计 | 排序依赖使用机会、尺度与成熟期 |
| 建立子任务 | 所重视的特征、终端偏好与停止问题 | 目标是否仍尊重主任务的奖励与价值 |
| 学习 options 与子价值 | 内部动作策略与停止行为 | 后果模型的目标随技能学习而改变 |
| 学习模型 | 累计奖励、终点与时长后果 | 数据权限、覆盖与模型过期 |
| 规划 | 用模型更新价值或选择行动 | 错误模型与计算分配可能放大偏差 |
| 筛选与维护 | 保留、修改或退役整条对象链 | 删除成本、重新学习和长期收益 |

先冻结当前主策略 $\mu$ 及其奖励率估计 $\widehat g^\mu$、差分价值估计 $\widehat v^\mu$。给定特征 $\phi_i$ 与奖金强度 $\kappa$，子问题选择内部策略 $\pi_o$ 与停止规则 $\beta_o$。令 T 为停止时刻，得到 OaK/NeurIPS 讲义第 24 页的差分目标；这里把主策略和 option 策略的符号明确分开。

$$
\max_{\pi_o,\beta_o}J_i(\pi_o,\beta_o;s),\qquad
 J_i=\mathbb E_{\pi_o,\beta_o}\!\left[\sum_{k=t+1}^{T}(R_k-\widehat g^\mu)
 +\widehat v^\mu(S_T)+\kappa\phi_i(S_T)\mid S_t=s\right].
$$

从时刻 t 出发，首先收到 $R_{t+1}$。到达 $S_T$ 的奖励 $R_T$ 仍计入沿途和式，然后终端价值与特征奖金各计一次。T 是 option 停止而非世界终止；主任务在之后继续。

参考奖励率来自主策略，不是待学习 option 自身的平均奖励。减去它相当于按 $T-t$ 个等时长原始步计算时间的机会成本；$\widehat v^\mu(S_T)$ 再估计回到主策略后的相对后果。若每步经过的物理时间不同，就需要按实际时长扣除对应奖励率，不能仍把每次决策当成一个相同单位。

手算：技能执行两步后停止，奖励依次为 2、4，参考奖励率为 1，终点主差分价值为 3，特征值为 2，$\kappa=0.5$。这次样本的子回报是 $(2-1)+(4-1)+3+0.5\times2=8$。终点的奖励 4 没有被丢弃，终点价值也没有沿途每步重复计入。

这个式子不是“无论代价都尽快到达一个坐标”。沿途外部奖励仍然计入，终点仍由主价值衡量，特征奖金只是改变某些后果的偏好。若允许永不停止，而且存在高于参考奖励率的循环，子问题可能没有有限最优值。教学实现应先限定有限窗口或 proper stopping 类，再研究解除该约束的平均奖励子问题。

折扣的到达目标小实验可以解释终端奖金怎样改变停走决策，但它和上面的差分目标不是同一个数学问题。若进一步省略主任务终端价值，则又改变了子目标。把这些简化写清楚，才能知道一个学习曲线检验了哪一条接口。

$$
C_{\rm build}+N C_{\rm use}<N C_{\rm primitive}
\quad\Longleftrightarrow\quad
N>\frac{C_{\rm build}}{C_{\rm primitive}-C_{\rm use}},\quad
C_{\rm primitive}>C_{\rm use}.
$$

这是固定单次使用成本的简单收支模型：构建费用包括探索、技能训练和模型拟合。它只计算资源回收门槛，不是策略收益定理。持续世界还要加上维护、失效与行为机会成本。

给定 waypoint 后，可以依次学习内部策略、拟合技能后果，再用模型规划；也可以让有限候选竞争或改变动力学。这样的实验能检验若干接口，但候选集合、停止规则和主任务模型仍可能由设计者提供。它不是完整的 OaK 自主构造，也没有单独验证特征发现。

一个可检验的下一步是先固定特征和子问题，只研究 option 改变后模型如何跟踪；再固定这条接口，比较手工候选、随机候选与效用选择。每步记录主任务累计收益、技能获取成本、模型校准、实际 backup 数与对象退役。先知道哪个过程带来收益，再增加自主构造范围。

<a id="rlss-oak-executable-chain"></a>

## 可运行算例：子问题改变后，旧模型为何使主任务选错

深入阅读：用另一组有限计算进一步检查子任务、技能、模型和规划的依赖。

下面把四个环节接起来：冻结主策略的奖励率与差分价值；求解包含停止选择的子问题；拟合该技能的后果；用后果做一次主任务规划。环境只有五个状态，每一步都能手算。这是原创机制诊断，不是作者 benchmark，也不是完整 OaK。状态、特征、两步时限和主策略由实验者给定；技能用精确动态规划求解，后果模型再由完整执行样本估计。

| 当前状态 | 动作 → 下一状态 | 期望外部奖励 | 主策略 μ | 终端特征 φ |
| --- | --- | --- | --- | --- |
| H：共同起点 | advance → J | 0 | 选择 advance | 0 |
| J：路口 | usual → A | 3 | 选择 usual | 1 |
| J：路口 | deferred → B | 0 | 不选 | 1 |
| J：路口 | feature → C | 0 | 不选 | 1 |
| A：常规分支终点 | return → H | 0 | 选择 return | 0 |
| B：延后收益分支终点 | return → H | 4 | 选择 return | 0 |
| C：高特征分支终点 | return → H | 0 | 选择 return | 2 |

表中的奖励是期望。实际采样时，每一步再独立加上等概率的 +1 或 −1。转移本身确定，每步耗时一个单位，世界没有终止状态。主策略沿 H→J→A→H 循环，三步的期望总奖励为 3。B 和 C 虽然不在主循环上，从它们出发仍按主策略返回 H。

$$
g^\mu=1,\qquad g^\mu+h^\mu(s)=r(s,\mu(s))+h^\mu(s'),\qquad
 h^\mu(H)=0,\quad (h^\mu(H),h^\mu(J),h^\mu(A),h^\mu(B),h^\mu(C))=(0,1,-1,3,-1).
$$

固定 h(H)=0 消除差分价值的加法常数。比如 h(B)=4−1+h(H)=3，而 h(C)=0−1+h(H)=−1。代码直接解这些线性方程，并检查每个状态的残差。

这个主循环有周期。这里采用 Poisson 方程和再生定义：从给定状态按 μ 走到 H，计算沿途奖励减去平均奖励率的期望和，并规定 h(H)=0。不需要把周期链上可能不收敛的普通无穷和当作数值答案。

option 只能在 H 启动，至少执行一步，最多执行两步。到达 J 后，它可以立即停止，也可以执行 usual、deferred、feature 中的一个动作，到达对应终点后停止。两步时限是设计约束，防止把未解决的无限时域停止问题藏在实现里。J 上的停走选择则确实由子问题求解，不是预先指定到某个终点。

$$
\begin{gathered}Z_\kappa(s)=h^\mu(s)+\kappa\phi(s),\qquad F_0(s)=Z_\kappa(s),\\
 F_\ell(s)=\max\!\left\{Z_\kappa(s),\ \max_a\mathbb E[R-g^\mu+F_{\ell-1}(S')\mid s,a]\right\}.\end{gathered}
$$

F 表示还剩 ℓ 步时，允许现在停止的子问题价值。初始 H 不允许停止，必须先执行 advance，再使用 F₁(J)。停止达到最大值时 β=1；否则 β=0，内部策略选取最大化的动作。相等时约定优先停止。这里求的是精确有限时域解，不是声称用 TD 从样本学会了该解。

从 H 出发只有四条可能的停止路径。因此无需相信实现，也能独立列出完整答案。所有式子先计入沿途奖励减去每步的主奖励率，最后只加入一次终端价值与奖金。

| 完整 option 路径 | 时长 | 子问题期望回报 Jκ | 主任务偏离值，不含奖金 |
| --- | --- | --- | --- |
| H→J，立即停止 | 1 | κ | 0 |
| H→J→A，停止 | 2 | 0 | 0 |
| H→J→B，停止 | 2 | 1 | 1 |
| H→J→C，停止 | 2 | −3+2κ | −3 |

例如 B 路径的回报是 $(0-1)+(0-1)+3=1$。到 B 的那一刻还没有收到返回 H 时的奖励 4；该未来后果由 $h^\mu(B)=3$ 表示。若在 κ=0 时删掉终端主价值，四条路径的值变成 −1、1、−2、−2，算法改选即时奖励为 3 的 A 路径。它解决的是另一个问题，不再评价回到主策略后的后果。

![四条停止路径的子回报随特征奖金改变，最优上包络依次选择 B、J、C。](https://yingwen.io/crl-code/diagnostics/rlss-oak/subproblem-options.svg)

由精确枚举与动态规划独立得到。κ<1 时选 B；1≤κ≤3 时在 J 停止；κ>3 时选 C。边界相等时优先停止。图不是训练曲线。

κ=0、2、4 分别产生“走到 B”“在 J 停止”“走到 C”三个 option。这同时展示了内部策略与停止规则的来源。不同 κ 定义不同目标，不能把不同曲线上的数值增大称为跨目标的学习进步。尤其在 κ=4 这个固定子问题中，C 路径的值 5 高于 B 路径的 1，却可能更差地服务主任务。

精确求解内部策略和停止规则。递归状态包含剩余时长；初始不准停止，终端值只计算一次。execute 返回完整 option 样本。

```python
@dataclass
class Option:
    kappa: float
    with_terminal_bias: bool
    value: float
    decisions: dict


def solve_option(kappa, with_terminal_bias=True):
    """Exact finite-horizon DP over (state, remaining time).

    Stop is allowed after the first action. At the horizon it is compulsory.
    A stop adds h_mu(s) + kappa*phi(s) ONCE. A continuation adds only R-g_mu.
    The stop action wins ties. This yields beta in {0,1}; it is not sampled TD.
    """
    if kappa < 0 or not math.isfinite(kappa):
        raise ValueError("kappa must be finite and nonnegative")
    decisions = {}

    def terminal(state):
        return (BIAS[state] if with_terminal_bias else 0.0) + kappa * PHI[state]

    @lru_cache(None)
    def value(state, remaining, may_stop):
        if remaining == 0:
            decisions[state, remaining, may_stop] = "stop"
            return terminal(state)
        best = terminal(state) if may_stop else -math.inf
        choice = "stop" if may_stop else None
        for action, (successor, reward) in WORLD[state].items():
            candidate = reward - GAIN + value(successor, remaining - 1, True)
            if candidate > best + 1e-12:
                best, choice = candidate, action
        decisions[state, remaining, may_stop] = choice
        return best

    optimum = value("H", HORIZON, False)
    return Option(kappa, with_terminal_bias, optimum, decisions)


def execute(option, rng=None):
    """A complete option execution. RNG=None gives the exact mean trajectory.

    The world never terminates. The return transition from the endpoint is NOT
    included in this sample: it is represented by the frozen terminal bias.
    """
    state, remaining, total, duration = "H", HORIZON, 0.0, 0
    while True:
        action = option.decisions[state, remaining, duration > 0]
        if action == "stop":
            return total, duration, state
        state, reward = WORLD[state][action]
        if rng is not None:
            reward += 1.0 if rng.random() < 0.5 else -1.0
        total += reward
        duration += 1
        remaining -= 1
```

接着只预测技能的外部后果：累计外部奖励、持续时间与终点分布。κφ 是为了构建技能加入的偏好，不属于主环境奖励；不能混进主任务模型。主规划器比较两种选择：在 H 按 μ 做一个原始步，或先执行当前 option，然后恢复 μ。

$$
B^\mu_o(H)=\bar R_o(H)-g^\mu\bar\tau_o(H)+\sum_s P_o(s\mid H)h^\mu(s),\qquad
 B^\mu_{\rm primitive}(H)=0-1+h^\mu(J)=0.
$$

因为 h(H)=0，B 也等于相对冻结 μ 的一次偏离优势。它不是反复执行这个选择后的平均奖励率，更不是全局最优收益。高层相等时回退到原始 μ 动作。

| 技能版本 | 后果模型：奖励、时长、终点 | 主备份 B | 每次到 H 都执行它的真实 gain |
| --- | --- | --- | --- |
| κ=0 | 0，2，B | 1 | 4/3 |
| κ=2 | 0，1，J | 0 | 1 |
| κ=4 | 0，2，C | −3 | 0 |

最后一列另算：每次到 H 采用表中的 option，停止后跟随 μ 直到再次回到 H。B 的再生回路三步获得期望奖励 4；C 的回路三步获得 0；在 J 停止则回到 μ 的三步奖励 3。代码既计算回路总奖励除以总时长，也另解所形成原始策略的奖励率方程，二者一致。这是独立的执行评价，不是把主备份 1 或 −3 直接读成长期收益。

现在在同一个 option 槽里把 κ 从 0 改为 4。环境没有改变，但技能由 B 路径变成了 C 路径。冻结旧模型仍报 B=1，规划器因此选中当前技能；实际执行的却是主备份 −3 的 C 路径。重估当前模型后，它报 B=−3，规划器回退 μ。恢复的是选择的正确性，奖励率由 0 回到 1；这没有让新技能变好，更没有超过旧技能的 4/3。若要保留旧能力，应另研究保存旧版本而非原地覆盖的机制。

后果样本均值与主任务规划。模型不储存子目标奖金；规划偏离值和重复调用的真实奖励率由不同函数计算。

```python
@dataclass
class OutcomeModel:
    """MC sufficient statistics: E[external reward], E[duration], P(endpoint).

    Kappa, feature bonuses, and terminal value NEVER enter the reward field.
    A model version is part of the protocol; changing an option changes targets.
    """
    count: int = 0
    reward_sum: float = 0.0
    duration_sum: float = 0.0
    endpoints: dict = field(default_factory=lambda: {s: 0 for s in STATES})

    def observe(self, sample):
        reward, duration, endpoint = sample
        if duration < 1 or endpoint not in self.endpoints:
            raise ValueError("only complete positive-duration option samples")
        self.count += 1
        self.reward_sum += reward
        self.duration_sum += duration
        self.endpoints[endpoint] += 1

    def target(self):
        """One option then return to mu: Rbar - g_mu*tau_bar + Pbar*h_mu.

        h_mu(H)=0, so this also equals its deviation advantage at H.
        This scalar is NOT the new policy's long-run reward rate.
        """
        if not self.count:
            raise ValueError("unobserved model: fall back to the primitive policy")
        return (self.reward_sum - GAIN * self.duration_sum
                + sum(self.endpoints[s] * BIAS[s] for s in STATES)) / self.count

    def record(self):
        if not self.count:
            raise ValueError("no moments without data")
        return {"count": self.count, "external_reward": self.reward_sum / self.count,
                "duration": self.duration_sum / self.count,
                "endpoint_probabilities": {s: self.endpoints[s] / self.count for s in STATES},
                "main_backup": self.target()}


def exact_model(option):
    result = OutcomeModel()
    result.observe(execute(option))
    return result


def repeated_policy_gain(option):
    """Independently evaluate a repeated high-level choice by a regenerative cycle.

    At H execute this option (or one mu action if None); thereafter follow mu
    until H. Repeat. Sum expected external reward / elapsed primitive time.
    Neither the kappa bonus nor any bias term belongs to this true gain.
    """
    if option is None:
        state, total = WORLD["H"][MU["H"]]
        duration = 1
    else:
        total, duration, state = execute(option)
    while state != "H":
        state, reward = WORLD[state][MU[state]]
        total += reward
        duration += 1
    return total / duration


def planned_gain(model, current_option):
    # Compare the option with the one-step mu backup at H, exactly zero here.
    # Ties favor the primitive baseline. Evaluator, not agent, knows the true gain.
    return repeated_policy_gain(current_option if model.target() > 0.0 else None)
```

采样对照先执行旧技能 64 次。切换后，比较冻结旧模型、继续混合新旧样本、清空旧统计后只估计新版本。每个种子最多再执行新技能 256 次；两个更新方案共享同一批新样本。共有 64 个随机种子。每次探测从 H 重新开始，一个完整样本耗费两个原始步：旧版本 128 步，新版本最多 512 步。探测重启是一项明确的数据权限，不是严格 single-life 实验。

$$
\mathbb E[\widehat B_{\rm pooled}(n)]=-3+4\frac{64}{64+n}.
$$

旧技能终点 B 的差分价值为 3，新技能终点 C 为 −1。混合样本的期望备份因此逐渐从 1 走向 −3。奖励噪声均值为零；样本数量是固定的。这个式子不表示每条随机样本路径都单调。

![旧模型保持正的主备份，混合样本缓慢转负，分版本模型围绕新技能的真实值负三波动。](https://yingwen.io/crl-code/diagnostics/rlss-oak/model-backup-refresh.svg)

实际运行的 64 种子均值；阴影为均值上下 1 个标准误，不是置信区间。灰线给出当前技能的精确备份与混合模型的解析期望。每个种子的值保存在原始 JSON。横轴是新版本样本数，每个样本两个真实步。

| 新样本数 | 混合模型：均值 ± 标准误 | 分版本模型：均值 ± 标准误 |
| --- | --- | --- |
| 1 | 0.9144 ± 0.0235 | −3.4063 ± 0.1846 |
| 24 | −0.1044 ± 0.0221 | −3.0013 ± 0.0422 |
| 256 | −2.2096 ± 0.0094 | −3.0074 ± 0.0104 |

![旧模型使高层重复选择奖励率为零的新技能；分版本重估选择奖励率为一的主策略；混合模型逐步改正。](https://yingwen.io/crl-code/diagnostics/rlss-oak/selection-gain.svg)

纵轴是模型选出的再生策略经真实环境解析评价后的奖励率，再对 64 个种子取均值；阴影为上下 1 个标准误，不是置信区间。这不是训练期间的累计回报。探测成本已另报，但未从此率中扣除。

分版本方案一个样本就能选对，是这个诊断特意保留的简单性质：新技能的两步噪声总和最多为 2，故其单样本备份至多为 −1，已低于原始动作的 0。这不证明一般问题只需一个样本。分版本的优势来自避免错误目标混合，而不是更复杂的估计器；在不知道版本变化、终点也随机或价值同时变化时，还需要另外的实验。

本算例没有从采样学习 option，没有自主构造特征，也没有求解不断改变表示的完整架构。它把一个可独立定位的困难讲清楚：子目标、技能实现和后果模型不是同一个对象。子目标可以尊重外部奖励而仍牺牲主任务收益；主规划器必须用外部后果重新评价它，且评价使用的模型必须对应当前实际执行的技能。

下载后可直接运行。只需 Python 3.10+ 标准库；test 检查方程、121 个 κ 的独立枚举和模型接口，run 重建全部图表数据。

```bash
python3 examples/rlss_oak_diagnostics.py test
python3 examples/rlss_oak_diagnostics.py run --output results.json
```

[完整代码（MIT）](../examples/rlss_oak_diagnostics.py) · [实际运行的全部结果 JSON](https://yingwen.io/crl-code/diagnostics/rlss-oak/results.json)。可以先把 κ 固定，再去掉终端主价值；也可以保留旧版本为单独技能。每次只改变一条接口，比较子问题解、主备份和真实 gain 是否出现不同变化。

<a id="lesson-integrated-setting"></a>

## 把技能学习与预测模型接入同一闭环

深入阅读：回到学习技能内部策略的七状态研究原型，明确它与主文固定技能教学运行的区别。

另一个七状态原型放开内部策略学习，并加入奖励变化。它与主线使用相同环几何，每个原子步向左或向右，以 0.1 概率停留；到达当前奖励位置得 1，每步另付 0.02。这里奖励位置在实验半程改变；这个时刻不提供给 agent。状态直接可观测，因此本实验不混入状态估计误差。控制采用平均奖励差分更新，实际比较保留全部学习期的外部累计收益。它们分别是更新所依据的目标和有限运行中被测量的表现。

本页展示的协议固定为 1200 步，在前 600 步之后改变奖励位置。当前教学脚本用总预算的一半规定变化时刻；直接更改 --steps 会同时改变世界的变化时间。因此，不能把不同预算的结果当作同一个世界过程的前缀比较。研究长时间扩展时，应先将变化日程与停止记录的预算分开固定。

| 对象 | 具体定义 | 学到什么 |
| --- | --- | --- |
| 状态 | 当前环位置 | 本例给定，不研究状态发现 |
| 子目标 | 到位置 1 或位置 4 | 本例给定，不声称自动构建目标 |
| 技能内部策略 | 子任务 Q 的 ε-greedy 策略 | 用每条真实经验学习两个子任务 |
| 技能终止 | 到达目标或执行六步 | 规则给定；不等于环境重置 |
| 技能模型 | 外部奖励和、时长、终点概率 | 从完整技能执行学习三类预测 |
| 控制与规划 | 原子动作和技能上的差分 Q | 真实更新与有限次模型期望备份 |

$$
\delta_t^i=c_{t+1}^i+0.95\,\mathbf1[S_{t+1}\ne g_i]\max_a q_t^i(S_{t+1},a)-q_t^i(S_t,A_t).
$$

子任务到达目标时累计量为 1，否则为 −0.01。人工累计量只用于技能学习，不进入主任务模型。真实环境仍然继续。

子任务更新使用主任务采集的原子经验，因此通常是离策略 Q-learning。内部策略在一次高层动作执行期间固定，在动作结束后才提交新策略。这样，当前执行样本对应一个明确的技能版本。这里只学习内部策略；终止规则和子目标仍由设计者给定。

<a id="lesson-definition-answer-use"></a>

## 连接模块之前：谁定义对象，谁学习，谁使用？

深入阅读：逐项检查原型的问题由谁给定、答案由谁学习，以及控制器怎样使用它们。

七状态环给出了各模块的具体对象。将这些对象与前面的原子循环放在一起，可以按三个问题核对：问题由谁规定，从经验更新什么，更新后的答案由谁读取。主任务评价保持独立；辅助预测或技能子任务可以使用不同信号。

| 对象 | 谁定义它 | 从经验更新什么 | 谁读取答案 |
| --- | --- | --- | --- |
| 主任务回报 | 任务或评价协议给定外部奖励及时间目标 | 主任务 critic、策略或差分价值 | 实际动作或技能选择器 |
| 辅助 GVF | 设计者或另一个构建机制给定 cumulant、continuation、目标策略 | 给定问题的预测参数 | 状态构建、辅助表示或明确的决策/模型接口 |
| 技能子任务 | 设计者或子任务生成器给定内部目标和停止条件 | 低层策略及其子任务价值；部分方法也学终止 | 被调用时产生原子动作 |
| option 后果模型 | 执行策略和停止规则确定预测语义，planner 决定所需后果 | 实际外部奖励、持续时间与终点预测 | 高层价值的模型备份 |
| 对象构建与选择规则 | 设计者给出候选族、资源限制和评价办法 | 候选生成、保留或淘汰规则 | 之后提出的问题与可用技能集合 |

最后一行不是前四行附带完成的功能。TD 误差可用于学习一个问题的答案，却不会自动选择最有用的 cumulant。Q-learning 可把指定到达目标变成技能，却不会自动说明为什么这个目标值得保存。构建算法还需要说明候选空间、效果信号、预算和淘汰规则；主回报的改进须经实际对照验证。

七状态原型的定义来源可以逐项追踪：GOALS=(1,4) 由设计者给定；available 排除已在对应目标上的启动；subtask_update 用到达奖金与步成本训练低层 Q；内部动作来自已提交的贪心策略加探索；终止条件是到达目标或六步上限。没有任何一步声称 TD 自动生成了这些规则。

同一原型中，OptionModel.update 只接收真实外部奖励和、实际时长与终点。OptionModel.target 才将这三项预测组合成主任务规划目标。低层到达奖金只进入 subtask_update，不进入该模型，也不记入真实收益。完整执行数据回答的是实际执行版本的技能，而不是刚刚更新后的另一策略。

$$
\underbrace{c^o\longrightarrow q_{\rm sub}^o\longrightarrow\pi_o}_{\text{learn how to execute}}\qquad\underbrace{(R,\tau,S_{\rm end})\longrightarrow(\widehat R_o,\widehat D_o,\widehat P_o)\longrightarrow Q_{\rm main}(s,o)}_{\text{learn whether to invoke}}
$$

左侧回答怎样执行子任务，右侧回答它对主任务是否值得调用。二者通过实际执行的策略连接，不是把两个奖励直接相加。若采用 Option-Critic，低层策略与终止也可以直接由主任务回报训练；这是另一条明确的设计路线。

构造型的共同检查顺序是：先找定义的来源，再检查学习目标，最后找到读取结果的控制过程。直接读取与共享参数是两条不同的作用路径：预测可以作为 actor 输入，也可以仅通过辅助梯度改变表示。检验前者时固定参数、比较是否读取答案；检验后者时重新训练有无辅助更新的系统，并匹配预算。只在部署时移除一个预测头，不能排除它在训练时已经形成的作用。

Sutton 与 Barto §17.5 将增量函数逼近、帮助未来学习的表示、学得模型上的规划、自主选择问题、行为与学习的相互作用列为研究议题，并另外讨论实际交互的安全性。这些议题帮助我们检查完整架构缺了什么，却不提供把模块相连就会成功的定理。一个组合方案应说明目前哪些对象给定、哪些已经学习、哪些仍是假设，并分别展示局部正确性和系统收益的证据。

<a id="lesson-model-questions"></a>

## GVF 何时真正成为规划知识

深入阅读：完整推导原型的 Monte Carlo 三头模型及其差分控制更新。

本节继续分析上面的变化技能原型，代码是 integrated_agents/_system.py。它的三个模型头都使用完整技能样本，规划次数与元学习设置也不同于主线单文件算例。

$$
R_o(s)=\mathbb E[\sum_{k=0}^{\tau-1}R_{t+k+1}\mid s,o],\quad D_o(s)=\mathbb E[\tau\mid s,o],\quad P_o(s,j)=\Pr(S_{t+\tau}=j\mid s,o).
$$

外部奖励和、单位时钟累积、终止位置指示量构成三个终止预测问题。目标策略是实际技能策略。

这些条件期望的样本只在一次技能完成后才全部可得。启动时保存 s、o 和已提交的策略版本；执行期间用真实外部奖励累加 $R=\sum_{k=0}^{\tau-1}R_{t+k+1}$，每次环境转移给时长加一；停止时观察终点 $s'$。于是三个标签分别是 $R,\tau$ 和终点 one-hot 向量。六步上限从本次启动计时，是停止规则的一部分；若改成逐步 TD 模型，必须把已执行时长也纳入该停止问题的递归信息。当前实现直接学习完整执行的后果。

$$
\begin{gathered}\widehat R_o^+(s)=\widehat R_o(s)+\xi(R-\widehat R_o(s)),\quad \widehat D_o^+(s)=\widehat D_o(s)+\xi(\tau-\widehat D_o(s)),\\\widehat P_o^+(s,j)=\widehat P_o(s,j)+\xi\bigl(\mathbf1[s'=j]-\widehat P_o(s,j)\bigr).\end{gathered}
$$

OptionModel.update 同时拟合这三个完整样本标签，相当于对应终止 GVF 的 Monte Carlo 更新。首次样本取 ξ=1，建立概率和为 1 的模型行；以后近期模型用 0.2，累计模型在第 n 次样本用 1/n。模型预测的时长始终按原子步计。

为什么三个均值已经够本页 planner 使用？在一次备份期间固定奖励率 g 和后继值 $V(s')=\max_{o'}Q(s',o')$，则线性期望给出 $\mathbb E[R-g\tau+V(S_{t+\tau})\mid s,o]=R_o(s)-gD_o(s)+\sum_jP_o(s,j)V(j)$。奖励、时长和终点可以相关，这个分解不要求它们独立；它依赖的是备份对三项分别线性。若要预测风险分布，或采用折扣后的终点价值，则需重新规定模型输出，不能沿用三个普通均值就认为信息仍然足够。

$$
Q_*(s,o)=R_o(s)-g_*D_o(s)+\sum_jP_o(s,j)\max_{o'\in\mathcal O(j)}Q_*(j,o'),\qquad\delta=R-\bar g\tau+\max_{o'\in\mathcal O(s')}Q(s',o')-Q(s,o).
$$

第一式描述固定技能集合、最优奖励率与起点无关时的 SMDP 差分最优方程；第二式才使用学习器当前 Q、ḡ 和本次执行样本。奖励率单位是奖励/原子步，随机时长不能省略。差分价值的共同加常数不改变方程或动作排序。

实际调用 Agent.complete 时，先用旧 Q、g 计算这一次真实执行的 δ，再写 Q←Q+0.1δ、g←g+0.002δ；随后更新三个模型量，执行四次模型备份。每次 OptionModel.target 读取已经更新的 g 和当前 Q，备份只写一个 Q 格。真实执行与模型计算共享后继价值，却有不同的后果来源；人工子任务奖金只在 subtask_update 内出现。

这里用本次真实时长 τ 扣除奖励率，且不按时长缩放步长；[平均奖励章](average.md#lesson-duration)的期望时长归一化更新是另一条算法。Wan、Naik 与 Sutton（2021）的式 (3–5) 讨论了这里这类直接扩展，式 (6–9) 才引入期望时长估计与归一化。固定技能下的理论不覆盖本例的常步长、变化技能闭环。模型的近期或累计记忆也需在噪声与适应之间比较。技能时长改变还会改变每个原子步的规划量，所以同时报告真实步数、模型备份数和子任务更新数。

<a id="experiment-integrated_recent_model"></a>

### 实验：预测有了实际消费者：完成一个技能后发生什么

子任务策略、GVF式后果预测和规划怎样接成真实闭环？预测准确度与整段生命期收益应怎样分开观察？

**环境与可用信息。** 七状态环，观测就是当前状态0—6。原始动作左移或右移；0.1概率停在原位，否则向选定方向移动。每步外部奖励为“到达当前奖励位置的指示量−0.02”。前600步奖励位置为1，随后为4；变化时刻不传给智能体。环境不终止、不重置。子目标1和4由设计者给定；技能在到达相应目标或执行满6个原始步时停止。

**设置。** 5个种子0—4，各1200个真实环境步。高层ε=0.2；技能内ε=0.1。每个原始转移用Q-learning更新两个给定子任务，步长0.25、折扣0.95；到达子目标得1，否则−0.01。高层差分SMDP步长0.1，奖励率增量为0.002倍TD误差。技能完成后才提交内部贪心策略；预算尾端尚未完成的技能不伪造终止。 模型对完整动作的外部奖励、时长与终点，首样本直接写入，随后用固定步长0.2更新；每次完成后随机选已建模的状态—动作做4次规划。对照保留子任务与模型学习，只关闭规划。技能策略改变后，两者都清除相应模型及高层Q。

**检验的机制。** 每个原始步更新给定子任务；完成动作后先用实际执行的旧技能语义更新高层与后果模型，再规划，最后提交新技能。奖励、时钟和终点是三个不同预测对象；规划真正消费它们，而非把辅助损失接上网络就宣称有用知识。

**测量。** 主图是最近100个真实步的外部奖励。配套生命期奖励率避免只看恢复末端，模型备份数展示额外计算，版本失效数展示接口维护。

```bash
python3 implementations/integrated_agents/integrated_recent_model.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](../assets/crl-figures/result-integrated_recent_model.svg)

横轴：environment_steps。纵轴：最近100个真实步的平均外部奖励。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 七状态环中，实际行为最近 min(step,100) 个原始转移的平均外部奖励。每步奖励为到达当前奖励位置的指示量减0.02。该窗口包括探索、技能执行和尚未完成的技能；不是冻结贪心策略的价值，也不是算法内部的 estimated_rate。

**step：怎样计时。** step 只数真实转移，奖励位置在预算中点改变，不重置世界或学习器。高层真实备份只在一个原子动作或 option 完成后进行；近期模型与累计模型每次完成各做4次模型备份，故 model_backups=4×real_backups。不规划的对照仍学习模型，但 model_backups 始终为0。每个真实步还更新两个给定子任务，subtask_backups=2×step。相同真实步数未必包含相同规划计算。

**怎样汇总。** 窗口每步滑动，但只在第1步、每20步和预算末尾记录。变化后的前99步仍可含旧阶段奖励。average_reward 是从第1步累计的实际奖励率；estimated_rate 则由每次完成后的差分误差递增，二者的定义不同。技能策略改变时，版本化方法清除对应模型和高层 Q，不清空奖励窗口。取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** lifetime_reward 与 average_reward×step 可互查，并可用累计量差恢复检查点之间的总奖励。step≥120的20步检查点可用相隔100步的累计量重算 value；其余稀疏窗口不一定能重建。预算末尾 pending_duration>0 时，其奖励已计入真实收益，但这段尚未产生高层备份或完整段模型样本。model_reward_mae 只平均最近至多100个“此前已有对应模型”的完整段奖励预测误差，既不是全状态误差，也没有检验时长和终点分布；模型清除会改变哪些样本进入它。

计算位置：[integrated_agents/_system.py](../implementations/integrated_agents/_system.py) · [integrated_agents/integrated_recent_model.py](../implementations/integrated_agents/integrated_recent_model.py) · [integrated_agents/integrated_cumulative_model.py](../implementations/integrated_agents/integrated_cumulative_model.py) · [integrated_agents/integrated_no_planning.py](../implementations/integrated_agents/integrated_no_planning.py)

</details>

**结果分析。** 最终规划版末窗奖励0.328，无规划为0.218；生命期奖励率分别0.2698与0.1957。规划版平均额外做3580次模型备份。第600步无规划末窗还更高，说明不能用单一末端改善概括整段训练。

**结论边界。** 闭环是可运行教学原型，不是完整STOMP或OaK。状态和目标手工给定，选项停止固定；没有自主状态构建、目标发现、算力调度学习或长期安全证明。

**继续实验。** 跟踪一次两步技能：记录原始奖励、人工子任务奖励、时长、模型目标及规划目标。检查子任务奖金没有混入外部奖励模型，未完成技能没有因为绘图结束而强行完成。

[源码](../implementations/integrated_agents/integrated_recent_model.py) · [逐种子记录](https://yingwen.io/crl-code/results/integrated_recent_model/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/integrated_recent_model/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/integrated_recent_model/curves.json)

<a id="frontier-architecture-stomp-contract"></a>

## STOMP 的下一步：技能变了，模型究竟失效在哪里

深入阅读：回到 STOMP 原文，比较已有理论、完整学习链和本章教学原型。

Reward-Respecting Subtasks（Artificial Intelligence，2023）把子任务、option、模型和规划放在一条学习链上。它保留真实沿途奖励，用停止价值表达子任务偏好。研究价值不止在“用了层次结构”，而在于每个对象都有对应的后果查询；该论文的实验并不等于完整非线性持续架构已经建立。

$$
r_o(s)=\mathbb E_o\!\left[\sum_{j=0}^{\tau-1}\gamma^jR_{t+j+1}\mid s\right],\qquad M_o(s,s')=\mathbb E_o[\gamma^\tau\mathbf1\{S_{t+\tau}=s'\}\mid s],\qquad T_oV=r_o+M_oV
$$

这里固定 option 和环境，采用真实外部奖励的后果模型。M 是联合编码终点和持续时间的折扣质量，不是归一化后的终点概率。子任务的停止 bonus 不应再被计入这个真实奖励模型。

$$
|\widehat T_oV(s)-T_oV(s)|\le |\widehat r_o(s)-r_o(s)|+\|\widehat M_o(s,:)-M_o(s,:)\|_1\|V\|_\infty
$$

这是直接由三角不等式得到的单次规划误差界，不是原论文的新的收敛定理。它说明只报告终点分类准确率不够：持续时间、奖励尺度与被查询的 V 都影响 backup。

技能策略或停止函数更新后，即使外部世界完全不变，右侧模型真值也改变。表示更新后，同一输入向量还可能不再代表同一情形。一个持续架构需要决定重学什么、哪些规划查询暂时不可靠、以及将有限真实交互分给哪种校准。模型低误差、技能高成功率和整体回报是三种不同指标。

| 最小结构消融 | 保持不变 | 定位的困难 |
| --- | --- | --- |
| 冻结技能，学习模型 | 环境和表示 | 模型估计本身是否正确 |
| 只改技能策略或停止 | 环境和表示 | 同名 option 的后果漂移 |
| 只改表示 | 原始轨迹与技能 | 模型坐标和旧知识是否兼容 |
| 最后打开规划 | 同样真实交互预算 | 模型查询是否实际改善控制，而非只增加计算 |

原文没有给出已确认的作者公开工程。第三方 ramos-ai/STOMP 自称复现且披露部分数值差异，不能把“曲线形状相似”当正确性证明。教材的闭环教学实现用于验证接口、时间和预算；要声称复现 STOMP，仍需对齐子任务停止值约定、原模型学习算法和规划实验。

<a id="lesson-version-contract"></a>

## 技能一旦改变，哪些知识也会失效

深入阅读：用同名技能改变路线的算例，检查哪些旧后果和高层价值随之失效。

同名技能不一定始终是同一策略。昨天它向左到目标，今天学会向右；奖励、时长与终点模型可能随之改变。旧高层 Q 也可能还在评价昨天的技能。模型误差不只是数据太少，还可能是预测对象已经改变。

![同一技能在同一网格里，内部策略从右右改成下右右，终点从G变F，累计奖励从3变负3，时长从2变3；旧模型的平均奖励备份5与新行为对应的负6分离。](../assets/crl-figures/concept-credit3-skill-version.svg)

蓝线是技能实际走的路线，圆环是本次停止位置。每步 −1，到 G 额外得 5；G、F 都令 option 停止，世界仍继续。底部固定规划器当前的奖励率估计与后继差分值，只改变技能版本。紫色虚线表示沿用旧模型，蓝色实线表示新策略的真实后果。原创计算，不把给定的规划快照称作最优解。

这里另用一个两行三列的确定性网格。两个版本都只从左上角启动，停止规则也都固定为“到 G 或 F 即停止”。版本 1 连续向右，两步到 G，外部奖励是 $(-1,4)$。版本 2 先向下，再连续向右，三步到 F，奖励是 $(-1,-1,-1)$。改变的是内部策略；世界、奖励和停止规则都没有变化。

$$
\widehat R^{(1)}-\bar g\widehat\tau^{(1)}+h(G)=3-1\times2+4=5,\qquad R^{(2)}-\bar g\tau^{(2)}+h(F)=-3-1\times3+0=-6.
$$

暂固定 $\bar g=1,h(G)=4,h(F)=0$，检查同一次平均奖励模型备份。继续读取版本 1 的模型会高估 11：其中累计奖励差 6，时间扣除差 1，后继值差 4。这个局部差异无需环境非平稳，也无需估计噪声。

若世界未变且状态表示未变，原子动作的后果模型仍然有效；失效的是由新内部策略组合出的技能模型，以及依赖旧技能后果的高层估计。完全清空所有知识和完全保留所有知识都忽略了这个依赖结构。可以重学该技能模型，也可以在可靠的原子模型上重新推演；两者都需要把模型标记为当前策略版本后再供规划读取。

本例在提交内部策略时比较新旧贪心动作。若策略改变，则使该技能的所有模型与高层 Q 失效。原子动作模型保持。它是保守、可检查的接口规则，代价是丢弃旧知识。渐进迁移、限制策略变化和联合模型适应是其他路线，不能仅凭复用一个数组声称已经完成迁移。

**算法：真实时间、学习时间与策略版本分开记录**

1. 真实原子步：
  1. 执行已提交的技能策略，环境只推进一次
  1. 用真实转移更新子任务 Q
  1. 累加技能的外部奖励和真实时长
1. 技能终止时：
  1. 更新主任务 Q、奖励率、奖励/时长/终点预测
  1. 用同版本的模型执行有限次规划
  1. 提交新的技能策略
  1. 失效受策略变化影响的模型与高层 Q
  1. 选择下一个高层动作

运行预算用尽时，尚未完成的技能只是被观测截断。其奖励仍计入生命期收益，但不能伪装成完整终止样本写入技能模型。技能自身的六步时限则是事先定义的终止条件。二者都叫“结束”会掩盖一个实质性的目标错误。

<a id="experiment-integrated_stale_options"></a>

### 实验：同一个技能名字，可能已经不是同一个模型查询

内部策略改变后，只保留过去的模型与高层价值会怎样？为什么参数版本是语义问题，而不只是工程日志？

**环境与可用信息。** 七状态环，观测就是当前状态0—6。原始动作左移或右移；0.1概率停在原位，否则向选定方向移动。每步外部奖励为“到达当前奖励位置的指示量−0.02”。前600步奖励位置为1，随后为4；变化时刻不传给智能体。环境不终止、不重置。子目标1和4由设计者给定；技能在到达相应目标或执行满6个原始步时停止。

**设置。** 5个种子0—4，各1200个真实环境步。高层ε=0.2；技能内ε=0.1。每个原始转移用Q-learning更新两个给定子任务，步长0.25、折扣0.95；到达子目标得1，否则−0.01。高层差分SMDP步长0.1，奖励率增量为0.002倍TD误差。技能完成后才提交内部贪心策略；预算尾端尚未完成的技能不伪造终止。 两者均用近期后果模型与每次完成后4次规划。诊断对照故意关闭版本失效：内部策略改变仍沿用旧模型与旧高层Q。参照在策略贪心动作发生变化时清除相应模型格及整个该技能Q列；目标名称保持不变。

**检验的机制。** 同名option的新策略有不同的奖励、时长与终点分布。旧经验不是新技能的同分布样本；高层Q也曾评价旧行为。清除模型而不处理高层价值仍不能完全修复这种语义错位，但清除本身会产生重学成本。

**测量。** 比较奖励曲线、生命期奖励率和失效次数。失效数为0不代表系统更稳定，在此只是开关关闭；也不能将单次最终回报相近解释成接口一致。

```bash
python3 implementations/integrated_agents/integrated_stale_options.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/integrated_stale_options/curves.svg)

横轴：environment_steps。纵轴：最近100个真实步的平均外部奖励。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 最终沿用旧模型的末窗奖励0.320，版本维护参照0.328，差异很小；生命期奖励率为0.2318与0.2698。参照平均触发18.6次失效。有限结果提示暂态差别，却没有证明完全清空是最优维护策略。

**结论边界。** 这是故意违规接口的诊断，不是推荐算法。闭环行为及模型备份数随之变化；缺少等数据、等计算下的部分修正、重要性校正或模型版本混合对照。

**继续实验。** 先冻结外部奖励与所有表示，只改变一个技能的一个状态动作。列出哪些模型格和Q项的定义变了；再比较清空、仅降权旧样本与单独保存版本的方式，并计入内存和重学开销。

[源码](../implementations/integrated_agents/integrated_stale_options.py) · [逐种子记录](https://yingwen.io/crl-code/results/integrated_stale_options/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/integrated_stale_options/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/integrated_stale_options/curves.json)

<a id="lesson-drift"></a>

## 表示漂移与模块一致性

深入阅读：隔离共享特征变化，计算它对预测、控制与模型读出的连带影响。

先考虑只有一个特征的例子。对同一个观测 $x=1$，编码器输出 $z=\phi x$，初始 $\phi=1$。一个预测头输出 $2z$，两个控制值分别为 $Q_L=z$、$Q_R=1.5$，模型预测物理后继观测 $\hat x'=0.5z$。假定真实预测目标为 2，两动作的真实回报分别为 1、1.5，真实后继观测为 0.5。因此三个读出开始时都正确。

![同一条纹观测经共享编码器得到的特征从1变为2，未更新的预测、控制和模型读出随之改变；对这个已知的纯缩放，同步缩放读出权重可以恢复答案。](../assets/crl-figures/concept-depth-shared-feature-drift.svg)

第一幅只更新编码器。第二幅蓝点是原答案，绿点是新答案；控制中的虚线是未变的右动作值 1.5，左动作值越过它后会改变贪心动作。模型输出的是物理观测，不是新的隐空间坐标。第三幅将三个输入权重各减半，展示纯缩放的精确补偿。原创算例，世界与读出头的学习目标保持不变。

$$
L_{\rm aux}(\phi)=\tfrac12(\phi x-2)^2,\quad\nabla_\phi L_{\rm aux}=(\phi x-2)x=-1,\quad\phi^+=\phi-1\times(-1)=2.
$$

假设一个新辅助任务希望该观测的编码为 2，本次只用它更新编码器。这里只展开一条明确的梯度路径，没有同时更新任何读出头。

更新后预测从 2 变为 4，模型预测从 0.5 变为 1。左动作估计从 1 变为 2，超过保持 1.5 的右动作估计，于是贪心选择也改变了。世界没有改变，三个读出目标没有改变，各头的参数也没有改变；错误由共享输入的变化产生。辅助损失降到了零，却不能代表整个智能体的知识更准确。

这个例子恰好有已知的可逆坐标变换 $z^+=2z$。若把三个输入权重同时除以 2，读出恢复为预测 2、控制值 $(1,1.5)$ 和物理后继 0.5。一般的非线性表示变化未必保留信息，也未必知道逆变换，所以不能照搬这一步补偿。本例模型的输出是物理观测，输入补偿已经够用；若输出改为隐空间后继，同一个物理后果在新表示下的编码也会改变，还必须变换模型输出。学习新特征时，因而要同时考察它区分历史的能力，以及已有预测、策略和模型能否继续正确地使用它。

| 设计选择 | 怎样保持接口 | 代价或未解决点 |
| --- | --- | --- |
| 固定状态编码 | 整个实验中保持坐标语义 | 无法研究自动 state construction |
| 保存原始经验重编码 | 新表示版本上重算状态和训练模型 | 额外存储/计算，严格 streaming 下可能不允许 |
| 慢表示 / 快预测器 | 限制表示变化，让依赖者追踪 | 时间尺度需要调节，仍无任意漂移保证 |
| 显式版本与兼容检查 | 模型记录表示版本，失配时作废或迁移 | 作废丢知识，迁移本身也是学习问题 |
| 联合一致性训练 | 共享或约束 representation/model/value | 损失冲突、梯度泄漏和新优化问题 |

处理网络单元回收也类似：如果一个隐藏特征被替换，依赖它的预测头、模型输入输出、eligibility trace 以及元梯度敏感性是否仍有旧语义？应明确哪些状态清零、哪些变换、哪些保留。把一个塑性机制接到架构里，影响不只发生在该层权重。

<a id="course-architecture-coordinate-maintenance"></a>

## 更换一个特征，为什么必须维护整条依赖链？

深入阅读：展开表示坐标改变后，各个读出需要怎样保持含义。

特征的生成、检验与淘汰把固定网络变成了持续构造过程。OaK 将特征、子问题、options、模型和规划连接起来，但并未使它们的参数自动保持一致。先研究一个最有利的情况：新特征只是旧特征的可逆线性换坐标。连这种不丢信息的变化都需要维护多个对象。真正生成或删除特征只会更困难。

$$
\widetilde x=Ax,\qquad \widetilde w=A^{-\top}w,\qquad \widetilde F=AFA^{-1},\qquad \widetilde b=A^{-\top}b.
$$

A 固定且可逆；原价值为 $w^\top x$，模型为 $Fx$ 与 $b^\top x$。逐项代入可验证新坐标下价值、奖励预测和下一特征预测描述相同对象。仅改 encoder 而保留旧 w、F、b，则不再是同一个预测系统。

模型输出也要变换，因为它预测的是特征。递归状态更新相应变为 $\widetilde f(\widetilde x,u)=Af(A^{-1}\widetilde x,u)$。若线性资格迹满足 $e_t=\Gamma_t\lambda e_{t-1}+x_t$，对整个历史一致换坐标后应有 $\widetilde e_t=Ae_t$。旧 trace 的数组槽位并不自带语义。

$$
\Delta\widetilde w_{\rm same}=A^{-\top}\alpha\delta e,\qquad \Delta\widetilde w_{\rm plain}=\alpha\delta\widetilde e=\alpha\delta Ae.
$$

保持旧预测只解决了当前功能等价。若仍使用相同标量步长，下一次普通梯度更新通常不等价。需要相应的预条件矩阵，或接受更新几何改变；只有特定坐标变化可保留原更新。

一维手算：令新特征为旧特征的两倍，读出权重减半，当前价值完全不变。但若仍用相同步长做 TD(0)，权重更新是原来的两倍，对价值造成的改变是原来的四倍。把步长改为原来的四分之一，才恢复这项线性更新的等价性。这也是“预测没变，所以优化器不用改”不成立的最小反例。

新生成的特征通常不是可逆换坐标。此时没有一个 A 能完整迁移所有旧知识。删除一维可能使与它有关的子目标无法定义，也可能让一个旧 option 的停止检测失去依据。重置依赖对象是一种明确但有损的处理；保留、迁移与重新校准则需要新的证据，不能默认为无成本。

| 发生变化的对象 | 必须追踪的依赖 | 保守处理的代价 |
| --- | --- | --- |
| 状态特征坐标 | 读出头、递归状态、模型输入与输出、trace | 重新适应期间的预测损失 |
| 特征定义被删除 | 引用它的 cumulant、终点奖金与终止规则 | 相关问题或技能可能必须退役 |
| 内部策略或停止规则 | option 的奖励/终点/时长模型、主任务估值 | 旧执行数据不再直接对应新对象 |
| 参数槽位被复用 | 梯度迹、动量、二阶矩、元梯度敏感度 | 清零避免旧语义泄漏，但丢失统计经验 |

输出权重大小也不是普适的特征效用。把特征乘 100、读出权重除 100，不改变任何预测，却会使基于权重幅度的排序改变。课程的 Online Representation Search 在特定 LTU 表示与在线监督设定中研究生成与测试；推广到可变尺度的神经表示时，须明确归一化、特征频率、成熟期和任务相关性。

**算法：资源有界的对象维护规程；不是完整 OaK 算法或性能保证**

1. 为每个特征和技能分配稳定标识，而不是仅使用数组位置。
1. 提交一次替换前，列出受影响的预测、子问题、模型、trace 与优化器状态。
1. 若是可验证的可逆坐标变换：迁移对象，并检查迁移前后预测。
1. 若是新语义：重置或标记受影响对象，禁止把旧模型当作已校准模型。
1. 保留合法的原始动作接口，执行下一真实步。
1. 记录替换、重新学习及计算开销；所有开销计入同一生命期预算。

检验应分两阶段。先在无新信息的纯换坐标实验中验证功能等价与更新等价；再真正替换特征，衡量学习恢复、被删除知识的损失和下游模型错误。只证明新特征能降低一个监督损失，还不足以证明整个持续智能体受益。

<a id="rlss-consumer-credit"></a>

## 谁在使用这个抽象：从梯度信用到知识的保留与改变

深入阅读：沿知识的使用者追踪反馈，区分预测误差和下游用途。

一个特征可以被多个价值预测器使用，也可以定义某个技能的到达目标。技能又被后果模型和规划器使用。正向信息流因此形成依赖链。OaK 提出的反向消费者信用询问：哪些下游模块依赖这个对象，愿意为它的保留和稳定性付出多少资源？它与一次 TD 误差沿资格迹分配参数梯度，不是同一个问题。

| 对象 | 消费者 | 消费者可能需要的稳定性 |
| --- | --- | --- |
| 状态特征 | 奖励价值、辅助预测、子问题定义 | 输入含义不突然改变；必要变化有迁移或重学安排 |
| option | 执行策略、后果预测器 | 内部策略和停止规则改变后，旧后果不能继续冒充当前模型 |
| 后果模型 | 规划器与行为选择 | 常用查询的误差、时长语义和有效范围可追踪 |

“有人使用”只是保留对象的一个理由，不是永久保护权。下游可能依赖了错误预测，也可能存在更便宜的替代。反之，当前没有被使用的技能，可能只是在访问分布中暂时没有机会。消费者信用必须在即时依赖、未来用途、维护成本与改变风险之间作出选择。

$$
U_{i\to j}^{\rm probe}=\mathbb E_{x\sim d_{\rm probe}}\!\left[
 L_j^{\setminus i}(x)-L_j(x)\right].
$$

一种可检查的诊断定义：固定其他参数与探测分布，移除对象 i 后观察消费者 j 的损失变化。正值表示这次删除使 j 变差。它测当前依赖，不是 OaK 已确定的效用算法，也不估计未来重新学习后的最优用途。

设目标为 x，两个完全相同的特征都等于 x，输出权重均为 1/2。原平方误差为零。固定其余权重，删掉一个特征后，误差变为 $x^2/4$；但若允许剩下的权重重新学到 1，误差又可为零。相同删除动作，在“立即不动其他参数”和“允许重新适应”两个问题下有不同代价。

不同消费者的损失也未必同量纲。概率预测、物理量预测与主奖励价值的误差不能直接相加。需要规定比较尺度、优先级与资源预算，并检验这些局部指标是否改善主任务或未来学习。把各头损失都降低，仍不证明其预测值得长期维护。

一个有用的实验把“能否删除”与“怎样迁移”分开。先冻结消费者，测即时功能扰动；再给相同再学习预算，测恢复成本；最后放入自然经验流，测长期收益和遗忘。探测数据不应偷偷进入训练。对于罕见但重要的能力，还要明确探测机会和保存探测记录的成本。

可塑性机制负责让候选有机会进入。消费者反馈则约束它们何时可以改变或退出。前者单独使用可能不断破坏依赖，后者单独使用可能把现有表示锁死。成熟期、有限保护预算、依赖版本和逐步迁移都是可研究的机制；没有一个局部启发式自动保证整条发现循环改善长期回报。

<a id="lesson-learning-continuity"></a>

## 同一套知识怎样持续学习

深入阅读：进一步连接问题发现、流式资源、知识保留和特征替换的相互作用。

预测、技能和模型都有了定义以后，仍有四个不同的问题。新经验到来时能做多少计算？怎样改进学习规则？哪些旧知识还值得保留？当前表示是否还学得动新问题？这四个问题分别通向流式学习、元学习、知识保留与学习可塑性。它们作用于前面同一套知识，不是另外四个互不相干的模块。

先沿一次交互看更新顺序。保存当前策略产生动作时的概率，收到真实后果后，用更新前的网络计算各预测误差及梯度。每个 GVF 按自己的累积信号与延续规则更新；当前技能累计外部奖励、折扣与原始时长。规划读取相应技能的后果模型，改变下一次选择。[流式学习](streaming.md#lesson-components)进一步要求，这些计算不重新读取旧转移，而且每步耗时与持久内存受限。资格迹保留过去方向，模型计算复用已学后果；二者都要占用这份预算。

如果要自动决定预测什么、保留哪些技能，更新顺序中还需加入对用途的评价。[GVF 发现](gvf.md#gvf-discovery)可以让下游损失评价候选问句；[技能发现](options.md#lesson-branches)可以用主任务收益、图结构或可区分后果产生候选行为。元梯度则追踪这些选择经过后续学习产生的影响。预测误差小、技能容易区分、元损失下降分别回答局部问题，完整智能体仍需以外部收益和取得这些收益的成本评价。

最后考虑旧知识的去留。改变 GVF 的问句或 option 的内部策略，可能改变答案本身；只改变表示，则可能在问题不变时破坏旧答案。[知识保留](retention.md#lesson-agent-memory)要区分两者。[特征替换](plasticity.md#lesson-consumers)还会影响所有读出头、资格迹和元敏感度：清空新坐标上的旧状态可以避免错误继承，但也会丢失尚未兑现的信用。没有重放权限时，恢复只能依靠之后获得的经验。整体设计的困难正在于这些相互影响；一个局部更新有利，并不足以推出整条学习过程更好。

<a id="lesson-integrated-code"></a>

## 独立实现、对照与实际结果

深入阅读：读取已有多种子实验与模型记忆作业，区分过程核验和长期收益比较。

| 实现入口 | 改变的唯一因素或模块 | 必须一起检查 |
| --- | --- | --- |
| [近期模型闭环](https://yingwen.io/zh/continual-rl/code/integrated_recent_model/) | 可运行的完整有限原型 | 真实收益、内部率估计、模型成本 |
| [无规划对照](https://yingwen.io/zh/continual-rl/code/integrated_no_planning/) | 只移除模型备份 | 仍然学习模型，不额外增加环境步 |
| [累计模型对照](https://yingwen.io/zh/continual-rl/code/integrated_cumulative_model/) | 只改变模型旧数据权重 | 稳态噪声与改变后的延迟 |
| [失效接口反例](https://yingwen.io/zh/continual-rl/code/integrated_stale_options/) | 故意沿用旧技能模型和 Q | 不是推荐算法；偶然高回报不证明接口正确 |
| [仅原子动作](https://yingwen.io/zh/continual-rl/code/integrated_primitive_only/) | 移除整个技能链 | 行动集合与计算成本同时变化，不能归因于单一更新项 |

读[完整实现说明](https://yingwen.io/zh/continual-rl/code/guides/integrated-agents/)可逐项核对模型契约、公式到函数的对应关系和测试。主图展示最近 100 个真实步的实际外部奖励。原始记录还包含全程收益、规划数、技能版本失效次数和模型误差。模型误差只在已有模型的后续完整执行上记录，必须同时检查有效样本数。

这些实现已经组成[模型记忆与规划研究作业](https://yingwen.io/zh/continual-rl/code/#study-model-memory-planning)。先不新增训练：从已保存的 1200 步检查点记录重算全程奖励率、变化后奖励率和最后窗口读数，标出各自时间分母。检查点不能还原每一次模型写入；逐步接口另用作业中的固定序列短例，沿外部奖励、时长、模型版本和备份追到下一行动。已有三组、各五个种子的运行也不等于完整四格对照，不能把短例的过程检查并入性能数据。

提交时，用一段已有记录解释两个不同问题：近期模型是否更快改变预测，改变后的预测是否通过规划影响了全程收益。若没有相同数据的隔离诊断，只能报告完整学习器的差异，不能把差异全归给模型记忆。状态、子目标与终止规则都仍是给定条件；神经模型、自动发现与跨模块元学习属于后续独立问题。

<a id="lesson-evaluation"></a>

## 从固定策略评价到持续学习过程

传统实验常在训练结束后冻结策略，再评价若干回合。这个协议回答最终策略有多好，却不能完整描述一个始终学习的智能体。后者的对象包括参数更新规则、记忆、优化器状态与资源分配；相同当前策略可能因为内部学习状态不同而具有不同的未来表现。

$$
H_t=(O_0,A_0,R_1,O_1,\ldots,O_t),\qquad W_{t+1}=U(W_t,A_t,R_{t+1},O_{t+1}),\qquad A_t\sim b(\cdot\mid W_t).
$$

$H_t$ 是完整交互历史，$W_t$ 沿用前文的完整持久状态，包含行动摘要、执行进度、参数及学习状态。环境可由历史条件分布描述；智能体必须用有限 $W_t$ 近似利用它，而不能存储无限历史。这里 U 合并了一步内所有声明的学习与规划计算。

Elelimy、Szepesvari、White 与 Bowling 在 RLC 2025 提出以 history process 和面向持续学习的 deviation regret 重新讨论这一评价对象。其启发是把学习中的变化和偏离行为的后果纳入形式化，而不只比较一个与时间无关的最优策略。这是一条研究立场，并不意味着有限 MDP 或平均奖励理论失效；关键是说明模型表达了环境什么信息，以及评价量是否覆盖学习的代价与收益。

| 评价条件 | 回答的问题 | 实现时的约束 |
| --- | --- | --- |
| 冻结参数的策略评测 | 已学行为在指定分布上有多好 | 不把这条曲线当作完整生命期表现 |
| 持续学习的全程收益 | 适应期间的损失能否被后续收益补偿 | 恢复、探索、失败期间同样计时 |
| 相同数据的学习 probe | 内部学习能力是否改变 | 固定数据顺序、容量与更新预算 |
| 相同实时计算预算 | 更多规划或预测是否值得其延迟 | 报告峰值内存、每步耗时及动作频率 |

AgarCL 将这些耦合问题放在同一环境中：细胞的质量改变移动速度与视觉尺度，部分可观测性要求记忆，吞噬与避敌要求长期控制。这里可以区分两种变化：固定规则下状态相关的动力学仍可构成平稳的完整状态 MDP，而智能体看到的分布与控制尺度会随行为持续改变。仅从观测变化不能推出环境转移核本身随时间变化。

平台的完整游戏没有统一回合重置，但被吞噬的细胞会局部重生，其余世界状态继续存在。动作包含连续方向和分裂等离散选择；DQN 基线将方向离散化，PPO/SAC 使用混合动作接口。论文测试的可塑性方法未在完整游戏中建立持续胜任能力，并用小游戏分离探索、信用分配等困难。因此，一个有用的复现路线是先核对小游戏中的模块机制，再进入完整游戏，保留失败、重生和学习成本。

<a id="frontier-architecture-robot-adaptation"></a>

## 机器人适应的近期证据：预训练、恢复速度与长期保持要分开

深入阅读：以机器人持续适应研究检验本章的知识维护与评价问题。

An Analysis of Streaming Deep Reinforcement Learning for Adaptive Continual Learning in Robotics（2026 预印本）从 PPO 预训练策略出发，比较突变条件下的流式适应。它在部分任务中观察到改进，也报告更复杂操控场景中性能下降与恢复有限。研究协议使用 episodic MDP，主要考察前向适应，不等于无重置单生命期或旧知识保持测试。

$$
u_i\leftarrow\eta u_i+(1-\eta)(1-h_i^2)\sum_k|w_{ik}|
$$

原文针对 tanh 采用这个 contribution utility 变体，将激活幅度替换为导数容量。它改变了被替换单元的选择标准；不能只称“使用 CBP”而遗漏这处差异。h 接近饱和时该因子小，具体效果仍依赖单元与输出权重的联合状态。

| 观察到的现象 | 不足以推出什么 | 需要的额外实验 |
| --- | --- | --- |
| 变化后成功率恢复快 | 旧任务能力被保留 | 变化后返回旧条件；单独测冻结策略表现 |
| 某种归一化更好 | 部署阶段单一组件造成全部提升 | 对齐预训练网络；原文 LN 组的预训练也含 LN |
| 流式比 batch 好 | 更少数据或总计算一定更优 | 同时报告真实交互、额外梯度计算、墙钟及调参预算 |
| 小单元替换有帮助 | 整个智能体无限期保持可塑性 | 更长多次变化，记录替换频率、饱和、保持与安全代价 |

作者公开 stream-rl-robotics 工程，适合沿预训练检查点、突变配置、optimizer、单元替换和冻结评估五条路径阅读。先复现一个变化条件，再做多个连续变化；不要把单次恢复图解释成完整终身智能体证据。教材的小链实验与这个工程作用互补：前者检查公式与接口，后者研究这些机制在更复杂动力学中能否共同工作。

<a id="rlss-learning-development-tests"></a>

## 积累、准备、开放与结构：持续学习的四种不同主张

深入阅读：区分当前表现与持续学习能力，为完整学习器选择测量。

一个智能体始终更新权重，不表示它积累了可复用知识；保留旧知识，也不表示它更善于学习未来任务。OaK 讲义用 accretive、preparatory、open-ended、structural 描述学习的不同作用。把这些词转成不同的实验问题，才能避免由一条回报曲线推出过多结论。

| 作用 | 具体问题 | 怎样检验 | 不能替代它的指标 |
| --- | --- | --- | --- |
| 知识积累 accretive | 过去经验是否形成以后仍可使用的能力？ | 在相同资源内，检验旧能力保持和跨情境复用 | 参数、技能或数据库条目数量增加 |
| 为未来学习做准备 preparatory | 现在的经验是否降低之后获取新能力的成本？ | 新目标出现后允许同等学习，比较达到指定水平的经验与计算，并计入准备成本 | 新目标到来前的即时回报 |
| 开放式构造 open-ended | 可学习的问题和解法能否超出预设有限目录？ | 与固定候选库比较，检验新组合在未知情境中的用途 | 在一个固定菜单上切换更多任务 |
| 结构变化 structural | 经验是否改变知识的组织、连接或模块？ | 允许与禁止结构变化，在匹配资源下比较保持、学习速度与成本 | 仅记录权重改变，或仅展示结构增大 |

例如，智能体先在一个地图上完成若干到达任务。它可能保留门口位置的预测、到达门口的技能以及技能的后果模型。以后目标换到另一房间，旧技能可以直接帮助行动；即使不能立即使用，已有模型也可能加快新路线学习。前者主要检验复用，后者主要检验准备。若门口由实验者标好，则两者都没有证明自主发现子问题。

公平比较需要把准备期算进去。一种方法把全部生命周期累计奖励作为主指标，再把新目标出现后的学习曲线作为解释性指标。另一种方法明确询问给定准备预算能否降低后续样本需求。二者回答的问题不同，不能只截取适应最快的一小段并忽略前期支出。

$$
C_{\rm life}=C_{\rm prepare}+\sum_{j=1}^{M}C_{\rm adapt}^{(j)}+C_{\rm maintain}.
$$

这是成本分解，不是收益定理。各项必须用一致单位计算，或分别报告真实经验、计算与内存；不应把三种资源随意加成一个数字。

有限内存下的积累不要求对象数永久增长。压缩、重组和有选择地遗忘，也可能保留更多有用能力。开放式构造同样不要求无限保留对象；它要求生成和组合规则不只是对固定目录做索引。另一方面，结构不断改变本身也不是进步：错误的删除可能使旧技能失效，过度保护则可能阻碍新学习。

这里的开放性描述可构造的知识与子问题。多智能体章节中的开放式种群学习则关注怎样不断产生新的对手、伙伴和训练目标。二者处在不同层次，也可以在同一系统中结合；不应只凭 open-ended 这个共同名称把它们当成同一个实验问题。

这四个维度可以组合，也可以彼此分离。固定结构的网络可以改善未来学习；动态增添模块的系统也可能只记住新任务而没有迁移。应分别问“保留了什么”“为谁做了准备”“能构造什么新对象”“改变了哪些依赖”，再把答案连接到主任务的长期收益。

<a id="lesson-schedule"></a>

## 真实交互与规划的更新调度

深入阅读：展开真实更新与规划更新的不同频率，以及共享计算预算下的调度问题。

**算法：算法伪代码**

1. 初始化 state memory、Q、ḡ、GVF、model 与固定规划预算 B
1. 每个真实环境步：
  1. 1. 用当前 agent state 计算行为分布；采样动作并保存 b(a|z)
  1. 2. 环境推进一次，返回真实 reward 和 observation
  1. 3. 用明确的表示版本计算 z′，保留本步必要中间量
  1. 4. 用更新前参数同时计算 control/GVF/model targets
  1. 5. 更新各学习器；模型只吸收此次真实 experience
  1. 6. 做恰好 B 次模型 backup；轮转选已知 state-action
  1. 7. 记录真实收益、预测误差、模型误差、更新成本
  1. 8. 若启用表示/技能重构，按版本规则处理依赖模块
  1. 9. 携带 memory 到下一步；不在记录窗口或隐藏变化时 reset

第 4 步的“同时”指语义上共同读取旧快照，不要求并行硬件执行。第 6 步的每次 planning 可以读取前一次 planning 更新后的 Q，这是异步迭代；要区别于第 4 步对真实经验目标的快照约定。代码显式写出这两个边界，避免一边修改 Q 一边意外改变本步 GVF 目标。

备份顺序怎样影响下一次交互？把这个问题缩到一项能完整走完的配送任务。从 S 选近路，一步到 D 得 2；选远路则依次经过 A、B、G，三步奖励为 0、0、6。D、G 是任务终点，送达后由协议把位置重置为 S，不增加奖励或环境步；Q、模型和计数全部保留。因此每趟都从 S 开始。这里采用折扣回合目标，折扣为 0.9，来隔离传播和动作选择；它不是前面[六状态](architectures.md#lesson-code)和[七状态](architectures.md#lesson-integrated-setting)原型的不间断交互协议。

学习器只获得当前状态、合法动作和实际动作后的奖励与后继，初始 Q 全为零、模型为空。先由采集协议指定走近路一趟、远路一趟，共四个真实步，不做模型规划。每条经验都更新当前动作价值，并把刚看见的后果写入模型。取步长 1，则直接更新和模型备份使用同一计算：

$$
Q(s,a)\leftarrow\begin{cases}r,&d=1,\\r+0.9\max_{a'}Q(s',a'),&d=0,\end{cases}\qquad d=\mathbf1[\text{本次到达任务终点}].
$$

终止时尾值为零。直接更新的 r、s′、d 来自环境；模型备份的三项只从此前真实观察建立的表中读取。学习器不接收路线真实回报作为训练标签。

![近路S到D一步奖励2，远路S经A、B到G三步奖励0、0、6；四次真实更新后，S近、S远、A、B的Q为2、0、0、6。](../assets/crl-figures/architecture-walkthrough-experience.svg)

沿蓝色实线按 t=1…4 读两条实际轨迹；D、G 的双圈表示送达终止。下方柱高是四次直接更新后的 Q，每条已见模型边有一次真实观察。初始 Q=0、模型为空，α=1、γ=0.9；重返 S 不计环境步且保留学习器。原创确定性计算，[逐步数据](https://yingwen.io/crl-figures/architecture-walkthrough-data.json)由配套实现生成。

四步后，Q(S,近)=2，Q(S,远)=0，Q(A,前进)=0，Q(B,送达)=6。远路确实拿到了奖励，但收到 6 时直接更新只改 B 的值；先前离开 A、S 时，还没有这个后继估计。现在回到 S，下一动作尚未选定，给规划器恰好两次调用。两种调度都只查询刚才观察过的 S→A 和 A→B，只改变这两次调用的顺序。

$$
\begin{aligned}\text{采样顺序:}\quad&Q(S,\text{远})\leftarrow0+0.9\times0=0,\quad Q(A,\text{前进})\leftarrow0+0.9\times6=5.4;\\\text{反向顺序:}\quad&Q(A,\text{前进})\leftarrow5.4,\quad Q(S,\text{远})\leftarrow0+0.9\times5.4=4.86.\end{aligned}
$$

两次模型调用发生时，真实环境时钟都停在 t=4。每次备份读取前一次备份之后的 Q。模型预测和真实观察次数均不改变。

于是采样顺序下，S 比较 2 与 0，下一步选近路；反向顺序下比较 2 与 4.86，下一步选远路。独立地对两条完整奖励序列求和，可得近路回报 2、远路回报 $0+0.9\times0+0.9^2\times6=4.86$。这项回报计算供读者核对，实际程序仍从四条经验和两次备份产生估计。若再给采样顺序一次 S→A 的备份，它也会得到 4.86；当前动作差异来自行动之前完成了哪些计算。

![两种调度各做两次模型调用，然后执行三个真实动作：采样顺序连续三次近路，反向顺序走完远路；两者窗口奖励都为6，但四条模型边的真实观察次数分别为4、1、1、1与1、2、2、2。](../assets/crl-figures/architecture-walkthrough-closed-loop.svg)

上部紫色虚线沿模型调用时钟 k 前进，真实时钟固定在 t=4；下部橙色实线是随后 t=5、6、7 的实际动作，此时 k 固定为 2。蓝点每个代表一条真实观察。为隔离这次调度的后果，这三个真实步仍学习 Q 和模型，但不再追加规划；任务终止后按协议回到 S。左右使用相同的四条初始经验、相同两次模型调用和相同三个后续真实步。

接着让两个学习器实际行动，新的采样便分开了。采样顺序连续走三次近路，奖励为 (2,2,2)，模型中 S→D 的真实观察次数从 1 增到 4；反向顺序走 S→A→B→G，奖励为 (0,0,6)，三条远路边的次数各从 1 增到 2。这里的后果确定，重复观察不会再改变已经正确的模型点预测，却改变了经验分配。计算调度先改变行为，再通过行为决定模型下一次用什么数据学习。

两个分支在这三个真实步内都得到外部奖励 6，不能把起点折扣价值由 2 变为 4.86 换算成这个窗口的累计收益提升。近路完成了三趟，远路完成一趟；允许无成本重返 S 是这个计数成立的条件。若配送后的返程也消耗时间，必须把返程加入环境再比较。计算预算还需单独记：此处两者都完成七次真实更新、七次模型记录和两次模型备份；这些是操作次数，不是实测墙钟耗时。

这一算例把 Dyna 的循环落实到下一条数据：真实经验同时更新价值与模型，有限的模型计算改变实际动作，新动作决定后续经验。Sutton 与 Barto §8.2 用这个循环组织直接学习与规划，Alberta Plan 的 Step 9 进一步把“选择哪些状态、按什么顺序更新”作为 search control 问题。这里固定两个已知模型行，只检查一次调度窗口；自动选择模型查询、表示和技能仍需各自的方法与证据。

下载[单文件 Python 算例](../tutorials/architecture-walkthrough.py)后，在仓库根目录运行下方命令；也可在任意目录使用这个文件的完整路径。只用 Python 3.10+ 标准库，不需要仓库、网络或训练数据。JSON 中 warmup 是实际采集，planning 是两次模型调用，follow 是学习后的实际行动；每条记录保留 t、k、Q 与真实观察次数。先遮住 follow，预测两个学习器会采到什么，再查看输出。

确定性有限算例；Python 使用 Fraction 有理数完成计算，输出十进制 JSON。

```sh
python3 tutorials/architecture-walkthrough.py
```

如果 GVF 数量 K、技能数 O 或网络规模持续增长，每步成本也可能增长。一个终生系统需要给新对象分配预算并淘汰低价值对象：例如只更新部分预测问题、限制模型缓存、分配固定 planning calls。自动增长不等于固定预算下的持续学习。

<a id="research-openmind-physical-time"></a>

## 身体不会等待更新：用真实平台检验架构调度

深入阅读：把原始交互步换成物理时间时，检查行为与计算的计时关系。

把一次梯度更新插入控制循环以后，机器人仍在运动，摄像头仍有延迟，电机也可能继续发热。架构因此不只是一张模块图，还必须说明动作何时发出、学习何时执行、观测对应哪个时刻，以及反射保护能否打断已选动作。更多预测或规划若拖慢响应，就改变了智能体真正面对的控制问题。

Physical Atari（RLC 2026，Javed、Modayil 等）用机器人驱动真实手柄，以摄像头读取持续运行的 Atari 画面及奖励标记。论文采用先发动作、再执行学习的 reactive 调度，并报告约 165 ms 的平台响应延迟；该数值尚未包含神经网络选动作的时间。其学习器保留目标网络与经验回放，所以“真实时间在线学习”与“无回放流式学习”必须分别说明。

它在六个游戏的重复运行中累计约 145 小时，无需人工干预；同样制造的另一个机器人身体却会降低已学策略表现。这支持把个体硬件差异纳入部署适应研究，也支持平台的可靠性主张。它仍是受控的机械操作任务，并不由此证明任意机器人能无限运行、自动修复或长期保持可塑性。

Open Ant（RLC 2026，Lupu、Spieler、Javed、De Asis、Martin、Steenstrup、Modayil）把问题移到行走身体：在有限场地内，根据越界条件改变运动奖励方向，使机器人来回行走。奖励取位移在目标方向的投影，减少走到场地边缘就必须回合重置的需求。控制命令、观测与 MuJoCo 配套模拟共同构成可修改的研究接口。

$$
R_t=(p_t-p_{t-1})^\top u_t,\qquad \widehat r_t=\frac1N\sum_{k=t-N+1}^{t}\frac{R_k}{\Delta t}
$$

p 是平面位置，u 是当时奖励方向，固定步长下的后式衡量单位物理时间的进展。原论文的 24 维观测包含目标方向相对身体朝向的二维单位向量；目标不是完全隐藏的。复现须保留这一权限，并另行说明是否增加了切换标记、世界坐标等信息。

Open Ant 报告 SARSA(λ) 与 SAC 可在约一小时内从身体经验学会行走，但两者的动作抽象、更新与回放协议不同。主实验各做五次 80 分钟试验；电源与通信缆线缠绕时仍由人暂停、解缠并放回起始配置。仿真中策略的性能排序还会在真机上改变。因此，非回合任务设计、少干预的工程能力与完全无人干预的持续学习是需要分别验证的主张。

对本章的集成架构，可提出一个尚待实施的检验：固定同一身体与真实时间预算，逐渐增加 GVF 数量或规划次数，同时记录观测至动作的延迟分布、遗漏更新、奖励率、人工干预与保护反射。再比较同步更新和固定截止时间的调度。若收益只在暂停世界的模拟器中增加，而在同一小时真机经验中降低，新增模块的计算代价已经改变了结论。

思考：保护反射替换了高层选定的动作以后，离策略学习器应记录提议动作还是实际执行动作？若机械修复需要二十分钟，生命期评价是否计时？这些约定应在实验前写明，因为它们决定了知识、行为与真实经验之间是否保持一致。

<a id="lesson-branches"></a>

## 模块假设与研究路线

深入阅读：在已明确的模块关系上，比较进一步的架构研究取向。

| 假设 | 只改哪一处 | 必需的机制检查 |
| --- | --- | --- |
| 更好的 state 减少混叠 | memory / representation | 同观测不同历史的预测与动作是否可区分 |
| GVF 作为知识改善控制 | 共享特征或决策输入 | 固定控制预算，比较真实预测与随机辅助任务 |
| 新技能节省规划深度 | option 和 model 接口 | 模型奖励/时长是否为真实外部后果 |
| 规划帮助适应 | 模型与 planning budget | 模型误差、旧数据权重、每步真实计算成本 |
| Meta-learning 改善未来学习 | 步长/目标/更新规则 | 是否穿过未来数据，是否多用元训练分布 |
| 替换机制维持长期能力 | 特征或技能的 generate-and-test | 被替换对象的全部依赖状态是否一致处理 |

Alberta Plan 与 OaK 将状态、预测、控制、规划、时间抽象和元学习组织为相互依赖的研究方向。它们是理解长期目标的研究纲领；具体实验仍需给出可执行的子系统、接口与预算。完整整合及统一评估属于持续推进的开放问题。

“架构 A 比 B 好”很容易混入更多计算、更多先验、更多模型查询或更丰富目标。更有解释力的问题是：“在相同真实步数、固定模型容量和固定每步更新预算下，自动 option model 比原子模型减少多少规划误差或适应延迟？”先给出可测量接口，才有可积累的结论。

<a id="rlss-alberta-roadmap"></a>

## Alberta Plan 的研究顺序与组成部分之间的依赖

深入阅读：将本章的部件关系放回 Alberta Plan 的长期研究路线。

课程借 Alberta Plan 说明为什么要返回基础算法。固定特征下的持续监督学习，已经需要同时处理噪声、追踪和未知步长；在此基础上才逐渐加入预测、控制和模型。后面的完整智能体也会反过来揭示前面模块遗漏的条件，因此路线不是必须逐级通关的单链。

| 原计划阶段 | 提出的问题 | 在本书中的落点 |
| --- | --- | --- |
| 1–2：给定特征的持续监督学习、监督特征发现 | 怎样持续学习；哪些特征值得保留 | 函数逼近、元步长、可塑性与表征搜索 |
| 3–6：持续 GVF、actor–critic、平均奖励 GVF、持续控制 | 单一流怎样支持多个预测和行为改进 | 预测知识、平均奖励与完整学习器评价 |
| 7–9：平均奖励规划、单步模型的持续逼近原型、搜索控制与探索 | 模型如何支持规划；计算和经验投向何处 | 模型规划、规划预算与探索 |
| 10–12：STOMP、OaK、智能增强原型 | 子任务如何产生抽象；完整系统如何帮助另一智能体 | 目标构造、options、系统依赖和人机协作 |

第 12 步的 intelligence amplification 不是“把所有模块拼完”的别名。它研究学习系统如何增强另一智能体的行动、感知和认知。原计划还明确把其他智能体放进环境。这与用一个主要智能体的经验流定义问题并不矛盾。

任何依赖图都要接受反向检验：新特征是否帮助实际预测？新 option 是否带来值得其训练成本的规划收益？更准的模型是否改善行为？单个小实验回答其中一问，不能替整条闭环回答全部问题。课程中的方法论立场也不等于普遍定理或技术发展时间表。

<a id="research-architecture-knowledge-contracts"></a>

## 用查询规格连接预测、技能与规划

深入阅读：进一步讨论知识规格和使用条件如何在多个模块之间保持一致。

完整架构中最容易遗漏的是知识的条件。相同的“ψ”可能指 SF 累计、flow 条件特征或 Laplacian 规划坐标；相同的“零样本”可能指无新奖励训练、无任务策略训练或无部署环境数据。模块名称不足以保证彼此兼容。可将每项知识记录为一份可检查的查询规格：它在什么状态版本、行为、时域与目标下预测什么，谁会使用它。

$$
K_j=(q_j,\,\mathcal D_j,\,\nu^{\rm state}_j,\,\nu^{\rm behavior}_j,\,\widehat y_j,\,\epsilon_j,\,c_j),\qquad q_j=(\pi_j,C_j,\gamma_j,\text{readout}_j)
$$

这是拟议的架构元数据规格：查询、验证域、状态与行为版本、预测器、所用校准误差及成本。ε 只有在明示验证协议下才有含义，不默认是全环境的数学上界；readout 指下游要从该知识算什么。

| 模块知识 | 明确依赖 | 允许的下游查询 | 不能据此宣称 |
| --- | --- | --- | --- |
| SF / FB | 动力学、策略族、特征与奖励读出 | 被表示覆盖的奖励价值或任务条件行为 | 任意动力学变化立即迁移 |
| SF² | 条件特征、flow 投影、目标策略及生成器 | 未来占用采样，或给非线性 critic 的特征 | 任意 GVF 可线性读出 |
| 方向技能 / OKB | 基础策略、SF、元策略及调用规则 | 组合执行；额外模型可支持规划 | 技能组合自动得到准确后果模型 |
| DINO-WM / 2-AC | 冻结视觉编码、动作语义、离线动力学数据 | 给定目标下的动作条件特征预测与搜索 | 在线自动状态/技能构造已完成 |
| VE / 分布等价模型 | 策略/价值或统计摘要查询族 | 该族范围内的 Bellman 或风险计算 | 模型还原全部世界，或保证任意新风险目标 |

例如编码器升级后，旧 DINO 特征终点与新目标坐标不能直接比较；低层停止规则改变后，旧 option model 的 τ 与终点不再对应当前执行；主任务改成风险规避后，旧 mean-value 等价模型的规格不够。失配来自明确依赖，而不是看到回报下降后笼统称“遗忘”。

**算法：这是一项可实现的集成研究设计，不等于现成 OaK 实现**

1. 接口维护骨架（拟议）：
  1. 真实 transition 只推进环境一次，保存采样时行为与状态版本
  1. 分发给各自明确的 prediction/model/control learner
  1. 问题或技能更新后标记依赖知识过期，安排真实后续验证
  1. planner 只调用查询条件与版本匹配的知识；失配时退回已有合法接口
  1. 记录旧知识重新校准、局部迁移与作废的成本
  1. 固定总容量与每步更新预算，比较接口检查对失败/恢复的作用

OaK 与 Alberta Plan 提出经验产生预测、子任务、options、模型与规划的长期研究链。上述近年工作分别强化其中若干接口，但外部数据、固定表示或已给定奖励族仍可能参与启动。研究价值在于明确这些模块怎样互相提供学习信号，并检验未见变化下的真实收益，而非将多个论文模块接在一起就宣称完成完整架构。

<a id="research-architecture-future-utility-and-budget"></a>

## 从预训练能力到有限预算下的持续知识维护

深入阅读：在有限资源下研究一种知识未来可能带来的用途。

预训练模型降低在线学习的起点成本，持续架构还需要决定学什么、何时补齐、何时淘汰。OKB 的行为基补齐、GVF 问题发现、generate-and-test 的特征维护针对不同对象，可以共享“未来是否有用”的评价思想，但不能把它们直接视为同一个算法。特别是低预测误差可能仅说明问题太容易，高误差也可能来自不可约噪声。

Pilarski 的 Creating an Exocerebellum 讲座提供了一个具体接口：机器先预测动作后果，再通过振动等反馈帮助使用者协调动作。这里有三个不同的学习问题：怎样学准给定预测，怎样选择值得预测的问题，怎样选择向使用者传达的信息及其时机。讲义第 60 页明确指出，预测单元与下游传达方式仍需人为选择和设计。因此，扩大 GVF 数量不等于自动学会有用的交流；“外部小脑”是研究构想与功能类比，不是神经机制的等价证明。

$$
U_j(W)=\underbrace{J_W(\text{with }K_j)-J_W(\text{without }K_j)}_{\text{同协议下的未来收益差}}-\lambda_C\Delta C_j-\lambda_M\Delta M_j,\qquad \sum_jM_j\leq M_{\max},\quad \sum_jC_{j,t}\leq C_{\max}
$$

这是资源受限知识效用的拟议检验量，不是新引用定理。$J_W$ 必须明确是未来真实收益、查询误差减少或其他指标，不能混用；with/without 需要配对随机性、冻结副本或独立控制实验，不能读取同一生命的反事实结果。

例子：一个新预测头使充电风险估计更准确，却增加每步延迟，以致机器人错过控制频率；另一个技能提高覆盖，但长期不被规划器查询。这两者可能有预测或探索价值，却未必有净控制效用。只有在同样真实时间、容量与更新预算下测到未来收益，才能判断应否保留。

| 加入候选后的检验 | 回答的问题 | 应保留的边界 |
| --- | --- | --- |
| 未来固定行为预测 | 新知识是否减少未见后果误差 | 不把 replay 拟合当作未来检验 |
| 冻结副本读出 | 现有表示是否让查询可低成本恢复 | probe 不写回正在评价的 agent |
| 影子规划 | 知识是否改变正确的动作排序 | 不把模型里的收益当作真实环境收益 |
| 配对控制 | 实际调用是否改善未来互动 | 计入探索、失败和恢复成本 |
| 预算压力与淘汰 | 有限容量是否仍能持续补齐 | 记录旧能力损失与未来再学习成本 |

从 DINO-WM 或 V-JEPA 2-AC 启动的系统，可先冻结 encoder 只持续更新后果预测，之后才加入状态、问题或技能生成；从 SF² 启动则可先测试新查询可读出程度，再决定是否需扩容。每次新增机制都应有同预算对照，避免把更多数据、更多参数或更多规划混进架构结论。

尚未闭合的研究问题包括：任务和查询不断新增时如何分配信用；知识版本改变时怎样保留兼容接口；新动力学需要探索时怎样支付机会成本；被淘汰知识在未来再次出现时怎样恢复。这些是全局持续学习的问题，与某个静态基准上最终分数提高不同，应以一条完整经验流的收益与资源轨迹验证。

<a id="lesson-check"></a>

## 自测与集成实验

- 主线的最后一步延续为零，为什么起点资格仍为0.8？延续关闭本次尾值，活动迹仍须把已收到的末步误差分配给之前预测，写完才清除。
- 第9步的后续验证读哪一个预测？读第8步启动前缓存的0.191，不能读已用本次奖励训练后的0.3465。
- 同一随机种子是否保证两个学习器得到同一经验？前九步恰好一致；第十步行动不同后，真实数据已不同。
- 删除planner的元梯度是否也删除实际规划？不删除；停止求导改变计算图，实际备份的前向值仍可改变行动。

- 为什么行为概率必须随经验保存？它描述采样当时分布；更新后策略不能倒过来解释历史动作。
- 同一个 reward 能否同时用于平均奖励控制和折扣 GVF？可以，但 discount、target policy 与意义不同，需要两个目标和估计器。
- 为什么模型有 loss 下降，planning 仍可能伤害行为？训练分布、规划查询分布和非平稳旧数据不同；小局部误差可以被反复 backup 放大。
- 在日志上切成多个窗口，能清空 recurrent state 吗？不应无声明清空，否则改变了任务协议。
- 只有多个模块接在一起就算 OaK 实现吗？不算；需逐项给出目标构造、学习器、接口、调度、预算以及开放环节。

深入阅读的变化技能原型练习：画出一次真实原子步到下一次高层选择的执行顺序，指出谁读旧值、谁写参数；解释该原型的技能六步停止为何可产生完整模型样本，而 1200 步记录截止未必可以；从已有作业记录重算全程收益，并说明这些记录还不能区分哪些机制。把规划预算改为每原始步固定，或分离模型与高层 Q 的失效处理，可以形成后续对照方案。写出你预期会看到的差异，以及什么结果会否定这项解释。

状态研究另用六状态骨架：移除 cue memory，让不同历史映射到同一状态。额外规划不能恢复已被输入表示丢弃的信息。不要把这个信息限制与优化不稳定混成同一个问题。

## 本章的实验设计

每次消融都对应一个接口假设。报告中间量和长期收益。完整系统的改进不能直接归因于其中一个模块。

设定：先固定技能和消费者检验预测模块，再比较预测开关×技能开关；在明确收益链后接模型或规划。

- 每个输出有明确消费者、参数版本与可用时刻。
- 改变诊断 oracle 不会通过中间模块进入部署输入。
- 完整快照包含各模块状态和事件队列；恢复能力按真实支持声明。

对照：四格交互设计与同容量基座；已训练系统断链与无模块重新训练分别报告；准确模块诊断及固定/在线版本

记录：预测准确、技能成功与真实回报三层指标；表示/目标/模型版本漂移；全部模块的计算、存储及延迟

[具体测试规程](../docs/experiment-handbook.md#handbook-planning)

## 学习与研究衔接

状态、预测、技能和规划一面共享经验，一面改变彼此的输入与目标。一个技能刚学好时，旧模型可能低估它，使规划很少调用它，进而延缓模型修正；状态改进也可能帮助多个预测同时辨认历史。完整架构应通过分别冻结、恢复和继续学习来辨认这些作用，并检验发现与维护知识的成本是否换来更多生命期收益。

[分册导读](../docs/learning-route-continual-rl.md) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-architectures) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=architectures) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=architectures)

## 从本章进入实践

[技能与规划](https://yingwen.io/zh/continual-rl/code/#practice-skills)：一个多步行为怎样成为可学习、可预测、可规划的动作？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

给定状态后可以估计价值；观测不足时，还要学习保留哪些历史。GVF 规定预测什么，RTRL 计算递归敏感度，资格迹组织时间信用。应分别检验信息是否进入状态、反馈能否教会这种保留，以及有限预测预算怎样分配，而不是把三者当作替代算法。

- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [Artifacts as Memory Beyond the Agent Boundary](https://yingwen.io/zh/continual-rl/research/#recent-openmind-artifacts-memory)

#### 时间信用分配与离策略多步学习

当前反馈如何修正过去的决策与预测，哪些历史信息可以压缩成迹？

前向回报定义目标，后向迹组织计算。离策略修正、条件期望迹、梯度目标和递归敏感度分别改变不同对象；需先固定参数时序与采样条件，再讨论深度及持续控制。

- [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://yingwen.io/zh/continual-rl/research/#recent-vtrace-impala)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

教材可以先给定目标和技能集合；持续构造还要决定哪些行为值得练习、维护或放弃。谱结构、路径奖励、时间距离和语言先验提供不同候选偏置。先固定候选比较选择与组合，再改变生成器，才能辨认下游收益究竟来自哪一步。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [MaestroMotif: Skill Design from Artificial Intelligence Feedback](https://yingwen.io/zh/continual-rl/research/#recent-maestromotif-semantic-skills)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Loss of plasticity in deep continual learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-backpropagation)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Learning from experience instead of curated datasets](https://yingwen.io/zh/continual-rl/research/#recent-oak-network-idbd)
- [Discovering state-of-the-art reinforcement learning algorithms](https://yingwen.io/zh/continual-rl/research/#recent-disco-rl)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [Position: Lifetime tuning is incompatible with continual reinforcement learning](https://yingwen.io/zh/continual-rl/research/#recent-lifetime-tuning)
- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [How Should We Meta-Learn Reinforcement Learning Algorithms?](https://yingwen.io/zh/continual-rl/research/#recent-meta-algorithm-search-comparison)
- [Physical Atari: A Robust and Accessible Platform for Real-time Reinforcement Learning on Robots](https://yingwen.io/zh/continual-rl/research/#recent-openmind-physical-atari)
- [The Open Ant: A Robot Platform for Reinforcement Learning Research](https://yingwen.io/zh/continual-rl/research/#recent-openmind-ant-platform)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文规定对象与条件，架构路线提出组织方式，算法实验检验局部机制。撤掉阶段间冻结后，一个模块会改变另一个模块的学习问题；有限预算应优先维护哪条知识，成为新的决策。先检验两模块反馈和资源分配，再扩大整机，而不是由组件分别有效推断长期组合收益。

- [Rethinking the Foundations for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-rethinking-crl-foundations)
- [Plasticity as the Mirror of Empowerment](https://yingwen.io/zh/continual-rl/research/#recent-plasticity-mirror-empowerment)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Towards model-free RL algorithms that scale well with unstructured data](https://yingwen.io/zh/continual-rl/research/#recent-nibbler-predictive-features)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)

### Towards model-free RL algorithms that scale well with unstructured data

Joseph Modayil, Zaheer Abbas

arXiv 预印本 · 2023 · 支持方法与理论

#### 研究问题

大量原始观测中只有少数局部组合与奖励有关，智能体能否逐步构造有用的预测特征？

#### 关键机制

Nibbler 将预测问题的构造和预测结果的复用结合起来：选择局部输入、学习与奖励相关的通用价值预测，再将预测作为后续学习的特征。GVF 在这里不是一个新的优化器，而是描述“预测什么、在什么行为下预测”的问题接口。

#### 证据

作者在组合式合成环境中增加观测规模，报告了利用任务结构的样本效率。环境可以具有指数增长的状态组合，但学习器不必显式枚举全部状态。

#### 条件与限制

这不是对任意高维观测的线性样本复杂度保证。局部可分解结构、候选问题与特征构造规则仍是关键条件；从该实验族迁移到视觉控制需要额外验证。

#### 阅读与实验

把一条预测完整写成累积量、延续条件、目标策略和输入特征四项，再指出它如何进入主任务的价值函数。区分问题生成带来的收益与增加参数量带来的收益。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/abs/2311.02215)：问题族、Nibbler 构造过程与扩展性实验。

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

### Loss of plasticity in deep continual learning

Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton

Nature · 2024 · 直接研究持续学习

#### 研究问题

一个长期训练的网络如何保留继续形成新特征的能力？

#### 关键机制

Continual Backpropagation 在梯度学习之外持续生成并测试特征。它估计单元的效用和成熟度，少量替换低效用的成熟单元，并协调新单元的输入、输出和相关状态。维护新的可学习方向是一个持续过程，而不是等到任务切换后整体重启。

#### 证据

论文在长序列监督学习与强化学习问题中展示可塑性损失，并检验特征替换的作用。作者仓库包含 generate-and-test 与优化器状态处理。

#### 条件与限制

原文明确说明，其效用主要考虑当前数据，CBP 并不解决遗忘。对新数据持续学得动与旧功能仍被保留是不同结果；有限长序列也不保证无限生命中的任意适应。替换率、效用和成熟度仍需选择。

#### 阅读与实验

在同一替换预算下消融成熟度、效用与随机替换，先检验新目标学习。随后让旧情境返回，测被回收功能的损失；若再用旧功能代价约束回收，应作为新增机制检验，不能把收益归给原始 CBP。

#### 原文与相关入口

- [Nature 原文](https://doi.org/10.1038/s41586-024-07711-7)：长期可塑性实验与 continual backpropagation。
- [作者代码](https://github.com/shibhansh/loss-of-plasticity)：关注 lop/algos/gnt.py 及替换时的优化器状态。

#### 作者代码

[论文作者公开的实验实现。](https://github.com/shibhansh/loss-of-plasticity)

论文任务、持续反向传播与 generate-and-test。

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

### Mastering diverse control tasks through world models

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, Timothy Lillicrap

Nature · 2025 · 支持方法与理论

#### 研究问题

同一套世界模型训练与控制方法，能否减少跨任务重新设计损失和超参数的需求？

#### 关键机制

DreamerV3 从经验学习递归潜在状态、奖励和延续预测，再在潜在想象轨迹上学习 actor 和 critic。尺度稳健的表示与损失设计使同一配置可以适用于多种任务。模型是用于决策的学习接口，不必生成完整真实世界。

#### 证据

论文在大量视觉和状态控制任务上报告了广泛表现。关键含义是共享算法配置；这些结果主要来自分别训练的任务智能体，不是一个智能体按顺序学会全部任务。

#### 条件与限制

经验重放、批量训练和模型想象都有资源成本。模型偏差、表示遗忘与长期任务切换仍需要专门实验，不能由多任务覆盖范围自动推出持续学习能力。

#### 阅读与实验

把状态更新、模型训练、想象起点和策略更新四种分布分别写清。比较真实交互步数之外，还应记录想象步数和优化次数。

#### 原文与相关入口

- [Nature 原文](https://www.nature.com/articles/s41586-025-08744-2)：方法和任务协议；区分共享配置与单智能体持续学习。
- [作者维护的实现](https://github.com/danijar/dreamerv3)：公开实现的版本与论文实验环境应分别记录。

#### 作者代码

[作者发布的重实现；不把当前分支当作原论文实验的冻结快照。](https://github.com/danijar/dreamerv3)

DreamerV3 的作者维护公开实现及运行配置。

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

### Plasticity as the Mirror of Empowerment

David Abel, Michael Bowling, Andre Barreto, Will Dabney, Shi Dong, Steven Hansen, Anna Harutyunyan, Khimya Khetarpal, Clare Lyle, Razvan Pascanu, Georgios Piliouras, Doina Precup, Jonathan Richens, Mark Rowland, Tom Schaul, Satinder P. Singh

NeurIPS 2025 · 2025 · 定义与架构观点

#### 研究问题

环境改变智能体的能力，与智能体改变环境的能力，能否放在统一的信息论框架中？

#### 关键机制

论文用广义有向信息描述两个方向：环境对智能体的影响对应一种可塑性，智能体对环境的影响对应赋能。统一表达使二者的关系和权衡可以被形式化，而不仅用神经元休眠或短期奖励间接描述。

#### 证据

贡献主要是概念定义和理论关系，提供研究长期交互的新坐标。它没有把信息量指标直接等同于某个具体神经网络算法的长期回报。

#### 条件与限制

信息论可塑性与“新目标拟合速度”不是相同估计量，也不等于参数变化越大越好。有限数据下怎样稳健估计这些信息量，需要额外方法。

#### 阅读与实验

分别举出高环境影响但低奖励、高赋能但不学习的过程。说明为什么两类能力与任务成功都需要独立评价。

#### 原文与相关入口

- [NeurIPS 2025 原文](https://papers.nips.cc/paper_files/paper/2025/hash/f04957cc30544d62386f402e1da0b001-Abstract-Conference.html)：统一定义、理论关系与解释。
- [作者预印本](https://arxiv.org/abs/2505.10361)：便于检索定义和证明。

### Rethinking the Foundations for Continual Reinforcement Learning

Esraa Elelimy, David Szepesvari, Martha White, Michael Bowling

RLC 2025 / RLJ · 2025 · 定义与架构观点

#### 研究问题

如果智能体终生交互且世界不断变化，传统形式化和评价对象遗漏了什么？

#### 关键机制

论文重新审视状态、时间与累计奖励评价中的隐含假设，并讨论以交互历史和偏离遗憾等对象描述持续学习。研究重点从“在固定任务上最终收敛到什么”转向“在持续过程里，什么样的行为比较才有意义”。

#### 证据

这是形式化与研究基础的论证，提出可继续研究的定义和问题；不是一个已经完成全部工程验证的通用智能体。

#### 条件与限制

对常见形式化局限的讨论不意味着 MDP、折扣回报或平均奖励在各自条件下无效。评价框架还需要与具体可计算算法和实验协议连接。

#### 阅读与实验

选择一个有不可逆代价的环境，分别写出最终任务分数、终生在线收益和比较策略集合。检验它们是否会给同一行为排出不同顺序。

#### 原文与相关入口

- [RLC 2025 原文](https://rlj.cs.umass.edu/2025/papers/RLJ_RLC_2025_243.pdf)：形式化动机、定义与论证。
- [RLJ 论文入口](https://rlj.cs.umass.edu/2025/papers/Paper243.html)：作者与正式收录信息。

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

### The Cell Must Go On: Agar.io for Continual Reinforcement Learning

Mohamed A. Mohamed, Kateryna Nekhomiazh, Vedant Vyas, Marcos M. José, Andrew Patterson, Marlos C. Machado

arXiv 预印本 · 2025 · 评价与实验协议

#### 研究问题

如何在持续、动态的高维交互里，同时研究记忆、探索、信用分配与学习能力保持？

#### 关键机制

AgarCL 提供持续运行的游戏环境，并用分解的小任务暴露不同困难。完整环境把这些机制放回同一交互循环，小任务则便于定位失败原因。游戏中的复活事件与把整个世界和智能体都重新开始不是同一种重置。

#### 证据

论文提供环境、基线与可塑性方法比较；部分常见修复在其测试中改善有限。这说明保持可塑性并不能单独代替记忆、探索和长期信用分配。

#### 条件与限制

一个游戏不能代表全部真实持续问题。小任务与完整游戏的协议需要分别阅读；此处仅按可确认的预印本状态收录，不把投稿信息写成会议录用。

#### 阅读与实验

先在一个小任务中验证机制，再检验它在完整环境中的作用是否仍存在。将世界重置、角色复活、参数重置和数据清空分开记录。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2505.18347)：环境设计、分解任务与基线结果。
- [作者环境仓库](https://github.com/machado-research/AgarCL)：环境安装、接口和运行示例；算法基线与环境本体分开。

#### 作者代码

[Machado 研究团队的环境实现。](https://github.com/machado-research/AgarCL)

AgarCL 环境与示例，非所有算法结果的单一训练脚本。

### Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning

Jiaheng Hu, Jay Shim, Chen Tang, Yoonchang Sung, Bo Liu, Peter Stone, Roberto Martín-Martín

RLC 2026 · 2026 · 直接研究持续学习

#### 研究问题

大规模预训练的视觉—语言—动作模型，是否仍需要复杂机制才能顺序学习控制任务？

#### 关键机制

论文研究对预训练 VLA 进行顺序强化学习，并以低秩适配等相对简单的训练流程检验持续学习。预训练表示、可更新参数子空间和 RL 目标共同决定迁移与遗忘，不能只把结果归因于单一保留正则项。

#### 证据

作者在多种 VLA 与长期任务基准上比较，并提供实验代码。RLJ 的 RLC 2026 论文页与论文脚注分别给出正式入口和作者仓库。

#### 条件与限制

预训练数据和算力属于外部资源，任务与重置协议也影响难度。该结果不意味着从零训练的网络不会遗忘，更不意味着任意无边界任务流只需微调。

#### 阅读与实验

固定预训练模型，分别改变可训练参数量和任务顺序。报告预训练资源、每任务在线数据、回放或重置条件，再与传统 CRL 方法比较。

#### 原文与相关入口

- [作者论文](https://arxiv.org/abs/2603.11653)：VLA 持续学习设置、机制与比较。
- [RLC 2026 / RLJ 论文页](https://rlj.cs.umass.edu/2026/papers/Paper84.html)：会议原文入口；PDF 脚注链接作者代码。
- [作者实现](https://github.com/UT-Austin-RobIn/continual-vla-rl)：持续 VLA 的训练与评价代码。

#### 作者代码

[UT Austin RobIn 实验室的原论文仓库。](https://github.com/UT-Austin-RobIn/continual-vla-rl)

预训练 VLA 的顺序 RL 训练和论文评价。

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

### IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Volodymyr Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, Koray Kavukcuoglu

ICML 2018 · 2018 · 支持方法与理论

#### 研究问题

actor采样策略落后于learner时，如何校正状态价值与策略更新？

#### 关键机制

V-trace用截断ρ校正当前TD误差，用独立截断c控制后续误差传播，再用下一状态V-trace目标构造actor优势。ρ上限还决定表格固定点对应的截断策略。

#### 证据

原文分析固定点并检验分布式多任务训练。固定版本作者代码明确区分clipped_rhos、cs、反向scan与pg_advantages。

#### 条件与限制

IMPALA保存短轨迹并批量训练，不属于严格单样本流式协议。截断后价值可能对应不同于原目标的策略；信用章bandit示例显示0.8变为0.5。

#### 阅读与实验

独立改变策略滞后、ρ上限和c上限。记录目标策略变化与传播长度，不把两种截断都只解释为方差控制。

#### 原文与相关入口

- [ICML原文](https://proceedings.mlr.press/v80/espeholt18a.html)：V-trace固定点与分布式实验。
- [作者固定实现](https://github.com/google-deepmind/scalable_agent/blob/6c0c8a701990fab9053fb338ede9c915c18fa2b1/vtrace.py)：from_importance_weights与下一状态actor目标。

#### 作者代码

[原作者团队仓库的固定版本。](https://github.com/google-deepmind/scalable_agent/tree/6c0c8a701990fab9053fb338ede9c915c18fa2b1)

IMPALA原始TensorFlow实现与V-trace；运行需要原项目环境。

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

### TD-MPC2: Scalable, Robust World Models for Continuous Control

Nicklas Hansen, Hao Su, Xiaolong Wang

ICLR 2024 · 2024 · 支持方法与理论

#### 研究问题

如何让短期动力学与长期价值分工，并在动作选择时继续使用模型？

#### 关键机制

TD-MPC2 学习无需观测 decoder 的潜在动力学、奖励、价值与策略先验。决策时优化有限动作序列，用终点价值补上未展开的后果；执行第一步后，利用新观测重新规划。

#### 证据

正式会议原文报告 104 个在线任务和单一大型多任务智能体的实验。官方仓库包含模型训练与计划接口，适合与 Dreamer 的想象 actor 学习比较计算位置。

#### 条件与限制

跨任务共享超参数和多任务能力不是单条生命流中持续适应的证据。replay、任务条件、模型更新、决策延迟等成本需进入 CRL 协议；长程 critic 错误不能被短期模型精度自动修复。

#### 阅读与实验

固定模型，对比无终点价值、不同 horizon 和不同规划预算；再固定预算比较部署 actor 与决策时搜索。环境变化后同时记录模型校准、critic 误差和恢复收益。

#### 原文与相关入口

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。
- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

#### 作者代码

[作者维护的原论文代码。](https://github.com/nicklashansen/tdmpc2)

TD-MPC2 的单任务/多任务训练和决策时规划。

### DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning

Gaoyue Zhou, Hengkai Pan, Yann LeCun, Lerrel Pinto

ICML 2025 · 2025 · 支持方法与理论

#### 研究问题

预训练视觉表示能否直接成为动作后果预测与目标规划的接口？

#### 关键机制

冻结 DINOv2 空间 patch 特征，用离线动作轨迹学习未来特征预测器；测试时优化动作序列，让预测特征接近目标图像特征。没有重建图像、奖励模型或逆模型，不表示没有动作条件的动力学训练。

#### 证据

ICML 原文在六类环境检验视觉目标规划，作者仓库公开数据、部分检查点、训练与 CEM 规划入口。零样本指给定已训练模型后解决目标，无额外任务策略训练。

#### 条件与限制

依赖视觉预训练与离线交互覆盖；patch 相近不总等于任务完成或风险相同。原实验不证明冻结视觉表示能适应长期新物体、新动作语义或隐藏状态。

#### 阅读与实验

分别改变背景、物体属性、控制动力学与目标分布。把冻结 encoder 和联合更新 encoder 分开，对照视觉距离、真实成功与模型误差，观察表示漂移的依赖成本。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。
- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

#### 作者代码

[原作者 Gaoyue Zhou 的论文配套仓库。](https://github.com/gaoyuezhou/dino_wm)

DINO 特征预测、离线环境数据与目标规划；README 公开部分环境检查点。

### V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas

arXiv 预印本（此处采用 2025 首稿） · 2025 · 支持方法与理论

#### 研究问题

无动作标注的视频预训练，与能接受机器人动作的规划模型之间还缺哪一步？

#### 关键机制

V-JEPA 2 先学被遮蔽视频的潜在特征预测；V-JEPA 2-AC 冻结编码器，再用机器人轨迹训练动作条件预测器。控制以目标图像的特征差为代价进行 MPC；视频理解、动作条件预测和真实控制是三个独立证据层。

#### 证据

2025 首稿报告以大规模视频预训练，再用不到 62 小时 DROID 交互视频后训练，在两个实验室以图像目标做真实机器人规划。论文单独讨论相机位置、长程规划与图像目标的局限。

#### 条件与限制

无任务奖励并不等于无动作、无机器人状态或无外部数据。零样本部署未持续更新模型，也未发现和维护 options。官方仓库现含 V-JEPA 2.1，复现首稿须记录配置和模型版本。

#### 阅读与实验

按视觉编码、动作坐标、后果模型、目标代价逐项做迁移检验。若引入在线更新，记录模型更新使旧目标接口失效的程度，测未来交互收益，而非仅用 frozen probe 证明 CRL。

#### 原文与相关入口

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。
- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

#### 作者代码

[Meta FAIR 官方仓库；首稿模型与后续版本需按配置区分。](https://github.com/facebookresearch/vjepa2)

官方视频表征与动作条件模型；数据、机器人部署条件与检查点分别核验。

### Artifacts as Memory Beyond the Agent Boundary

John D. Martin, Fraser Mince, Esra’a Saleh, Amy Pajak

arXiv 预印本 · 2026 · 支持方法与理论

#### 研究问题

完成任务所需的内部记忆，是否也取决于世界能替我们保存哪些历史信息？

#### 关键机制

把对过去观测提供确定信息的当前观测定义为 artifact。在特定条件下分析历史缩减，并通过环境中的地标与痕迹研究外部记忆效应。

#### 证据

论文给出关于下一观测信息的形式化结果，以及不同价值函数参数容量的实验。外部痕迹能影响完成任务所需的表示容量。

#### 条件与限制

关于观测信息的结论不是任意策略的控制充分性定理。参数数量不等于全部工作内存；论文对回放的计费与严格流式协议不同。

#### 阅读与实验

比较历史痕迹、直接动作提示和随机标记；另提出写读有成本的任务，记录内部状态、环境存储和移动时间。

#### 原文与相关入口

- [作者预印本](https://arxiv.org/html/2604.08756v1)：定义、信息结果、表示容量实验与限制。
- [John Martin 发表目录](https://jdmartin86.github.io/research/)：连接其规划、奖励与智能体边界的个人研究脉络。

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

深入阅读中的六状态、双动作、手工记忆骨架；主线七位置环的单文件程序在“沿运行记录走完同一次学习循环”一节下载。二者均不是完整 OaK/STOMP 性能复现。

[下载 lifelong_algorithms_lab.py](../examples/lifelong_algorithms_lab.py)

```sh
python examples/lifelong_algorithms_lab.py architectures
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto · §8.1–8.3、§17.1–17.5](http://incompleteideas.net/book/the-book-2nd.html)：§8.2 将直接学习、模型学习、search control、模型备份与实际行为放进同一 Dyna 循环；§17.5 讨论学得模型、问题选择与行为对学习的反馈。

- [Sutton · Dyna, an integrated architecture for learning, planning, and reacting](https://doi.org/10.1145/122344.122377)：模型学习、直接学习与模拟规划的经典接口；本页骨架受此启发，并明确另外加入平均奖励与 GVF。

- [Sutton、Bowling、Pilarski · The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173)：组件接口、有限资源与研究路线；Common Model 的角色划分是一种架构观点。

- [Xu、van Hasselt 与 Silver · Meta-Gradient Reinforcement Learning](https://arxiv.org/abs/1805.09801)：§1 以之后的样本评价先前学习，并区分完整参数敏感度与在线近似；本章有限技能上的局部步长规则另行定义。

- [Burega et al. · Meta-Gradient Search Control](https://arxiv.org/html/2406.19561v1)：§3 学习 Dyna 模型查询分布，经规划参数更新评价查询；这与本章只用后续预测误差调整奖励模型步长的实现不同。

- [Sutton et al. · Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://arxiv.org/abs/2202.03466)：子任务、option、后果模型与规划的原论文。本文写出的平均奖励差分形式来自后续 OaK 讲义，不与原论文的折扣实验混为同一设定。

- [Wan, Naik & Sutton · Average-Reward Learning and Planning with Options](https://papers.neurips.cc/paper_files/paper/2021/file/c058f544c737782deacefa532d9add4c-Paper.pdf)：§3 式 (3–5) 与 (6–9) 区分采样时长直接扩展和期望时长归一化；正式收敛结果不覆盖本章变化技能、常步长的完整教学系统。

- [Barreto et al. · Successor Features for Transfer in Reinforcement Learning](https://arxiv.org/abs/1606.05312)：有限奖励特征、固定策略 SF 与 GPI 的经典机制桥梁。

- [Richard Sutton · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：公开架构讲座入口。差分子问题依据 RLSS 收录的 OaK/NeurIPS 讲义第 24 页；学习的四种作用、消费者信用和增量规划讨论依据 OaK thinker 讲义。本文的成本记账、删除诊断和缓存反例用于澄清机制，不是作者已完成的通用算法。

- [作者代码 · average-reward-methods](https://github.com/abhisheknaik96/average-reward-methods)：control_agents.py 中的直接学习与 planning_update，适合比较奖励率更新调度。

- [Elelimy et al. · Rethinking the Foundations for Continual RL](https://rlj.cs.umass.edu/2025/papers/Paper243.html)：RLC 2025：history process、deviation regret 与持续学习评价对象。

- [Mohamed et al. · The Cell Must Go On: Agar.io for Continual RL](https://arxiv.org/abs/2505.18347)：2025 年提出、后续修订的 AgarCL：完整环境与分解小游戏，讨论探索、记忆、信用分配和可塑性。

- [AgarCL 作者环境](https://github.com/machado-research/AgarCL)：C++ 仿真与 Python 接口；局部重生、观测和混合动作需按环境配置理解。

- [AgarCL 作者基线 · PPO 混合动作实现](https://github.com/machado-research/AgarCL-benchmark/blob/main/PPO_multi_heads_full_action.py)：同仓库包含 DQN_full_action_set.py、SAC_full_action_set.py 及 recurrent 版本；完整游戏实验需匹配动作、参数搜索与运行预算。

- [Mahmood & Sutton — Online Representation Search and Its Interactions with Unsupervised Learning](https://www.eng.uwaterloo.ca/~jbergstr/files/nips_dl_2012/Paper%2019.pdf)：原始在线监督表示搜索，研究生成器与测试器；其设定不等于一般深度流式 RL。坐标变换与更新几何的推导为本节教学分析。

- [Sutton et al. · Reward-Respecting Subtasks · Artificial Intelligence 2023](https://doi.org/10.1016/j.artint.2023.104001)：子任务、技能、模型和规划的接口；最初预印本2022，AAAI2024条目为摘要重印。

- [An Analysis of Streaming Deep RL for Adaptive Continual Learning in Robotics](https://arxiv.org/html/2609.28807v1)：2026预印本；预训练后突变适应、优化器和tanh-CBP变体；episodic及前向迁移范围。

- [Streaming robotics · 作者工程](https://github.com/tjvitchutripop/stream-rl-robotics/)：论文直接链接；预训练、任务变化、流式更新和评价配置需联合复现。

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。

- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。

- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

- [Touati & Ollivier · Learning One Representation to Optimize All Rewards](https://arxiv.org/abs/2103.07945)：FB 表示的理论出发点；探索/经验覆盖、近似误差及奖励查询约定。

- [Pilarski · Creating an Exocerebellum（2024 讲座）](https://pilarski.github.io/talk/creating-an-exocerebellum/)：第 28–29 页区分控制与反馈通路；第 60 页明确预测问题选择及下游通信尚需人工设计。作为知识效用与智能放大的具体问题，不作为完整自主架构的证据。

- [Javed et al. — Physical Atari（RLC 2026）](https://keen-technologies.github.io/keen-website/research/physical-atari/static/physical_atari.pdf)：平台工程与真实学习论文；含延迟测量、先行动后学习、回放、跨身体迁移和可靠性记录。

- [Keen Technologies — Physical Atari 作者硬件与代码](https://github.com/Keen-Technologies/physical-atari-rlc)：硬件搭建、配置、智能体实现与实验脚本。

- [Lupu et al. — The Open Ant（RLC 2026）](https://arxiv.org/html/2607.18488v1)：非回合行走、SARSA 与 SAC、真机干预、模拟到真实排序差异；主体为平台贡献。

- [Openmind Research Institute — Open Ant 作者硬件与代码](https://github.com/Openmind-Research-Institute/open-ant)：硬件设计、维护说明、模拟器与 SARSA/SAC 入口；平台可用性不等于算法终生保证。

- [Sutton · Toward a New Approach to Model-based Reinforcement Learning](https://www.incompleteideas.net/papers/MBRL2.pdf)：课程指定阅读 Introduction 与 §1：近似 agent state、特征对当前表现与未来学习的用途、学习与规划的耦合。
