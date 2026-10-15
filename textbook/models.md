# 模型与后果预测：学什么，才能用于下一次决策？

执行一个动作或技能以后，会积累多少奖励、何时到哪里；这些预测怎样支持规划与任务变化后的迁移？

## 本章内容

- 从规划消费者反推模型必须保留的行为条件、奖励、终点与时间信息，说明一组 GVF 如何组成后果模型。
- 从随机持续时间的回报推导折扣双头、平均奖励三头及各自样本更新。
- 知道期望模型为何在线性价值下足够，以及非线性消费者、移动表示和技能版本如何破坏复用。
- 用代码区分模型证据更新和价值重算；进一步通过变化环境、SF/GPI、DR 与潜在模型比较不同复用边界。

## 阅读这条论证

规划需要比较尚未执行的行为。先从这项用途确定模型要回答的问题，再把技能的奖励、时长与终点写成可学习的预测。每个输出都要说明怎样由真实经验更新、怎样被后续价值使用。随后检查压缩、技能改变和共享表示会丢掉哪些信息。

[从用途规定预测对象](models.md#lesson-setting) → [推导技能后果模型](models.md#lesson-derive) → [改用平均奖励目标](models.md#lesson-average-model) → [检验预测是否足够](models.md#lesson-expectation) → [追踪一段技能后果](models.md#lesson-example) → [对应实际更新](models.md#lesson-code)

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [Options：多步决策、技能发现与可复用行为](options.md)：明确技能的执行、停止与后果。
- [Dyna：模型学习与规划](dyna.md)：理解模型如何被更新和使用。


### 条件模型

模型回答“从这个状态，假如做这个动作/遵循这个技能，会发生什么”。它必须包含条件行为；把经验中的平均下一状态当作所有动作共同的预测，无法比较行动。

### Option

一个可执行技能包含内部策略 $π_o$ 和停止概率 $β_o$。τ 是至少为 1 的原始步数。模型预测这个已指定行为的后果；改变其策略或停止函数，就是改变被建模的对象。

### 期望与采样

分布模型描述全部可能后果，样本模型随机生成一个后果，期望模型只输出某些统计量。平均值足够与否由下游计算决定，不由模型名字决定。

### 线性价值

φ(s) 是固定特征，w 是权重。线性指价值对特征线性；φ 本身可以是非线性编码。若编码也在学习，关于固定模型/表示的推导要重新检查。

$$
V_w(s)=w^\top\phi(s)
$$

<a id="problem-definition"></a>

## 本章的问题定义

预测动作或技能后果，以支持可改变的下游价值和规划；输出应由用途确定，而非统一要求重建全部观测。

### 给定条件与符号

- 被建模的动作或技能、外部奖励、原始时间单位，以及折扣或平均奖励的使用目标。
- 状态或固定特征、后续价值函数类、真实数据与模型容量预算；技能执行和停止规则的版本明确。

### 需要求解的对象

足以计算指定后果备份的奖励及终点统计；完整分布、样本模型、期望特征模型和SF预测的是不同对象。

### 信息与数据权限

技能 $o$ 的策略与停止函数给定，$\tau$ 为原始时长。改变技能或表示即改变题目；离策略一步学习须记录行为概率并检查支持。

$$
\mathcal B_oV(s)=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau V(S_{t+\tau})\mid s\right],\qquad \hat{\mathcal B}_oV\approx\mathcal B_oV\quad(V\in\mathcal V)
$$

$\mathcal V$ 为声明的下游价值类，$\gamma$ 为折扣。模型充分性以能否复算该类备份判断；奖励模型 $r_o$ 和折扣终点模型 $p_o^\gamma$ 是实现接口。若 $V_w(s)=w^\top\phi(s)$，折扣特征期望足够；一般非线性价值不满足此交换。

### 成立条件与解的含义

- 先固定环境、表示与被建模行为。折扣模型使用有界奖励与 $0\le\gamma<1$，终点几乎必然可达；平均奖励的技能后果模型另要求有限期望时长与可积段内奖励。折扣时长与终点联合建模，真实终止的后续价值固定为零。
- SF重加权需相同动力学、策略、折扣和特征语义，奖励变化限于给定特征的线性张成。

判断准则：已知技能上检验奖励和折扣终点向量，并对多组未拟合下游价值比较备份误差；包含随机终点均值失败反例及时间—终点相关例。

### 适用边界

- 平均下一状态代入非线性价值通常不等于价值期望。
- 模型重建损失下降不直接证明动作选择改善。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [Options 与技能发现](options.md)：技能规定执行和停止，模型估计该版本行为的奖励、时长及终点后果。

- 组合不同学习问题 · [时间抽象与规划](planning.md)：规划使用模型统计计算候选价值，其需求决定模型必须保留什么。

- 组合不同学习问题 · [通用价值函数与预测知识](gvf.md)：模型输出可以由一组明确的预测问题学习；要复用到不同后续价值，还需保留可重新组合的后果。

- 组合不同学习问题 · [平均奖励与差分价值](average.md)：平均奖励规划读取未折扣奖励、原始时长和普通终点分布；折扣终点核不能直接当成普通概率。

- 组合不同学习问题 · [时间信用分配与资格迹](credit.md)：逐步预测可用资格迹将新误差写回本次技能中的旧预测；每个预测有自己的延续和停止边界。

- 组合不同学习问题 · [持续学习的智能体架构](architectures.md)：共享表示、技能版本和模型读取时序共同决定预测在行动时是否仍有原来的含义。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

完整后果分布昂贵；压缩后的预测可能丢失下游价值计算所需的信息。

### 本章的核心思路

先写出要复算的备份，再确定每个预测的行为条件、累计量、停止规则和更新时间。

1. [从技能回报拆出后果预测](models.md#lesson-derive)：将段内奖励与终点后续值分开，推导各预测头的 GVF 递推；它们才可在尾值变化后继续复用。

2. [为平均奖励保留真实时长](models.md#lesson-average-model)：从逐步中心化奖励得到奖励总和减奖励率乘时长；分别预测三项后果，避免奖励率改变时连带改变模型题目。

3. [检验压缩是否保留所需期望](models.md#lesson-expectation)：先在线性后续价值下推导等价，再检查非线性、最大值和时间终点相关性各自需要的信息。

4. [让真实更新成为可查询的知识](models.md#lesson-code)：核对完整执行与逐步估计的时序，再运行统一智能体，检查所学后果是否真的进入规划目标。

结论与条件：线性价值与固定特征下的期望备份等价是恒等式；SF/GPI保证需相同动力学等条件及价值误差控制，不覆盖任意奖励或技能变化。

### 相关方法改变了什么

- 完整分布/样本模型：保留多后果或提供样本，可用于一般尾值但成本与采样方差不同。

- 期望特征模型：对指定线性尾值充分，不能只凭均值支持任意非线性控制。

- Successor features：预测固定策略的累计特征，用于奖励重加权，不等于技能终点模型。


<a id="lesson-setting"></a>

## 1. 模型要预测什么，取决于怎样使用它

价值预测把未来压缩成一个给定行为条件下的回报；规划还需要比较不同动作和不同后续决策。我们于是要保留可以重新组合的后果知识。需要保留多少，取决于后续计算：平均位移可能把绕障碍的左右两条路径平均成穿墙；只预测所需奖励和后续特征，却可能已经足够支持某类价值备份。先明确规划要计算哪个量，才能判断模型输出是否充分，并选择训练标签。

| 对象 | 模型的输出 | 直接用途 |
| --- | --- | --- |
| 一步分布模型 | P(s′,r,d\|s,a) | 对后续价值求期望，或采样模拟；d 表示真实终止 |
| 一步样本模型 | 给 (s,a) 产生一次随机 (r,s′,d) | Dyna / rollout；需多样本处理随机性 |
| Option model | 执行期间的外部奖励 + 时间加权终点后果 | 以技能为单位做 Bellman backup |
| 期望模型 | 下游所需特征的条件期望 | 线性价值的精确期望 backup |
| Successor features | 固定策略下未来特征的累计期望 | 奖励权重变化时重估该策略，再做 GPI |
| Latent/world model | 潜在状态、奖励、继续概率等 | 潜在轨迹上的控制学习或搜索 |

先假设环境是固定 MDP，状态或固定特征可观察，奖励有界，技能 $\pi_o,\beta_o$ 固定，且 $0\le\gamma<1$。本页的终点模型假设技能停止或真实终止几乎必然在有限时间发生，所以终点随机变量有定义。训练数据可以来自实际执行技能的轨迹，也可以来自支持其动作的其他行为策略。若同时改变环境、技能和表示，模型的预测目标也会漂移，除了静态收敛，还需要分析跟踪误差。

以 option 为例，模型接收 $(s,o)$，输出执行期间的外部奖励模型 $r_o(s)$ 与折扣终点核 $p_o^\gamma(\cdot|s)$；规划器用当前价值 $V$ 计算 $r_o+p_o^\gamma V$。这里的“执行期间”限制累计的时间段，不是指技能训练用的内部奖励。固定动力学、奖励定义和 option 后，这个分工允许后续价值改变时复用同一个后果模型。直接预测某个固定高层策略的完整回报也有用，但不能替代面向不同后续价值的后果接口。

这也回答了“已经学了 GVF，为什么还需要模型”。一个 GVF 指定累积信号、延续和目标策略，回答一个行为条件下的问题；规划要把某个行动的后果交给可改变的后续决策。比如两个动作当前都预测总回报为零，但一个通向 A、另一个通向 B。明天 A 的后续价值变成 10，旧的两个零值无法告诉规划器该选谁；动作条件的终点预测可以。多个 GVF 能组成所需模型，前提是问题族保留了消费者需要的信息。

$$
q_V(s,a)=\mathbb E[R+\gamma(1-D)V(S')\mid s,a]
       =r(s,a)+\sum_j p_a^\gamma(j|s)V(j),\quad
       p_a^\gamma(j|s)=\mathbb E[\gamma(1-D)\mathbf1\{S'=j\}\mid s,a]
$$

这是只执行一个原始动作的接口。当前动作明确作为条件，尾值 $V$ 由调用者提供；动作之后继续哪一个策略尚未固化进模型。若只学行为策略下的平均后果 $\sum_a b(a|s)p_a^\gamma$，动作差异就已经被平均掉。主线接下来把同一接口推到跨多步的技能；道路关闸的深入算例另行检查这种可重组知识怎样过期、又怎样获得修正。

<a id="lesson-model-repair"></a>

### 1.1 从五条记录到模型，再看关闸后的错误

深入阅读：在运输任务中隔离模型错误对不同目标的影响，再观察模型修正后的价值传播。

沿用目标章的运输世界：上路 S→A→X，下路 S→B→C→X，X、Y 为物理终点。每条边一步，目标 X 只在到达时给奖励 1，折扣为 0.9。智能体已记录两条路线的五个转移；可用动作和地点标识已知，后果表最初为空。这里的确定性假设让每个被访问的状态—动作对只需保存最近观察到的下一状态与物理终止标志。它没有从缺失记录推断道路的能力。

$$
\widehat M(s,a)\leftarrow(s',d_{\rm phys}),\qquad \widehat P(j\mid s,a)=\mathbf1\{j=s'\}
$$

这是按真实记录覆盖一个表格条目的模型更新。每个阶段内动力学确定且不再改变，才使单条新观察足以精确修正该行；随机环境需要估计后果分布。

规划器读取这个表，再用当前目标把后果转换成 reward 和 continuation。它不能查询评估器中的真实地图，也没有“闸门已改”的通知。图中的完整道路是给读者与精确诊断看的；学习器只取得五条旧记录，以及稍后真实经过 S→A→Y 得到的两条新记录。为了取得后者，需再允许一次回到 S 的重置，不能把 A 当成免费可访问的查询点。

![关闸后的真实上路从A通向Y，左图旧模型还虚线预测X，右图获得新记录并传播价值后改选下路。](../assets/crl-figures/goal-model-planning-walkthrough-repair.svg)

红箭头是当前真实后果，紫虚线是陈旧预测，数字为规划中的状态值。目标 X 下，旧模型预测上路回报 $0.9$，实际沿 S→A→Y 得到 $0$。两步新记录仅改变 A 的后果行；再做两轮共十次模型调用后，起点选择回报 $0.81$ 的下路。

模型的误差需要注明在哪个分布上量。旧表在五条历史记录上仍完全吻合；若评估器用当前确定性真值核对五个状态—动作对，只有 A 的出口错了。该行预测质量由 X 移到 Y，$L_1$ 距离为 $|1-0|+|0-1|=2$，五行均匀平均为 $2/5=0.4$。这项全表诊断使用真值，学习器并不因此获得隐藏后果。新记录修正后误差为零，因为此例只有这一行发生变化。

$$
\widehat y_g-y_g=\sum_j\left[\widehat P(j\mid s,a)-P(j\mid s,a)\right]\left[r_g(s,a,j)+\gamma(1-d_g(j))V(j)\right]
$$

固定同一个后续值 $V$，将两个备份相减即得模型误差怎样进入消费者。这里的状态足以决定物理终止，且奖励函数由任务已知；一般情形还要计入奖励、终止预测与表示误差。

A 这一行中，旧模型送入的是 X，target 为 $1$；当前后果 Y 物理终止但未成功，target 为 $0$。所以这一行的备份误差是 $1$，并通过上一步折扣变成 S 上路价值的 $0.9$ 误差。若改为目标 A，过程在进入 A 时已经结束，这个出口错误就不影响从 S 完成目标 A。模型误差怎样影响决策，取决于算价值时给这些后果赋予什么奖励与尾值；同一个全表误差因而不能直接代表所有目标的控制损失。

奖励权重变化时复用 SF，需要保持动力学、目标策略、特征及累计规则一致，见 [SF 原文 §3–4](https://arxiv.org/html/1606.05312)。关闸改变了未来访问的地点，旧累计特征本身就要更新。Option 模型也条件化于内部策略和停止规则；只改同名技能的执行而保留旧模型，会产生同类失配。[STOMP 原文 §4–5](https://arxiv.org/html/2202.03466v3)将模型学习与规划分别定义，正是为了让后果预测能够被明确的备份消费。

本例保留原子动作模型，所以可以从新目标重新算奖励与停止；若只缓存上路技能的“终点 X、时长两步、奖励 0.9”，这个摘要既不适合关闸后的上路，也不够回答在中途 A 提前结束的任务。要么保存足以回答新问题的后果信息，要么重新学习对应的模型。下面的 option 模型推导把“足以回答什么”写成具体接口。

独立复算入口：[goal_model_planning_walkthrough.py](../tutorials/goal_model_planning_walkthrough.py)；完整逐轮记录：[确定性数据 JSON](https://yingwen.io/crl-figures/goal-model-planning-walkthrough-data.json)。[规划章](planning.md#lesson-replanning)接着说明为什么模型改好以后，旧计划还不会在同一瞬间全部改好。

下载单文件后直接运行；Fraction 精确计算后以小数输出

```sh
python3 tutorials/goal_model_planning_walkthrough.py
```

<a id="lesson-model-change"></a>

### 1.2 闸门再打开时，智能体怎样才会知道

深入阅读：让智能体在没有变化通知时发现旧通路重新开放，分析真实验证与模型更新的区别。

刚才把两条新记录交给模型，便能逐项检查修正与传播。现在让控制器自己取得记录。仍用同一六地点运输图、目标 X 和折扣 0.9，只让 A 的出口在 X 与 Y 之间改变。目标发现节曾在这张基图上增加两条 probe 支路；这里保留原来的五条运输边，不开放那两条附加动作，以便单独观察“已知道路发生改变”的困难。下路始终通往 X。

智能体先真实走八次上路、一次下路，所有已访问后果都进入模型。这段共同暖身提供 $19$ 次道路动作、$9$ 次重置和奖励 $9$。随后每个控制器从 S 独立运行 $96$ 个真实交互步。进入 X 或 Y 结束本次送货；若预算尚余，终点到 S 的 reset 花一个交互步，奖励为零。相较前面的固定记录算例，这里把回到起点的代价也列入预算，避免较短路线凭免费重置获得隐含优势。

$$
G_{
m trip}=\sum_{j=1}^{\tau}\gamma^{j-1}R_{t+j},\qquad L_{96}=\sum_{t=0}^{95}R_{t+1},\qquad \gamma=0.9.
$$

$\tau$ 是一次送货的道路动作数，终点尾值固定为零；$G_{\rm trip}$ 是规划器比较路线的准则。$L_{96}$ 是固定真实交互预算内实际取得的奖励总数，包含 reset 消耗的时间。它们是两个明示的评价对象；代码没有把单次送货的折扣价值称为终身回报。暖身计入时，总预算为 $124$，总奖励为 $9+L_{96}$。

令 $t$ 表示比较期已完成的真实交互步数，包括重置。环境在动作发出前按自己的时钟改变 A：$t<12$ 时通 X，$12\le t<48$ 时通 Y，$t\ge48$ 时重新通 X。这个时间表仅用于构造环境与画真值参照，控制器收不到阶段标签、变化通知或当前后果表。一次交互只返回实际的 $(s,a,r,s',d)$；智能体可以记住自己多久没有走某条路。

![相同六地点运输图只改变A出口；环境时钟决定关闸重开，学习器只能从实际行走取得记录，终点回S另花一步。](../assets/crl-figures/model-change-walkthrough-task.svg)

先读道路，再读右侧真实时间段。上路成功时两步回报为 $0.9$，下路三步回报为 $0.81$；关闸的上路仍花两步，却到达奖励为零的 Y。图给读者完整地图，模型只能消费实际观察。比较期的 $96$ 步与每次回到 S 的五次模型行备份分别计数。

关闸比较容易被发现：正在走上路的智能体很快会到 Y，旧预测与当前观察产生冲突。重开却不碰触正在走下路的智能体。假如它总依据“上路通 Y”的模型选择下路，之后收到的经验都继续支持下路；世界变好了，已有行为却不再提供能揭示改善的记录。第 8 章 §8.3 的 blocking maze 与 shortcut maze 正是用这两个方向的变化说明模型错误怎样和探索联系起来。[Sutton 与 Barto，第二版，2020 版次文本，第 166–168 页](http://incompleteideas.net/book/the-book-2nd.html)。

由此出现两个不同问题。已经走到 A 并观察新后果后，旧证据应保留多大权重？长时间不走 A 时，又怎样决定是否值得花真实动作重新检查？先把前一个估计问题写清，才能分辨后一个覆盖问题有没有被解决。

<a id="lesson-model-aging"></a>

### 1.3 一条新观察应改动多少模型

深入阅读：沿同一闭环比较累计与近期证据，区分世界变化和随机后果。

对固定状态—动作对 $(s,a)$，把第 $k$ 次真实观察编码为后果向量 $Z_k(j)=\mathbf1\{S'=j\}$。这里 $k$ 只在真正执行这一对时增加；它与全局时钟 $t$ 不同。累计模型保存全部后果计数，近期模型给刚到达的观察更大权重。两者都在估计动作后果，尚未决定下次走哪条路。

$$
\begin{aligned}\widehat P_k^{\rm all}(j\mid s,a)&=\frac1k\sum_{i=1}^k Z_i(j)
       =\widehat P_{k-1}^{\rm all}(j\mid s,a)+\frac1k\bigl[Z_k(j)-\widehat P_{k-1}^{\rm all}(j\mid s,a)\bigr],\\
       \widehat P_k^{\rm recent}(j\mid s,a)&=(1-\alpha)\widehat P_{k-1}^{\rm recent}(j\mid s,a)+\alpha Z_k(j).
       \end{aligned}
$$

本算例取 $\alpha=1$，即保存最近一次实际后果；每个阶段内转移确定，所以新观察能直接描述刚走过的这一行。一般 $0<\alpha<1$ 时，第 $i$ 条旧观察的权重按后续对该行的访问次数衰减，而非按没有访问的真实时间自动衰减。

本例奖励与终止由到达地点决定，因而只需观察到下一地点即可重算它们；独立代码仍把 $(s',r,d)$ 合成后果，避免把不同奖励或终止的记录混为同一结果。记 $p=\widehat P(X\mid A,\mathrm{cross})$，模型给控制器的上、下路价值分别为 $0.9p$ 和 $0.81$。平分选下路，于是上路只有在 $p>0.9$ 时才被贪心选择。

$$
\widehat Q(S,\mathrm{upper})=\gamma\bigl[p\times1+(1-p)\times0\bigr]=0.9p,\qquad \widehat Q(S,\mathrm{lower})=\gamma^2=0.81.
$$

这是同一模型版本下的期望备份；X 与 Y 均无回合尾值。代码按 C、B、A、S 的两个动作逆向备份，共读五个模型行，得到完整的本次计划。这里没有额外的真实经验，也没有直接 Q-learning 更新。

先关闭复查。暖身已有八次上路成功，比较期又在 $t=2,5,8,11$ 完成四次成功。第一次关闸观察在 $t=14$ 完成，此时累计模型为 $p=12/13$，上路值 $10.8/13\approx0.831$ 仍大于 $0.81$，所以再走一次上路。第二次失败后 $p=12/14$，上路值约 $0.771$，才转向下路。近期模型在第一次失败后便把 $p$ 置零，下一次送货走下路。这一差异来自新旧证据的权重。

![上下两组阶梯曲线分别关闭和启用路线年龄复查；两种贪心模型都错过重开，复查后近期模型回到1而累计模型仍低于决策阈值。](../assets/crl-figures/model-change-walkthrough-estimates.svg)

纵轴是 A 出口到 X 的模型概率，橙线 $0.9$ 是起点改选上路所需的阈值；圆点与方块标真实经过 A 的时刻。灰虚线是真实变化，仅供读者对照。上图的近期估计虽然能快改，却在重开后保持零；下图在真实复查后才获得新的成功证据。数据由有限闭环逐步运行产生，没有平滑或随机重复。

模型“老了”与模型“估错了”不是同一个可观察事件。长时间未访问说明知识缺少近期检验，却没有告诉我们门是否真的变了；一次失败则可能来自变化，也可能来自原本随机的后果。为分清两者，暂时换一个估计问题：固定不变的门独立地以 $p=0.95$ 通向 X，预先指定地观察 $n=20$ 次。最后一条观察作为概率估计的均方误差为 $p(1-p)=0.0475$；累计频率的均方误差为 $p(1-p)/n=0.002375$。附带脚本对二项分布的 $21$ 种计数精确求和得到这些值。这个随机、平稳的反事实没有混入前面的确定性变化实验：它解释了为什么覆盖旧值在无噪声时反应快，却容易把随机失败当成结构改变。

保存模型版本 $v$ 能说明一次计划读到了哪份估计。这里每接收一条道路经验就增加版本；同一版本上的规划只改计算结果，不增加任何后果计数。反复使用一条旧 $A\to X$ 记录可以传播旧知识，却不能增加“门在当前时刻开放”的独立证据；若把重放次数直接加进频率分母，相当于人为改变样本权重。即使累计模型包含 X 与 Y 两种后果，其混合比例在变化环境中也只是历史频率，并非当前门真的随机通向两端。

<a id="lesson-model-recheck"></a>

### 1.4 把复查代价放回行动循环

深入阅读：将复查旧路线的真实行动代价计入收益，观察估计改变后行为是否随之改变。

现在加入一个完全由已知经验驱动的行为规则。每次回到 S，记 $\ell(a)$ 为最近执行该起点动作后完成的交互时钟，路线年龄为 $u_t(a)=t-\ell(a)$。若有路线满足 $u_t(a)\ge12$，就选其中最久未走的一条；否则读取当前模型的贪心路线。年龄相等按动作名排序，价值相等选下路。上、下路都受同一规则约束，控制器没有被告知哪一行会变。

**算法：模型只从真实记录更新；规划计算不推进环境时钟。两种时钟分别记账。**

1. 共同暖身：实际走 8 次上路、1 次下路；每次终点重置花 1 步
1. 比较期重复，直到真实交互计数达到 96：
  1. 若在 S：
    1. 冻结当前模型版本，逆序做 5 次期望行备份
    1. 先求贪心路线；启用复查时，再检查两个路线年龄
    1. 最老路线年龄达到 12，则本次真实改走该路线
  1. 若在 A、B、C：执行唯一道路动作
  1. 若在 X、Y：执行 reset，奖励 0，真实时钟加 1
  1. 获得真实后果；道路记录才更新模型和证据版本
  1. 按实际奖励计分；预算耗尽时立即停，不补齐回合或 reset

这个复查规则直接改变实际选路，没有给模型奖励添加 bonus。Dyna-Q+ 则在模型产生的规划奖励中加入随未尝试时间增长的项，再由更新后的价值影响行为；原书还允许规划未尝试动作并为它们规定初始模型。两者共享“长期未检验的后果值得再试”的动机，具体更新与覆盖机制不同。[原书 §8.3，第 168 页及脚注 1](http://incompleteideas.net/book/the-book-2nd.html)。这里也没有同时发现子任务、学习任意 option 或复现完整 STOMP/Dyna。

近期模型加复查在 $t=15,30,45$ 真实看到 A 通向 Y；三次失败都计入成本。重开后，它在 $t=58$ 从 S 选上路，$t=60$ 才收到 A 通 X 的新观察。该记录将 $p$ 从 0 改为 1；第 $61$ 步时回到 S，下一次贪心计划于是选择上路。行动取得证据，证据改变模型，模型经过规划又改变后续行动。累计模型加复查也会再看到 X，但旧 Y 计数仍占权重，比较期末 $p=14/18\approx0.778$，未越过 $0.9$；它接下来的上路访问仍由复查规则触发。

![固定96真实交互步比较累计环境奖励，近期复查在重开条件获得25，但无变化只得30低于贪心32，永久关闸只得20低于贪心24。](../assets/crl-figures/model-change-walkthrough-return.svg)

上图保留四种实际轨迹：累计贪心 $23$、近期贪心 $24$、累计复查 $22$、近期复查 $25$。下图固定近期估计，只改变是否复查，并并列始终开放与永久关闸两个对照；横向条长使用同一奖励尺度。所有分支先支付共同 $28$ 步暖身，曲线只画随后 $96$ 步。它们是单个确定性构造中的精确结果，不是跨任务平均性能。

| 重开条件 | 实际奖励 | 失败送货 | 道路动作 + reset | 模型行备份 |
| --- | --- | --- | --- | --- |
| 累计 · 贪心 | 23 | 2 | 71 + 25 = 96 | 130 |
| 近期 · 贪心 | 24 | 1 | 71 + 25 = 96 | 130 |
| 累计 · 复查 | 22 | 4 | 70 + 26 = 96 | 135 |
| 近期 · 复查 | 25 | 3 | 69 + 27 = 96 | 140 |

固定交互预算没有固定计算量：路线越短，越常回到 S，五行规划也调用得越频繁。表中每次道路动作对应一次模型统计更新；reset 只改变真实位置和时间。模型内存固定为五行、各后果计数及最后访问时刻，不保存完整训练轨迹；JSON 中的逐步轨迹属于只读教学记录。各分支在预算末端停在不同位置：近期复查刚到终点，其余重开分支还有未完成的送货。统计奖励只计已经发生的事件，既不补出未来成功，也不虚构最后一次重置。

复查没有固定方向的收益。门始终开放时，近期贪心得 32，近期复查因为定期绕下路只得 30；门关闭后不再打开时，两者分别得 24 与 20，复查增加了失败。重开条件中，累计复查虽然收集了更多更新证据，奖励 22 仍低于累计贪心的 23。新鲜度、当前预测误差和固定预算收益因此要分别观察：证据要足以改变决策，改变后的决策还要来得及偿还取得证据的代价。

还能做一个更强的辨析：对两个纯贪心控制器，把环境从“第 $48$ 步重开”改成“永不重开”，它们收到的全部动作、奖励和地点序列逐项相同，因为它们不再访问 A。仅处理这份历史的算法无法知道自己身处哪一个世界。增加旧模型规划次数也不会打破这种等价；需要一个能把行为带回相关位置的机制。年龄复查只是一种给定方案，阈值与任务代价的关系仍须另作比较。

独立脚本 [model-change-walkthrough.py](../tutorials/model-change-walkthrough.py) 只用 Python 标准库，以 Fraction 保存概率与回报，通过枚举经验模型的有限终点路径计算路线值；网页模块用逆序 Bellman 行备份，两种推导逐决策互查。下载单文件到空目录即可运行，输出三个环境条件、四个控制器的资源账与精确断言；加 --json 输出实际动作、模型版本与估计轨迹。网页图的全部数组另见 [数据 JSON](https://yingwen.io/crl-figures/model-change-walkthrough-data.json)。

固定 28 + 96 交互步；无第三方依赖、随机种子或训练任务。Python 枚举操作量不等于网页规划器的五行备份计数。

```sh
python3 tutorials/model-change-walkthrough.py
python3 tutorials/model-change-walkthrough.py --json
```

刚才的模型每次只预测一个道路动作的后果。若把整段行为当成技能来比较，就需要合并其中各步奖励，并保留停止位置及到达它所花时间的影响。技能策略和停止规则决定这些量的分布，因此也必须明确。接下来先固定它们，再从技能的回报分解推导奖励模型和折扣终点模型。

<a id="lesson-derive"></a>

## 2. Reward model 与折扣终点模型从哪里来

先固定一个 option 和一个后续价值函数 $V$。我们要计算的是：从 $s$ 执行这个技能，取得沿途奖励，再从停止位置按 $V$ 评价后续行为。把原始步的回报在技能停止时拆开，便得到下式。此处固定 $V$ 只为说明模型应保留什么；规划随后会不断改变 $V$。

$$
Q(s,o)=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}+\gamma^\tau V(S_{t+\tau})\mid S_t=s\right]
$$

先执行指定 option，再按下游策略继续。其内奖励的和与终点的后续价值可以分别建模，因为期望具有线性。

$$
\begin{aligned}r_o(s)&=\mathbb E_o\!\left[\sum_{k=0}^{\tau-1}\gamma^kR_{t+k+1}\mid s\right]\\ p_o^\gamma(j|s)&=\mathbb E_o\!\left[\gamma^\tau\mathbf1\{S_{t+\tau}=j\}\mid s\right]\\ Q(s,o)&=r_o(s)+\sum_jp_o^\gamma(j|s)V(j)\end{aligned}
$$

$p_o^γ$ 已把时间与终点联合加权。它不是普通归一化转移概率，行和是 $E[γ^τ]$。规划时不能再乘一次 γ，也不能把行重新归一化到 1。

第二个等号还用到了恒等式 $V(S_{t+\tau})=\sum_j\mathbf1\{S_{t+\tau}=j\}V(j)$。先把每个终点的指示乘上实际 $\gamma^\tau$，再取期望；没有假设时长、终点和奖励相互独立。模型保留 $r_o,p_o^\gamma$ 后，可以回答新的 $V$；只存当前 $Q(s,o)$，则已把后续行为压进一个数，无法这样重新组合。

本节把环境终止状态也存入终点向量，并规定 $V(\mathrm{terminal})=0$。预测知识与技能章则只存仍可决策的终点，在环境终止时输出零向量。这两种存储方式给出的备份相同：$p_{\rm terminal}^\gamma\,0$ 可以省去。对应的期望特征模型也应令终止特征为零，或在构造标签时屏蔽这部分；否则两种实现就不再等价。

完整执行可直接提供监督：观察奖励 $r_1,\ldots,r_\tau$ 后计算折扣和，为终点构造 one-hot 向量并乘 $\gamma^\tau$，再分别拟合条件均值。这不需要 bootstrap，但须等技能停止；长技能的估计方差和更新延迟都可能增大。下面用一步递推来学习相同对象。

考虑刚到 $s'$：若 option 停止，后续内部奖励为零，终点就是 $s'$；若继续，剩余奖励由 $r_o(s')$ 描述，剩余折扣终点由 $p_o^\gamma(\cdot|s')$ 描述。真实环境终止也必须停止技能。令 $D\in\{0,1\}$ 为真实终止标志，$\widetilde\beta=D+(1-D)\beta_o(S')$ 为合并后的停止概率。这里的 $\tau$ 取技能停止与真实终止中先发生的时间；采样窗口截断不算真实终止。

$$
\begin{aligned}r_o(s)&=\mathbb E_{\pi_o}[R+\gamma(1-\widetilde\beta)r_o(S')\mid s]\\ p_o^\gamma(j|s)&=\mathbb E_{\pi_o}\!\left[\gamma\{\widetilde\beta\mathbf1_{S'=j}+(1-\widetilde\beta)p_o^\gamma(j|S')\}\mid s\right]\end{aligned}
$$

两条分支都已经过一个原始步，所以终点项与继续项都乘折扣。奖励模型的停止分支没有额外奖励。这里保留显式终止状态，并固定其后续价值为零；终点模型仍记录到达它的折扣质量。

逐项对照 GVF 的统一递推 $v_q(s)=\mathbb E[c_q+\Gamma_qv_q(S')\mid s]$，就能看出如何学习这两个模型。$\Gamma_q$ 指定预测的延续方式；奖励头取 $c_q=R$，第 $j$ 个终点头取 $c_q=\gamma\widetilde\beta\mathbf1\{S'=j\}$，两者都用 $\Gamma_q=\gamma(1-\widetilde\beta)$ 和目标策略 $\pi_o$。奖励头逐步累计外部奖励，终点头只在停止分支累计对应终点的折扣质量；规划再把各终点预测乘以当前 V 并求和。这样，每个 GVF 的问题由前面的回报分解确定。终点 cumulant 中的这一次 $\gamma$ 不能省略：第一步就停止时，终点核应为 $\gamma\mathbf1_j$，而不是 $\mathbf1_j$。

$$
\begin{aligned}\widetilde\beta'&=d+(1-d)\beta_o(s')\\ \delta_r&=R+\gamma(1-\widetilde\beta')\hat r_o(s')-\hat r_o(s)\\ \delta_{p,j}&=\gamma[\widetilde\beta'\mathbf1_{s'=j}+(1-\widetilde\beta')\hat p_o^\gamma(j|s')]-\hat p_o^\gamma(j|s)\\ \hat r_o(s)&\leftarrow\hat r_o(s)+\alpha_r\rho_o\delta_r\\ \hat p_o^\gamma(j|s)&\leftarrow\hat p_o^\gamma(j|s)+\alpha_p\rho_o\delta_{p,j}\end{aligned}
$$

d 是本次经验中的真实终止标志。状态值式模型对当前动作取 $π_o$ 平均，off-policy 时 $ρ_o=π_o(a|s)/b(a|s)$；真实执行该 option 时 ρ=1。所有右侧使用旧参数；神经网络版本对 target 停止梯度，并对当前输出的梯度做半梯度更新。

方程右侧需要对动作和环境后果求期望。实际一步只给出 $(s,a,R,s',d)$，于是用这一个后果构造标签，用标签减当前预测形成误差。若动作来自行为策略 $b$，在 $b(a|s)>0$ 覆盖所有 $\pi_o(a|s)>0$ 的动作时，有

$$
\mathbb E_b[\rho_o Y\mid s]=\sum_a b(a|s)\frac{\pi_o(a|s)}{b(a|s)}\mathbb E[Y\mid s,a]=\mathbb E_{\pi_o}[Y\mid s]
$$

$Y$ 是上一式任一输出的单步标签，标签中的模型参数暂时固定。这个等式只把当前状态下的动作平均换成目标技能的平均；后续步骤由当前自举模型表示，模型误差仍会进入标签。它也没有修正行为策略访问各状态的频率。表格逐状态更新与共享参数的函数逼近因此仍要分别分析。

**算法：算法伪代码**

1. 固定 option $π_o$、$β_o$ 和表示版本；初始化 reward/end-state 模型
1. 每次看到原始 transition s,a,r,s′,d：
  1. 缓存旧 $r_o(s),r_o(s^{\prime}),p_o(s),p_o(s^{\prime})$
  1. 读取到达 s′ 后的 β；计算记录动作时的 $π_o(a|s)/b(a|s)$
  1. `stop = d + (1-d)*beta`
  1. `reward_target = r + gamma*(1-stop)*old_reward_next`
  1. `endpoint_target = gamma*(stop*one_hot(next_state) + (1-stop)*old_endpoint_next)`
  1. 分别更新 reward 和 endpoint 输出
1. 规划调用时：$r_o(s)$ + `dot(p_o(s), 当前 V)`；终止状态的 V 固定为零
1. 若 option/表示改变，标记模型过期并重新采样或持续更新

终止反例：本步奖励为 $1$、$\gamma=0.9$、技能本身的 $\beta_o(s')=0$，旧奖励模型在终止状态误估为 $5$。遗漏 $d$ 会得到目标 $1+0.9\times5=5.5$；正确目标为 $1$。即使控制价值在终止状态固定为零，也不能替代奖励模型自己的停止屏蔽。

同一条真实转移可以供多个技能的模型使用，但每个技能都要读取自己的 $\pi_o,\beta_o$，构造自己的标签和重要性比率。若用资格迹，延续也要按各自 $\Gamma_q$ 衰减；不能把主任务的一条迹原样共享给所有问题。共享编码器时，各头的旧预测和梯度应先在声明的同一参数版本下求出，再合成更新；某个头先改了表示，后续头再算 target，得到的就是另一种顺序算法。[在线信用章](credit.md#lesson-derive)解释迹保存的历史依赖，[流式章](streaming.md#lesson-components)讨论这些预测如何共享有限预算。

<a id="experiment-option_model"></a>

### 实验：为什么终点模型的一行不应归一化到1

同一个终点在不同时间到达，会怎样改变规划所需的模型？逐步TD与等整段完成的MC在学同一个对象吗？

**环境与可用信息。** 七格链，状态可观测，左右动作在边界截断。固定option以0.8概率右移，以0.2概率左移；每到达一个非终点状态以0.3概率停止。到6奖励1且强制停止，其他步奖励−0.02。每个option停止后从0—5均匀重新启动，因此这是模型识别实验，不是无重置持续控制。

**设置。** 5个种子，各1200个原始转移，γ=0.9、模型步长0.15，模型零初始化。TD每个原始步更新启动于当前状态的模型；MC等待完整执行，只更新本次启动状态。预算尾部的未完成option不作为MC完整样本。

**检验的机制。** 同时学习 $R_o(s)$ 与 $M_o(s,j)=\mathbb E[\gamma^\tau\mathbf1\{S_{t+\tau}=j\}]$。后者的行和是 $\mathbb E[\gamma^\tau]$，保留时间与终点的关联；归一化或再乘一次γ会改变规划语义。TD通过“现在停止／继续执行”的混合备份传播这两个对象。

**测量。** 图为六个起点的奖励模型及七维折扣终点模型的联合RMSE；真值由固定策略精确求解。另记录完成option数与起点0的终点行质量。

```bash
python3 implementations/extended_knowledge/option_model.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](../assets/crl-figures/result-option_model.svg)

横轴：environment_steps。纵轴：冻结reward/discounted-endpoint联合RMSE。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 六个起点各有一个奖励模型和七个折扣终点权重，共48个分量。与固定 option 模型的数值固定点比较，合并平方误差、除以48再开方。

**step：怎样计时。** step 是真实原始转移数；完成的 option 数另存 completed_options。这里学习模型，不执行任务策略改善。

**怎样汇总。** 取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** CSV 可重算这项联合误差的跨种子汇总。仅有 value 不能拆回奖励误差与终点权重误差；两者的尺度也会影响联合分数。

计算位置：[extended_knowledge/option_model.py](../implementations/extended_knowledge/option_model.py) · [extended_knowledge/option_model_monte_carlo.py](../implementations/extended_knowledge/option_model_monte_carlo.py) · [extended_knowledge/_common.py](../implementations/extended_knowledge/_common.py)

</details>

**结果分析。** 1200步TD联合RMSE为0.0308±0.0168，MC为0.0775±0.0115。这里TD每步都能更新，MC只在完成时更新起点；差异同时包含更新频率和bootstrap，不是单独比较一个无偏与有偏估计量。

**结论边界。** 技能策略与终止固定，模型表格化。没有非线性模型、选项变化或模型驱动控制；联合RMSE还把不同分量混在一起，需要查看分量误差。

**继续实验。** 将两个可能终点的时长改为不同分布，比较联合折扣质量与“平均折扣×终点概率”。再把M行归一化用于规划，手算一个后继价值向量，找出产生偏差的项。

[源码](../implementations/extended_knowledge/option_model.py) · [逐种子记录](https://yingwen.io/crl-code/results/option_model/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/option_model/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/option_model/curves.json)

<a id="lesson-average-model"></a>

### 2.1 平均奖励规划为何需要第三个预测头

折扣核通过 $\gamma^\tau$ 记录了时间的价值；平均奖励需要显式问“这个技能占用了多少真实时间”。考虑持续环境、固定技能，从每个建模起点最终停止，且 $\mathbb E[\tau]<\infty$、奖励有界。令 $\bar r_o$ 预测未折扣奖励总和，$L_o$ 预测原始步数，$P_o$ 预测普通终点分布。将轨迹仍拆为刚完成的一步与尚未完成的部分，得到：

$$
\begin{aligned}
     \bar r_o(s)&=\mathbb E_{\pi_o}[R+(1-\beta_o(S'))\bar r_o(S')\mid s],\\
     L_o(s)&=\mathbb E_{\pi_o}[1+(1-\beta_o(S'))L_o(S')\mid s],\\
     P_o(j|s)&=\mathbb E_{\pi_o}[\beta_o(S')\mathbf1_{S'=j}+(1-\beta_o(S'))P_o(j|S')\mid s].
   \end{aligned}
$$

三个 GVF 分别累积 $R$、$1$ 和 $\beta_o(S')\mathbf1_j$，共同延续为 $1-\beta_o(S')$。时长头在停止的最后一步仍加 1。用一条后果替代条件期望，就得到各头的 TD 标签；模型增量仍为 $\alpha\rho_o(y-\hat m)$。这些递推对应 [Wan 等，§5](https://arxiv.org/html/2110.13855)，固定问题的表格收敛还要求其覆盖与步长条件。

$$
\mathcal B_o^{g}h(s)=\bar r_o(s)-gL_o(s)+\sum_jP_o(j|s)h(j)
       =\mathbb E_o[R_{\rm sum}-g\tau+h(S_{t+\tau})\mid s]
$$

本次调用固定奖励率 $g$ 与差分价值 $h$。$g$ 的单位为奖励/原始步；$L_o$ 以原始步计，三个加项因此都有奖励单位。不要求奖励、时长和终点相互独立。只把折扣模型的 $\gamma$ 改成 1 会得到其中两个对象，却不会自动补出时间头。

为什么分别存 $\bar r_o$ 和 $L_o$，而不只学一个“平均奖励下的技能值”？如果 $g$ 从 0.4 改到 0.6，长度为 5 的技能即使后果完全没变，中心化奖励也会降低 1。分开的两个头可以立即组合出这个变化；若只拟合 $R_{\rm sum}-g\tau$，其预测目标会随奖励率估计一起漂移。环境奖励、动力学和技能固定时，$g,h$ 的改变只要求重新规划，不要求重新采集模型标签。

单位也会改变偏好：两个总是回到同一决策状态的技能，分别获得 2 奖励/1 原始步和 3 奖励/2 原始步。按每次调用的奖励会选后者；按真实时间的奖励率应选前者。取正确基准 $g=2,h=0$，两个备份分数是 $0$ 与 $-1$。若时长随机，长期比较用总奖励与总时长之比，而不是逐次比值 $R_{\rm sum}/\tau$ 的平均。

三个预测问题不要求采用同一种估计器。本章后面的单文件交互例让奖励头逐原始步做 TD(λ)，时长与终点头则在技能完成时取近期完整后果的均值；旧的 integrated_agents/_system.py 原型把三个头都放在完成时更新。两者都不是这里逐步三头 TD 的完整实现。选择取决于延迟、方差和内存预算；[平均奖励章](average.md#lesson-duration)另行讨论奖励率和差分价值如何学习。

<a id="lesson-expectation"></a>

## 3. 期望模型为什么有时够用，有时必错

![抛物线上两个等概率后果的价值均为一，平均后果零的价值却为零。](../assets/crl-figures/concept-research-model-expectation.svg)

绿色线段连接两个真实后果的价值；橙点表示把平均后果送入同一个价值函数。两种运算给出不同答案。数值由 research-mechanisms.mjs 精确计算；下文给出线性情形的等价条件。

状态很多时，显式存每个终点的概率昂贵。若下游价值对特征线性，就可以把求和移入特征期望，预测一个固定维度的向量 $m_o(s)$。这不是近似技巧，而是给定前提下的恒等式；但前提本身非常重要。

$$
\begin{aligned}m_o(s)&=\mathbb E_o[\gamma^\tau\phi(S_{t+\tau})\mid s]\\ \mathbb E_o[\gamma^\tau V_w(S_{t+\tau})\mid s]&=w^\top m_o(s)\\ y_m&=\gamma[\widetilde\beta'\phi(s')+(1-\widetilde\beta')\hat m_o(s')]\end{aligned}
$$

m 的 TD 递推只需把 one-hot 终点替换为特征，并沿用真实终止与技能停止的合并概率。状态可以高维，但下游 V 必须对这套特征线性，才能把期望与价值计算交换。

若采用终止状态零特征的约定，令 $\phi(s')=0$ 当 $d=1$。目标可等价写为 $y_m=\gamma(1-d)[\beta_o(s')\phi(s')+(1-\beta_o(s'))\hat m_o(s')]$。这时真实终止的目标向量为零；它与显式终止状态的 one-hot 约定输出不同，但在终止后价值为零的规划中一致。

反例：$X$ 以相同概率为 $-1$ 或 $+1$，$V(x)=x^2$。均值模型给 $\mathbb E[X]=0$，代入价值得到 $0$，而真正的 $\mathbb E[V(X)]=1$。只保存均值会丢掉方差信息。扩展特征为 $(x,x^2)$、学习完整分布或采样多个后果，分别提供了恢复所需信息的途径。

控制还包含最大值运算。即使每个动作的 $Q$ 对特征线性，$V(s)=\max_aQ(s,a)$ 通常仍非线性，所以 $\mathbb E[\max_aQ(S',a)]$ 不等于 $\max_aQ(\mathbb E[S'],a)$。采用线性状态价值 $V_w$、再在当前状态比较不同动作模型，是另一种规划接口；预测中的线性等价不能直接推广到任意 Q 控制。

$$
y(s)=\max_{o\in\mathcal O(s)}[\hat r_o(s)+w^\top\hat m_o(s)],\qquad
      w^+=w+\alpha[y(s)-w^\top\phi(s)]\phi(s)
$$

这里先对每个当前候选的后果取期望，再在候选之间比较；固定本次 $w$ 构造标签并停止其梯度。这是线性状态价值的规划更新，不是把平均后继当作一个实际状态继续展开任意非线性网络。[Expectation Models for Control，§2–3](https://arxiv.org/html/2104.08543)明确讨论这种接口。

还要区分两个“线性”。上述等价要求价值 $V_w=w^\top\phi$ 对特征线性；产生 $\hat m_o$ 的模型则可以是输入的非线性函数，因为真实的条件均值未必对起点特征线性。这个恒等式也没有消除模型容量不足、状态混叠或规划查询超出训练范围造成的误差。尤其当 $\phi_\xi$ 继续学习时，同一终点的特征会改变，模型拟合的标签也随之改变；若模型输入同样使用这些特征，起点的描述也在变化。此时新 $w,\phi$ 要与模型使用同一套坐标，可以重新编码保存的数据并更新模型，或在一段学习与规划中冻结共同的表示版本。

随机时间与终点也可能相关。假设一半概率走一步到价值 $10$ 的 A，另一半概率走三步到价值 $0$ 的 B，$\gamma=0.9$。正确贡献是 $0.5\times0.9\times10=4.5$；若把 $\mathbb E[\gamma^\tau]=0.8145$ 与平均终点价值 $5$ 相乘，得到 $4.0725$。用平均时长 $2$ 得到的 $\gamma^2\times5=4.05$ 也不同。需要保存的是时间与终点的联合折扣后果。

<a id="course-model-fit-versus-query"></a>

## 线性模型的两种分布：经验决定学到什么，规划决定先算什么

期望模型在线性价值下足以计算一次期望备份，并不意味着任意学到的模型都正确。Linear Dyna 还有一个容易混淆的结论：固定线性模型后，充分覆盖的规划起点分布可以不改变其固定点。这个结论不表示训练模型时的数据分布无关。下面把两层分布拆开。

冻结策略和特征，令 $x=\phi(s)\in\mathbb R^d$，模型为 $\widehat x'=Fx$、$\widehat r=b^\top x$，价值为 $v_w=w^\top x$。从固定数据集 $\mathcal D=\{(x_k,r_k,x'_k)\}$ 分别最小二乘拟合下一特征与奖励。设 $C_{\mathcal D}=\sum_kx_kx_k^\top$ 可逆。

$$
D_{\mathcal D}=\sum_kx_k{x'_k}^\top,\quad z_{\mathcal D}=\sum_kx_kr_k,\quad F^\top=C_{\mathcal D}^{-1}D_{\mathcal D},\quad b=C_{\mathcal D}^{-1}z_{\mathcal D}.
$$

这是普通最小二乘的正规方程。F 是下一特征的线性投影，并不要求真实下一特征一定由线性动力学生成。输入分布改变时，投影一般也改变。

$$
\delta_{\rm model}(x)=b^\top x+\gamma w^\top Fx-w^\top x,\qquad \mathbb E_{x\sim\mu}[\delta_{\rm model}(x)x]=C_\mu[b+\gamma F^\top w-w].
$$

$\mu$ 是规划时自行选择的起点分布，$C_\mu=\mathbb E_\mu[xx^\top]$。若它满秩，且 $I-\gamma F^\top$ 可逆，零期望更新的候选解是 $w=(I-\gamma F^\top)^{-1}b$，不依赖 $\mu$。

$$
(C_{\mathcal D}-\gamma D_{\mathcal D})w=z_{\mathcal D}.
$$

将最小二乘模型代回候选解，恰好得到同一固定数据集的 LSTD 方程。这就是模型学习与 TD 固定点的联系。数据集换了，三个经验矩阵会改变，所得价值也可以改变。

一维手算：模型训练数据只有两条，均为 x=1、r=1，而下一特征分别为 0 与 1。因此 F=1/2、b=1。取 γ=0.8，固定点 w=1/(1−0.4)=5/3。规划只查询 x=1，或以相同概率查询 x=−2 与 x=2，都会给出同一候选固定点；后者的二阶矩是前者四倍，更新速度与稳定步长却不同。若模型训练数据改为下一特征恒为 1，F 变为 1，固定点变为 5。

固定点存在、迭代收敛、预测真实未来，是三个命题。原论文对线性 TD 规划给出了模型稳定性与采样/步长条件；不能只检查矩阵可逆就宣称 TD 迭代稳定。例如一维 F=2、γ=0.9、b=1 有代数解 w=−1.25，但 TD 的平均增量为 α(1+0.8w)，会把偏离该解的误差继续放大。

深度模型通常既非全局线性，也不让所有规划查询共享一个可精确满足的残差为零方程。此时查询分布改变可能改变近似固定点，模型拟合分布外的查询还可能产生错误后果。表示更新会同时改变模型输入、模型输出和价值坐标；增加规划次数可能只是更快解出一个过时模型的问题。

因此规划实验至少拆成两项。第一项冻结同一个模型，只改变查询分布，检查数值收敛和计算效率。第二项固定查询规则，只改变模型训练数据或遗忘率，检查模型偏差与适应。把两项同时改变后只画收益曲线，无法判断改进来自更好知识还是更好的计算分配。

<a id="rlss-linear-planning-stability"></a>

## 同一个线性模型，TD 规划与残差下降为何不同

冻结策略及模型，令下一特征为 $Fx$、奖励为 $b^\top x$，值为 $w^\top x$。模型残差 $\delta(x)=b^\top x+\gamma w^\top Fx-w^\top x$ 对参数是线性的。TD 只沿当前特征更新，残差下降则同时对当前值与下一值求导。二者有相同的可精确满足方程，但没有相同的稳定性条件。

$$
\Delta w_{\rm TD}=\alpha\delta(x)x,\qquad
\Delta w_{\rm RG}=\alpha\delta(x)(x-\gamma Fx).
$$

第二式是最小化固定模型残差平方的一步梯度下降。这里 Fx 是模型提供的确定期望，不需要从同一随机转移复制两个独立样本；它与直接对环境单样本 TD-error 平方求梯度的问题不同。

$$
A=I-\gamma F^\top,\quad C=\mathbb E[xx^\top]\succ0,\quad
L(w)=\tfrac12(b-Aw)^\top C(b-Aw),\quad \nabla^2L=A^\top CA.
$$

若 A 可逆，Hessian 正定，残差目标有唯一极小点 A⁻¹b。确定全梯度下降还要选择合适步长；随机迭代还要样本和步长条件。存在唯一极小点不等于任意常数步长都会收敛。

一维取 $F=2,\gamma=.9,b=1,x=1$。代数解为 $w^*=-1.25$。TD 的误差满足 $e^+=(1+.8\alpha)e$，对正步长会放大。残差下降为 $e^+=(1-.64\alpha)e$，在 $0<\alpha<3.125$ 时收缩。但这个代数解并不是“不稳定模型中无限奖励和”的证明：模型本身的长期语义仍需单独检验。

还要避免把谱半径、矩阵范数与二次型混成一个条件。原线性 Dyna 收敛论文使用其明确的数值半径和随机逼近假设。课件中的口头化标题不能替代原定理。仅检查特征值位于单位圆内，不能保证对任意规划起点协方差都稳定。

$$
F=\begin{pmatrix}0&3\\0&0\end{pmatrix},\quad
\rho(F)=0,\qquad
\max_{\|x\|_2=1}x^\top Fx=\tfrac32.
$$

F 的两步幂已为零，但一个方向上的瞬时二次型可以大于一。这是区分两种矩阵量的反例，不是单凭它就宣布每个采样分布下 TD 都发散。

接回持续学习，固定模型定理只是内层参照。模型学习、策略改善和表示变化会让 F、b、C 同时移动。分别冻结这三者做测试，才能区分求解器不稳定、模型不正确和表示已过期。增加规划次数可能放大其中任何一个问题。

<a id="lesson-successors"></a>

## 4. Successor features 与 GPI：不重学动力学地换奖励

深入阅读：动力学与策略固定而奖励权重变化时，展开 successor features 的复用及 GPI。

另一个复用问题是：环境动力学和被评价的目标策略不变，奖励的偏好改变。采集经验的行为策略可以不同，但估计仍需满足相应覆盖和校正条件。比如同一路线会产生“耗时、电量、采到的资源”三个信号；早上偏好快，晚上偏好省电。如果奖励是这些信号的线性组合，就能先预测整段未来信号，再乘新的偏好权重。这是 SF 的基本分解，不是预测“下一状态均值”。

$$
\begin{aligned}r_w(s,a,s')&=\phi(s,a,s')^\top w\\ \psi^\pi(s,a)&=\mathbb E_\pi\!\left[\sum_{k=0}^\infty\gamma^k\phi(S_{t+k},A_{t+k},S_{t+k+1})\mid S_t=s,A_t=a\right]\\ Q_w^\pi(s,a)&=\psi^\pi(s,a)^\top w\end{aligned}
$$

φ 是每步可观察的特征信号，ψ 是同一目标策略下的向量预测；每个分量都是一个价值预测问题。等式需要同样的动力学、策略、折扣与特征语义，奖励变化仅发生在 w。

将累计和拆成第一步与后续，就得到向量 Bellman 方程：标量奖励换为向量 $\phi$，标量价值换为向量 $\psi$。下一动作按同一个目标策略平均或采样。若对每个特征分量分别取最大，不同分量可能对应彼此冲突的行为，就不再是这个策略的未来特征预测。

$$
\begin{aligned}y_\psi&=\phi(s,a,s')+\gamma(1-d)\sum_{a'}\pi(a'|s')\hat\psi^\pi(s',a')\\ \hat\psi^\pi(s,a)&\leftarrow\hat\psi^\pi(s,a)+\alpha[y_\psi-\hat\psi^\pi(s,a)]\end{aligned}
$$

这是动作条件的表格 expected-TD：当前动作已作为条件给定，环境转移的样本不需要再乘当前动作概率比；后续动作明确按 π 求平均。如果改成状态式 ψ(s) 预测，当前动作平均和 off-policy 修正就必须相应改变。

只有一个旧策略时，重新加权得到的是该策略的新价值。为了改善控制，可保存多个 $\pi_i$ 的 SF，先计算每个候选在新奖励下的价值，再在每个状态选择候选中评价最高的动作，这就是 generalized policy improvement。它可以逐状态组合动作，而非只在 episode 开头选一个旧策略并始终照做。

$$
\pi_{\rm GPI}(s)\in\arg\max_a\max_i\left[\hat\psi^{\pi_i}(s,a)^\top w_{\rm new}\right]
$$

顺序是先沿特征维度与 w 做点积，再沿策略维度取 max，最后沿动作维度选 argmax。把两个 max 或特征求和维度弄错，会产生另一种算法。

**算法：算法伪代码**

1. 准备固定策略集合 $π_1,\ldots,π_n$ 与各自 SF 估计
1. 每个观察 transition：
  1. 对每个 i，用 $π_i$ 的下一动作分布构造向量 TD target，更新 $ψ_i$
1. 若奖励权重变为 $w_{new}$：
  1. 不改 SF，先计算 $Q_i(s,a)=ψ_i(s,a)^T w_{new}$
  1. $\mathrm{score}(a)=\max_i Q_i(s,a)$
  1. 执行 $\arg\max_a\mathrm{score}(a)$，并继续收集真实数据
1. 如果学习了新的专门策略，把它连同其 SF 加入集合；预算有限时须选择保留项

在精确 $Q$、相同动力学和折扣下，GPI 不劣于被比较的各个策略；近似保证取决于统一价值误差界。奖励不在当前 $\phi$ 的线性张成空间，或动力学改变导致旧 $\psi$ 失效时，需要重新估计相应误差。SF 预测策略产生的累计特征，option 定义执行与停止，option model 预测停止时后果；三者可以组合，但承担不同职责。

SF 例子：两个动作分别产生特征 $(1,0)$ 和 $(0,1)$，随后进入动作编号对应的状态，$\gamma=0.9$。策略 $\pi_0$ 始终选动作 $0$，$\pi_1$ 始终选动作 $1$。先做动作 $1$ 再跟 $\pi_0$，SF 为 $(9,1)$；先做动作 $0$ 再跟 $\pi_1$，SF 为 $(1,9)$。当 $w=(1,2)$ 时，GPI 对两个动作的分数为 $19,20$，故选动作 $1$；改为 $w=(2,1)$ 后，无需重学 SF 就会改选动作 $0$。这里复用的是不变动力学和不变目标策略下的预测。

<a id="experiment-successor_features_gpi"></a>

### 实验：后继特征复用的究竟是动力学还是奖励

奖励权重已知地变化时，为什么无需将所有长期预测从零重学？固定基础策略为什么可能已足够？

**环境与可用信息。** 单状态、两动作、无终止；每个动作都返回同一状态。转移特征是动作的二维one-hot向量。γ=0.8；前600步奖励权重为(0.2,1)，随后为(1,0.2)，变化后的权重直接提供给算法。行为始终均匀选动作。

**设置。** 5个种子，各1200个真实动作。两个基础策略分别总选动作0与总选动作1；所有策略—动作的SF从0起、步长0.1，更新读取同一旧副本。GPI用已学SF与给定权重选动作；对照始终使用基础策略0。

**检验的机制。** 用 $\psi^\pi(s,a)^\top w$ 估计新奖励下价值，再对库内策略与动作取最大值。策略索引不能去掉，否则会把不同未来行为的占据量混在一起。权重变化不改变转移特征的累计定义。

**测量。** 图为所选确定性策略的解析折扣价值，不是训练行为的回报。此单状态问题中，总选奖励为1的动作值为5，总选奖励为0.2的动作值为1。

```bash
python3 implementations/continual/successor_features_gpi.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/successor_features_gpi/curves.svg)

横轴：environment_steps。纵轴：冻结选择策略的解析折扣价值。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 已记录的检查点中，GPI在前后两阶段均为5；固定策略0前半段为1，变化后也为5。基线后半段变好是因为任务奖励改到它偏爱的动作，不是它开始适应。五种子策略价值一致，也不代表SF数值完全相同。

**结论边界。** 新奖励权重有特权地直接提供；没有奖励辨识、未知目标发现或表示迁移。单状态任务也不能说明GPI在长程复杂导航上的优势。

**继续实验。** 不再直接告知奖励权重，添加独立的在线奖励回归。将奖励辨识误差与SF误差分开记录，再观察变化后的延迟来自哪一个模块。

[源码](../implementations/continual/successor_features_gpi.py) · [逐种子记录](https://yingwen.io/crl-code/results/successor_features_gpi/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/successor_features_gpi/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/successor_features_gpi/curves.json)

<a id="lesson-example"></a>

## 5. 两次原始转移怎样构造完整技能模型

先核对折扣：技能从状态 $0$ 经过 $1$ 到 $2$，在 $2$ 停止，奖励分别为 $1,2$，$\gamma=0.9$。模型初值为零，步长为一。按从后往前的顺序做一步更新，可以在两次 backup 中算出结果；若只按在线顺序做一步 TD、不用迹，第一步尚读不到后方的新信息。资格迹或完整回报更新提供了不必再次访问的另一条路径。

| 更新 | reward target | discounted endpoint target |
| --- | --- | --- |
| 1→2，β(2)=1 | 2 + 0 = 2 | 0.9 × one-hot(2) |
| 0→1，β(1)=0 | 1 + 0.9×2 = 2.8 | 0.9 × [0,0,0.9] = [0,0,0.81] |
| 给 V(2)=10 做一次规划 | 2.8 | 0.81×10 = 8.1，合计 10.9 |

这个 backup 与直接计算两步回报相同，无需重新模拟技能内的动作。若把终点向量错误归一化为 $[0,0,1]$，会得到 $12.8$；若额外再乘 $\gamma$，会得到 $10.09$。两个错误分别丢掉或重复计算了时间折扣。

现在放回真实交互。七个状态组成环，向左或向右一步有 0.1 概率原地停滞；到达状态 1 得 $0.98$，其余一步得 $-0.02$，环境不重置。给定两个固定技能：L 向左到 4，R 向右到 1；均在到达目标或执行两步时停止，启动后至少做一步。为了使“两步上限”后的剩余后果明确，奖励 GVF 的坐标为 $(o,n,s)$，其中 $n\in\{2,1\}$ 是剩余原始步预算。它预测技能停止前的未折扣外部奖励，规划读取 $w_{o,2,s}$。

单文件例的第 4、5 个真实步执行 R：$6\to0\to1$，两个相关预测初值为零，$\alpha=0.25,\lambda=0.8$。第一步误差为 $-0.02$，起点权重变成 $-0.005$。第二步的到达使技能停止，bootstrap 变成零，但起点的活跃资格迹仍为 $0.8$；误差 $0.98$ 将起点再改动 $0.25\times0.8\times0.98=0.196$。所以规划读取的奖励模型是 $0.191$，不是本次完整回报 $0.96$：有限步长的 TD 估计并不等于单次标签。最后一次误差用完迹，才清空迹。

![真实技能轨迹6到0到1，第二步奖励通过起点资格迹0.8把奖励预测从负0.005更新到0.191。](../assets/crl-figures/agent-cycle-credit.svg)

图取自 agent_cycle_walkthrough.py 的同一条真实执行。信用更新写入的是规划稍后读取的奖励模型；终点技能停止不表示持续环境重置。

| 模型输出 | 由哪种真实证据更新 | 完成这次技能之后 |
| --- | --- | --- |
| 奖励总和预测 | 每个原始步的 TD(λ)；迹给之前的预测分配信用 | Rhat(6,R) = 0.191 |
| 期望时长 | 每个起点—技能最近至多 4 次完整时长的均值 | Lhat(6,R) = 2 |
| 终点分布 | 同一组完整后果的终点频率 | Phat(1\|6,R) = 1 |

一个完整后果还不足以学准随机动力学；这里的终点概率 1 只是当前经验频率。没有完整后果的模型行暂不参与规划。奖励头不重放历史样本，时长和终点却保留至多 $14\times4$ 条宏动作后果，因此这是有界模型记忆的例子，不是“完全不保存经验”。它还用下一次相同起点—技能的真实回报评价旧奖励预测并调整步长；这个元目标局限于模型预测，不是整个智能体回报的梯度。[规划章](planning.md#lesson-cycle)接着展示这些输出怎样影响下一个动作。

<a id="lesson-code"></a>

## 6. 实现与实验：期望、时间和策略条件

先下载 [agent_cycle_walkthrough.py](../tutorials/agent_cycle_walkthrough.py)，在仓库根目录执行下列命令。reward_td_step 对应逐步奖励头，Agent.model 从有界完整后果读取时长和终点，Agent.plan 才把三头交给当前价值。JSON 中 atomic、option_end、planning 三类事件分别记录证据到达、完整后果和内部计算；不是三个不同环境。

一个文件、Python 标准库；有限逐步算例与接口测试，不是性能实验

```sh
python3 tutorials/agent_cycle_walkthrough.py --steps 10 --seed 0
python3 tutorials/agent_cycle_walkthrough.py --test
```

Monte Carlo/TD option model、随机时长联合模型、SF 向量更新及 GPI；数组和更新时序都可直接检查。

```python
def option_episode_target(rewards, gamma, endpoint, n_states):
    reward_target = sum(gamma**k * r for k, r in enumerate(rewards))
    endpoint_target = [0.0] * n_states
    endpoint_target[endpoint] = gamma**len(rewards)
    return reward_target, endpoint_target


def model_td_update(reward_model, endpoint_model, state, next_state, reward,
                    beta_next, alpha=0.1, gamma=0.9, rho=1.0, terminal=False):
    """Fixed option; n is a discounted endpoint distribution, not normalized.

    True environment termination forces stopping, independently of beta.
    The explicit terminal endpoint has V=0 during planning; its discounted
    probability mass is retained. A rollout cutoff is not true termination.
    All right-hand sides use pre-update values (self-loops included).
    """
    old_reward, next_reward = reward_model[state], reward_model[next_state]
    old_row, next_row = endpoint_model[state][:], endpoint_model[next_state][:]
    stop = 1.0 if terminal else beta_next
    r_target = reward + gamma * (1.0 - stop) * next_reward
    p_target = [gamma * (stop * float(j == next_state)
                        + (1.0 - stop) * next_row[j])
                for j in range(len(old_row))]
    reward_model[state] = old_reward + alpha * rho * (r_target - old_reward)
    endpoint_model[state] = [v + alpha * rho * (target - v)
                             for v, target in zip(old_row, p_target)]
    return r_target, p_target


def model_backup(reward, discounted_endpoints, values):
    return reward + sum(p * v for p, v in zip(discounted_endpoints, values))


def mixed_duration_model(outcomes, gamma, n_states):
    """outcomes: (probability, rewards, endpoint). Keep time/end correlation."""
    if not math.isclose(sum(p for p, _, _ in outcomes), 1.0):
        raise ValueError("Outcome probabilities must sum to one")
    r_bar, p_bar = 0.0, [0.0] * n_states
    for probability, rewards, endpoint in outcomes:
        r, p = option_episode_target(rewards, gamma, endpoint, n_states)
        r_bar += probability * r
        p_bar = [old + probability * new for old, new in zip(p_bar, p)]
    return r_bar, p_bar


def successor_feature_step(psi, state, action, features, next_state, next_action_probs,
                           alpha=0.1, gamma=0.9, terminal=False):
    """Action-conditioned SF for a fixed target policy, with known feature signal.

    The observed action conditions the prediction, so the next action is averaged
    under the target policy; no current-action importance ratio is needed here.
    psi[s][a] is a feature vector. Snapshot before mutation handles self-loops.
    """
    old = psi[state][action][:]
    next_vectors = [row[:] for row in psi[next_state]]
    next_features = [sum(pr * vec[j] for pr, vec in zip(next_action_probs, next_vectors))
                     for j in range(len(features))]
    target = [x + (0.0 if terminal else gamma * xn)
              for x, xn in zip(features, next_features)]
    psi[state][action] = [x + alpha * (y - x) for x, y in zip(old, target)]
    return target


def gpi_action(successor_feature_bank, state, reward_weights):
    """Greedy policy improvement over a bank of evaluated fixed policies."""
    action_scores = [max(sum(x*w for x,w in zip(psi[state][a], reward_weights))
                         for psi in successor_feature_bank)
                     for a in range(len(successor_feature_bank[0][state]))]
    return max(range(len(action_scores)), key=action_scores.__getitem__), action_scores


def learn_tiny_sf():
    bank = []
    for fixed_action in (0, 1):
        psi = [[[0.0, 0.0] for _ in range(2)] for _ in range(2)]
        probabilities = [float(a == fixed_action) for a in range(2)]
        for _ in range(400):
            for s in range(2):
                for a in range(2):
                    successor_feature_step(psi, s, a, [float(a==0), float(a==1)],
                                           a, probabilities, alpha=1.0)
        bank.append(psi)
    return bank
```

无第三方依赖；所有目标都可用上面的数字核算

```sh
python3 examples/knowledge_algorithms_lab.py models
python3 examples/knowledge_algorithms_lab.py test
```

运行得到 reward model 为 $2.8$，终点向量为 $[0,0,0.81]$，backup 为 $10.9$；随机时间与终点联合贡献为 $4.5$。SF/GPI 在 $w=(1,2)$ 时选择动作 $1$，两个动作的分数约为 $[19,20]$。逐项改变价值、持续时间或奖励权重，可以观察模型究竟保留了哪一种可复用信息。

- 改动 A：只改变 endpoint value，冻结 option 与模型，验证模型可被不同价值函数重复调用。
- 改动 B：令短路径终点与长路径终点交换，验证即使平均时长和未加权终点分布没变，正确价值仍可改变。
- 改动 C：保持线性 reward 但改变动力学，冻结 SF；观察立即迁移不再正确，再比较持续更新 ψ 的恢复速度。
- 改动 D：给 critic 加非线性，分别比较均值输入、特征扩展和多样本估计；报告 target 偏差而非只看训练 loss。

<a id="lesson-reward-aware"></a>

## 7. 当环境结构也需要考虑代价：Default Representation

深入阅读：在特定 KL 控制问题中推导 Default Representation，比较奖励放在模型内部或读出端的区别。

SR 和 Laplacian 表示主要刻画在某种默认行为下哪些状态容易互相到达，但相同的图结构可以对应完全不同的实际代价。两条通道几何长度相同，一条却持续消耗大量资源：仅根据连通性构造的技能仍可能偏爱这条通道。Reward-Aware Proto-Representations（NeurIPS 2025）研究如何让用于技能与奖励塑形的结构表示反映沿途奖励。

理解 Default Representation（DR）要先限定控制问题。设有限的非终止状态集合为 $N$、有限的终止集合为 $T$，默认策略诱导转移 $P^{\pi_d}$，每个非终止奖励 $r(s)$ 都是有限的负数，终止奖励也有限。在线性可解控制中，控制器改变下一状态分布，同时为偏离默认分布付出 KL 代价，温度 $\lambda>0$ 控制偏离的价格。它不是任意标准动作 MDP 都自动具备的结构。

$$
v^*(s)=r(s)+\lambda\log\sum_{s'}P^{\pi_d}(s'|s)\exp\!\left(v^*(s')/\lambda\right)
$$

这个软 Bellman 方程来自最大化“期望后续价值减去对默认转移的 KL 代价”。默认分布没有支持的后果不能凭空控制出来；边界状态的价值由终止收益给定。

令 desirability 为 $z(s)=\exp(v^*(s)/\lambda)$，把两侧指数化。非线性的对数求和变成 $z$ 的线性递推，再把非终止状态和终止状态分开，就得到 DR。

$$
\begin{aligned}D_r&=\operatorname{diag}\!\left(\exp(r_N/\lambda)\right)\\ z_N&=D_r\left(P_{NN}^{\pi_d}z_N+P_{NT}^{\pi_d}z_T\right)\\ Z_{NN}&=\left[D_r^{-1}-P_{NN}^{\pi_d}\right]^{-1}\\ z_N&=Z_{NN}P_{NT}^{\pi_d}z_T,\qquad z_T=\exp(r_T/\lambda)\end{aligned}
$$

$Z$ 把内部动力学与沿途代价编码在一起；固定这两项而改变终点收益时，可以重用 $Z$。有限状态和严格负奖励给出 $q=\max_{s\in N}e^{r(s)/\lambda}<1$。由于 $P_{NN}^{\pi_d}$ 的行和不超过 1，$\|D_rP_{NN}^{\pi_d}\|_\infty\le q$；线性递推因而收缩，逆矩阵存在。终点收益并未混进内部模型。

矩阵可逆还不等于每个状态都有有限控制价值。由非负矩阵级数 $z_N=\sum_{k\ge0}(D_rP_{NN}^{\pi_d})^kD_rP_{NT}^{\pi_d}z_T$ 可见，要使 $z(s)>0$，还需从 $s$ 出发存在默认转移支持的有限路径到终点。若每个非终止状态都满足这项可达性，才可由 $v^*(s)=\lambda\log z(s)$ 得到处处有限的值。反例是只有自环、无法离开的状态：令 $r=-\lambda\log2$，则 $Z=(2-1)^{-1}=1$ 完全有限，但没有终点入口，$z=0$、$v^*=-\infty$。控制器不能通过一个默认概率为零的出口逃离。

DR 与 SR 的联系可直接算出。若所有非终止状态的奖励都等于 $\lambda\log\gamma$，则 $D_r=\gamma I$，因而 $Z_{NN}=\gamma(I-\gamma P_{NN}^{\pi_d})^{-1}$，即 SR 的常数倍。奖励不均匀时，每经过一个状态都受到不同的指数权重，表示便能区分低代价和高代价区域。

$$
Z=D_r+D_rPZ,\qquad Z(s,:)\leftarrow Z(s,:)+\alpha\left[e^{r(s)/\lambda}\left(\mathbf e_s+Z(s',:)\right)-Z(s,:)\right]
$$

这是在默认策略下采样、只考虑非终止行列时的递推；进入终止状态后，内部矩阵的后续行取零。若行为不同，需明确所估计的默认转移或使用相应分布修正。它近似 SR 的向量 TD，但当前状态奖励决定的缩放同时作用于 one-hot 与后续行，不能只替换 SR 的 $\gamma$。

手算一条通道：默认行为从状态 0 必然进入状态 1，再进入终点；令两个非终止状态的指数奖励权重均为 0.5。则内部矩阵第一行为 (0.5,0.25)，第二行为 (0,0.5)。若状态 0 的代价增大，使权重降为 0.25，第一行变成 (0.25,0.125)。到达状态 1 的默认路径没有变，但跨过代价区的权重下降了。这正是奖励感知结构与纯连通结构的差异。

论文将 DR 用于构造奖励感知的谱特征，再用于奖励塑形、技能发现与迁移实验。它提供的是另一种构造长期结构的准则，而非把所有模型都换成一个新矩阵。若环境奖励变化，DR 本身通常也要更新；相反，奖励线性权重变化而动力学不变时，固定策略 SF 可以保持不变。两者分别把奖励放在不同位置，因此具有不同的复用边界。

| 表示 | 固定哪些条件才能复用 | 直接改变什么 |
| --- | --- | --- |
| SR / SF | 动力学、目标策略、折扣、特征语义 | 用新奖励权重重新评价同一策略 |
| DR | 默认动力学、内部奖励、KL 控制约定 | 通过终点收益或奖励感知结构重新求解 |
| Option model | 技能策略与停止规则、动力学、奖励约定 | 用新的后续价值评价同一技能 |
| STOMP 子任务与模型 | 子任务可改变，后果模型须匹配当前技能 | 由奖励相关子任务产生行为，再预测真实后果 |

作者 Reward-Aware-Proto-Representations 仓库适合按“默认转移与奖励 → DR → 特征 → 技能/塑形 → 外部回报”阅读。实验中应分别改变内部代价和终点奖励：前者测试结构能否重新适应，后者测试已有结构能否复用。若两者同时变化，就难以解释改进来自哪一种能力。

<a id="lesson-branches"></a>

## 8. 从表格模型到潜在世界模型

深入阅读：从表格后果预测扩展到潜在世界模型，进一步处理表示、技能与环境的不同漂移。

潜在世界模型把观测历史压缩为 z，再学从 z 和动作预测下一潜在状态。一个常见训练结构包含：利用当前观测得到后验表示、仅凭过去表示和动作得到预测先验、重建或预测观测、预测 reward 与继续概率，并用一致性/KL 项约束先验后验。目标是让不接触未来真实观测的想象轨迹仍保持决策相关的信息，而不是在训练时用未来观测泄漏答案。

$$
\begin{aligned}z_t&\sim q_\theta(z_t\mid h_t,o_t),\qquad h_{t+1}=f_\theta(h_t,z_t,a_t)\\ \hat z_{t+1}&\sim p_\theta(z_{t+1}\mid h_{t+1})\\ \mathcal L_{\rm model}&=\mathcal L_{\rm obs}+\mathcal L_{\rm reward}+\mathcal L_{\rm continue}+\mathcal L_{\rm regularize}\end{aligned}
$$

这四项说明观测、奖励、继续概率和潜在先验—后验之间的分工；各项的实际权重、分布形式与梯度路径由具体算法规定。重建项使用真实观测，想象轨迹只能使用预测先验。

Dreamer 类方法在真实序列上学习模型，再从后验状态启动想象 rollout 来训练 actor/critic；部署可直接执行学到的 actor。MuZero 类方法学习对 reward、value、policy 有用的潜在递推，并在决策时做搜索，不要求重建全部像素。这两类模型“学来做什么”不同，不能以画面是否逼真作为统一分数。

在持续学习中，一个后果预测变差可以有不同原因。道路改变时，需要新经验修正环境规律；技能策略改变时，需要重新估计新行为的后果；编码器改变时，即使道路和行为都相同，旧潜在状态也可能不再适合当前模型。应分别改变这些条件，才能知道模型在跟踪什么。技能与编码器版本可帮助识别样本的含义，保存的原始经验可用于重新编码，近期数据加权可改变旧后果的影响；缩短想象跨度则减少一次规划对连续预测的依赖。这些处理分别针对不同困难，仍需检查修正后的模型是否改变了实际决策。

| 模型路线 | 特别擅长的复用 | 需要额外检查 |
| --- | --- | --- |
| Tabular option model | 技能后果被任意新 V 调用 | 持续时间、终止、技能版本 |
| Expectation model | 线性值函数改变时快速重估 | 特征充分性、非线性和 max 的交换 |
| SF / GPI | 同动力学下换奖励权重 | 目标策略固定、reward 特征可表达性 |
| Generative latent model | 多步想象与分布性后果 | 表示充分性、分布外误差、多步滚动 |
| Value-equivalent / decision-oriented model | 保留下游决策所需量 | 保证通常相对于某类策略/价值，不是全世界精确模型 |

<a id="experiment-integrated_cumulative_model"></a>

### 实验：累计样本更多，不一定更适合当前模型查询

外部奖励或技能改变后，全历史平均模型与固定步长模型会怎样影响适应？只看末尾窗口会得出相同结论吗？

**环境与可用信息。** 七状态环，观测就是当前状态0—6。原始动作左移或右移；0.1概率停在原位，否则向选定方向移动。每步外部奖励为“到达当前奖励位置的指示量−0.02”。前600步奖励位置为1，随后为4；变化时刻不传给智能体。环境不终止、不重置。子目标1和4由设计者给定；技能在到达相应目标或执行满6个原始步时停止。

**设置。** 5个种子0—4，各1200个真实环境步。高层ε=0.2；技能内ε=0.1。每个原始转移用Q-learning更新两个给定子任务，步长0.25、折扣0.95；到达子目标得1，否则−0.01。高层差分SMDP步长0.1，奖励率增量为0.002倍TD误差。技能完成后才提交内部贪心策略；预算尾端尚未完成的技能不伪造终止。 两者每个完成的高层动作后做4次规划。候选模型用1/n步长累计均值；对照第一次样本全量写入，之后固定步长0.2。两者均在内部技能策略改变时失效相应模型及高层Q。

**检验的机制。** 模型分别估计完成动作的外部总奖励、总时长和终点指示量；平均奖励规划使用“总奖励−奖励率×时长＋终点价值”。累计平均长期保留旧后果，固定步长更重视近期后果，但方差更大。子任务人工奖励不进入这个外部后果模型。

**测量。** 同时看最近100步奖励与生命期奖励率；模型奖励误差只在模型已存在且动作完成的样本上记录，不能当全状态无偏误差。

```bash
python3 implementations/integrated_agents/integrated_cumulative_model.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](../assets/crl-figures/result-integrated_cumulative_model.svg)

横轴：environment_steps。纵轴：最近100个真实步的平均外部奖励。每种方法 1200 environment_steps；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

<details>
<summary>这张曲线的 value 与 step</summary>

**value：评价什么。** 七状态环中，实际行为最近 min(step,100) 个原始转移的平均外部奖励。每步奖励为到达当前奖励位置的指示量减0.02。该窗口包括探索、技能执行和尚未完成的技能；不是冻结贪心策略的价值，也不是算法内部的 estimated_rate。

**step：怎样计时。** step 只数真实转移，奖励位置在预算中点改变，不重置世界或学习器。高层真实备份只在一个原子动作或 option 完成后进行；近期模型与累计模型每次完成各做4次模型备份，故 model_backups=4×real_backups。不规划的对照仍学习模型，但 model_backups 始终为0。每个真实步还更新两个给定子任务，subtask_backups=2×step。相同真实步数未必包含相同规划计算。

**怎样汇总。** 窗口每步滑动，但只在第1步、每20步和预算末尾记录。变化后的前99步仍可含旧阶段奖励。average_reward 是从第1步累计的实际奖励率；estimated_rate 则由每次完成后的差分误差递增，二者的定义不同。技能策略改变时，版本化方法清除对应模型和高层 Q，不清空奖励窗口。取该记录时刻的值；不先对曲线上的时间点求平均。 先在每个完整运行内计算 value，再在同一 step 上跨运行种子求均值和样本标准差（分母 n−1）。时间点不是独立重复；确定性计算即使换用种子也可能完全相同。标准差带不是置信区间，也不是单次观测的取值范围；图中的带可能越过奖励或误差的可行边界。

**从记录能重算什么。** lifetime_reward 与 average_reward×step 可互查，并可用累计量差恢复检查点之间的总奖励。step≥120的20步检查点可用相隔100步的累计量重算 value；其余稀疏窗口不一定能重建。预算末尾 pending_duration>0 时，其奖励已计入真实收益，但这段尚未产生高层备份或完整段模型样本。model_reward_mae 只平均最近至多100个“此前已有对应模型”的完整段奖励预测误差，既不是全状态误差，也没有检验时长和终点分布；模型清除会改变哪些样本进入它。

计算位置：[integrated_agents/_system.py](../implementations/integrated_agents/_system.py) · [integrated_agents/integrated_recent_model.py](../implementations/integrated_agents/integrated_recent_model.py) · [integrated_agents/integrated_cumulative_model.py](../implementations/integrated_agents/integrated_cumulative_model.py) · [integrated_agents/integrated_no_planning.py](../implementations/integrated_agents/integrated_no_planning.py)

</details>

**结果分析。** 最终累计模型末窗奖励0.364，高于近期模型0.328；但生命期奖励率为0.247，低于近期模型0.270。800步时累计模型末窗为0.098，近期为0.114。结论依赖是否重视过渡期损失，不能只选终点排名。

**结论边界。** 两条闭环轨迹与技能版本变化也不同，因此不是同一固定数据集上的纯模型回归比较。只有一次奖励位置改变，尚无多次变化或变化速度扫描。

**继续实验。** 先冻结技能并用同一日志训练两种模型，单独测奖励与时长误差。再恢复闭环行动。比较这两阶段，区分模型记忆机制与它改变数据分布后的反馈。

[源码](../implementations/integrated_agents/integrated_cumulative_model.py) · [逐种子记录](https://yingwen.io/crl-code/results/integrated_cumulative_model/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/integrated_cumulative_model/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/integrated_cumulative_model/curves.json)

<a id="rlss-one-step-and-jump-models"></a>

## 单步模型与跳跃模型：误差、数据和规划成本一起比较

单步模型可以反复组合，但每次使用都把前一步的预测作为下一步输入。跨多步直接预测后果的模型减少这种组合次数。它并不免费：更长的后果通常更晚才可观察，也依赖内部策略、停止规则以及访问覆盖。比较二者时应同时计算建模数据、估计误差和规划预算。

$$
\widehat F^k-F^k=\sum_{j=0}^{k-1}\widehat F^{k-1-j}(\widehat F-F)F^j.
$$

固定线性模型的望远镜恒等式；不要求两个矩阵可交换。每一项表示一个局部模型误差在其前后动力学中的传播。

$$
\|\widehat F^k-F^k\|\leq kL^{k-1}\varepsilon,\qquad
\|F\|,\|\widehat F\|\leq L,\quad\|\widehat F-F\|\leq\varepsilon.
$$

这是相容矩阵范数下的上界。收缩动力学可能抑制误差，扩张动力学可能放大误差。直接学 k 步模型 G 的误差 ε_k 必须由数据估计，不能假设它与单步 ε 相同后宣称普遍更优。

对 option，模型不只预测一个远端状态。它还要给出累计奖励、停止后状态或特征，以及用于主目标的时间权重。内部策略或停止规则一变，所预测的随机变量也变。更复杂的模型、更多规划次数都不能修复旧模型的语义已经失效这一事实。

$$
\|\widehat V-V\|_\infty\leq\frac{\|\widehat T V-TV\|_\infty}{1-\kappa},\qquad 0\leq\kappa<1.
$$

假设学习模型的规划算子为 κ 压缩，V 是真实算子的固定点，V̂ 是学习算子的固定点。加入并减去 T̂V，再用压缩性即可得到。更多 backup 只减小到 V̂ 的数值误差；模型偏差项仍然存在。

| 对照 | 固定什么 | 它能够回答什么 |
| --- | --- | --- |
| 精确原始模型 vs 精确 option 模型 | 真实奖励与技能，计入每种 backup 成本 | 时间抽象是否减少求解所需计算 |
| 同数据学习单步与多步模型 | 数据来源、真实步、拟合与内存预算 | 直接后果估计是否值得其统计成本 |
| 技能变化，模型不变 vs 模型跟踪 | 同一变化与执行权限 | 规划错误来自数值收敛还是后果过期 |

例如，在四房间中预先提供最短路径 options，再比较价值传播速度，可以检验技能怎样帮助规划，但不能证明技能能够自行发现。同样，若人为指定单步与多步模型的误差，再比较误差传播，得到的是算子诊断；要比较模型学习能力，还需匹配训练数据、拟合计算和查询分布。

持续任务中的关键问题不是“模型越准越好”一句话，而是哪种后果值得持续校准。未再执行的 option 缺少新证据；依靠离策略更新又需要覆盖与明确的问题定义。主动验证技能、检测模型过期与规划查询分配因此互相耦合。

<a id="research-model-query-equivalence"></a>

## 模型充分性由规划查询与风险目标共同决定

本章的期望模型已说明线性价值下哪些统计足够。Value Equivalence（NeurIPS 2020）进一步把模型规格写成查询集合：指定哪些策略与后续函数，要求模型在这些查询上生成正确 backup。模型可以舍弃不影响这些计算的细节；当规划器或奖励改变时，原先可忽略的细节也可能变成必要知识。

$$
(T_M^\pi v)(s)=\mathbb E_{M,\pi}[R+\gamma v(S')\mid s],\qquad \widehat M\equiv_{\Pi,\mathcal V}M\ \Longleftrightarrow\ T_{\widehat M}^\pi v=T_M^\pi v\quad\forall\pi\in\Pi,\ v\in\mathcal V
$$

等价针对指定策略与函数族。用一个当前 critic 拟合 targets，只检验一个有限且会变化的查询集合；VE 不是说任意小模型都足够，也不是把 reward-only 预测称为完整环境模型。

Proper Value Equivalence（NeurIPS 2021）考察多步算子与策略价值固定点，给出适当策略族下的规划充分性。Distributional Model Equivalence（NeurIPS 2023）揭示它的另一边界：保持期望值不足以保持风险敏感决策。后者需保留回报分布或与指定风险目标相容的统计摘要。

手算反例：动作 A 确定获得 1；动作 B 以各半概率获得 −9 或 11；两者期望均为 1。只保持均值的模型可以把它们当作同一动作，但最差一半的平均回报分别为 1 和 −9。若目标由最大期望改为最大 lower-tail CVaR，旧模型无法回答新问题。这个反例不需要非平稳环境，改变查询规格本身就能造成模型不足。

$$
\mathcal S(\nu)=(\mathbb E_\nu[G],\mathbb E_\nu[G^2],\ldots),\qquad \mathcal S(\mathcal T^\pi\eta)=\mathcal T_{\mathcal S}^\pi\mathcal S(\eta)
$$

第二式表达摘要的 Bellman 闭合要求：更新后的统计应能由已有统计正确计算。完整分布保留更多信息；有限矩、分位点与投影各有局限，均值加方差通常不能识别任意尾部风险。不要把任意摘要都默认为满足这条闭合式。

**算法：模型 loss、查询误差和决策结果是三类证据**

1. 模型用途实验（拟议）：
  1. 固定真实数据、编码器、模型容量与优化预算
  1. 分别训练状态预测模型、当前价值等价模型、指定分布摘要模型
  1. 在未用于拟合的策略/后续价值/奖励查询上测 target 误差
  1. 对相同候选动作同时报告均值、选定尾部指标与真实动作排序
  1. 只改变风险目标，再只改变动力学，区分规格不足与环境漂移

持续模型研究可以由此提出清楚的假设：固定容量模型应按未来规划查询而非仅按观测频率分配表示。怎样发现新查询、保留旧风险事件，并及时判断模型的等价规格失效，仍是开放问题。模型等价定理也不提供有限采样下的安全保证。

<a id="research-model-frozen-visual-dynamics"></a>

## 冻结视觉特征后的动力学：DINO-WM 与 V-JEPA 2-AC

两条近年的视觉模型路线都把“看到什么”与“执行动作后发生什么”分开，但外部经验来源不同。DINO-WM（ICML 2025）在 DINOv2 patch 特征上用离线行为轨迹拟合动力学；V-JEPA 2（2025 首稿）先以无动作标注的视频学潜在预测，再冻结编码器，用机器人交互轨迹训练 2-AC 动作条件预测器。没有像素 decoder，并不意味着没有动力学数据。

$$
z_t=e_{\rm frozen}(o_t),\qquad \hat z_{t+1}=f_\theta(z_{t-L+1:t},a_{t-L+1:t}),\qquad \mathcal L_{\rm pred}=\sum_{k=1}^{H}\ell(\hat z_{t+k},\operatorname{sg}[e_{\rm frozen}(o_{t+k})])
$$

这是两类方法的教学性共同接口，历史长度、动作/机器人状态输入、损失距离与预测 rollout 方式以各原文为准。冻结 e 稳定了模型输出坐标；训练看到真实未来观测，不意味着测试规划可以读取它。

例子：同一物体的视觉颜色改变，冻结 encoder 可能依然给出相近的任务特征，模型因而迁移；摩擦系数改变却会使相同动作产生不同位置，即使视觉编码完全稳定，f 也必须更新。若把 e 也在线更新，第三种问题出现：旧模型预测的坐标与当前目标图像的坐标可能不再相同。三类变化需要独立实验。

| 数据与接口 | DINO-WM | V-JEPA 2 / 2-AC |
| --- | --- | --- |
| 预训练来源 | DINOv2 图像特征 | 大规模图像/视频的潜在预测 |
| 动作条件阶段 | 离线行为轨迹上预测 patch 特征 | 机器人轨迹上后训练动作条件模型 |
| 目标 | 观测目标的特征距离 | 机器人图像目标的潜在距离 |
| 部署使用 | 优化动作序列并滚动重规划 | 2-AC 模型支持图像目标 MPC |
| 持续更新证据 | 原文主要检验离线学习后规划 | 首稿主要检验冻结模型零样本机器人部署 |

先在作者公开检查点和已支持环境上验证训练/规划接口，再用单变量扰动做预测误差与真实目标成功率的联合评价。图像目标很近时仍可能物理碰撞；视觉目标代价也可能漏掉执行中的负奖励。因此若研究奖励控制，还需要额外 reward/风险接口，不能把视觉相似度默认成环境目标。

两个工作给 CRL 的机会是稳定且可迁移的起始表示，但未来学习必须另行检验。可比较冻结 e 仅更新 f、联合更新 e/f、以及使用兼容约束的三种方案，固定每步训练预算，测新后果学习、旧查询保留与控制恢复。预训练资源与在线维护资源应各自记录；首稿结果不是自动问题发现、option 构造或终生世界模型维护的证明。

<a id="lesson-check"></a>

## 9. 失败诊断与自测答案

- 模型 loss 很低，规划却变坏：检查训练分布与 planner 查询分布、是否漏掉罕见高代价事件、是否只预测均值而下游非线性。
- 长技能被过分偏爱：检查折扣终点模型是否错误归一化、是否用平均 τ 替换随机 τ、是否漏记沿途 reward。
- 换奖励后 SF 迁移失败：检查新奖励是否真在特征张成空间、旧策略 SF 是否准确、动力学是否同时变了。
- 同名技能更新后旧模型失效：不是随机异常，而是条件行为已改变；要么重估模型，要么让版本与参数成为显式条件。

自测 1：$p_o^\gamma$ 的行和小于一是否代表概率遗漏？答：它包含时间折扣，行和为 $\mathbb E[\gamma^\tau]$。自测 2：把每个奖励特征各自的最优累计值相加，能否得到新奖励的最优值？答：各分量的最优策略可能冲突，SF 必须条件化于同一策略。自测 3：为什么 stopping bonus 不进入 $r_o$？答：它是训练技能的辅助目标，而主任务模型预测执行技能真正得到的环境奖励。

## 本章的实验设计

模型预测误差与决策损失需要分别测量。对规划无关变量的准确预测，不一定改善策略。

设定：固定同一真实数据集训练后果模型，再在独立轨迹和规划实际访问的状态上评价。加入随机时长与终点相关的 option。

- 折扣后果模型保留持续时间与终点的相关性。
- 奖励、continuation 和后继特征目标分别核对。
- 模型生成的虚拟奖励不计作真实环境收益。

对照：准确模型诊断与学习模型；随机分布测试与规划访问分布测试；固定消费者只替换模型

记录：一步及多步预测误差；动作排序、价值误差和模型使用分布；真实执行表现及模型计算成本

[具体测试规程](../docs/experiment-handbook.md#handbook-planning)

## 学习与研究衔接

模型需要保留什么，取决于规划随后会问什么。线性尾值可以读取期望终点特征，非线性尾值通常还需要更多分布信息。状态或技能改变后，应分别检查输入含义、真实后果与采样覆盖；只重新拟合一个起点的备份，可能掩盖其他条件问题已经无法区分。模型维护由此既有统计问题，也有信息是否仍在的问题。

[分册导读](../docs/learning-route-continual-rl.md) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-models) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=models) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=models)

## 从本章进入实践

[技能与规划](https://yingwen.io/zh/continual-rl/code/#practice-skills)：一个多步行为怎样成为可学习、可预测、可规划的动作？

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 从历史构造状态与预测知识

当前观测不够时，应记住什么、预测什么，又怎样在线学习？

给定状态后可以估计价值；观测不足时，还要学习保留哪些历史。GVF 规定预测什么，RTRL 计算递归敏感度，资格迹组织时间信用。应分别检验信息是否进入状态、反馈能否教会这种保留，以及有限预测预算怎样分配，而不是把三者当作替代算法。

- [Does Zero-Shot Reinforcement Learning Exist?](https://yingwen.io/zh/continual-rl/research/#recent-zero-shot-forward-backward)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

#### 子任务、技能与经验获取

哪些行为值得成为可复用技能，技能怎样帮助探索和新任务？

教材可以先给定目标和技能集合；持续构造还要决定哪些行为值得练习、维护或放弃。谱结构、路径奖励、时间距离和语言先验提供不同候选偏置。先固定候选比较选择与组合，再改变生成器，才能辨认下游收益究竟来自哪一步。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Reward-Aware Proto-Representations in Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-reward-aware-proto-representations)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 后果模型、知识保留与规划

学会预测后果，何时能真正改善决策？

给定模型可研究怎样规划；模型也在学习时，规划会选择性地查询误差，并改变以后的数据。Dreamer、STOMP 和 DRAGO 分别研究想象控制、随机时长行为模型和旧知识保留。新的比较应固定规划查询与总预算，检验哪些后果误差真正改变选择，哪些维护值得继续。

- [Reward-Respecting Subtasks for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-stomp-reward-respecting)
- [Mastering diverse control tasks through world models](https://yingwen.io/zh/continual-rl/research/#recent-dreamerv3-world-models)
- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)
- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [Laplacian Keyboard: Beyond the Linear Span](https://yingwen.io/zh/continual-rl/research/#recent-laplacian-keyboard)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-distributional-model-equivalence)
- [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://yingwen.io/zh/continual-rl/research/#recent-tdmpc2-decision-time-model)
- [DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning](https://yingwen.io/zh/continual-rl/research/#recent-dino-wm-feature-planning)
- [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://yingwen.io/zh/continual-rl/research/#recent-vjepa2-action-conditioned)
- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 持续控制、平均奖励与重置

当学习、行动和恢复占用同一条时间轴时，应优化什么，又怎样探索？

平均奖励改变跨时间目标；中心化改变估计的参照；重置协议改变转移和控制权限；后验采样改变探索。它们可以组合，但不能由同一条改名的更新式替代。

- [Reset-free Reinforcement Learning with World Models](https://yingwen.io/zh/continual-rl/research/#recent-morefree-reset-free-models)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [Knowledge Retention in Continual Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-drago-model-retention)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文规定对象与条件，架构路线提出组织方式，算法实验检验局部机制。撤掉阶段间冻结后，一个模块会改变另一个模块的学习问题；有限预算应优先维护哪条知识，成为新的决策。先检验两模块反馈和资源分配，再扩大整机，而不是由组件分别有效推断长期组合收益。

- [General Agents Contain World Models](https://yingwen.io/zh/continual-rl/research/#recent-general-agents-world-models)
- [The OaK Architecture: A Vision of SuperIntelligence from Experience](https://yingwen.io/zh/continual-rl/research/#recent-oak-architecture)
- [Constructing an Optimal Behavior Basis for the Option Keyboard](https://yingwen.io/zh/continual-rl/research/#recent-option-keyboard-basis)
- [The Value Equivalence Principle for Model-Based Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-value-equivalence-models)
- [Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations](https://yingwen.io/zh/continual-rl/research/#recent-successor-flow-features)

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

### Reward-Aware Proto-Representations in Reinforcement Learning

Hon Tik Tse, Siddarth Chandrasekar, Marlos C. Machado

NeurIPS 2025 · 2025 · 支持方法与理论

#### 研究问题

仅编码可达关系的表示，怎样进一步反映奖励与行动成本？

#### 关键机制

论文研究 default representation，将奖励或成本纳入对未来状态关系的表示，并给出动态规划与 TD 学习方法。由此提取的谱特征可以参与技能发现、奖励塑形和迁移。它沿着 SR 的后果预测思路前进，但不再把奖励完全留到最后的线性读出阶段。

#### 证据

作者提供表格问题中的推导，并用表示、技能和迁移实验展示奖励信息如何改变学得的结构。代码包含 SR、DR 的计算和在线表示学习实验。

#### 条件与限制

把奖励纳入表示会改变迁移边界：奖励或内部成本变化后，原表示可能需要重学。论文结果不能解释为任意新奖励下都能免费零样本迁移。

#### 阅读与实验

固定转移图，只改变一处通行成本，比较 SR 与 DR 的谱方向。随后检查新的 eigenoption 是改变了可达性，还是改变了对路径代价的偏好。

#### 原文与相关入口

- [论文与版本记录](https://arxiv.org/abs/2505.16217)：NeurIPS 2025；后续版本修订不改变会议年份。
- [作者实现](https://github.com/httse9/Reward-Aware-Proto-Representations)：从 minigrid_basics/examples 的表示计算与技能实验开始。

#### 作者代码

[原论文作者仓库。](https://github.com/httse9/Reward-Aware-Proto-Representations)

奖励感知表示、谱特征与相关 MiniGrid 实验。

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

### Knowledge Retention in Continual Model-Based Reinforcement Learning

Haotian Fu, Yixiang Sun, Michael Littman, George Konidaris

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

当前任务不再访问旧区域时，世界模型怎样避免忘记那些区域的动力学？

#### 关键机制

DRAGO 将生成式旧经验、旧模型知识和探索结合起来。模型学习不只跟随当前奖励驱动的数据分布，还尝试维持对曾经学过区域的预测能力。它关注知识保留发生在模型里，而不只是保存旧策略输出。

#### 证据

论文在 MiniGrid 和连续控制任务中检验模型保留与任务表现，并给出原作者实现。关键设定包括共享状态与动力学、变化的任务奖励。

#### 条件与限制

已知任务切换、旧模型和数据资源是协议的一部分。共享动力学下的保留不能直接外推到动力学本身任意变化，也不是禁止重放的严格流式算法。

#### 阅读与实验

分别测量旧区域模型误差、旧任务回报与新任务学习速度。若模型误差改善而控制没有改善，再检查规划是否真正查询了被保留的知识。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/fu25f.html)：任务设定、模型保留和探索机制。
- [作者代码](https://github.com/YixiangSun/drago)：原文链接的 DRAGO 实验实现。

#### 作者代码

[作者 Yixiang Sun 的仓库；不使用同名第三方项目。](https://github.com/YixiangSun/drago)

DRAGO 的模型、重放、探索与实验。

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

### Bridging Successor Measure and Online Policy Learning with Flow Matching-Based Representations

Haosen Shi, Jianda Chen, Sinno Jialin Pan

ICLR 2026 · 2026 · 支持方法与理论

#### 研究问题

能否直接学习多步未来状态的分布，并把它压缩为适合控制学习的特征？

#### 关键机制

SF² 以 flow matching 估计 successor measure，将条件向量场分解为未来位置及生成时间的投影与当前状态动作特征的乘积。特征进入 TD3/SAC 的 critic；线性的是向量场对条件特征的分解，critic 本身可以非线性。

#### 证据

正式原文给出 mixture Bellman 结构、生成式 bootstrap 与控制实验，并提供作者 JAX/Brax 仓库。实验研究在线收集数据下的 off-policy 控制，并使用 replay、批次与目标网络。

#### 条件与限制

“online policy learning”不代表 strict streaming。生成时间不是环境时间；向量场线性不保证任意奖励价值线性。文中与 SR 的小生成时间联系是近似动机，未证明递归 agent state 或任意持续变化下的充分性。

#### 阅读与实验

对齐模型调用与梯度预算，拆分直接预测、bootstrap、critic 联合训练。冻结特征后比较线性与非线性读出，再测新奖励和动力学变化，才能检验预测知识的可复用程度。

#### 原文与相关入口

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。
- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

#### 作者代码

[正式论文摘要直接链接的作者代码。](https://github.com/Shiien/successor-flow-representation-implementation)

基于 JAX/Brax 的 SF² 控制实验；不包含自动 GVF 问题发现或完整持续架构。

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

### The Value Equivalence Principle for Model-Based Reinforcement Learning

Christopher Grimm, André Barreto, Satinder Singh, David Silver

NeurIPS 2020 · 2020 · 支持方法与理论

#### 研究问题

模型容量有限时，必须预测全部状态细节，还是只须保持规划会查询的量？

#### 关键机制

Value equivalence 以策略集合和函数集合定义模型规格：模型对这些函数进行这些策略的 Bellman backup，应与真实环境相同。扩大查询族会缩小可接受模型族；它把“决策相关”从口号变成有条件的等价关系。

#### 证据

论文给出等价模型类的性质及有限实验，并解释若干隐式模型方法。后续 Proper Value Equivalence（NeurIPS 2021）研究策略价值固定点等价及规划充分性。

#### 条件与限制

少数当前 critic 的 backup 相同，不说明所有新奖励、新策略或风险目标都相同。精确算子等价与神经损失在样本上较小不同；奖励或查询族变化后须重新验证。

#### 阅读与实验

保存独立的 planner 查询集，直接测 target 误差和动作排序。用未参与模型拟合的价值函数检验迁移，并与像素误差对照，找出模型实际保留的信息。

#### 原文与相关入口

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。
- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

### Distributional Model Equivalence for Risk-Sensitive Reinforcement Learning

Tyler Kastner, Murat A. Erdogdu, Amir-massoud Farahmand

NeurIPS 2023 · 2023 · 支持方法与理论

#### 研究问题

模型正确预测期望回报，能否同时支持避开低概率灾难的决策？

#### 关键机制

论文证明 proper value equivalence 对风险敏感规划不足，再以回报分布与统计摘要定义更强的模型等价。完整分布覆盖更多风险度量，有限摘要则限制可支持的风险目标；相应 Bellman 闭合性质决定摘要能否递推。

#### 证据

正式原文包含理论、表格反例与大规模实验，并直接给出 distribution-equivalence 作者仓库。它检验的是特定风险敏感目标下的模型学习与规划接口。

#### 条件与限制

正确均值和方差不自动保证尾部概率或 CVaR；有限 quantile 表示与投影也有近似误差。静态模型等价不保证新环境中的风险校准，更不等于安全约束保证。

#### 阅读与实验

构造均值相同、尾部不同的两动作，先验证期望控制无法区分，再用指定风险度量评价。训练分布、投影和风险目标必须匹配，不能在评估时随意换风险函数。

#### 原文与相关入口

- [NeurIPS 2023 原文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)：proper VE 的不足、统计摘要与 Bellman 闭合。
- [作者实现](https://github.com/tylerkastner/distribution-equivalence)：原文第 7 节直接链接的实验代码。

#### 作者代码

[正式原文第 7 节提供的作者仓库。](https://github.com/tylerkastner/distribution-equivalence)

分布模型等价与风险敏感实验；不提供任意任务的安全证书。

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


<a id="chapter-code"></a>

## 下载与运行

表格 option model、SF/GPI 与条件反例；不包含完整深度世界模型训练。

[下载 knowledge_algorithms_lab.py](../examples/knowledge_algorithms_lab.py)

```sh
python3 examples/knowledge_algorithms_lab.py models
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Sutton & Barto — 第二版 §17.2：用 GVF 学习 option models](http://incompleteideas.net/book/the-book-2nd.html)：模型预测对象、GVF 分解与 option 备份。本文沿首项不折扣的约定，从一步递推单独核对终点 cumulant 中的折扣。

- [Sutton, Precup & Singh — Options 的 Bellman 模型](https://doi.org/10.1016/S0004-3702(99)00052-1)：原始框架。先比较完整轨迹定义与一步递推，再核对 discount 已包含在 transition model 中。

- [Wan, Naik & Sutton — Average-Reward Learning and Planning with Options](https://arxiv.org/html/2110.13855)：§5 式 (18)–(23)：奖励、时长、终点三类模型及其逐步学习；持续环境与固定技能的理论条件不同于本地变化环境原型。

- [Sutton et al. — Reward-Respecting Subtasks，模型部分](https://arxiv.org/html/2202.03466v3)：原文第 4 节将 reward model 与 expectation model 分开；子任务 stopping bonus 不能混入环境 reward model。

- [Wan et al. — Planning with Expectation Models](https://arxiv.org/abs/1904.01191)：原文。理解线性状态价值下期望模型的充分性及函数逼近前提。

- [Planning with Expectation Models for Control](https://arxiv.org/abs/2104.08543)：控制与期望模型的接口；重点比较线性预测的等价条件与包含非线性 Q/max 的控制更新。

- [Barreto et al. — Successor Features for Transfer in RL](https://arxiv.org/abs/1606.05312)：原文。先查奖励分解，再查目标策略条件与 GPI 改善保证的精确/近似前提。

- [DeepMind — Option Keyboard 作者工程](https://github.com/google-deepmind/deepmind-research/tree/f5de0ede8430809180254ee957abf36ed62579ef/option_keyboard)：研究团队代码；keyboard_agent.py 的策略、cumulant、动作三个维度与本页 GPI 计算顺序对照。不是所有 SF 方法的统一实现。

- [Reward-Aware Proto-Representations in RL · NeurIPS 2025](https://arxiv.org/abs/2505.16217)：DR 的定义、线性可解控制推导与谱特征用途；注意内部奖励固定和终点收益改变是不同的迁移条件。

- [Reward-Aware Proto-Representations 作者实现](https://github.com/httse9/Reward-Aware-Proto-Representations)：从默认转移和奖励构造 DR，比较奖励感知特征、奖励塑形与技能发现；读取数值求解与谱处理设置。

- [DR 矩阵构造 — rep_utils.py](https://github.com/httse9/Reward-Aware-Proto-Representations/blob/master/minigrid_basics/examples/rep_utils.py)：compute_SR 与 compute_DR 并排给出两个矩阵的区别；get_representation 中的对称化决定了随后分解的谱对象。

- [DR 在线表示与发现 — ROD_DR.py](https://github.com/httse9/Reward-Aware-Proto-Representations/blob/master/minigrid_basics/examples/ROD_DR.py)：learn_representation 对应 DR 的采样递推；compute_eigenvector 从访问过的状态计算谱特征。结合数据采集循环阅读，辨认其经验转移实际对应的默认行为分布。

- [Hafner et al. — DreamerV3](https://arxiv.org/abs/2301.04104)：原文。区分真实序列模型学习与想象序列 actor/critic 学习，保留完整损失与超参数。

- [Danijar Hafner — DreamerV3 作者维护实现](https://github.com/danijar/dreamerv3)：作者维护的公开重实现；README 明确其为 reimplementation，并非原内部训练代码。先追 model/imagined rollout/actor 三个接口。

- [Schrittwieser et al. — MuZero](https://arxiv.org/abs/1911.08265)：原文。预测 reward、policy、value 的潜在模型与决策时树搜索；不以观测重建作为必要接口。

- [Sutton et al. — Dyna-Style Planning with Linear Function Approximation and Prioritized Sweeping（UAI 2008）](https://arxiv.org/abs/1206.3285)：§3.1 固定模型下的规划分布与稳定条件；§3.2 定理 3.3 说明最小二乘模型的固定点等于同一数据的 LSTD 解。arXiv 上传年份不是发表年份。

- [作者原文](https://arxiv.org/abs/2209.14935)：2022 首稿，ICLR 2023；比较奖励表示、SF 与 FB。

- [作者研究平台](https://github.com/facebookresearch/controllable_agent)：README 直接关联两篇 FB 论文；该仓库已经归档。

- [ICLR 2026 正式原文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/48acf4b231771e693f42305b4c9b4c9f-Abstract-Conference.html)：第 2–3 节和算法附录；区分 flow 时间、环境时间与近似 SR 联系。

- [原文链接的作者实现](https://github.com/Shiien/successor-flow-representation-implementation)：SAC/TD3、flow 特征、对照和 sweep 配置。

- [NeurIPS 2025 原文与补充材料入口](https://proceedings.neurips.cc/paper_files/paper/2025/hash/0ab48777def88e73b50746a6011be0b0-Abstract-Conference.html)：算法 1–3、附录 A.3 的两个最优子程序假设及代码声明；未在本教材运行补充代码。

- [Option Keyboard 的经典桥梁](https://proceedings.neurips.cc/paper/2019/file/251c5ffd6b62cc21c446c963c76cf214-Paper.pdf)：cumulant 组合、GPE/GPI 与技能接口。

- [NeurIPS 2020 原文](https://papers.nips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)：VE 依赖策略与函数集合。

- [Proper Value Equivalence · NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/400e5e6a7ce0c754f281525fae75a873-Abstract.html)：多步算子、固定点与规划充分性；不是任意潜在网络的保证。

- [NeurIPS 2023 原文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/b0cd0e8027309ea050951e758b70d60e-Abstract-Conference.html)：proper VE 的不足、统计摘要与 Bellman 闭合。

- [作者实现](https://github.com/tylerkastner/distribution-equivalence)：原文第 7 节直接链接的实验代码。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html)：短期预测、终点价值、多任务协议。

- [作者实现](https://github.com/nicklashansen/tdmpc2)：训练、模型与 plan 函数分别阅读。

- [ICML 2025 原文](https://proceedings.mlr.press/v267/zhou25t.html)：正式发表入口；早期 ICLR 投稿页不能替代此状态。

- [作者项目代码](https://github.com/gaoyuezhou/dino_wm)：train.py、plan.py、数据与已公开模型检查点范围。

- [2025 首稿](https://arxiv.org/abs/2506.09985v1)：action-free 预训练、2-AC 后训练、真实规划与第 4.3 节限制；此处不赋予未核实会议状态。

- [Meta FAIR 官方实现](https://github.com/facebookresearch/vjepa2)：包含 V-JEPA 2、2-AC 和较新的 2.1；版本不能混用。

- [Richard Sutton · The OaK Architecture](https://oaklab.ai/posts/the-oak-architecture)：公开架构讲座入口。差分子问题依据 RLSS 收录的 OaK/NeurIPS 讲义第 24 页；学习的四种作用、消费者信用和增量规划讨论依据 OaK thinker 讲义。本文的成本记账、删除诊断和缓存反例用于澄清机制，不是作者已完成的通用算法。

- [Sutton · Toward a New Approach to Model-based Reinforcement Learning](https://www.incompleteideas.net/papers/MBRL2.pdf)：课程指定阅读 Introduction 与 §1：近似 agent state、特征对当前表现与未来学习的用途、学习与规划的耦合。

- [Sutton et al. · Dyna-Style Planning with Linear Function Approximation and Prioritized Sweeping](https://proceedings.mlr.press/r6/sutton08a.html)：UAI 2008 原文。固定模型的 TD/残差迭代、收敛条件，以及从状态前驱到特征前驱的优先扫描。
