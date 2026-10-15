# 可塑性：梯度通路、有效学习率与预测干扰

在相同新数据与更新预算下，学习能力为什么可能下降，又该怎样诊断与恢复？

## 本章内容

- 用 matched aged/fresh probe 区分能力丧失、探索失败和普通遗忘。
- 从梯度通路推导 dormant-unit 机制，具体实现 ReDo 与 CBP 的选择、替换、成熟期和优化器状态处理。
- 理解 CReLU、正则化、网络重置、plasticity injection 等分支的不同作用及保留代价。

## 阅读这条论证

保留良好的旧答案，并不保证还能迅速学会新问题。从固定学习目标和预算出发，先用梯度几何解释学习速度，再研究特征替换怎样恢复可用方向。替换还会影响旧预测、资格迹和元学习状态；这些代价必须与恢复的学习能力一起考察。

[定义学习能力的比较](plasticity.md#lesson-setting) → [从梯度推到误差变化](plasticity.md#lesson-derive) → [恢复无效梯度通路](plasticity.md#lesson-redo) → [持续生成与检验特征](plasticity.md#lesson-cbp) → [追踪替换的连带影响](plasticity.md#lesson-consumers) → [对应实现与测量](plasticity.md#lesson-code)

<a id="chapter-prerequisites"></a>

## 预备知识与符号

### 需要哪些基础

已掌握下面的概念即可直接阅读；需要回顾时再打开对应章节。

- [知识保留：经验重放、参数约束与模型记忆](retention.md)：先区分保留旧知识和获得新知识。
- [深度价值学习：DQN、Double DQN 与目标的时间顺序](../foundations/deep/deep-value.md)：理解特征梯度、目标变化与神经网络更新。


### 梯度链式法则

网络输出对某个权重的梯度由下游权重、激活导数与输入相乘；任一环节接近零都可能使参数难以变化。

### ReLU

$\operatorname{ReLU}(z)=\max(0,z)$；$z<0$ 时导数为零。一个单元在当前数据上不激活，并不意味着它在所有未来状态上都无用。

### 优化器状态

Adam 保存一阶矩、二阶矩及时间计数；只重置参数而不处理这些状态，可能让新参数继承旧方向和尺度。

<a id="problem-definition"></a>

## 本章的问题定义

长时间训练后，网络在同样新数据与优化预算下变得难以学习；需先排除探索和任务难度差异。

### 给定条件与符号

- 同容量的新学习器与已长期训练的学习器、匹配的新目标、数据顺序和优化预算。
- 固定探针评价、优化规则与归一化协议，以及允许的单元替换和参数扰动预算。优化器与统计状态属于各自历史，不默认清零。

### 需要求解的对象

可学习性退化的可识别诊断及恢复机制，并量化恢复对旧功能和在线收益的代价。

### 信息与数据权限

probe数据对各条件一致且不向主在线学习器提供额外世界经验；重置参数、优化器和环境是不同干预。

$$
\mathcal P_K(M;\mathcal D_{\rm train},\mathcal D_{\rm eval})=L_{\mathcal D_{\rm eval}}(M)-L_{\mathcal D_{\rm eval}}\!\left(U^K(M;\mathcal D_{\rm train})\right)
$$

$M$ 包括参数、优化器及运行统计。$U$ 只使用训练数据执行 K 次更新；更新前后均在同一独立评价集上只读评价 M 内的预测器，不更新参数或统计。比较改进量需控制初始损失；此诊断不是在线回报目标，机制还要回到真实控制评价。

### 成立条件与解的含义

- 容量、目标难度、数据、优化器与统计更新匹配，初始误差或可比较误差区间受控。
- dormant、特征秩和梯度范数是诊断代理；低活动不证明单元对所有未来状态无用。

判断准则：匹配probe上测固定预算误差曲线及aged/fresh差异；选择性重置检查精确输出扰动、优化器清理和年龄；联合报告旧能力损失与在线恢复。

### 适用边界

- dormant比例下降不自动证明在线收益增加。
- 部分网络重置不等于新的所有坐标都拥有全新优化器语义。

### 与其他问题的关系

关系类型描述本章相对于所链接问题的变化。“特例”表示本章增加条件；“推广”表示本章放宽条件。目标、近似方法和数据协议的改变另行区分。

- 组合不同学习问题 · [知识保留与再适应](retention.md)：恢复学习通路可能删除旧预测；固定问题上的保持损失与再次学习速度分别衡量这项代价。

- 组合不同学习问题 · [通用价值函数与预测知识](gvf.md)：共享特征可能服务多个预测问题；只按当前任务的活动或读出评分，会遗漏其他问题仍在使用的贡献。

- 组合不同学习问题 · [时间信用分配与资格迹](credit.md)：新生特征不能仅凭同一参数槽位继承旧资格；清除受影响的迹同时放弃其中尚未兑现的时间信用。

- 组合不同学习问题 · [元学习与学习规则的适应](meta.md)：元步长调整已有方向的尺度，特征替换改变可用方向；替换还需处理优化器状态和元敏感度。

- 改变信息或数据协议 · [流式更新与稳定性](streaming.md)：无replay时缺少从旧样本重建功能的通路，候选效用的覆盖、统计时间尺度和替换预算更需明确。

<a id="problem-solution"></a>

## 从问题到方法

### 直接求解的难点

相同标称步长可因梯度通路、尺度、曲率或优化器历史产生不同学习速度。

### 本章的核心思路

先用匹配数据隔离可学性，再针对被识别的通路或几何故障干预并测保留代价。

1. [规定新学习的目标与预算](plasticity.md#lesson-setting)：在匹配起始误差、数据和更新预算后比较学习曲线，同时区分探针拟合、独立输入泛化与在线控制。

2. [由参数更新推到残差变化](plasticity.md#lesson-derive)：Jacobian把误差传入参数，再传回所有输入；其可用方向和尺度解释了相同误差为何仍有不同学习速度。

3. [恢复休眠单元的梯度通路](plasticity.md#lesson-redo)：ReDo重采输入连接、将输出连接归零；先学出输出权重后，输入特征才重新获得下游梯度。

4. [按贡献与年龄生成新特征](plasticity.md#lesson-cbp)：CBP由删除特征的下游扰动构造效用代理，以成熟期保护新特征，再按有限名额替换并清理相应状态。

5. [追踪替换对其他知识的影响](plasticity.md#lesson-consumers)：对各预测头计算被删除的贡献，再检查策略、模型、资格迹和元敏感度；局部低效用不等于所有消费者都不再需要。

结论与条件：单隐藏层选择性替换的输出差和理想尺度不变层的SGD尺度关系可代数检验；不构成普遍可塑性恢复或无遗忘定理。

### 相关方法改变了什么

- ReDo/CBP：按不同评分检测并替换特征，需测稀有状态覆盖和成熟期。

- NaP/正则：改变更新几何及有效学习尺度，不等于单元替换。

- 优化器重置/参数重置：分别干预历史尺度与表示，matched对照可定位原因而代价不同。


<a id="lesson-setting"></a>

## 1 · 可塑性的操作性定义

函数逼近使有限参数可以服务许多状态，也使学习的历史改变了以后的梯度通路与更新尺度。一次训练得到低误差，还不能说明同一网络在很久以后仍容易学习。要研究这种能力，先让两个同样结构的网络接收匹配的新目标和数据：一个已经训练很久，另一个刚初始化。控制起始误差和学习预算后，若前者改进更慢，就得到可塑性下降的证据。单独的在线回报下降还混入探索、任务难度与估值误差，因而不足以完成这个诊断。

$$
\mathcal P_K(M;\mathcal D_{\rm train},\mathcal D_{\rm eval})=L_{\mathcal D_{\rm eval}}(M)-L_{\mathcal D_{\rm eval}}\!\left(U^K(M;\mathcal D_{\rm train})\right)
$$

这是一个 $K$ 步改进量。$M$ 包含参数、优化器与归一化统计，$U$ 指定完整更新；$L_{\mathcal D}(M)$ 是用该状态中的预测器在数据 $\mathcal D$ 上只读评价的损失，不更新参数或统计量。训练只使用 $\mathcal D_{\rm train}$，两次评价使用同一份独立且匹配的 $\mathcal D_{\rm eval}$。还应控制初始损失，避免把更少的可改进空间误判为较差学习能力。

这一定义针对一项具体学习要求，不是网络脱离任务后的一项常数属性。若关心多种未来预测，应先给出问题分布 $q\sim\nu$，再报告 $\mathbb E_{q\sim\nu}[\mathcal P_K(M;q)]$ 及各类目标的结果。这里用起始损失减去末损失；Lyle 等原文使用指定基线减去探针末损失。起点不匹配时，两种读数可以排序不同。训练集拟合速度和独立评价集上的改进也应分别记录：能迅速记住新标签，还不等于能对新输入学习出有用的规律。

为什么把优化器也放进 $M$？令 $L(w)=(w-1)^2/2$、$w=0$，采用 $m^+=0.9m+\nabla L(w)$、$w^+=w-0.1m^+$。若旧动量为 $m=-1$，新参数为 $0.19$，损失下降；若旧动量为 $m=2$，新参数为 $-0.08$，损失反而上升。网络参数与当前损失完全相同，下一步学习能力仍可不同。比较“只重置优化器”的条件，就是为了区分这类影响与表示退化。

完整 aged/fresh 对照至少固定网络容量、目标难度、数据顺序、更新数、优化规则和归一化协议。已有优化器状态与运行统计是待比较学习历史的一部分，不能在主对照中不加说明地清零。为了拆解原因，还应增加“只重置优化器”“只重置部分权重”“保留 replay 但重置网络”等条件。若 fresh 用额外探索或外部 task reset，就不再是同一学习能力测试。

| 诊断量 | 能提示什么 | 不能单独证明什么 |
| --- | --- | --- |
| 新目标上的固定预算 loss reduction | 局部可学习性 | 在线探索是否成功 |
| Dormant 比例与梯度范数 | 激活或梯度通路变化 | 全部 plasticity loss 都由死 ReLU 导致 |
| 特征矩阵的有效秩 | 表征多样性退化 | 秩越大一定回报越高 |
| Hessian/曲率代理、权重范数 | 优化几何与尺度变差 | 一个相关曲线就是因果机制 |

CRL 中即使环境外部不变，策略、bootstrap target 与所访问状态也持续变化。因此可塑性问题不要求人为切任务；但人为目标切换能帮助在小实验里隔离机制。

<a id="lesson-paired-probe"></a>

### 同一反转问题：对称轨道、等函数干预与重新学习

深入阅读：保持初始函数相同而改变参数构成，区分旧表现与后续学习速度。

沿用保持章的左右线索预测：$x=\pm1$ 等权，旧目标 A 为 $x$，新目标 B 为 $-x$，预测器 $f(x)=uvx$。给定旧检查点 $(1,1)$，以步长 $1/4$ 在 B 上做八次完整梯度更新；另一个副本先变为 $(2,1/2)$，保留完全相同的初始预测。两层都没有激活函数，困难来自双线性参数化与更新轨迹。这样可以检查：没有死 ReLU，是否仍有一种走不到可表示目标的情形？

$$
\begin{aligned}u^+&=u-\alpha(uv+1)v,\qquad v^+=v-\alpha(uv+1)u,\\q^+&=q-\alpha(q+1)(u^2+v^2)+\alpha^2(q+1)^2q,\qquad q=uv.\end{aligned}
$$

第二行是把两个更新后的权重相乘得到的精确等式，不是一阶近似。相同 $q$ 的两个网络具有相同预测与损失，但 $u^2+v^2$ 可以不同，因此下一步的预测变化也不同。这里两份副本该量分别为 2 与 $17/4$。

对称旧解满足 $u=v=a$，一次同步梯度步仍保持相等。在 B 上，$a^+=a[1-(a^2+1)/4]$。当 $0<a\leq1$ 时，括号处于 $[1/2,3/4)$，所以 $a$ 保持正值并趋向 0；预测系数 $q=a^2$ 永远不会变成新目标要求的 $-1$。新损失趋于 $1/2$，旧损失也趋于 $1/2$。损失下降了，却没有学会 B。

$$
\nabla L_B(0,0)=0,\qquad \nabla^2L_B(0,0)=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad L_B(1,-1)=0.
$$

原点的 Hessian 有特征值 $+1$ 和 $-1$，因此它是鞍点；函数类中确有零损失解。失败是这个确定性优化器与精确对称初始化形成的不变轨道造成的。它并非容量不足，也没有证明普通随机初始化会落入同一轨道。

等预测变换 $(1,1)\mapsto(2,1/2)$ 改变了参数几何与梯度尺度。其第一步是 $(7/4,-1/2)$，已经跨到 $uv<0$；八步后 $L_B\approx1.08\times10^{-14}$。这是一项保持初始功能的机制干预，不是 ReDo、CBP 或通用重置算法。若把很小的非对称扰动加入原起点，对称轨道也会被破坏，但固定八步内能恢复多少仍需实际计算。

现在复制两份 B 的第八步检查点，先关闭更新测 A，再分别开放八次 A 梯度更新。保持测试只读参数；重新学习发生在独立副本上，不写回 B 检查点。所有副本使用同一数据、步长和更新数。固定 fresh 参照从 (1,0) 直接学习 A，用来显示起始误差不同的影响。

看图前先预测：旧预测损坏更多的橙色副本，重新学习 A 时一定更慢吗？参数图中的哪一项信息，是只看预测值无法获得的？

![参数平面上两条等预测双曲线与两个学习轨迹；对称轨迹趋于原点，重参数化轨迹跨越v零点。下方从B检查点复制后再学A的损失曲线显示旧误差与重新学习速度可以反向排序。](../assets/crl-figures/retention-plasticity-walkthrough-geometry.svg)

原创确定性算例。上图箭头是 B 的第一步更新，双曲线分别是旧目标 uv=1 与新目标 uv=−1 的解集；蓝线受限于 u=v。下图横轴 0 是只读保持测试，1–8 是副本上重新学习 A；紫色参照从 (1,0) 新学 A，没有接受 B 的训练。橙色副本起初旧误差更大，却恢复得更快。

| 副本 | B 的初始→八步损失 | B 的八步改进量 | B 检查点上 A 的只读→再学八步损失 |
| --- | --- | --- | --- |
| 对称旧解 | 2 → 0.503174 | 1.496826 | 0.496836 → 0.401097 |
| 等预测重参数化 | 2 → 约 $1.08\times10^{-14}$ | 约 2 | 约 2 → 约 $1.00\times10^{-8}$ |
| 固定 fresh 参照 | 0.5 → 0.00008165 | 0.499918 | 从 (1,0) 新学 A：0.5 → 0.00008165 |

这也说明为什么要同时保留初始损失、末损失与改进量。对称旧解的绝对改进量大于 fresh，末损失却更差，因为它原本有更大的改进空间。两个旧解副本的起始预测和损失一致，因而更直接地隔离了参数化对这项探针的影响；fresh 则只是一个给定初值的参照，不代表随机初始化群体。

把结果用于诊断时，可以依次问：旧功能在关闭更新后是否还在；相同新目标在固定预算内是否能学；旧目标回来时是否要重新学习、代价是多少。Lyle 等原文 §2–3 将可塑性定义到指定目标分布、优化器和更新预算；本例只取一个反转目标，既不估计广泛任务上的可塑性，也不把给定旧解的困难归因于未经模拟的长期训练历史。[原文](https://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf)。

[下载单文件标准库脚本](../tutorials/retention_plasticity_walkthrough.py)后运行 `python3 tutorials/retention_plasticity_walkthrough.py`。Python 逐输入累计梯度，JavaScript 使用解析期望梯度；两者逐步核对参数、损失与重新学习轨迹。脚本还保留精确分数的首步检查，避免把先改 u、再用新 u 改 v 的顺序更新误当成这里的同步梯度下降。

<a id="lesson-derive"></a>

## 2 · 从学习目标到可用的梯度方向

为什么相同的新目标误差会对应不同的学习速度？先冻结一个包含 $n$ 个输入的探针集和标签，令预测向量为 $f_\theta\in\mathbb R^n$，残差 $r=f_\theta-y$，损失 $L=\|r\|^2/(2n)$。Jacobian $J$ 的第 $i$ 行是 $\nabla_\theta f_\theta(x_i)^\top$。链式法则先把输出误差传到参数，再把一次参数移动传回所有输入：

$$
\begin{aligned}\nabla_\theta L&=\frac1nJ^\top r,\qquad\Delta\theta=-\frac\alpha nJ^\top r,\\r^+&\approx r+J\Delta\theta=\left(I-\frac\alpha nJJ^\top\right)r.\end{aligned}
$$

对参数线性的预测器，此式为一次完整梯度步的精确等式；对可微非线性网络，它是本次更新的局部线性化。固定 ReLU 激活区域时可以局部使用；跨越激活边界后应重新计算 J。它描述探针上的拟合，不直接给出独立评价误差。

令 $K=JJ^\top$。在线性模型中，若某个残差方向是 $K$ 的特征向量、特征值为 $\lambda$，一次更新就把该分量乘以 $1-\alpha\lambda/n$。$\lambda=0$ 的分量不动；正特征值很小的分量改得慢；过大的 $\alpha\lambda/n$ 则造成振荡甚至增长。两个网络即使 $\|r\|$ 相同，也可能因 $J$ 不同或残差指向不同而学得很不一样。因此误差大小、参数数量、活动单元比例，都不能单独替代一次受控学习探针。

接着把这条关系展开到单个特征，就能看出何处适合干预。以下单隐藏层把梯度受阻的原因暴露出来：

$$
\begin{aligned}f(x)&=\sum_i v_i h_i(x),\qquad h_i(x)=\max(0,u_i^\top x),\\L&=\tfrac12(f(x)-y)^2,\\\nabla_{u_i}L&=(f-y)v_i\mathbf1[u_i^\top x>0]x,\\\partial_{v_i}L&=(f-y)h_i(x).\end{aligned}
$$

若当前数据上 $u_i^\top x<0$，这个单元的输入权重与输出权重梯度均为零。参数容量与能够通过梯度使用的容量因而不同。

若这个单元在整个探针分布上都不激活，它对应的输入、输出坐标在 $J$ 中都不起作用。增加这些坐标的步长仍然是零乘以步长；恢复输入覆盖或改变特征才可能重新打开通路。反过来，所有单元都活动时，$J$ 的行仍可能近乎相关，因而某些残差方向很难改变。ReDo/CBP 针对可用特征的生成与替换；归一化、[元步长](meta.md)和更新约束主要改变已有方向上的尺度与几何。

$$
s_i=\frac{\mathbb E_{x\sim D}[|h_i(x)|]}{\frac1H\sum_{j=1}^H\mathbb E_{x\sim D}[|h_j(x)|]}
$$

ReDo 用层内相对活动度辨认 dormant 单元，H 为该层单元数，D 是检测用数据分布。若整层全零，分母为零，实现必须明确处理，而不是让 NaN 静默决定选择。

一个活动度低的单元可能在稀有但重要的状态上有用。D 的覆盖、检测频率和阈值决定了“无用”的含义。把检测 batch 换成当前一个观测，会极大改变噪声和选择；这不是不需要评估的实现小节。

<a id="lesson-redo"></a>

## 3 · ReDo：检测休眠，再恢复可用的梯度通路

**算法：算法伪代码**

1. 按既定周期收集检测 batch；前向记录各层隐藏激活
1. 对每层计算 `mean(abs(h_i))` 和相对 score $s_i$
1. 选择 $s_i$ ≤阈值 的单元；阈值与检测频率是超参数
1. 对选中单元：
  1. 输入权重按原初始化分布重新采样，输入 bias 重新初始化
  1. 对应输出权重置零，避免新随机特征立即注入任意输出
  1. 清理相关优化器状态；核对 target network 与参数同步规则
1. 恢复正常 RL 更新；记录替换数量与替换前后的预测变化

“输出权重置零”只保证新随机特征在替换瞬间不产生新贡献，不保证旧输出完全保留。因为旧单元的贡献也被删掉了；只有旧贡献恰好为零时，替换瞬间才严格不变。低活动度通常减小扰动，但不是函数等价变换。

$$
f_{\rm after}(x)-f_{\rm before}(x)=-\sum_{i\in\mathcal R}v_i h_i(x)
$$

假定其余参数和 bias 不变且新输出权重设为零，R 是替换集合。这是精确的单隐藏层输出差，可用于检查重置实现有没有删除未选中的单元。

下一个样本上，若新激活 $h_{\rm new}\ne0$ 且误差非零，即使输出权重为零，输出权重仍有梯度。输出连接建立后，输入权重才重新获得下游梯度。新生单元需要学习时间，频繁再次替换会打断这一过程。

![单个ReLU单元的四帧连接图：输入为负导致休眠，回收后激活恢复但输出连接为零，随后两步梯度下降逐渐恢复两条学习通路。](https://yingwen.io/crl-figures/concept-crl-mechanisms-plasticity.svg)

固定 x=1、目标 y=1，用平方损失和步长 0.1；每帧的金色箭头表示该快照上非零的反向梯度。回收瞬间输出仍为零，第一步先把输出权重变为 0.05；下一步输入权重才变为 0.504875，图内作了四舍五入。下方圆点标出预激活在 ReLU 曲线上的位置。这是一个选中单元的精确机制演示，不是 ReDo 的整体训练结果。

<a id="experiment-extended-redo"></a>

### 实验：实验 · 回收低活动单元，不保证这个任务学得更好

按照相对活动度替换单元以后，网络是否确实更快拟合变化的函数？

**环境与可用信息。** 四维输入各自均匀分布于 [−1,1]。前半段目标是前三维的线性组合（系数 1、0.5、−0.3），加上第四维平方的 0.2 倍。后半段仅将线性部分反号。模型为 4–16–1 ReLU 网络，种子控制 PyTorch 默认初始化。

**设置。** 种子 0–4，各训练 1200 个样本，SGD 步长 0.03、无动量。每 50 步，用最近 32 个输入计算相对平均活动度，替换不超过 0.1 的单元。新入边取 U(−0.5,0.5)，偏置与出边清零。对照不替换，其余训练流相同。

**检验的机制。** 替换删除旧单元的输出贡献，并提供新的随机特征。出边为零只阻止新特征立即注入任意输出，不保证删掉旧贡献后函数完全不变。新单元还需要后续样本建立输出连接。

**测量。** 主图在同一组 64 个独立输入上冻结测当前函数 MSE。原始日志记录累计替换数；旧函数图仅描述旧答案的变化。由于两阶段答案冲突，旧误差大不能单独说明可塑性丧失。

```bash
python3 implementations/extended_adaptation/redo.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/extended-redo/curves.svg)

横轴：training_samples。纵轴：冻结当前函数均方误差。每种方法 1200 training_samples；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 五次运行共分别替换 4、6、6、3、6 个单元。第 600 步平均 MSE 为 0.00664，对照 0.00740；末尾则为 0.00838，对照 0.00704。初期微小优势没有转化为末端优势，不能从“发生了回收”推断有效学习容量一定提高。

**结论边界。** 只有一次函数切换，没有长期 aged/fresh 对照，因此不是长期可塑性丧失的验证。单隐层、无动量优化器，也未检验深层网络或 Adam 状态重置。

**继续实验。** 增加等数量随机替换对照，并记录每次替换前后的预测变化。随后固定新任务难度，比较长期训练网络与 fresh 网络的同预算学习速度，才检验“恢复可塑性”的因果说法。

[源码](../implementations/extended_adaptation/redo.py) · [逐种子记录](https://yingwen.io/crl-code/results/extended-redo/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/extended-redo/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/extended-redo/curves.json)

<a id="lesson-cbp"></a>

## 4 · CBP：特征的持续生成与检验

Continual Backpropagation 在常规反向传播之外，持续维护候选特征的效用与年龄，优先替换已经成熟却低效用的单元。效用不必等同于活动度：高活动但没有下游用途的特征，也可能值得替换。作者实现提供 contribution、zero-contribution、adaptable-contribution 等选项；必须写清实际使用哪一种。

选择规则可以从“删除它会改变多少下游信号”来理解。设某层的下一层预激活为 $z_j=\sum_i W_{{\rm out},ji}h_i+b_j$。删除第 $i$ 个单元后，$\Delta z_j=-W_{{\rm out},ji}h_i$，所以 $\mathbb E\|\Delta z\|_1=\mathbb E|h_i|\sum_j|W_{{\rm out},ji}|$。这直接解释了为何同时使用激活幅度与输出权重。原 Nature 论文使用输出权重绝对值的和；下面的教学函数使用均值，同一层输出宽度固定时二者只差常数，层内排序相同。

$$
\begin{aligned}c_{i,t}&=\mathbb E_B[|h_i|]\,\operatorname{mean}_j|W_{{\rm out},ji}|,\\u_{i,t}&=\beta u_{i,t-1}+(1-\beta)c_{i,t},\\\widehat u_{i,t}&=u_{i,t}/(1-\beta^{a_i}).\end{aligned}
$$

这是 contribution 效用的一个明确版本；ai 是自出生以来的更新次数。它同时考虑激活和下游权重，并用年龄进行 EMA 偏差修正。不要把此式泛称为所有 CBP 实验的唯一评分。

从零初始化展开递推有 $u_{i,t}=(1-\beta)\sum_{k=1}^{a_i}\beta^{a_i-k}c_{i,k}$，权重总和为 $1-\beta^{a_i}$。除以它能去掉初始零值带来的缩小；只有瞬时贡献均值不变时，才可解释成对该均值的无偏修正。持续变化时它仍是近期加权统计。并且小的预激活扰动还要经过后续非线性和读出，才成为预测误差或控制代价；这份效用是局部删除代价的代理。

$$
c_{i,t}^{\rm adapt}=\frac{\mathbb E_B[|h_i-\widehat\mu_i|]\,\operatorname{mean}_j|W_{{\rm out},ji}|}{\operatorname{mean}_k|W_{{\rm in},ik}|+\varepsilon}
$$

adaptable-contribution 风格还考虑非恒定贡献及输入权重尺度。本式显式加入 ε 以定义零分母；对应原代码/实验的均值、范数及 ε 选择应按实际配置复核。

**算法：算法伪代码**

1. 1. 用当前样本完成普通梯度更新；保留检测所需激活
1. 2. 单元年龄 ai += 1；更新激活均值与效用 EMA
1. 3. 只让 ai>成熟阈值 的单元进入候选集合 E
1. 4. replacement_credit += replacement_rate × |E|
1. 5. 本步替换 floor(replacement_credit) 个候选，扣掉整数部分
1. 6. 在候选内按 bias-corrected utility 从小到大选择
1. 7. 重采输入参数，输出连接归零；必要的 bias 补偿按指定变体处理
1. 8. 清零新生单元的 age、utility、activation statistics
1. 9. 清理相关 optimizer moments / coordinate step；继续反向传播

步骤 4–5 是累计预算变体，也可用概率方式实现小于一个单元的期望替换率。成熟期有明确的计算原因：新单元的输出权重为零，因此贡献评分也为零；若立即按最低分再次淘汰，它还没有机会完成“先学输出连接、再学输入特征”的两阶段过程。替换率为零时，算法退化成不替换的原学习器。

Adam 的一阶矩、二阶矩以及时间计数都与参数年龄有关。某些作者实现使用可逐坐标清零的 AdamGnT；普通框架只给整个张量一个 step 时，局部重置无法等同于所有坐标的全新 Adam。复现时应保留原优化器语义，或把替代策略作为一个新的实验条件。

<a id="experiment-extended-continual_backprop"></a>

### 实验：实验 · 生成得更多，不等于检验得更有效

成熟期、贡献效用与累计替换预算如何控制新特征的生命周期？这种机制在单次切换上有无收益？

**环境与可用信息。** 与 ReDo 配对使用同一四维切换回归、4–16–1 ReLU 网络和 64 个冻结评价输入。第 601 步目标线性部分反号，二次部分不变。

**设置。** 五种子各 1200 个样本，每步一次无动量 SGD，步长 0.03。贡献效用采用衰减 0.99 的 EMA，并按单元年龄校正；年龄严格大于 20 才可替换。每步按合格单元数乘 0.005 积累替换名额，选择最低效用者，重置规则同 ReDo。

**检验的机制。** 活动度乘下游权重用于估计贡献；成熟期保护新单元；分数名额累积后才执行整数次替换。它避免“每步不足一个就永远不换”，但也引入持续扰动。这里没有实现 centered/adaptable 效用或均值偏置补偿。

**测量。** 将 replaced_units 与当前函数 MSE 一起读。单元替换数不是奖励，也不是可塑性本身；需要检查替换后的实际学习。相同 SGD 次数不表示效用统计和选择过程没有额外计算。

```bash
python3 implementations/extended_adaptation/continual_backprop.py --steps 1200 --seeds 0 1 2 3 4 --out results/MY_NEW_RUN
```

在[完整代码包](https://yingwen.io/crl-code/learning-code.zip)的根目录运行。

![实测学习曲线](https://yingwen.io/crl-code/results/extended-continual_backprop/curves.svg)

横轴：training_samples。纵轴：冻结当前函数均方误差。每种方法 1200 training_samples；训练种子 0、1、2、3、4。曲线是种子均值，阴影是 ±1 个样本标准差，不是置信区间。相同交互量不保证相同计算量。

**结果分析。** 每个种子都累计替换了 85 个单元位置。末尾平均 MSE 为 0.00773，不替换基线为 0.00704。曲线接近，未显示该短实验中的稳定收益；不能因为替换比 ReDo 多，就认为维护了更多有用知识。

**结论边界。** 这是明确的 contribution 变体组件，不是 CBP 全论文复现。一次切换、16 单元与 1200 样本不足以支持长期可塑性结论。

**继续实验。** 固定总替换次数，再比较最低效用、随机选择与去掉成熟期。将目标改为多次同难度变化，同时加入 fresh 探针；把“选择规则有效”和“随机重生即可”分开检验。

[源码](../implementations/extended_adaptation/continual_backprop.py) · [逐种子记录](https://yingwen.io/crl-code/results/extended-continual_backprop/raw-runs.zip) · [配置与来源](https://yingwen.io/crl-code/results/extended-continual_backprop/manifest.json) · [绘图数据](https://yingwen.io/crl-code/results/extended-continual_backprop/curves.json)

<a id="rlss-renewal-consumer-contract"></a>

## 替换神经元改变了什么：删除扰动、成熟期与局部学习目标

Continual Backpropagation 将常规梯度学习与少量单元更新结合。它先估计单元在当前数据上的贡献，再在足够成熟的单元中选择低效用对象。重新初始化其输入连接，并把输出连接置零。后一步只保证新随机特征暂时不贡献输出，不保证删除旧单元时网络函数不变。

$$
y(x)=\sum_i v_i h_i(x),\qquad
v_j^+=0\quad\Longrightarrow\quad y^+(x)-y(x)=-v_jh_j(x).
$$

其余参数暂不改变。旧贡献只有本来为零，替换才对当前输出无影响。例如旧输出权重 0.3、激活为 1 与 0.7，两个输入的输出分别改变 −0.3 与 −0.21。

$$
u_i\leftarrow\eta u_i+(1-\eta)|h_i|\sum_k|v_{ik}|,\qquad
a_i\leftarrow a_i+1,\qquad E=\{i:a_i>m\}.
$$

这是原研究使用的 contribution utility 形式；a 是年龄，m 是成熟阈值。偏差校正及层间实现应按对应作者版本核对。效用是对当前分布的启发式估计，不是未来一切任务的重要性。

$$
B\leftarrow B+\rho|E|,\qquad
k=\min(\lfloor B\rfloor,|E|),\qquad B\leftarrow B-k.
$$

一种明确的批量小数预算实现：在本步合格集合中替换 k 个最低效用单元。原论文算法在小替换率下逐个处理；推广为批量时须写清合格集与预算的时序，不能把 ρ 的含义悄悄改成全层比例。

例如 512 个单元始终合格，替换率为十万分之一，10000 步积累 51.2 次请求，整数机制实现 51 次，余下 0.2 留待后续。实际训练中新生单元不合格，所以实际数量通常更少。若每步直接取整，则这个例子一次也不会替换。年龄不是装饰性统计，它决定测试窗口与实际探索速度。

删除和引入还具有不对称性。新单元输出权重为零时，其输入权重从主损失获得的梯度通常为零；输出权重仍可学习。新特征若长期不激活，或从未与主误差相关，就可能尚未获得使用机会便再次被淘汰。成熟期延缓淘汰，却不保证候选能得到有效训练信号。

$$
y=v\,\sigma(u^\top x),\qquad
\nabla_u L=\frac{\partial L}{\partial y}\,v\,\sigma'(u^\top x)x,\qquad
\frac{\partial L}{\partial v}=\frac{\partial L}{\partial y}\,\sigma(u^\top x).
$$

v=0 时，输入端主任务梯度为零，输出端梯度未必为零。这个代数事实说明候选学习的入口；它不证明必须采用某一种局部目标。

Sutton 的 Decentralized Neural Networks 讲义讨论另一条研究思路：保留已有效工作的主干，让边缘候选通过局部目标、连接建立和效用传播参与结构搜索。此处 DNNs 指“去中心化神经网络”，不是深度神经网络的通称。讲义把结构、权重和步长视为不同适应层，并明确尚无完整实证系统。

| 过程 | 局部反馈 | 与总体目标的关系 |
| --- | --- | --- |
| 权重学习 | 预测或控制损失的梯度 | 直接优化当前代理目标，仍受采样与自举限制 |
| 步长适应 | 参数变化对后续误差的影响 | 调节现有学习通路，不自动产生新特征 |
| 结构搜索 | 候选激活、被使用程度、贡献与年龄 | 需检验局部目标是否确实改善主任务或未来学习 |

“经常激活”不能替代“有助于学习”。恒为一的重复单元容易达到活跃目标，却可能没有新信息；“输出权重增大”也依赖参数单位。要检验局部目标，可在固定候选生成器下比较主任务学习速度、旧功能扰动与跨目标迁移，而不是只报告候选自身的分数。

接入多预测 CRL 时，一个单元可能同时服务奖励价值、GVF、option 停止判断和模型。删除它前，需要确定哪些消费者有权保护它；替换后则要维护相应的优化器统计、资格迹和模型坐标。仅在单一监督头上降低误差，没有证明这条依赖链已经闭合。

<a id="lesson-consumers"></a>

## 替换一个特征以后，谁还在使用旧知识

一个特征可能同时服务价值、策略、多个 [GVF](gvf.md) 和 [option model](options.md#lesson-option-models)。先看共享特征上的线性读出 $f^{(q)}(x)=\sum_i v_i^{(q)}h_i(x)$。替换集合 $\mathcal R$ 并把对应输出连接归零时，每个消费者都承受一项确定的删除：

$$
\Delta f^{(q)}(x)=-\sum_{i\in\mathcal R}v_i^{(q)}h_i(x),\qquad\mathbb E_{d_q}|\Delta f^{(q)}|\leq\sum_{i\in\mathcal R}|v_i^{(q)}|\,\mathbb E_{d_q}|h_i|.
$$

这是线性读出下的精确变化与三角不等式上界。每个问题可以有不同的参考分布 dq。只用当前 critic 的输出权重或当前访问分布评分，可能漏掉对另一预测头或稀有旧状态重要的特征。深层网络还要考虑下游变换。

若特征在参考分布上近乎常数，可先把被删贡献的均值并入读出偏置：$b_q^+=b_q+\sum_{i\in\mathcal R}v_i^{(q)}\mu_i$。输出变化就成为 $-\sum_{i\in\mathcal R}v_i^{(q)}(h_i-\mu_i)$。这解释了中心化贡献为何可能比单看活动幅度更合适，但它只补偿指定均值；输入相关的信息仍被删除。教学替换函数没有做这项偏置补偿。

预测扰动还要经过使用者，才成为行为代价。例如，删特征后两项动作价值都平移相同常数，贪心动作可能不变；若扰动使它们的大小次序交换，动作便立刻改变。一个 option 模型的终点预测变化还会影响多次规划备份。保持策略头的即时输出，并不自动保持所有预测头、模型和递归状态的语义。

时间信用也依赖这些坐标。[资格迹](credit.md)保存旧访问对参数的资格；新生特征若继承同一槽位的旧迹，下一条误差就会沿旧特征的资格写入新参数。清除被替换坐标的迹、优化器矩和元学习敏感度，可以定义一个明确的重新开始，但同时放弃那部分尚未兑现的旧信用。若希望无损地保留它，必须给出坐标变换或重新计算规则；仅把参数槽位编号沿用下来并不够。

严格流式条件下没有旧 buffer 来重建被删功能，低效用选择因而更依赖有限统计的时间尺度。可以把多预测头的重要性、历史参考分布与当前学习进展共同用于候选评分；这是一条待检验的设计方向。应先在副本上测各消费者的预测变化、固定预算适应和额外计算，再回到连续控制中检验收益。允许 replay 的网络重置实验则多了一条从旧经验重建的路径，不能直接搬成无 replay 的结论。

<a id="lesson-normalization"></a>

## 5 · NaP：归一化后的有效学习率

深入阅读：进一步推导归一化网络中的有效学习率，分析仍有梯度却学得慢的情况。

单元保持活动时，网络仍可能逐渐学得更慢。一个原因是归一化改变了参数尺度与函数变化的关系。考虑无偏置、忽略数值稳定项的理想尺度不变层：把进入归一化的权重乘以正数，输出保持不变。设损失也具有这种不变性，用链式法则可得：

$$
L(cw)=L(w),\quad \nabla L(cw)=c^{-1}\nabla L(w),\quad c>0.\qquad \eta_{\rm eff}^{\rm SGD}=\frac{\eta}{\|w\|^2}.
$$

令单位方向为 $w/\|w\|$，一次普通 SGD 的方向变化与权重范数的平方成反比。这里的有效学习率公式针对普通梯度步；归一化梯度的对应尺度为 $\eta/\|w\|$，不能直接把两者等同于所有 Adam 更新。

因此，同样的标称步长下，权重范数从 2 增长到 4，理想 SGD 的有效步长变为原来的四分之一。这个效应不会被“休眠单元比例仍然很低”排除。Lyle 等在 NeurIPS 2024 的 Normalize-and-Project（NaP）将非线性之前的归一化，与定期恢复每层初始权重范数结合，使学习率调度显式化。

$$
\widetilde W_\ell\leftarrow\operatorname{OptimizerStep}(W_\ell),\qquad W_\ell^+\leftarrow\frac{\|W_{\ell,0}\|_F}{\|\widetilde W_\ell\|_F}\widetilde W_\ell.
$$

式中假定投影前范数非零。投影保持权重方向，仅恢复范数。归一化的可学习缩放与偏置是另一组参数，需要按具体变体单独约束；它们不自动满足这里的尺度不变条件。

归一化还有第二个作用：均值和方差使单元之间的梯度耦合，位于 ReLU 前的归一化可以给部分不激活的预激活量传递其他单元的梯度。这与直接替换单元是不同机制。有效学习率保持恒定也并非总是最优：价值估计需要一定程度的收敛，论文中的部分 Rainbow 实验仍需要显式衰减。研究问题因而是适应速度与估计噪声的调度，而非无条件维持最大更新幅度。

随机环境把这种区分变得更重要。Bordne 与 Biedenkapp（EWRL 2026）在三个 DMC 任务的 SAC 训练中分别加入观测和动作噪声，同时记录回报、critic 学习探针和 Q 值。他们以当前预测加抽样扰动作为探针目标，使起始损失较可比；这是局部拟合能力的诊断，与本章独立训练/评价数据的协议可分别报告。

在较难的所测条件下，重置常改善学习；跨任务汇总时，观测噪声下叠加 LayerNorm 未胜过单独重置。严重失败还伴随 Q 值急剧增大，较容易任务则出现噪声下更高的探针可塑性。这些单任务实验使我们需要同时检查“还能否拟合固定目标”与“在线 TD 目标是否稳定”。实际诊断可冻结探针标签和数据，再观察改变噪声、重置或归一化后，哪一项先恢复；观测噪声带来的部分可观测性也应单独控制。[原文 §4–6](https://andrebiedenkapp.github.io/assets/pdf/paper/26-ewrl-plastic.pdf)。

<a id="lesson-churn"></a>

## 6 · C-CHAIN：控制更新在参考状态上的影响

深入阅读：用参考状态上的预测变化约束干扰，讨论保持与必要适应的取舍。

共享参数让一次局部学习同时改变其他输入的预测。称这种改变为 prediction churn。设当前训练输入为 $x$，参考输入为 $\bar x$，标量网络为 $f_\theta$，一次梯度步为 $\Delta\theta=-\eta\nabla_\theta L_x$。对参考预测做一阶展开：

$$
\begin{aligned}\Delta f(\bar x)&\approx\nabla_\theta f_\theta(\bar x)^\top\Delta\theta\\&=-\eta\underbrace{\nabla_\theta f_\theta(\bar x)^\top\nabla_\theta f_\theta(x)}_{K_\theta(\bar x,x)}\frac{\partial L_x}{\partial f_\theta(x)}.\end{aligned}
$$

$K_\theta$ 是两个输入的梯度内积，即经验神经切线核的一个元素。它可以为正或负；在 $x$ 上降低损失并不意味着在 $\bar x$ 上也降低损失。该一阶式在更新较小时解释局部干扰，并非有限大步长下的精确等式。

ICML 2025 的 C-CHAIN 在参考状态上约束这种变化。其值函数形式可写成下面的函数空间正则：当前网络在参考状态的全部动作价值，靠近近期冻结网络的输出。它与 EWC 的参数距离不同，也与长期保存旧任务教师不同，主要控制训练过程中的预测变化。

$$
L(\theta)=L_{\rm TD}(\theta)+\lambda\,\mathbb E_{s\sim D_{\rm ref}}\|Q_\theta(s,\cdot)-\operatorname{sg}[Q_{\theta^-}(s,\cdot)]\|_2^2.
$$

$\theta^-$ 是近期参数快照，$\operatorname{sg}$ 表示停止梯度。作者 MinAtar 实现独立采样 TD batch 与参考 batch，因此样本可能重叠；它还维护有界的历史网络，并根据损失尺度调节正则系数。

**算法：算法伪代码**

1. 从经验库采样 TD batch，构造停止梯度的 bootstrap target
1. 独立采样参考状态，选择一个近期冻结网络 $Q_{\theta^-}$
1. 计算 TD 损失与参考状态上的全动作价值差
1. 对两项加权和反向传播并更新当前网络
1. 按配置保存近期网络快照、淘汰超出窗口的快照
1. 用运行中的损失统计调整正则强度，记录预测改变与新目标学习速度

实现不需要显式构造完整神经切线核；核只用于解释干扰机制。过强的约束会妨碍必要的价值修正，参考分布遗漏的状态也不受保护。参考 batch、历史网络和额外前向传播均有成本，所以该方法的经验重放条件与严格流式学习不同。一个直接实验是匹配更新数与内存预算，在固定 probe 上比较替换、范数投影和短期函数约束分别改善哪类故障。

<a id="lesson-example"></a>

## 7 · 手算：替换扰动与学习恢复

深入阅读：逐项核算一次特征替换的即时输出扰动与后续学习变化。

网络只有两个单元，x=1，输入权重 u=[−1,1]、输出权重 v=[0.5,2]。激活 h=[0,1]，输出 f=2；活动度的层均值为 0.5，scores=[0,2]，阈值 0.1 只选第一个。把它的输入改成 0.5、输出改成 0，输出仍为 2，因为旧贡献为零。

若误选第二个单元，它的旧贡献为 2，输出会从 2 降为 0。这个变化不是数值误差，而是更新定义直接造成的。对神经网络实验应记录预测扰动、策略 KL 和短期回报下降，不能只记录 dormant 数量减少。

再用单单元学习 y=1。aged 网络 u=−1、v=1，在 x=1 上所有相关梯度为零，平方损失一直为 0.5。fresh 网络 u=0.5、v=0，第一步 α=0.1 更新输出权重 v→0.05，输入 u 暂时不变；此后输出通路非零，输入也能学习。两者用相同 20 次样本训练，本页程序验证后者损失下降。这个人为构造只证明死 ReLU 机制，不代表所有 aged 网络都会失效。

成熟期例子：三个单元效用 [0,0.1,1]、年龄 [0,21,21]，阈值 20，rate=0.25。最年轻的单元虽效用最低也不进入候选；两个候选每步积累 0.5 替换名额，第二步才替换效用 0.1 的成熟单元。

<a id="lesson-code"></a>

## 8 · 替换器实现与诊断实验

单输入/单输出隐藏层的机制实现；moment 数组分别表示被替换输入/输出权重的状态。完整网络还需处理 bias、卷积轴和 target 参数。

```python
def redo_indices(mean_abs_activations, threshold=0.1):
    """Relative activity score; an all-zero layer is entirely dormant."""
    if not mean_abs_activations:
        return []
    denominator = sum(mean_abs_activations) / len(mean_abs_activations)
    if denominator == 0:
        return list(range(len(mean_abs_activations)))
    return [i for i, a in enumerate(mean_abs_activations)
            if a / denominator <= threshold]


def cbp_candidates(utilities, ages, maturity, rate, credit=0.0):
    """Accumulated-budget variant: protect young units, rank eligible utility."""
    if not 0 <= rate <= 1:
        raise ValueError("replacement rate outside [0,1]")
    eligible = [i for i, age in enumerate(ages) if age > maturity]
    credit += rate * len(eligible)
    count = min(len(eligible), int(credit))
    selected = sorted(eligible, key=lambda i: (utilities[i], i))[:count]
    return selected, credit - count


def cbp_contribution_utility(old_utility, mean_absolute_activation,
                             mean_absolute_outgoing, age, decay=0.99):
    """EMA plus age correction for the explicitly chosen contribution variant."""
    if age < 1 or not 0 <= decay < 1:
        raise ValueError("increment age before updating utility")
    instantaneous = mean_absolute_activation * mean_absolute_outgoing
    updated = decay * old_utility + (1. - decay) * instantaneous
    return updated, updated / (1. - decay**age)


def recycle_one_unit(incoming, outgoing, first_moment, second_moment,
                     index, new_incoming):
    """Scalar-input, scalar-output hidden unit; reset its incoming/outgoing state.
    first/second_moment[i] each contain [incoming_parameter, outgoing_parameter].
    """
    incoming[index] = new_incoming
    outgoing[index] = 0.0
    first_moment[index] = [0.0, 0.0]
    second_moment[index] = [0.0, 0.0]


def relu_prediction(incoming, outgoing, x=1.0):
    return sum(v * max(0., u*x) for u, v in zip(incoming, outgoing))


def train_relu_toy(incoming, outgoing, target=1., steps=20, alpha=0.1):
    incoming, outgoing = incoming[:], outgoing[:]
    for _ in range(steps):
        error = target - relu_prediction(incoming, outgoing)
        old_u, old_v = incoming[:], outgoing[:]
        for i in range(len(incoming)):
            incoming[i] += alpha * error * old_v[i] * (old_u[i] > 0)
            outgoing[i] += alpha * error * max(0., old_u[i])
    return 0.5 * (target - relu_prediction(incoming, outgoing)) ** 2


def plasticity_demo():
    u, v = [-1., 1.], [0.5, 2.]
    before = relu_prediction(u, v)
    moments1, moments2 = [[3., 4.], [3., 4.]], [[5., 6.], [5., 6.]]
    recycle_one_unit(u, v, moments1, moments2, 0, 0.5)
    print("plasticity", {"redo_selected": redo_indices([0., 2.]),
          "prediction_before_after": (before, relu_prediction(u, v)),
          "cleared_optimizer_state": moments1[0] + moments2[0],
          "matched_probe_loss_aged": train_relu_toy([-1.], [1.]),
          "matched_probe_loss_fresh": train_relu_toy([0.5], [0.]),
          "scope": "synthetic dead-ReLU mechanism, not an empirical LoP diagnosis"})
```

输出应包括替换前后 (2,2)、清零的 optimizer state，以及 aged/fresh probe loss

```sh
python examples/lifelong_algorithms_lab.py plasticity
python examples/lifelong_algorithms_lab.py test
```

示例覆盖全零层、相对活动度、成熟期、分数替换预算与替换扰动。fresh 网络是固定数据诊断的对照条件。作者 ReDo 代码中的 weight_recyclers.py、CBP 的 lop/algos/gnt.py 分别给出完整框架中的选择与生成流程；C-CHAIN 的 MinAtar agent 展示参考采样和快照队列。本页小实验实现替换机制，不包含这些论文的完整深度 RL 训练。

| 函数 | 对应推导中的哪一步 | 调用者还须处理什么 |
| --- | --- | --- |
| redo_indices | 由每个单元的平均绝对活动算层内相对分数；全零层全部入选 | 激活采集、检测周期和实际网络层 |
| cbp_contribution_utility | 更新贡献 EMA，按已递增的 age 做偏差修正 | 后续候选筛选及新生统计清零 |
| cbp_candidates | 只在成熟集合中排序；累计替换名额并保留小数余额 | 真正生成新特征和更新年龄 |
| recycle_one_unit | 重采指定输入、输出归零，并清两个矩数组 | bias、逐坐标 step、迹、其他预测头与 target 网络 |
| train_relu_toy | 读取旧输入/输出权重，同步做固定目标梯度步 | 无多层网络、探索、自举或 Adam；不代表完整论文实验 |

- 先做固定数据 probe，分离探索与学习能力；再返回在线 RL 看是否仍有实际收益。
- 匹配参数量与 FLOPs，特别是 CReLU 改变特征维度、模块扩容和 injection 增加容量的情况。
- 与随机等量替换、只清 optimizer state、全网络重置和只改激活函数对照，辨认选择规则是否真的有用。
- 同时报告新任务适应、旧技能保留、替换后短期损失和每步额外成本，避免只选择一种有利指标。

<a id="lesson-branches"></a>

## 9 · 干预位置与选择依据

深入阅读：根据退化原因选择干预，并分别检查知识保留与新问题学习。

$$
\operatorname{CReLU}(z)=[\max(0,z),\max(0,-z)]
$$

对 z≠0 至少一支对 z 有非零导数，但最终输出梯度仍取决于下游权重与损失。它不是整个网络梯度永远不为零的保证，并且会改变宽度/参数预算。

| 方法线 | 干预对象 | 应排除的替代解释 |
| --- | --- | --- |
| ReDo | 当前分布下低活动单元周期性回收 | 收益是否仅来自额外随机扰动？ |
| CBP / generate-and-test | 低效用成熟特征的持续替换 | 是否只需随机替换？成熟期与效用是否都重要？ |
| CReLU / 激活改造 | 激活及梯度通路 | 增加的特征维度是否解释提升？ |
| LayerNorm / 参数正则 | 尺度、曲率与优化几何 | 是否只是更好的超参数而非长期机制？ |
| Primacy-bias resets | 周期重置一部分网络，保留经验 | 收益可能依赖 replay 重建能力，不自动适于无 replay 流式学习 |
| Plasticity injection | 接入可训练新分支并控制初始输出 | 额外参数和 optimizer state 如何计入固定预算？ |

最重要的研究逻辑是先诊断，再选干预。若 aged 与 fresh 在同一数据上学得同样快，但在线 aged 不再到达新状态，应转向探索或 agent state；若输出尺度越来越大而 dormant 比例不变，优先检查优化几何；若旧技能特别珍贵，则回收规则要加入保留约束。

<a id="research-parseval-geometry"></a>

## Parseval：持续维护尺度与方向几何

ReDo/CBP 改动单元，Parseval regularization 则让仍在使用的矩阵保持较好的几何条件。它不等待某个单元完全休眠才干预，而是持续约束不同输出方向的相关性与尺度。其机制应与 NaP 的有效学习率和 C-CHAIN 的函数变化分别检验。

$$
\Omega(W)=\lambda\|WW^\top-sI\|_F^2,\qquad\nabla_W\Omega=4\lambda(WW^\top-sI)W
$$

W 为输出维度×输入维度。对平方 Frobenius 范数微分得到该梯度；s>0。它约束行内积与范数，不把坐标锁到旧值。

$$
\|WW^\top-sI\|_F^2=\sum_i(\|w_i\|^2-s)^2+\sum_{i\ne j}\langle w_i,w_j\rangle^2
$$

范数与方向是两个独立作用。weight decay 不能保证方向分开；每行归一化也不能消除两行重合。

两行均为 $(1,0)$、$s=1$，范数已正确，但正则仍为 $2\lambda$；改成 $(1,0)$ 与 $(0,1)$ 后为零。若 W 有 3 行只有 2 列，$\operatorname{rank}(WW^\top)\le2$，不可能等于 $sI_3$。需改变约束侧、分组或层宽，不能把不可实现的零残差当目标。

非零奇异值靠近同一尺度不等于整个非线性网络等距。激活导数、输入分布、残差与输出层仍决定完整 Jacobian。作者保留输出层自由度，并检验额外尺度与层等容量补偿，说明可学习性与函数表达之间存在取舍。

**算法：目标级骨架；并非独立参数 reset**

1. 按基础 RL 方法构造 actor/critic 代理损失与停止梯度 target
1. 对选定隐藏层计算 `W @ W.T`，并加入 $λ\|WW^T-sI\|^2$
1. 同一次反传更新任务与正则项；保留 optimizer 状态
1. 记录任务误差、Gram 残差、奇异值和新目标 probe

| 消融 | 隔离的因素 |
| --- | --- |
| 仅正交初始化 vs 持续正则 | 有利几何是否在训练中流失。 |
| 仅范数、仅非对角项、完整项 | 尺度与方向的不同贡献。 |
| 固定宽度 vs 容量补偿 | 改善是否依赖新增表达能力。 |

可从 wechu/parseval_reg 的 agent.py 对应正则，main.py 对应任务序列。验证时匹配交互、梯度次数、参数量和调参预算，并比较 aged/fresh 在同一新目标数据上的拟合速度及旧功能。稳定秩提高本身不证明任意未来目标可学习；几何改善而真实控制无改善也应保留。

<a id="lesson-check"></a>

## 10 · 自测与研究练习

- 为什么 low-utility 不是“永远无用”？效用根据某个近期分布估计；稀有历史状态和未来任务可能改变结论。
- 为什么新输出权重置零后仍能成长？第一步输出权重有非零激活梯度，后续输入权重才得到非零下游信号。
- 清零全张量 Adam step 可以当作局部重置吗？不可以，它也改变未替换坐标的偏差修正；必须明确这项额外干预。
- ReDo 能否证明可塑性下降就是 dead neurons？不能。存在其他机制，也存在休眠是结果而非根因的可能。

动手题：在 toy 网络中让旧单元的小贡献从 0 缓慢变到 0.1，逐次核对替换前后输出差正好等于删掉的贡献。再把随机 reset 与 score-based reset 设置为相同替换次数，用 matched probe 比较改善，避免将替换率不同误判为选择规则更好。

## 本章的实验设计

使用新任务学习速度检验可塑性。保留原参数与重新初始化的对照必须具有相同的新增数据预算。

设定：让 learner 具有不同训练年龄，再在相同新增数据下学习新信号。用 fresh 模型、优化器重置与不同替换机制拆解原因。

- 新信号对普通强基线确实可学。
- 替换包含按算法约定处理入边、出边和 optimizer 状态。
- 诊断探针引入的任务变化与原主轨迹分开。

对照：完全 fresh、仅重置优化器；相同替换数量的随机替换；相同新增数据、容量与总计算

记录：新任务达到标准的样本量；aged/fresh 学习曲线与失败；激活、梯度、秩及替换成本

[具体测试规程](../docs/experiment-handbook.md#handbook-continual)

## 学习与研究衔接

旧问题仍答得好，不能说明网络还学得会新问题。可以先给训练年龄不同的网络相同新数据和目标，比较它们的学习速度，再回到在线交互检查探索影响。替换特征或调整有效步长若有帮助，还要测量旧功能的损失，以及技能和模型重新适配的成本；这些结果共同决定干预是否值得长期使用。

[分册导读](../docs/learning-route-continual-rl.md) · [本章实验](https://yingwen.io/zh/continual-rl/labs/#experiment-plasticity) · [资源](https://yingwen.io/zh/continual-rl/library/?chapter=plasticity) · [学者](https://yingwen.io/zh/continual-rl/people/?chapter=plasticity)

## 持续强化学习：近期研究与原始实现

从问题设定进入机制，再比较证据、成立条件和实验资源。理论结果、算法实验、基准和架构观点承担不同作用。

### 问题支线

#### 流式协议下的稳定更新

只有当前经验和有限状态时，学习如何保持数值稳定与有效信用分配？

流式是数据使用协议，资格迹是时间信用机制，归一化和 Intentional 是尺度控制，Adam 是一种自适应更新。先对齐允许保存什么、每步计算多少和使用哪版算法，再比较效果。

- [Streaming Deep Reinforcement Learning Finally Works](https://yingwen.io/zh/continual-rl/research/#recent-stream-x)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)

#### 新学习能力、知识保留与负迁移

学得慢是失去学习能力、旧知识有害，还是必须保护的知识发生干扰？

可塑性看新知识能否学会，保留看旧能力是否下降，负迁移看过去学习是否使新任务差于从头学习。网络回收、函数正则、双学习器和预训练适配对应不同机制，不应只用一个平均回报解释全部现象。

- [The Dormant Neuron Phenomenon in Deep Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-redo-dormant-neurons)
- [Understanding Plasticity in Neural Networks](https://yingwen.io/zh/continual-rl/research/#recent-understanding-plasticity)
- [Loss of plasticity in deep continual learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-backpropagation)
- [Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn](https://yingwen.io/zh/continual-rl/research/#recent-c-chain-churn)
- [Prevalence of Negative Transfer in Continual Reinforcement Learning: Analyses and a Simple Baseline](https://yingwen.io/zh/continual-rl/research/#recent-reset-and-distill)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)
- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-upgd-utility-protection)
- [Parseval Regularization for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-parseval-continual-geometry)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 学习规则本身的适应

谁在调整学习过程，依据哪些经验，付出多少外部训练成本？

在线步长元梯度、跨任务算法发现、知识整合与局部更新控制并非同一设定。逐项写清智能体内部的更新、设计者的预训练和调参，以及测试时仍能变化的量，才能判断真正的适应来自哪里。

- [Step-size Optimization for Continual Learning](https://yingwen.io/zh/continual-rl/research/#recent-step-size-optimization)
- [Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-fame-fast-meta-learners)
- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-trac-online-regularization)

#### 持续问题与可比较实验

一个基准究竟检验了哪种困难，又把哪些适应工作留给设计者？

离线固定数据、已知任务序列、持续动态世界和预训练模型适配具有不同资源与信息。需要记录任务边界、未来信息、重置、预训练、数据访问和总计算，而不是把所有 benchmark 分数放进同一张排名表。

- [The Cell Must Go On: Agar.io for Continual Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-agarcl)
- [Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning](https://yingwen.io/zh/continual-rl/research/#recent-continual-vla-simple-recipe)

#### 完整智能体与研究基础

长期能力应怎样定义，各个机制又怎样共同产生它？

形式化论文规定对象与条件，架构路线提出组织方式，算法实验检验局部机制。撤掉阶段间冻结后，一个模块会改变另一个模块的学习问题；有限预算应优先维护哪条知识，成为新的决策。先检验两模块反馈和资源分配，再扩大整机，而不是由组件分别有效推断长期组合收益。

- [Plasticity as the Mirror of Empowerment](https://yingwen.io/zh/continual-rl/research/#recent-plasticity-mirror-empowerment)

### The Dormant Neuron Phenomenon in Deep Reinforcement Learning

Ghada Sokar, Rishabh Agarwal, Pablo Samuel Castro, Utku Evci

ICML 2023 · 2023 · 支持方法与理论

#### 研究问题

网络参数数量没有变，为什么越来越多隐藏单元不再对输出产生有效贡献？

#### 关键机制

ReDo 用相对激活量识别低活跃单元，重新初始化其输入连接，并处理输出连接，使被回收单元可以重新参与学习。它针对的是可用表示容量，而不是直接惩罚旧任务表现变化。

#### 证据

论文记录深度 RL 中的休眠单元现象，并比较回收机制对多个任务学习的影响。实现进入作者所在团队的 Dopamine 代码库。

#### 条件与限制

低激活只是可塑性问题的一种诊断，不能覆盖曲率变化、优化器状态和负迁移。回收也可能损坏低频但重要的旧知识，需要与保留指标共同评价。

#### 阅读与实验

同时记录休眠比例、新目标拟合速度与旧任务冻结表现。三者发生不同方向变化时，不要用单个表示指标替代整个持续学习结论。

#### 原文与相关入口

- [ICML 2023 原文](https://proceedings.mlr.press/v202/sokar23a.html)：休眠定义、回收规则与实验。
- [Dopamine ReDo 实现](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：作者团队公开代码中的 ReDo 模块。

#### 作者代码

[论文作者团队发布的实现，不是本教材的简化版本。](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)

Dopamine 中的 ReDo 神经元回收与实验实现。

### Understanding Plasticity in Neural Networks

Clare Lyle, Zeyu Zheng, Evgenii Nikishin, Bernardo Avila Pires, Razvan Pascanu, Will Dabney

ICML 2023 · 2023 · 支持方法与理论

#### 研究问题

学习变慢一定意味着网络已饱和或特征秩下降吗？

#### 关键机制

论文通过新目标拟合实验研究可塑性，并分析优化几何与曲率的影响。某些表示统计与学习能力下降会同时出现，却不是所有设置中的充分解释。评价对象从“网络看起来是否健康”转向“在受控更新预算内还能学会什么”。

#### 证据

受控探针与 RL 实验展示了不同机制之间的区别，并检验网络设计和优化过程的作用。它为可塑性研究提供诊断方式，而不是单一通用修复算法。

#### 条件与限制

探针目标、优化器和步数会改变测得的可塑性。相关性不等于所有控制任务中的因果机制；探针训练也不能写回被评价的在线智能体。

#### 阅读与实验

复制同一个检查点，在副本上拟合两类新目标。保持训练预算一致，并报告探针过程与真实环境回报之间的区别。

#### 原文与相关入口

- [ICML 2023 原文](https://proceedings.mlr.press/v202/lyle23b.html)：可塑性探针、优化几何与诊断边界。

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

### Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn

Hongyao Tang, Johan Obando-Ceron, Pablo Samuel Castro, Aaron Courville, Glen Berseth

ICML 2025 · 2025 · 直接研究持续学习

#### 研究问题

一次局部更新为什么会在其他输入上引发大幅预测变化，并损害后续学习？

#### 关键机制

C-CHAIN 抑制相对于近期参考网络的函数输出变化，降低一次更新在其他样本上造成的 churn。论文把该现象与经验神经切线核及学习动力学联系起来。正则化对象是函数变化，不是直接把所有参数锁在旧值附近。

#### 证据

作者在持续 Gym Control、ProcGen、DMC 和 MinAtar 序列中比较，并提供对应环境和算法代码。

#### 条件与限制

近期函数稳定性不等于长期任务知识保留；参考样本和参考网络也占资源。若环境突然发生真实变化，过强抑制输出变化可能延迟必要适应。

#### 阅读与实验

将 churn 按旧分布、新分布分别计算，并同时画适应速度。这样才能区分“减少无关干扰”和“阻止有用改变”。

#### 原文与相关入口

- [ICML 2025 原文](https://proceedings.mlr.press/v267/tang25g.html)：机制、理论分析与持续实验。
- [作者代码](https://github.com/bluecontra/C-CHAIN)：四类持续环境的基线和 C-CHAIN 对照实现。

#### 作者代码

[作者仓库，README 说明依赖的 TRAC、CleanRL 与 MinAtar 基础实现。](https://github.com/bluecontra/C-CHAIN)

持续控制环境与 C-CHAIN 对照实验。

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

### Parseval Regularization for Continual Reinforcement Learning

Wesley Chung, Lynn Cherif, David Meger, Doina Precup

NeurIPS 2024 · 2024 · 直接研究持续学习

#### 研究问题

仅在初始化时保持良好的权重几何，是否足以让很晚出现的新任务仍容易学习？

#### 关键机制

在选定隐藏层加入 $\lambda\|WW^\top-sI\|_F^2$，持续约束行向量的范数与角度；输出层及额外尺度设计保留表达能力。它维护学习的几何条件，并不直接保存旧任务标签或预测。

#### 证据

作者在 Gridworld、CARL、MetaWorld 任务序列中检验，并拆分范数与角度约束。稳定秩、Jacobian 与熵属于诊断量，不单独构成可塑性或保留的因果证明。

#### 条件与限制

约束会限制函数类；输出行数大于输入维度时，全部行正交不可实现。非线性门控仍能切断梯度。有限任务序列的结果不保证无限生命内有效，也不是无任务信息的万能机制。

#### 阅读与实验

同预算比较仅初始化正交、持续范数约束、持续角度约束和完整正则；同时记录新目标拟合、真实回报、旧功能与额外计算。

#### 原文与相关入口

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。
- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

#### 作者代码

[仓库明确标为 NeurIPS 2024 官方实现。](https://github.com/wechu/parseval_reg)

PPO、任务序列、正则化与网络结构消融。


<a id="chapter-code"></a>

## 下载与运行

替换、optimizer 状态与合成 aged/fresh 机制诊断；不是 ReDo/CBP 全论文复现。

[下载 lifelong_algorithms_lab.py](../examples/lifelong_algorithms_lab.py)

```sh
python examples/lifelong_algorithms_lab.py plasticity
```

<a id="lesson-sources"></a>

## 参考文献与实现

- [Bordne & Biedenkapp — Towards Understanding the Impact of Plasticity Loss on RL in Stochastic Environments · EWRL 2026](https://andrebiedenkapp.github.io/assets/pdf/paper/26-ewrl-plastic.pdf)：噪声、局部学习探针与 TD 稳定性的对照实验。

- [Sokar et al. · The Dormant Neuron Phenomenon in Deep RL](https://proceedings.mlr.press/v202/sokar23a.html)：ReDo 活动度定义、重置流程与实验。

- [ReDo 作者实现 · Google Dopamine](https://github.com/google/dopamine/tree/master/dopamine/labs/redo)：作者发布的 Dopamine 版本；核心文件为 weight_recyclers.py 及各 recycled agent。

- [Dohare et al. · Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7)：Methods 的 CBP 更新、成熟期与替换规则；Extended Discussion 区分可塑性、保持与未来迁移。

- [Loss of plasticity · 作者实现](https://github.com/shibhansh/loss-of-plasticity)：使用对应任务的原配置，检查输入连接、输出连接及 optimizer 状态的重置。

- [Lyle et al. · Understanding Plasticity in Neural Networks](https://proceedings.mlr.press/v202/lyle23b.html)：强调曲率等机制，避免将全部 plasticity loss 简化为 dead ReLU。

- [Abbas et al. · Loss of Plasticity in Continual Deep RL](https://proceedings.mlr.press/v232/abbas23a.html)：循环 Atari 与 CReLU 的原始研究，配合预算匹配理解激活改造。

- [Nikishin et al. · The Primacy Bias in Deep RL](https://proceedings.mlr.press/v162/nikishin22a.html)：保留经验但周期重置部分网络的机制；与严格 streaming 条件不同。

- [Lyle et al. · Normalization and effective learning rates in reinforcement learning](https://papers.nips.cc/paper_files/paper/2024/hash/c04d37be05ba74419d2d5705972a9d64-Abstract-Conference.html)：NeurIPS 2024：NaP、归一化的梯度耦合，以及显式与隐式学习率调度。

- [Tang et al. · Mitigating Plasticity Loss in Continual RL by Reducing Churn](https://arxiv.org/abs/2506.00592)：ICML 2025：用参考状态的函数空间约束控制预测改变。

- [C-CHAIN 作者实现 · MinAtar Double DQN](https://github.com/bluecontra/C-CHAIN/blob/main/crl_minatar/agents/double_dqn_c_chain.py)：参考 batch、全动作 Q 正则、近期网络队列和损失尺度自适应；完整运行还需对应环境及训练配置。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/5b76d77e7095c6480ed827b85f0c2878-Paper-Conference.pdf)：Algorithm 1–2、正则化联系与持续实验。

- [作者论文 v3](https://arxiv.org/html/2405.16642v3)：区分凸理论、RL 经验结果与初期表现限制。

- [Fast TRAC: A Parameter-Free Optimizer for Lifelong Reinforcement Learning · 作者实现](https://github.com/ComputationalRobotics/TRAC)：trac.py、PyTorch/JAX optimizer 包与控制/视觉实验。 作者项目页与仓库均明确标为官方实现。

- [ICLR 2024 原文](https://proceedings.iclr.cc/paper_files/paper/2024/file/8e5f0591943d8dae5702af12dcdcd2f6-Paper-Conference.pdf)：效用定义、近似、不同 UPGD 变体与 PPO 实验。

- [作者预印本](https://arxiv.org/abs/2404.00781)：流式监督协议与 RL 证据范围。

- [Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning · 作者实现](https://github.com/mohmdelsayed/upgd)：权重/特征效用实验、流式任务及 PPO 实现。 论文首页明确链接的作者仓库；README 的短实现是一个指定变体。

- [NeurIPS 2024 原文](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6df4efa20adf8ef9acb80e94072a429-Paper-Conference.pdf)：目标函数、容量限制、角度/范数消融及持续任务协议。

- [作者版本记录](https://arxiv.org/abs/2412.07224)：正式会议年份为 2024。

- [Parseval Regularization for Continual Reinforcement Learning · 作者实现](https://github.com/wechu/parseval_reg)：PPO、任务序列、正则化与网络结构消融。 仓库明确标为 NeurIPS 2024 官方实现。
