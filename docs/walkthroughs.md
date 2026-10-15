# 按教材查找逐步算例

先在正文完成手算，再运行对应的独立脚本。输入、运行命令和图中数值的解释见各节；完整训练与基线另在实验入口中提供。

## 第 I 册 · 经典强化学习

### 多臂老虎机：估计、探索与直接策略学习

- [7 · 源码与实验观察](../foundations/tabular/bandits.md#lesson-code) · [bandit_tracking_walkthrough.py](../tutorials/bandit_tracking_walkthrough.py)
- [7 · 源码与实验观察](../foundations/tabular/bandits.md#lesson-code) · [bandit_exploration_walkthrough.py](../tutorials/bandit_exploration_walkthrough.py)

### MDP、回报与价值：序列决策的数学对象

- [7 · 模型表示与实验观察](../foundations/tabular/mdps.md#lesson-code) · [mdp_dp_walkthrough.py](../tutorials/mdp_dp_walkthrough.py)

### Monte Carlo：完整回报、探索控制与离策略评价

- [3.1 · 估计改变行动，行动产生下一条回报](../foundations/tabular/monte-carlo.md#mc-control-loop) · [mc-control-walkthrough.py](../tutorials/mc-control-walkthrough.py)
- [5.6 · 从一条真实路径走过 Q、C、W 和下一版行为](../foundations/tabular/monte-carlo.md#mc-offpolicy-loop) · [mc-offpolicy-control-walkthrough.py](../tutorials/mc-offpolicy-control-walkthrough.py)
- [8 · 实验观察与估计诊断](../foundations/tabular/monte-carlo.md#lesson-code) · [monte-carlo-walkthrough.py](../tutorials/monte-carlo-walkthrough.py)

### TD 预测与控制：SARSA、Expected SARSA、Q-learning 和 Double Q

- [3 · 控制的三种下一动作](../foundations/tabular/temporal-difference.md#td-control-targets) · [control_targets_walkthrough.py](../tutorials/control_targets_walkthrough.py)

### 多步学习：n-step、Tree Backup 与 Q(σ)

- [6 · 完整 n-step 执行顺序](../foundations/tabular/multistep.md#multistep-algorithm) · [control_targets_walkthrough.py](../tutorials/control_targets_walkthrough.py)

### 学习与规划：Dyna、优先扫描和执行时搜索

- [7 · 可运行模型循环](../foundations/tabular/planning.md#lesson-code) · [planning-budget-walkthrough.py](../tutorials/planning-budget-walkthrough.py)
- [7 · 可运行模型循环](../foundations/tabular/planning.md#lesson-code) · [stochastic-model-walkthrough.py](../tutorials/stochastic-model-walkthrough.py)
- [7 · 可运行模型循环](../foundations/tabular/planning.md#lesson-code) · [dyna-planning-walkthrough.py](../tutorials/dyna-planning-walkthrough.py)

### 函数逼近预测：从回归到 TD 固定点

- [共享几何 · 一次更新究竟帮助了谁？](../foundations/approximation/prediction.md#prediction-sharing) · [shared_gradient_walkthrough.py](../tutorials/shared_gradient_walkthrough.py)

### 特征、泛化与半梯度控制

- [4.1 · 在走廊里执行更新，再看它选出的经验](../foundations/approximation/features-control.md#features-sarsa-corridor) · [feature-control-walkthrough.py](../tutorials/feature-control-walkthrough.py)

### 持续控制与平均奖励

- [2.1 · 同一个服务站：决策次数与真实秒数](../foundations/approximation/average-control.md#average-rate-cycles) · [average_rate_walkthrough.py](../tutorials/average_rate_walkthrough.py)

### 离策略函数逼近：覆盖、发散与稳定更新

- [8 · 两种独立计算检查同一例子](../foundations/approximation/off-policy.md#lesson-code) · [offpolicy-geometry-walkthrough.py](../tutorials/offpolicy-geometry-walkthrough.py)

### 多步回报、资格迹与 True-online TD

- [6 · 两个独立实现逐前缀对照](../foundations/approximation/traces.md#lesson-code) · [shared_gradient_walkthrough.py](../tutorials/shared_gradient_walkthrough.py)

### 策略梯度、基线与 Actor–Critic

- [3 · 从状态动作遍历到策略轨迹采样](../foundations/approximation/policy-gradient.md#policy-sampling) · [policy-gradient-sampling.py](../tutorials/policy-gradient-sampling.py)
- [4.1 · 同一个例子：占用怎样收集全部 score](../foundations/approximation/policy-gradient.md#policy-occupancy-example) · [policy-gradient-walkthrough.py](../tutorials/policy-gradient-walkthrough.py)

## 第 II 册 · 深度强化学习

### 深度价值学习：DQN、Double DQN 与目标的时间顺序

- [2.2 · 固定标签，与让标签参与求导](../foundations/deep/deep-value.md#lesson-gradient-paths) · [dqn_shared_feature_walkthrough.py](../tutorials/dqn_shared_feature_walkthrough.py)

### 策略梯度：从轨迹概率到 GAE 与 actor–critic

- [6 · 从数值核验到实际 on-policy 训练](../foundations/deep/policy-gradient.md#lesson-code) · [gae_ppo_walkthrough.py](../tutorials/gae_ppo_walkthrough.py)

### 策略更新的尺度：TRPO 与 PPO

- [6 · 数值检验：TRPO 的步长与 PPO 的轨迹](../foundations/deep/trust-region.md#lesson-example) · [gae_ppo_walkthrough.py](../tutorials/gae_ppo_walkthrough.py)

### 连续动作的价值优化：DDPG 与 TD3

- [3.1 · 预测上升，为什么真实回报反而下降？](../foundations/deep/deterministic-control.md#lesson-critic-error) · [continuous_control_walkthrough.py](../tutorials/continuous_control_walkthrough.py)

### 最大熵连续控制：SAC 的价值、密度与温度

- [6 · soft target 与密度手算](../foundations/deep/entropy-control.md#lesson-example) · [continuous_control_walkthrough.py](../tutorials/continuous_control_walkthrough.py)

### 不完全可观测：信念状态、信息行动与递归记忆

- [7 · 可运行 filter 与信息价值](../foundations/deep/partial-observability.md#lesson-code) · [belief-planning-walkthrough.py](../tutorials/belief-planning-walkthrough.py)
- [7 · 可运行 filter 与信息价值](../foundations/deep/partial-observability.md#lesson-code) · [recurrent-replay-walkthrough.py](../tutorials/recurrent-replay-walkthrough.py)

### 探索与不确定性：后验、乐观估计和时间一致行动

- [6.1 · 把远端信息接到一个未知奖励模型](../foundations/deep/exploration.md#deep-exploration-task) · [deep-exploration-walkthrough.py](../tutorials/deep-exploration-walkthrough.py)

### 分布强化学习：Bellman 分布、分位数与风险目标

- [1 · 同一均值可以来自不同的回报分布](../foundations/deep/distributional.md#lesson-setting) · [distributional-walkthrough.py](../tutorials/distributional-walkthrough.py)
- [4 · 分位数回归与均值决策](../foundations/deep/distributional.md#lesson-quantiles) · [quantile-walkthrough.py](../tutorials/quantile-walkthrough.py)
- [共享网络：一个输入的下降怎样影响另一个输入](../foundations/deep/distributional.md#lesson-neural-shared) · [neural-distribution-walkthrough.py](../tutorials/neural-distribution-walkthrough.py)

### 离线强化学习：数据支持、策略评估与保守改进

- [8 · 估计器与目标函数的可执行核](../foundations/deep/offline.md#lesson-code) · [offline_support_walkthrough.py](../tutorials/offline_support_walkthrough.py)

### 模型学习与规划：MPC、短模型 rollout 和潜在想象

- [8 · 学模型、枚举规划与想象回报核](../foundations/deep/model-based.md#lesson-code) · [belief-planning-walkthrough.py](../tutorials/belief-planning-walkthrough.py)

### 约束强化学习：占据测度、拉格朗日与可行策略

- [8 · 占据流、解析混合与 primal–dual 核](../foundations/deep/constraints.md#lesson-code) · [constraint-control-walkthrough.py](../tutorials/constraint-control-walkthrough.py)

### 自对弈与开放式多智能体学习：评估、目标与策略种群

- [8 · 算例与原始工程：从标签到种群评价](../foundations/deep/multi-agent-populations.md#lesson-code) · [selfplay_search_labels.py](../tutorials/selfplay_search_labels.py)

## 第 III 册 · 持续强化学习

### 强化学习问题的形式化：交互、目标与持续学习

- [9.2 奖励带噪声时，哪一个学习器更好？](../textbook/objectives.md#lesson-noisy-lifetime) · [lifetime_evaluation_walkthrough.py](../tutorials/lifetime_evaluation_walkthrough.py)

### 奖励假设与奖励设计

- [13. 运行、阅读原始代码与选择基准](../textbook/reward-design.md#lesson-code) · [reward-design-walkthrough.py](../tutorials/reward-design-walkthrough.py)

### 平均奖励：奖励率、差分价值与持续控制

- [7.1 · 贯穿算例：让预测与控制共用一个时钟](../textbook/average.md#average-rate-task) · [average_rate_walkthrough.py](../tutorials/average_rate_walkthrough.py)

### 智能体状态：从经验学习怎样记住

- [3. 让写入门学习，再看实际行动](../textbook/state.md#lesson-learned-memory) · [learned-memory-walkthrough.py](../tutorials/learned-memory-walkthrough.py)
- [四步延迟提示：让保留系数也能学习](../textbook/state.md#lesson-delayed-recurrent) · [recurrent_state_walkthrough.py](../tutorials/recurrent_state_walkthrough.py)
- [中途更新参数：活动、trace 与求导对象怎样改变](../textbook/state.md#lesson-online-versions) · [state-online-walkthrough.py](../tutorials/state-online-walkthrough.py)
- [学到的递归系数怎样改变行动与下一条经验](../textbook/state.md#lesson-recurrent-control) · [recurrent-control-walkthrough.py](../tutorials/recurrent-control-walkthrough.py)
- [门后继续走：无重置活动、在线变参与后续经验](../textbook/state.md#lesson-continuing-control) · [continuing-state-walkthrough.py](../tutorials/continuing-state-walkthrough.py)
- [新模式与长走廊：保存信息和保存梯度各花什么资源](../textbook/state.md#lesson-delayed-memory-budget) · [delayed-state-walkthrough.py](../tutorials/delayed-state-walkthrough.py)

### 通用价值函数与预测知识

- [15 · 自动发现：什么预测值得占用一个位置？](../textbook/gvf.md#gvf-discovery) · [discovery_mechanisms.py](../tutorials/discovery_mechanisms.py)

### 时间信用分配：从资格迹到深度梯度学习

- [从两步到三步：同一经验流上的预测版本](../textbook/credit.md#lesson-prefix-walkthrough) · [credit_meta_walkthrough.py](../tutorials/credit_meta_walkthrough.py)

### 流式强化学习：交互协议与更新稳定性

- [10 · 机制实验与作者实现](../textbook/streaming.md#lesson-code) · [streaming_state_walkthrough.py](../tutorials/streaming_state_walkthrough.py)

### 学习规则的适应：在线元梯度与跨任务元学习

- [沿同一条经验流，区分资格与步长敏感度](../textbook/meta.md#lesson-trace-sensitivity) · [credit_meta_walkthrough.py](../tutorials/credit_meta_walkthrough.py)

### 知识保留：经验重放、参数约束与模型记忆

- [同一反转问题：忘得更多，也可能学得更快](../textbook/retention.md#lesson-shared-switch) · [retention_plasticity_walkthrough.py](../tutorials/retention_plasticity_walkthrough.py)

### 可塑性：梯度通路、有效学习率与预测干扰

- [同一反转问题：对称轨道、等函数干预与重新学习](../textbook/plasticity.md#lesson-paired-probe) · [retention_plasticity_walkthrough.py](../tutorials/retention_plasticity_walkthrough.py)

### 子目标构建：从经验到技能与规划

- [1.1 同一条运输记录，换目标时究竟改什么](../textbook/goals.md#lesson-shared-world) · [goal_model_planning_walkthrough.py](../tutorials/goal_model_planning_walkthrough.py)
- [3.2 从真实练习到技能、模型和下一次送货](../textbook/goals.md#lesson-learning-cycle) · [learned_subgoal_cycle.py](../tutorials/learned_subgoal_cycle.py)
- [3.3 对照：在已观察模型上求解同一到达问题](../textbook/goals.md#lesson-discovery-loop) · [goal_discovery_walkthrough.py](../tutorials/goal_discovery_walkthrough.py)

### Options：多步决策、技能发现与可复用行为

- [动手：从一个固定技能走到模型规划](../textbook/options.md#lesson-option-practice) · [option_model_walkthrough.py](../tutorials/option_model_walkthrough.py)
- [8. Eigenoptions：把环境的慢变化方向变成行为](../textbook/options.md#lesson-eigenoptions) · [discovery_mechanisms.py](../tutorials/discovery_mechanisms.py)

### 模型与后果预测：学什么，才能用于下一次决策？

- [1.1 从五条记录到模型，再看关闸后的错误](../textbook/models.md#lesson-model-repair) · [goal_model_planning_walkthrough.py](../tutorials/goal_model_planning_walkthrough.py)
- [1.4 把复查代价放回行动循环](../textbook/models.md#lesson-model-recheck) · [model-change-walkthrough.py](../tutorials/model-change-walkthrough.py)
- [6. 实现与实验：期望、时间和策略条件](../textbook/models.md#lesson-code) · [agent_cycle_walkthrough.py](../tutorials/agent_cycle_walkthrough.py)

### 规划：把模型中的经验转成更好的决策

- [1.1 模型已经修正，起点何时改选路线](../textbook/planning.md#lesson-replanning) · [goal_model_planning_walkthrough.py](../tutorials/goal_model_planning_walkthrough.py)
- [10. 实现与实验：Dyna、优先传播与 MPC](../textbook/planning.md#lesson-code) · [agent_cycle_walkthrough.py](../tutorials/agent_cycle_walkthrough.py)

### 持续探索：新奇、不确定性、学习进展与恢复

- [1.4 · 同一运输任务：为了获取哪条经验而练习技能？](../textbook/exploration.md#lesson-data-cycle) · [learned_subgoal_cycle.py](../tutorials/learned_subgoal_cycle.py)
- [6 · 技能发现与时间一致的探索](../textbook/exploration.md#lesson-skills) · [deep-exploration-walkthrough.py](../tutorials/deep-exploration-walkthrough.py)
- [9 · 奖励时序实现与探索诊断](../textbook/exploration.md#lesson-code) · [exploration-information-walkthrough.py](../tutorials/exploration-information-walkthrough.py)

### 持续智能体架构：从一次经验到下一次行动

- [7. 沿运行记录走完同一次学习循环](../textbook/architectures.md#lesson-cycle-code) · [agent_cycle_walkthrough.py](../tutorials/agent_cycle_walkthrough.py)
- [状态与技能也从经验中形成时](../textbook/architectures.md#lesson-learning-components) · [learned-memory-walkthrough.py](../tutorials/learned-memory-walkthrough.py)
- [状态与技能也从经验中形成时](../textbook/architectures.md#lesson-learning-components) · [learned_subgoal_cycle.py](../tutorials/learned_subgoal_cycle.py)
- [状态与技能也从经验中形成时](../textbook/architectures.md#lesson-learning-components) · [knowledge_transport.py](../tutorials/knowledge_transport.py)
- [真实交互与规划的更新调度](../textbook/architectures.md#lesson-schedule) · [architecture-walkthrough.py](../tutorials/architecture-walkthrough.py)

## 心理与神经科学等视角：学习、心智与意识

### 有限理性：思考的价值与代价

- [3 · 怎样比较两张彩票？](../textbook/supplements/economics.md#reference-dependence) · [economic-choice-walkthrough.py](../tutorials/economic-choice-walkthrough.py)

### 模型与解释：怎样理解学习

- [4.3 恢复参数之后，还要比较更新规则](../textbook/supplements/explanation.md#explanation-identifiability-recovery) · [explanation-identifiability-walkthrough.py](../tutorials/explanation-identifiability-walkthrough.py)
