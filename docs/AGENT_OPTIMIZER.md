# Agent Optimizer Extension

DSPy 扩展层，为 Agent 级别的优化提供支持。

## 背景

DSPy 原生优化器（MIPROv2, BootstrapFewShot）主要针对单步 LLM 调用优化，对于复杂 Agent 系统存在以下局限：

| 局限 | 说明 |
|------|------|
| 无轨迹记录 | 只能看到最终结果，无法分析中间步骤 |
| 无 Credit Assignment | 无法知道哪个步骤贡献了成功/失败 |
| 多 Agent 支持有限 | 难以跟踪 Agent 间 Handoff |
| 缺少步骤级分析 | 无法定位具体失败点 |

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    AgentOptimizer                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Optimization Methods                               │ │
│  │  • Bootstrap - 从成功案例提取 few-shot             │ │
│  │  • Self-Refine - LLM 分析失败并改进                │ │
│  │  • Evolution - 遗传算法优化参数                    │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    Credit Assignment                     │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  • Uniform - 均匀分配                              │ │
│  │  • Decay - 时间衰减（越近贡献越大）                │ │
│  │  • Attention - 基于规则的重要性分配                │ │
│  │  • Counterfactual - 反事实分析                     │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Trajectory Recording                    │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Step Types:                                        │ │
│  │  • USER_INPUT → TOOL_CALL → AGENT_OUTPUT           │ │
│  │  • HANDOFF (Agent 间转移)                          │ │
│  │  • LLM_CALL (推理步骤)                             │ │
│  └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    Agent Wrappers                        │
│  ┌───────────────────┐  ┌────────────────────────────┐  │
│  │ DSPyReActWrapper  │  │ MultiAgentWrapper          │  │
│  │ (DSPy ReAct)      │  │ (自定义多 Agent 系统)      │  │
│  └───────────────────┘  └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 安装

```python
# 已包含在 dspy_guardrails 中
from dspy_guardrails.agent_optimizer import (
    AgentOptimizer,
    OptimizationConfig,
    OptimizationMethod,
    wrap_agent,
    TrajectoryRecorder,
    assign_credit,
)
```

## 快速开始

### 1. 包装 DSPy ReAct Agent

```python
import dspy
from dspy_guardrails.agent_optimizer import wrap_agent, AgentOptimizer

# 创建 DSPy ReAct Agent
class MyAgent(dspy.Module):
    def __init__(self):
        self.react = dspy.ReAct(MySignature, tools=[tool1, tool2])

    def forward(self, query):
        return self.react(query=query)

# 包装 Agent
agent = MyAgent()
wrapped = wrap_agent(agent, agent_type="react")

# 运行并记录轨迹
result = wrapped.run("查询航班状态", record=True)
print(result.response)
print(result.trajectory.summary())
```

### 2. 轨迹分析

```python
from dspy_guardrails.agent_optimizer import assign_credit

# 分配 Credit
assign_credit(result.trajectory, method="decay")

# 查看关键步骤
for step in result.trajectory.steps:
    if step.step_type == StepType.TOOL_CALL:
        print(f"Tool: {step.action_name} | Credit: {step.credit:.3f}")

# 获取失败点
failure = result.trajectory.get_failure_point()
if failure:
    print(f"Failure at: {failure.action_name} - {failure.error}")
```

### 3. 运行优化

```python
from dspy_guardrails.agent_optimizer import (
    AgentOptimizer,
    OptimizationConfig,
    OptimizationMethod,
)

# 定义评估指标
def my_metric(result):
    if not result.success:
        return 0.0
    # 自定义评分逻辑
    return 1.0 if "success" in result.response else 0.5

# 配置优化
config = OptimizationConfig(
    method=OptimizationMethod.BOOTSTRAP,
    num_demos=4,
    demo_selection="diverse",  # "best", "random", "diverse"
)

# 运行优化
optimizer = AgentOptimizer(metric=my_metric, config=config)
result = optimizer.optimize(
    agent=wrapped,
    train_tasks=["task1", "task2", "task3"],
    eval_tasks=["eval1", "eval2"],
)

print(result.summary())
# Optimization Result:
#   Success: True
#   Baseline: 62.50%
#   Optimized: 78.00%
#   Improvement: +15.50%
```

## API 参考

### Trajectory 类

```python
@dataclass
class Trajectory:
    trajectory_id: str
    task: str
    steps: List[Step]
    final_response: str
    outcome: Outcome  # SUCCESS, PARTIAL, FAILURE, ERROR
    final_reward: float

    # 分析方法
    def get_tools_used(self) -> List[str]
    def get_agents_visited(self) -> List[str]
    def get_handoff_chain(self) -> List[tuple]
    def get_critical_steps(self, threshold=0.5) -> List[Step]
    def get_failure_point(self) -> Optional[Step]
    def summary(self) -> str
    def to_dict(self) -> dict
    def to_json(self) -> str
```

### Step 类

```python
@dataclass
class Step:
    step_id: int
    step_type: StepType  # TOOL_CALL, HANDOFF, LLM_CALL, etc.
    agent_name: str

    # 输入/输出
    input_data: Dict
    thought: str
    action_name: str
    action_args: Dict
    result: Any
    error: Optional[str]

    # 性能
    latency_ms: float
    token_count: int

    # 优化相关
    reward: float
    credit: float
```

### Credit Assignment 方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `uniform` | 均匀分配 | 步骤重要性相近 |
| `decay` | 时间衰减 | 最后几步更关键 |
| `attention` | 规则启发式 | 成功工具调用优先 |
| `counterfactual` | 反事实分析 | 需要评估函数 |

```python
from dspy_guardrails.agent_optimizer import assign_credit

# 使用衰减分配
assign_credit(trajectory, method="decay", decay_factor=0.9)

# 使用注意力分配
assign_credit(trajectory, method="attention")
```

### OptimizationConfig

```python
@dataclass
class OptimizationConfig:
    method: OptimizationMethod = BOOTSTRAP
    num_trials: int = 20
    batch_size: int = 5
    patience: int = 5
    credit_method: str = "decay"

    # Bootstrap 参数
    num_demos: int = 4
    demo_selection: str = "diverse"  # "best", "random", "diverse"

    # Self-Refine 参数
    num_iterations: int = 5

    # Evolution 参数
    population_size: int = 10
    mutation_rate: float = 0.1
```

## 与 DSPy 原生优化器对比

### 实验结果 (航空客服 Demo)

| 方法 | 成功率 | 工具匹配 | 关键词得分 |
|------|--------|----------|-----------|
| DSPy ReAct (Baseline) | 62.50% | 78.12% | 70.83% |
| + agent_optimizer 扩展 | 80.00% | 100% | 80.00% |

### 何时使用扩展层

**推荐使用**：
- 多步骤 Agent（工具调用 > 2次）
- 多 Agent 系统（有 Handoff）
- 需要调试失败案例
- 需要细粒度优化分析

**不必使用**：
- 简单 QA（单步推理）
- 低延迟要求场景
- DSPy 原生优化已足够

## 文件结构

```
dspy_guardrails/agent_optimizer/
├── __init__.py      # 包导出
├── trajectory.py    # Step, Trajectory, TrajectoryRecorder
├── credit.py        # CreditAssigner, assign_credit
├── wrapper.py       # OptimizableAgent, DSPyReActWrapper, MultiAgentWrapper
└── optimizer.py     # AgentOptimizer, OptimizationConfig
```

## 限制与未来工作

### 当前限制

1. **优化效果待验证** - Bootstrap 方法已实现，但大规模效果需更多实验
2. **计算开销** - 轨迹记录增加约 5-10% 延迟
3. **LLM 依赖** - Self-Refine 需要额外 LLM 调用

### 未来方向

1. **GRPO/PPO 集成** - 强化学习优化策略
2. **多目标优化** - 同时优化成功率和延迟
3. **在线学习** - 边运行边优化
4. **攻防协同进化** - 与 redteam 模块集成

## 相关模块

- `dspy_guardrails.redteam` - 红队攻击测试
- `dspy_guardrails.testing` - 安全测试框架
- `dspy_guardrails.mcp` - MCP 协议安全
