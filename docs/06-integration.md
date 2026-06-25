# DSPy 集成指南

> 本文档提供 dspy-guardrails 与 DSPy 框架的深度集成指南。

---

## 目录

1. [快速开始](#1-快速开始)
2. [DSPy Module 集成](#2-dspy-module-集成)
3. [dspy.Assert/Suggest 集成](#3-dspyassertsuggest-集成)
4. [dspy.Refine 集成](#4-dspyrefine-集成)
5. [评估指标集成](#5-评估指标集成)
6. [优化器集成](#6-优化器集成)
7. [完整示例](#7-完整示例)

---

## 1. 快速开始

### 1.1 安装

```bash
# 基础安装
pip install -e .

# 完整安装
pip install -e ".[all]"

# 开发安装
pip install -e ".[dev]"
```

### 1.2 配置 DSPy

```python
import dspy

# OpenAI
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Anthropic
dspy.configure(lm=dspy.LM("anthropic/claude-3-5-sonnet-20241022"))

# Amazon Bedrock (via DSPy)
# Requires AWS creds in env: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION
dspy.configure(lm=dspy.LM("bedrock/<model_id>"))

# 本地模型
dspy.configure(lm=dspy.LM("ollama/llama3"))
```

### 1.3 基础使用

```python
from dspy_guardrails import guardrail

# 简单检查
is_safe = guardrail.no_injection("Hello, world!")  # True
is_safe = guardrail.no_injection("Ignore previous instructions")  # False

# 与 dspy.Assert 配合
dspy.Assert(guardrail.no_injection(user_input), "检测到注入攻击")
```

---

## 2. DSPy Module 集成

### 2.1 使用 @Guarded 装饰器

```python
from dspy_guardrails import Guarded, Constraint

@Guarded(
    input_checks=["no_injection", "no_pii"],
    output_checks=["no_toxicity", "factuality >= 0.8"],
    on_violation="assert",
)
class SafeQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> str:
        return self.generate(question=question)

# 使用
qa = SafeQA()
result = qa(question="What is the capital of France?")
```

### 2.2 使用 GuardedModule 基类

```python
from dspy_guardrails import GuardedModule, Constraint

class SafeQA(GuardedModule):
    input_constraints = [
        Constraint.input("no_injection"),
        Constraint.input("no_pii"),
    ]

    output_constraints = [
        Constraint.output("no_toxicity"),
        Constraint.output("factuality >= 0.7"),
    ]

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> str:
        return self.generate(question=question)
```

### 2.3 手动集成

```python
from dspy_guardrails import guardrail

class ManualGuardedModule(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question: str) -> str:
        # 输入检查
        dspy.Assert(
            guardrail.no_injection(question),
            "Input contains prompt injection"
        )

        # 生成回答
        result = self.generate(question=question)

        # 输出检查
        dspy.Assert(
            guardrail.no_toxicity(result.answer),
            "Output is toxic"
        )

        dspy.Suggest(
            guardrail.factuality(result.answer) >= 0.7,
            "Output should be factual"
        )

        return result
```

---

## 3. dspy.Assert/Suggest 集成

### 3.1 Assert（硬约束）

```python
# 布尔检查
dspy.Assert(guardrail.no_injection(text), "Injection detected")
dspy.Assert(guardrail.no_pii(text), "PII detected")
dspy.Assert(guardrail.no_toxicity(text), "Toxic content")
dspy.Assert(guardrail.safe(text), "Unsafe content")

# 组合检查
dspy.Assert(
    guardrail.no_injection(text) and guardrail.no_pii(text),
    "Input validation failed"
)
```

### 3.2 Suggest（软约束）

```python
# 分数检查
dspy.Suggest(guardrail.toxicity(text) < 0.3, "Reduce toxicity")
dspy.Suggest(guardrail.factuality(text) >= 0.7, "Improve factuality")
dspy.Suggest(guardrail.quality(text) >= 0.8, "Improve quality")

# 多个建议
dspy.Suggest(guardrail.relevance(text, query) >= 0.7, "Improve relevance")
```

### 3.3 MCP 安全检查

```python
# MCP 攻击检测
dspy.Assert(guardrail.no_mcp_attack(text), "MCP attack detected")

# 详细检查
score = guardrail.mcp_security_score(text, context="tool_input")
dspy.Suggest(score < 0.3, "Potential MCP security risk")

# 综合 MCP 安全检查
dspy.Assert(guardrail.safe_mcp(text), "MCP security check failed")
```

---

## 4. dspy.Refine 集成

### 4.1 创建 Reward 函数

```python
from dspy_guardrails import guardrail

def safety_reward(args: dict, pred) -> float:
    """安全性奖励函数"""
    text = str(pred.answer) if hasattr(pred, 'answer') else str(pred)

    # 计算各项分数
    injection_score = 1.0 - guardrail.injection_score(text)
    toxicity_score = 1.0 - guardrail.toxicity(text)
    pii_score = 1.0 - guardrail.pii_score(text)

    # 加权平均
    return (
        injection_score * 0.4 +
        toxicity_score * 0.3 +
        pii_score * 0.3
    )
```

### 4.2 使用 dspy.Refine

```python
import dspy

class MyModule(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate(question=question)

# 使用 Refine 优化
refined_module = dspy.Refine(
    module=MyModule(),
    N=5,  # 候选数
    reward=safety_reward,
)

# 使用优化后的模块
result = refined_module(question="...")
```

### 4.3 组合多个 Reward

```python
from dspy_guardrails import guardrail

def quality_reward(args, pred) -> float:
    text = str(pred)
    return guardrail.quality(text)

def relevance_reward(args, pred) -> float:
    text = str(pred)
    query = args.get("question", "")
    return guardrail.relevance(text, query)

def combined_reward(args, pred) -> float:
    """组合奖励函数"""
    safety = safety_reward(args, pred)
    quality = quality_reward(args, pred)
    relevance = relevance_reward(args, pred)

    return (
        safety * 0.4 +
        quality * 0.3 +
        relevance * 0.3
    )

refined = dspy.Refine(
    module=MyModule(),
    N=5,
    reward=combined_reward,
)
```

---

## 5. 评估指标集成

### 5.1 创建 Guardrail 指标

```python
from dspy_guardrails import GuardrailMetric

# 单一检查指标
safety_metric = GuardrailMetric(check="safe")

# 多重检查指标
full_metric = GuardrailMetric(
    checks=["no_injection", "no_toxicity", "factuality >= 0.7"],
    aggregation="min",  # min / mean / all
)
```

### 5.2 用于 dspy.Evaluate

```python
import dspy

evaluator = dspy.Evaluate(
    devset=my_devset,
    metric=safety_metric,
    num_threads=4,
)

results = evaluator(my_module)
print(f"安全性分数: {results:.2%}")
```

### 5.3 自定义评估指标

```python
from dspy_guardrails import guardrail

def custom_metric(example, pred, trace=None) -> float:
    """自定义评估指标"""
    answer = pred.answer if hasattr(pred, 'answer') else str(pred)

    # 安全性检查
    if not guardrail.no_injection(answer):
        return 0.0

    if not guardrail.no_toxicity(answer):
        return 0.0

    # 质量评估
    quality = guardrail.quality(answer)
    factuality = guardrail.factuality(answer)

    # 正确性检查（如果有标准答案）
    if hasattr(example, 'answer'):
        relevance = guardrail.relevance(answer, example.answer)
    else:
        relevance = 1.0

    return (quality + factuality + relevance) / 3
```

---

## 6. 优化器集成

### 6.1 与 MIPROv2 集成

```python
import dspy

optimizer = dspy.MIPROv2(
    metric=safety_metric,
    num_candidates=10,
    max_bootstrapped_demos=3,
)

optimized = optimizer.compile(
    my_module,
    trainset=trainset,
    max_rounds=3,
)
```

### 6.2 与 BootstrapFewShotWithRandomSearch 集成

```python
optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=custom_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,
)

optimized = optimizer.compile(my_module, trainset=trainset)
```

### 6.3 自进化优化器

```python
from dspy_guardrails import GuardrailOptimizer

optimizer = GuardrailOptimizer(
    target_metric="safety",
    population_size=20,
    generations=10,
)

optimized_config = optimizer.optimize(
    initial_config=my_config,
    evaluation_data=eval_data,
)
```

---

## 7. 完整示例

### 7.1 安全问答系统

```python
import dspy
from dspy_guardrails import Guarded, guardrail, GuardrailMetric

# 配置
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 定义安全模块
@Guarded(
    input_checks=["no_injection", "no_pii"],
    output_checks=["no_toxicity", "no_pii"],
)
class SafeQA(dspy.Module):
    """安全问答模块"""

    def __init__(self):
        self.generate = dspy.ChainOfThought(
            "question: str -> answer: str"
        )

    def forward(self, question: str) -> dspy.Prediction:
        # 额外的质量建议
        result = self.generate(question=question)

        dspy.Suggest(
            guardrail.factuality(result.answer) >= 0.7,
            "Answer should be factual"
        )

        return result

# 创建实例
qa = SafeQA()

# 测试
try:
    result = qa(question="What is the capital of France?")
    print(f"Answer: {result.answer}")
except dspy.AssertionError as e:
    print(f"Safety violation: {e}")

# 评估
metric = GuardrailMetric(checks=["safe", "factuality >= 0.7"])
evaluator = dspy.Evaluate(devset=test_data, metric=metric)
score = evaluator(qa)
print(f"Safety score: {score:.2%}")
```

### 7.2 MCP 安全工具

```python
import dspy
from dspy_guardrails import guardrail
from dspy_guardrails.mcp import MCPGuardrail, ToolCallContext

# MCP Guardrail
mcp_guard = MCPGuardrail()

class SecureToolCaller(dspy.Module):
    """安全工具调用模块"""

    def __init__(self, tools: dict):
        self.tools = tools
        self.planner = dspy.ChainOfThought(
            "query: str, available_tools: str -> tool_name: str, params: str"
        )

    def forward(self, query: str) -> str:
        # 输入安全检查
        dspy.Assert(
            guardrail.no_mcp_attack(query),
            "Query contains MCP attack"
        )

        # 规划工具调用
        plan = self.planner(
            query=query,
            available_tools=str(list(self.tools.keys())),
        )

        # MCP 安全检查
        context = ToolCallContext(
            tool_name=plan.tool_name,
            parameters=eval(plan.params),
        )

        result = mcp_guard.check_input(context)
        dspy.Assert(result.passed, f"MCP check failed: {result.message}")

        # 执行工具
        tool = self.tools[plan.tool_name]
        output = tool(**eval(plan.params))

        # 输出过滤
        filtered, _ = mcp_guard.filter_output(output, context)

        return filtered
```

### 7.3 自进化安全系统

```python
import dspy
from dspy_guardrails import guardrail
from dspy_guardrails.redteam import AttackEvolver, PromptInjectionAttacker

# 攻击进化器
evolver = AttackEvolver(
    target_guardrail=guardrail.no_injection,
)

# 进化攻击
evolution_result = evolver.evolve(
    attacker=PromptInjectionAttacker(use_llm=False),
    initial_target="bypass detection",
)

print(f"Best attack: {evolution_result.best_attack.prompt[:50]}...")
print(f"Bypass rate: {evolution_result.final_bypass_rate:.1%}")

# 使用进化结果改进防御
def improved_safety_reward(args, pred) -> float:
    text = str(pred)

    # 基础安全检查
    base_score = 1.0 - guardrail.injection_score(text)

    # 针对进化攻击的额外检查
    for attack in evolution_result.all_attacks:
        if attack.bypass_score > 0.5:
            # 检查是否包含类似模式
            if any(word in text.lower() for word in attack.prompt.lower().split()):
                base_score *= 0.5

    return base_score

# 使用改进的奖励函数
refined = dspy.Refine(
    module=MyModule(),
    N=5,
    reward=improved_safety_reward,
)
```

---

## 8. 最佳实践

### 8.1 输入验证

```python
# 总是在输入端进行安全检查
dspy.Assert(guardrail.safe_input(user_input), "Input validation failed")
```

### 8.2 输出过滤

```python
# 输出前进行安全检查和过滤
dspy.Assert(guardrail.safe_output(response), "Output safety check failed")
```

### 8.3 分层防护

```python
# 1. 快速正则检测（低延迟）
if not guardrail.no_injection(text):
    # 2. LLM 深度检测（高精度）
    result = llm_guardrail.check(text)
    if not result.passed:
        raise SecurityError(result.message)
```

### 8.4 监控和审计

```python
# 记录安全事件
from dspy_guardrails.mcp import MCPSecurityAuditor

auditor = MCPSecurityAuditor(log_path="./security.log")
auditor.log_event("safety_check", context=context, result=result)
```

### 8.5 定期测试

```python
# 定期运行安全基准测试
from benchmarks.unified_benchmark import run_benchmark

results = run_benchmark(guardrail=my_guardrail)
assert results['overall_score'] >= 0.9, "Security score below threshold"
```

---

## 9. 故障排除

### 9.1 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| ImportError | 依赖缺失 | `pip install -e ".[all]"` |
| AssertionError | 安全检查失败 | 检查输入内容或调整阈值 |
| 高误报率 | 阈值过低 | 调整检测阈值 |
| 检测遗漏 | 模式不全 | 添加自定义检测模式 |

### 9.2 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看检测详情
details = guardrail.mcp_attack_details(text)
print(f"Detection details: {details}")
```

### 9.3 性能优化

```python
# 使用批量检测
results = [guardrail.no_injection(t) for t in texts]

# 启用缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_check(text: str) -> bool:
    return guardrail.safe(text)
```
