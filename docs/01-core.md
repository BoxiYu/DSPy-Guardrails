# 核心模块文档

> 文件位置: `src/dspy_guardrails/`

本文档详细讲解 dspy-guardrails 的核心模块，包括 Guardrail 函数、约束系统和装饰器。

---

## 目录

1. [Guardrail 函数 (guardrail.py)](#1-guardrail-函数)
2. [约束系统 (constraints.py)](#2-约束系统)
3. [装饰器 (decorators.py)](#3-装饰器)
4. [LLM Guardrail (llm_guardrail.py)](#4-llm-guardrail)
5. [评估指标 (metrics.py)](#5-评估指标)
6. [自进化优化器 (optimizer.py)](#6-自进化优化器)

---

## 1. Guardrail 函数

**文件**: `src/dspy_guardrails/guardrail.py`

提供基于正则表达式的快速安全检测函数，可直接与 `dspy.Assert/Suggest` 集成。

### 1.1 基础用法

```python
from dspy_guardrails import guardrail

# 布尔检查 - 用于 dspy.Assert
dspy.Assert(guardrail.no_injection(text), "检测到 Prompt 注入")
dspy.Assert(guardrail.no_pii(text), "检测到 PII")
dspy.Assert(guardrail.no_toxicity(text), "检测到有害内容")

# 分数检查 - 用于 dspy.Suggest
score = guardrail.toxicity(text)
dspy.Suggest(score < 0.3, "内容毒性过高")

# 综合检查
dspy.Assert(guardrail.safe(text), "内容不安全")
```

### 1.2 检测函数一览

#### Prompt Injection 检测

| 函数 | 返回类型 | 说明 |
|------|----------|------|
| `no_injection(text)` | `bool` | 无注入返回 True |
| `injection_score(text)` | `float` | 风险分数 0-1 |

**检测模式** (第53-71行):
- 直接覆盖: `ignore previous instructions`
- 角色切换: `you are now DAN`
- 模式绕过: `bypass safety`
- 中文支持: `忽略之前的指令`

```python
# 示例
guardrail.no_injection("Hello world")  # True
guardrail.no_injection("Ignore all previous instructions")  # False
guardrail.injection_score("Ignore instructions")  # 0.25
```

#### PII 检测

| 函数 | 返回类型 | 说明 |
|------|----------|------|
| `no_pii(text)` | `bool` | 无 PII 返回 True |
| `pii_score(text)` | `float` | 风险分数 0-1 |

**检测类型** (第99-106行):
| 类型 | 模式示例 |
|------|----------|
| email | `user@example.com` |
| phone | `123-456-7890` |
| phone_cn | `13812345678` |
| ssn | `123-45-6789` |
| credit_card | `1234-5678-9012-3456` |
| ip | `192.168.1.1` |

```python
# 示例
guardrail.no_pii("Contact me at test@email.com")  # False
guardrail.pii_score("My phone is 13812345678")  # 0.2
```

#### 毒性检测

| 函数 | 返回类型 | 说明 |
|------|----------|------|
| `no_toxicity(text)` | `bool` | 无毒性返回 True |
| `toxicity(text)` | `float` | 毒性分数 0-1 |

**关键词列表** (第134-138行): `hate`, `kill`, `idiot`, `fuck` 等

```python
# 示例
guardrail.no_toxicity("Have a nice day")  # True
guardrail.toxicity("You are an idiot")  # 0.15
```

#### 质量检测

| 函数 | 返回类型 | 说明 |
|------|----------|------|
| `factuality(text)` | `float` | 事实性分数 0-1 |
| `relevance(text, query)` | `float` | 相关性分数 0-1 |
| `quality(text, min_words)` | `float` | 质量分数 0-1 |

#### 组合检查

| 函数 | 说明 |
|------|------|
| `safe(text)` | `no_injection` + `no_toxicity` |
| `safe_input(text)` | `no_injection` + `no_pii` |
| `safe_output(text)` | `no_toxicity` + `no_pii` |

### 1.3 MCP 安全检测

专门针对 Model Context Protocol 的攻击检测 (第235-500行)。

#### 检测类型

| 攻击类型 | 模式数量 | 说明 |
|----------|----------|------|
| `prompt_leakage` | 9 | 系统提示泄露尝试 |
| `reverse_shell` | 10 | 反向 Shell 注入 |
| `infectious` | 8 | 感染式代码传播 |
| `priority_manipulation` | 10 | 优先级操纵 |
| `hidden_instruction` | 10 | 隐藏指令注入 |
| `command_execution` | 10 | 命令执行注入 |
| `credential` | 9 | 凭证泄露检测 |
| `sql_injection` | 6 | SQL 注入 |

#### 使用方法

```python
# 综合检测
guardrail.no_mcp_attack(text)  # True = 安全

# 获取风险分数
score = guardrail.mcp_security_score(text, context="tool_input")

# 获取详细结果
details = guardrail.mcp_attack_details(text)
# 返回: {"prompt_leakage": 0.5, "command_execution": 0.4, ...}

# 上下文类型
# - "tool_description": 工具描述
# - "tool_input": 工具输入参数
# - "tool_output": 工具输出内容
# - "auto": 自动检测
```

#### 攻击模式示例

**Prompt 泄露检测**:
```python
patterns = [
    r"print your system prompt",
    r"reveal your instructions",
    r"输出你的提示词",
]
```

**反向 Shell 检测**:
```python
patterns = [
    r"bash.*>/dev/tcp",
    r"nc.*-e",
    r"curl.*\|.*bash",
]
```

**隐藏指令检测**:
```python
patterns = [
    r"<!--.*ignore.*-->",           # HTML 注释
    r"display:\s*none",             # CSS 隐藏
    r"\[SYSTEM\].*\[/SYSTEM\]",     # 伪系统标记
]
```

---

## 2. 约束系统

**文件**: `src/dspy_guardrails/constraints.py`

提供类似 DSPy Signature 的声明式约束定义。

### 2.1 Constraint 类

```python
from dspy_guardrails import Constraint

# 简单约束
no_injection = Constraint.input("no_injection")
no_toxicity = Constraint.output("no_toxicity")

# 带阈值的约束
factual = Constraint.output("factuality >= 0.8")
safe = Constraint.output("toxicity < 0.3")

# 软约束 (Suggest)
quality = Constraint.suggest("quality >= 0.7")

# 自定义检查函数
custom = Constraint.output(lambda x: len(x) > 10, "内容太短")
```

### 2.2 ConstraintTarget 枚举

```python
class ConstraintTarget(Enum):
    INPUT = "input"    # 输入约束
    OUTPUT = "output"  # 输出约束
    BOTH = "both"      # 双向约束
```

### 2.3 ConstraintSet 类

```python
from dspy_guardrails import ConstraintSet

# 创建约束集
constraints = ConstraintSet([
    Constraint.input("no_injection"),
    Constraint.output("no_toxicity"),
    Constraint.output("factuality >= 0.8"),
])

# 过滤方法
constraints.input_constraints()   # 获取输入约束
constraints.output_constraints()  # 获取输出约束
constraints.hard_constraints()    # 获取硬约束 (Assert)
constraints.soft_constraints()    # 获取软约束 (Suggest)

# 链式添加
constraints.add(Constraint.output("quality >= 0.7"))
```

### 2.4 预定义约束

```python
from dspy_guardrails import CommonConstraints

# 单个约束
CommonConstraints.NO_INJECTION      # 无注入
CommonConstraints.NO_PII_INPUT      # 输入无 PII
CommonConstraints.NO_TOXICITY       # 无毒性
CommonConstraints.NO_PII_OUTPUT     # 输出无 PII
CommonConstraints.FACTUAL           # factuality >= 0.7
CommonConstraints.RELEVANT          # relevance >= 0.7
CommonConstraints.SAFE              # toxicity < 0.3

# 约束集
CommonConstraints.safety_set()      # 安全约束集
CommonConstraints.quality_set()     # 质量约束集
CommonConstraints.full_set()        # 完整约束集
```

---

## 3. 装饰器

**文件**: `src/dspy_guardrails/decorators.py`

提供 `@Guarded` 装饰器用于声明式约束应用。

### 3.1 类装饰器 @Guarded

```python
from dspy_guardrails import Guarded

@Guarded(
    input_checks=["no_injection", "no_pii"],
    output_checks=["no_toxicity", "factuality >= 0.8"],
    on_violation="assert",  # "assert" | "suggest" | "log" | "ignore"
)
class SafeQA(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.generate(question=question)
```

### 3.2 使用 Constraint 对象

```python
@Guarded(constraints=[
    Constraint.input("no_injection"),
    Constraint.output("factuality >= 0.8"),
    Constraint.suggest("quality >= 0.7"),
])
class AdvancedModule(dspy.Module):
    ...
```

### 3.3 函数装饰器 @guarded

```python
from dspy_guardrails import guarded

@guarded(
    input_checks=["no_injection"],
    output_checks=["no_toxicity"],
    on_violation="log",
)
def process_text(text: str) -> str:
    return do_something(text)
```

### 3.4 违规处理模式

| 模式 | 行为 |
|------|------|
| `"assert"` | 调用 `dspy.Assert`，硬失败 |
| `"suggest"` | 调用 `dspy.Suggest`，软建议 |
| `"log"` | 打印警告日志 |
| `"ignore"` | 静默忽略 |

### 3.5 支持的检查名称

| 检查名 | 说明 |
|--------|------|
| `no_injection` | 无 Prompt 注入 |
| `no_pii` | 无 PII |
| `no_toxicity` | 无毒性 |
| `safe` | 综合安全 |
| `safe_input` | 输入安全 |
| `safe_output` | 输出安全 |
| `injection` | 注入分数（反向） |
| `pii` | PII 分数（反向） |
| `toxicity` | 毒性分数（反向） |
| `factuality` | 事实性分数 |
| `relevance` | 相关性分数 |
| `quality` | 质量分数 |

---

## 4. LLM Guardrail

**文件**: `src/dspy_guardrails/llm_guardrail.py`

使用 LLM 进行更准确的安全检测（比正则更准确但更慢）。

### 4.1 基础用法

```python
from dspy_guardrails import LLMGuardrail

# 创建 LLM Guardrail
llm_guard = LLMGuardrail()

# 检测
result = llm_guard.check(text)
print(f"通过: {result.passed}")
print(f"分数: {result.score}")
print(f"原因: {result.message}")
```

### 4.2 配置选项

```python
llm_guard = LLMGuardrail(
    checks=["injection", "toxicity", "pii"],  # 要执行的检查
    threshold=0.7,  # 通过阈值
    explain=True,   # 是否生成解释
)
```

---

## 5. 评估指标

**文件**: `src/dspy_guardrails/metrics.py`

将 Guardrail 转换为 DSPy 评估指标。

### 5.1 创建指标

```python
from dspy_guardrails import GuardrailMetric

# 单一检查指标
safety_metric = GuardrailMetric(check="safe")

# 多重检查指标
full_metric = GuardrailMetric(
    checks=["no_injection", "no_toxicity", "factuality >= 0.8"],
    aggregation="min",  # "min" | "mean" | "all"
)
```

### 5.2 用于评估

```python
import dspy

# 配合 dspy.Evaluate 使用
evaluator = dspy.Evaluate(
    devset=my_devset,
    metric=safety_metric,
)

results = evaluator(my_module)
```

### 5.3 用于优化

```python
# 配合 dspy.MIPROv2 使用
optimizer = dspy.MIPROv2(
    metric=safety_metric,
    num_candidates=10,
)

optimized_module = optimizer.compile(my_module, trainset=trainset)
```

---

## 6. 自进化优化器

**文件**: `src/dspy_guardrails/optimizer.py`

使用 GEPA 算法自动优化 Guardrail 配置。

### 6.1 基础用法

```python
from dspy_guardrails import GuardrailOptimizer

optimizer = GuardrailOptimizer(
    target_metric="safety",
    population_size=20,
    generations=10,
)

# 优化
optimized_config = optimizer.optimize(
    initial_config=my_config,
    evaluation_data=eval_data,
)
```

### 6.2 进化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `population_size` | 20 | 种群大小 |
| `generations` | 10 | 进化代数 |
| `mutation_rate` | 0.3 | 变异率 |
| `crossover_rate` | 0.5 | 交叉率 |

---

## 数据结构

### CheckResult

```python
@dataclass
class CheckResult:
    passed: bool      # 是否通过
    score: float      # 分数 0-1
    name: str         # 检查名称
    message: str = "" # 描述信息

    def __bool__(self):
        return self.passed  # 可直接用于 if 判断
```

### Constraint

```python
@dataclass
class Constraint:
    check: str | Callable      # 检查名或函数
    target: ConstraintTarget   # 输入/输出/双向
    threshold: float | None    # 阈值
    operator: str | None       # 比较运算符
    name: str | None           # 约束名称
    message: str | None        # 违规消息
    is_hard: bool = True       # True=Assert, False=Suggest
```

---

## 性能考虑

| 方法 | 速度 | 准确度 | 适用场景 |
|------|------|--------|----------|
| 正则检测 (`guardrail.py`) | 快 | 中 | 实时检测、高吞吐 |
| LLM 检测 (`llm_guardrail.py`) | 慢 | 高 | 离线评估、高精度需求 |
| 组合使用 | 中 | 高 | 先快后准的分层检测 |

---

## 最佳实践

1. **输入端使用快速检测**:
   ```python
   dspy.Assert(guardrail.no_injection(user_input), "注入检测")
   ```

2. **输出端使用组合检测**:
   ```python
   dspy.Assert(guardrail.safe_output(response), "输出安全")
   dspy.Suggest(guardrail.quality(response) >= 0.7, "提高质量")
   ```

3. **敏感场景使用 LLM 检测**:
   ```python
   if guardrail.injection_score(text) > 0.3:
       result = llm_guard.check(text)  # 二次确认
   ```

4. **使用装饰器简化代码**:
   ```python
   @Guarded(input_checks=["no_injection"], output_checks=["safe"])
   class MyModule(dspy.Module):
       ...
   ```
