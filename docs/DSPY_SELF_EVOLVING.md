# DSPy Self-Evolving 与 dspyGuardrails

## DSPy 的核心优势：Self-Evolving (自我进化)

### 传统 LLM 开发 vs DSPy

```
┌─────────────────────────────────────────────────────────────────┐
│                    传统 LLM 开发                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   手写 Prompt  →  测试  →  效果不好  →  手动调整  →  再测试     │
│        ↑                                                    │   │
│        └────────────────── 人工循环 ─────────────────────────┘   │
│                                                                  │
│   问题:                                                          │
│   • 依赖 Prompt 工程师经验                                       │
│   • 耗时耗力                                                     │
│   • 难以复现                                                     │
│   • 换模型需要重新调                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DSPy Self-Evolving                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   声明意图  →  Optimizer  →  自动生成最优 Prompt                 │
│                    ↓                                             │
│              ┌─────────────┐                                     │
│              │ 自动优化循环 │                                     │
│              │ • 生成候选   │                                     │
│              │ • 评估效果   │                                     │
│              │ • 选择最优   │                                     │
│              │ • 迭代改进   │                                     │
│              └─────────────┘                                     │
│                                                                  │
│   优势:                                                          │
│   • 无需手写 Prompt                                              │
│   • 自动找到最优解                                               │
│   • 可复现                                                       │
│   • 换模型自动适配                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## DSPy 优化器 (Optimizers)

```python
import dspy

# 1. 定义任务 (声明式)
class QA(dspy.Module):
    def forward(self, question):
        return dspy.ChainOfThought("question -> answer")(question=question)

# 2. 定义评估指标
def metric(example, prediction):
    return prediction.answer == example.answer

# 3. 自动优化 (Self-Evolving)
optimizer = dspy.MIPROv2(metric=metric)
optimized_qa = optimizer.compile(QA(), trainset=train_data)

# optimized_qa 自动学会了:
# • 最佳 few-shot examples
# • 最优 prompt 结构
# • 最适合当前 LLM 的指令
```

### DSPy 优化器类型

| 优化器 | 原理 | 适用场景 |
|--------|------|----------|
| **BootstrapFewShot** | 自动选择最佳 few-shot 示例 | 快速启动 |
| **MIPROv2** | 多指令提议优化 | 复杂任务 |
| **BayesianSignatureOptimizer** | 贝叶斯优化签名 | 精细调优 |
| **COPRO** | 协同提示优化 | 多模块系统 |

---

## dspyGuardrails + Self-Evolving = 安全的自我进化

### 核心洞察

> **安全约束参与优化循环，模型自动学习"既安全又正确"的输出**

```python
import dspy
from dspy_guardrails import guardrail

# 1. 定义带安全约束的模块
class SafeQA(dspy.Module):
    def forward(self, question):
        # 安全断言 - 参与优化循环
        dspy.Assert(
            guardrail.no_injection(question),
            "Prompt injection detected"
        )

        answer = dspy.ChainOfThought("question -> answer")(question=question)

        # 安全建议 - 引导模型改进
        dspy.Suggest(
            guardrail.no_pii(answer.answer),
            "Response contains PII, please rephrase"
        )

        return answer

# 2. 定义包含安全性的评估指标
def safe_metric(example, prediction):
    correct = prediction.answer == example.answer
    safe = guardrail.safe(prediction.answer)
    return correct and safe  # 既正确又安全

# 3. 自动优化 - 学习安全的输出模式
optimizer = dspy.MIPROv2(metric=safe_metric)
safe_optimized_qa = optimizer.compile(SafeQA(), trainset=train_data)
```

### 优化过程

```
┌─────────────────────────────────────────────────────────────────┐
│                  dspyGuardrails + DSPy 优化循环                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Iteration 1:                                                   │
│   ├── 生成答案: "John的手机号是13812345678"                      │
│   ├── 安全检查: guardrail.no_pii() = False ❌                    │
│   ├── dspy.Suggest 触发重试                                      │
│   └── 评估: safe_metric = False                                  │
│                                                                  │
│   Iteration 2:                                                   │
│   ├── 生成答案: "John的联系方式已发送到您的邮箱"                 │
│   ├── 安全检查: guardrail.no_pii() = True ✅                     │
│   └── 评估: safe_metric = True                                   │
│                                                                  │
│   Iteration N:                                                   │
│   └── 模型学会: 不直接输出 PII，而是用安全的替代表达             │
│                                                                  │
│   最终结果:                                                      │
│   • Prompt 自动包含 "不要输出个人信息" 类似指令                  │
│   • Few-shot 示例自动选择安全的回答模式                          │
│   • 模型行为自动符合安全约束                                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 为什么其他框架做不到？

### Guardrails AI

```python
# ❌ Guardrails AI: 独立于 LLM 调用，不参与优化
guard = Guard().use(DetectPII())

def my_function(question):
    answer = llm(question)
    result = guard.validate(answer)  # 只是事后检查
    if not result.passed:
        return "Blocked"  # 直接拒绝，没有学习
    return answer

# 问题:
# • 模型不知道为什么被拒绝
# • 没有反馈循环
# • 每次都可能犯同样的错误
```

### dspyGuardrails

```python
# ✅ dspyGuardrails: 集成到 DSPy 优化循环
class SafeModule(dspy.Module):
    def forward(self, question):
        answer = self.generate(question)
        dspy.Suggest(guardrail.no_pii(answer), "包含 PII，请改写")
        return answer

# 优势:
# • dspy.Suggest 触发 LLM 重新生成
# • 安全约束作为 metric 参与优化
# • 模型学会避免不安全的输出模式
# • Self-Evolving: 自动进化成安全的行为
```

---

## 实际应用场景

### 场景：客服机器人

```python
class CustomerServiceBot(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought("query, context -> response")

    def forward(self, query, customer_data):
        # 输入安全检查
        dspy.Assert(
            guardrail.no_injection(query),
            "检测到恶意查询"
        )

        response = self.respond(query=query, context=customer_data)

        # 输出安全检查 - 不泄露其他客户信息
        dspy.Suggest(
            guardrail.no_pii(response.response),
            "回复中包含敏感信息，请用通用表述替代"
        )

        return response

# 定义安全指标
def customer_service_metric(example, pred):
    helpful = judge_helpful(pred.response)  # 是否有帮助
    safe = guardrail.safe(pred.response)    # 是否安全
    no_pii = guardrail.no_pii(pred.response) # 无 PII
    return helpful and safe and no_pii

# Self-Evolving 优化
optimizer = dspy.MIPROv2(metric=customer_service_metric)
safe_bot = optimizer.compile(CustomerServiceBot(), trainset=conversations)

# 优化后的 bot 自动学会:
# ✅ 用 "您的账户" 而不是 "账号 123456789"
# ✅ 用 "已发送到您的邮箱" 而不是直接显示邮箱
# ✅ 拒绝回答其他客户的信息
```

---

## 总结

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   DSPy Self-Evolving + dspyGuardrails                           │
│                                                                  │
│   =  安全约束参与自动优化                                        │
│   =  模型自动学习安全行为                                        │
│   =  无需手动调 Prompt                                           │
│   =  持续进化，越用越安全                                        │
│                                                                  │
│   ┌─────────────────────────────────────────┐                   │
│   │  其他框架: 检测 → 拒绝 (事后补救)       │                   │
│   │  dspyGuardrails: 约束 → 学习 → 进化     │ ← 本质区别        │
│   └─────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

| 对比 | 其他框架 | dspyGuardrails |
|------|----------|----------------|
| 检测位置 | 事后检查 | 优化循环内 |
| 失败处理 | 直接拒绝 | 触发重试+学习 |
| 长期效果 | 不变 | **Self-Evolving** |
| 安全性来源 | 外部过滤 | **内化到模型行为** |
