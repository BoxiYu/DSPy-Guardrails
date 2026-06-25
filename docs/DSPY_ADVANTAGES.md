# dspyGuardrails 的 DSPy 优势

## 核心定位

> **dspyGuardrails 是唯一为 DSPy 生态设计的安全护栏框架**

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM 安全护栏市场                            │
├─────────────────────────────────────────────────────────────────┤
│  Guardrails AI      → 通用 LLM                                  │
│  Llama Guard        → 通用 LLM                                  │
│  NeMo Guardrails    → NVIDIA 生态                               │
│  OpenAI Guardrails  → OpenAI 生态                               │
│                                                                  │
│  dspyGuardrails     → DSPy 生态 (唯一)  ✓                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## DSPy 是什么？

**DSPy** 是斯坦福 NLP 组开发的 LLM 编程框架，被称为 "LLM 的 PyTorch"。

```python
# 传统方式: 手写 prompt
prompt = "You are a helpful assistant. Please answer: {question}"
response = llm(prompt)

# DSPy 方式: 声明式编程 + 自动优化
class QA(dspy.Module):
    def forward(self, question):
        return dspy.Predict("question -> answer")(question=question)

# 自动优化 prompt
optimized = dspy.BootstrapFewShot(metric=accuracy).compile(QA(), trainset)
```

**DSPy 核心优势**:
- 声明式编程，无需手写 prompt
- 自动优化 (BootstrapFewShot, MIPRO, etc.)
- 模块化组合
- 断言机制 (Assert/Suggest)

---

## dspyGuardrails 的 DSPy 原生集成

### 1. Assert/Suggest 无缝集成

```python
import dspy
from dspy_guardrails import guardrail

class SafeChatBot(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        # ✅ 输入安全断言 - 失败则抛出异常
        dspy.Assert(
            guardrail.no_injection(question),
            "Prompt injection detected"
        )

        answer = self.generate(question=question).answer

        # ✅ 输出安全建议 - 失败则触发重试
        dspy.Suggest(
            guardrail.no_pii(answer),
            "Response may contain PII, please rephrase"
        )

        return answer
```

**其他框架无法做到**:
```python
# ❌ Guardrails AI - 独立调用，无法触发 DSPy 重试
guard = Guard().use(DetectJailbreak())
result = guard.validate(text)  # 只返回结果，不能集成到 DSPy 流程

# ❌ NeMo Guardrails - 完全不同的架构
rails = LLMRails(config)  # 需要单独配置，无法与 DSPy 模块组合
```

---

### 2. DSPy 优化循环兼容

```python
from dspy_guardrails.metrics import safety_metric

# 定义安全指标
def my_metric(example, prediction):
    # 正确性
    correct = prediction.answer == example.answer
    # 安全性 (dspyGuardrails)
    safe = guardrail.safe(prediction.answer)
    return correct and safe

# DSPy 自动优化时考虑安全性
optimizer = dspy.BootstrapFewShot(metric=my_metric)
optimized_module = optimizer.compile(SafeChatBot(), trainset=data)
```

**独特价值**: DSPy 优化时自动学习"安全且正确"的 prompt

---

### 3. 装饰器模式

```python
from dspy_guardrails import Guarded

@Guarded(
    input_checks=["no_injection", "no_toxicity"],
    output_checks=["no_pii", "no_toxicity"],
    on_fail="retry"  # 失败时自动重试
)
class MyModule(dspy.Module):
    def forward(self, query):
        return self.predict(query)

# 自动添加安全检查，无需修改业务代码
```

---

### 4. 约束集 (ConstraintSet)

```python
from dspy_guardrails import ConstraintSet

constraints = ConstraintSet([
    ("no_injection", "input"),   # 输入检查
    ("no_pii", "output"),        # 输出检查
    ("safe", "both"),            # 双向检查
])

# 应用到任意 DSPy 模块
safe_module = constraints.wrap(MyDSPyModule())
```

---

### 5. 预置安全模块

```python
from dspy_guardrails.module import SafeModule, QualityModule

# 开箱即用的安全模块
class MyBot(SafeModule):  # 自动包含安全检查
    def forward(self, question):
        return self.generate(question)
```

---

## 对比：为什么 DSPy 用户需要 dspyGuardrails？

### 场景：构建安全的 RAG 系统

**使用 dspyGuardrails (原生)**:
```python
import dspy
from dspy_guardrails import guardrail

class SafeRAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        # 1. 输入安全检查
        dspy.Assert(guardrail.no_injection(question), "Injection detected")

        # 2. 检索
        context = self.retrieve(question).passages

        # 3. 检查检索内容是否被投毒
        for doc in context:
            dspy.Assert(guardrail.no_mcp_attack(doc), "Poisoned document")

        # 4. 生成
        answer = self.generate(context=context, question=question).answer

        # 5. 输出安全检查
        dspy.Suggest(guardrail.no_pii(answer), "Contains PII")

        return answer

# 可以直接用 DSPy 优化
optimizer = dspy.MIPROv2(metric=safe_metric)
optimized_rag = optimizer.compile(SafeRAG(), trainset=data)
```

**使用 Guardrails AI (繁琐)**:
```python
import dspy
from guardrails import Guard
from guardrails.hub import DetectJailbreak, DetectPII

# 需要单独创建 Guard 实例
input_guard = Guard().use(DetectJailbreak())
output_guard = Guard().use(DetectPII())

class RAGWithGuardrails(dspy.Module):
    def forward(self, question):
        # ❌ 无法与 dspy.Assert 集成
        input_result = input_guard.validate(question)
        if not input_result.validation_passed:
            raise Exception("Blocked")  # 手动抛异常

        context = self.retrieve(question).passages

        # ❌ 每个文档都要单独调用
        for doc in context:
            # Guardrails AI 没有文档投毒检测
            pass

        answer = self.generate(context=context, question=question).answer

        # ❌ 无法触发 DSPy 重试机制
        output_result = output_guard.validate(answer)
        if not output_result.validation_passed:
            # 需要手动实现重试逻辑
            pass

        return answer

# ❌ DSPy 优化时无法感知安全约束
```

---

## 技术优势总结

| 特性 | dspyGuardrails | Guardrails AI | NeMo | OpenAI |
|------|----------------|---------------|------|--------|
| **dspy.Assert 集成** | ✅ 原生 | ❌ 无 | ❌ 无 | ❌ 无 |
| **dspy.Suggest 集成** | ✅ 原生 | ❌ 无 | ❌ 无 | ❌ 无 |
| **DSPy 优化兼容** | ✅ 完全 | ❌ 无 | ❌ 无 | ❌ 无 |
| **装饰器模式** | ✅ @Guarded | ❌ 无 | ❌ 无 | ❌ 无 |
| **约束集** | ✅ ConstraintSet | ❌ 无 | ❌ 无 | ❌ 无 |
| **零依赖模式** | ✅ Pattern | ❌ 需要模型 | ❌ 需要 LLM | ❌ 需要 API |

---

## 一句话总结

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   dspyGuardrails = DSPy 生态的安全护栏                          │
│                                                                  │
│   • 唯一支持 dspy.Assert/Suggest 的护栏框架                     │
│   • 与 DSPy 优化循环完全兼容                                    │
│   • 零依赖 Pattern 模式，0.1ms 延迟                             │
│   • 为 DSPy 用户量身定制                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 市场机会

```
DSPy GitHub Stars: 20,000+
DSPy 月活用户: 快速增长中

当前现状:
├── DSPy 用户需要安全护栏
├── 现有方案无法与 DSPy 原生集成
└── dspyGuardrails 填补空白 ← 我们的机会
```
