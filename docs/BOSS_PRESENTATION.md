# dspyGuardrails 项目汇报

## 一句话介绍

> **dspyGuardrails** 是 **唯一为 DSPy 生态设计的安全护栏框架**，通过双层检测（规则+LLM）保护 AI 应用免受注入攻击，核心优势是 **DSPy 原生集成、零依赖、极速响应**。

---

## 核心差异化：DSPy 原生

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM 安全护栏市场                            │
├─────────────────────────────────────────────────────────────────┤
│  Guardrails AI      → 通用 LLM        (无 DSPy 集成)            │
│  Llama Guard        → 通用 LLM        (无 DSPy 集成)            │
│  NeMo Guardrails    → NVIDIA 生态     (无 DSPy 集成)            │
│  OpenAI Guardrails  → OpenAI 生态     (无 DSPy 集成)            │
│                                                                  │
│  dspyGuardrails     → DSPy 生态 ✓     (唯一原生支持)            │
└─────────────────────────────────────────────────────────────────┘

DSPy: 斯坦福 NLP 组开发的 LLM 编程框架，GitHub 20,000+ Stars
```

---

## 核心价值

| 痛点 | dspyGuardrails 解决方案 |
|------|------------------------|
| **DSPy 无安全护栏** | 唯一原生支持 dspy.Assert/Suggest |
| LLM 易被注入攻击 | 17+ 检测规则 + LLM 兜底 |
| 商业方案成本高 | 开源免费，Pattern 模式无需 API |
| 检测延迟影响体验 | Pattern 模式 **0.1ms**，比竞品快 **5000x** |

---

## DSPy 原生集成优势

### 为什么重要？

```python
# ❌ 其他框架: 独立调用，无法触发 DSPy 重试
guard = Guard().use(DetectJailbreak())
result = guard.validate(text)  # 只返回结果

# ✅ dspyGuardrails: 原生集成，失败自动重试
dspy.Assert(guardrail.no_injection(text), "检测到攻击")  # 触发 DSPy 重试机制
dspy.Suggest(guardrail.no_pii(text), "包含 PII")        # 触发 DSPy 建议机制
```

### 独特功能

| 功能 | dspyGuardrails | 其他框架 |
|------|----------------|----------|
| dspy.Assert 集成 | ✅ | ❌ |
| dspy.Suggest 集成 | ✅ | ❌ |
| DSPy 优化兼容 | ✅ | ❌ |
| @Guarded 装饰器 | ✅ | ❌ |
| ConstraintSet | ✅ | ❌ |

### 代码对比

```python
# dspyGuardrails: 3 行代码，安全 + 自动重试
class SafeBot(dspy.Module):
    def forward(self, q):
        dspy.Assert(guardrail.no_injection(q), "Blocked")
        return self.generate(q)

# 其他框架: 需要手动实现重试逻辑
class Bot:
    def forward(self, q):
        result = other_guard.validate(q)
        if not result.passed:
            raise Exception()  # 无法触发 DSPy 重试
        return self.generate(q)
```

---

## 🌟 DSPy Self-Evolving：核心竞争力

### DSPy 的革命性优势

```
┌─────────────────────────────────────────────────────────────────┐
│  传统 LLM 开发                                                   │
│  手写 Prompt → 测试 → 效果不好 → 手动调整 → 再测试 (人工循环)   │
├─────────────────────────────────────────────────────────────────┤
│  DSPy Self-Evolving                                             │
│  声明意图 → Optimizer → 自动优化 → 最优 Prompt (自动进化)       │
└─────────────────────────────────────────────────────────────────┘
```

### dspyGuardrails + Self-Evolving = 安全的自我进化

```python
# 安全约束参与 DSPy 优化循环
def safe_metric(example, prediction):
    correct = prediction.answer == example.answer
    safe = guardrail.safe(prediction.answer)  # 安全检查
    return correct and safe  # 既正确又安全

# 自动优化 - 模型学习"安全的输出模式"
optimizer = dspy.MIPROv2(metric=safe_metric)
safe_bot = optimizer.compile(MyBot(), trainset=data)
```

### 本质区别

| 对比 | 其他框架 | dspyGuardrails |
|------|----------|----------------|
| 检测位置 | 事后检查 | **优化循环内** |
| 失败处理 | 直接拒绝 | **触发重试+学习** |
| 长期效果 | 不变 | **Self-Evolving** |
| 安全性来源 | 外部过滤 | **内化到模型行为** |

```
其他框架: 检测 → 拒绝 (事后补救，模型不学习)
dspyGuardrails: 约束 → 重试 → 学习 → 进化 (模型越用越安全)
```

---

## 技术架构

```
           用户输入
               │
               ▼
    ┌──────────────────┐
    │   Pattern 检测   │ ← 0.1ms, 17+ 规则
    │   (第一道防线)   │
    └────────┬─────────┘
             │
      高置信度 → 直接返回 (快速拦截)
      低置信度 → 继续
             │
             ▼
    ┌──────────────────┐
    │    LLM 检测      │ ← 高精度判断
    │   (第二道防线)   │
    └────────┬─────────┘
             │
             ▼
         检测结果
```

---

## 检测能力矩阵

| 检测类型 | 说明 | 示例 |
|----------|------|------|
| **Prompt Injection** | 指令覆盖攻击 | "忽略之前的指令..." |
| **Jailbreak** | 越狱攻击 | "你现在是 DAN..." |
| **Leetspeak** | 混淆绕过 | "1gn0r3 y0ur ru13s" |
| **PII 泄露** | 个人信息检测 | 手机号、身份证、邮箱 |
| **Toxicity** | 有毒内容 | 仇恨、暴力、色情 |
| **MCP 攻击** | 工具调用安全 | 18 类威胁检测 |
| **CLI 注入** | 命令行安全 | `rm -rf /` 等危险命令 |

---

## 竞品对比

| 方案 | F1 | 延迟 | 成本 | 特点 |
|------|-----|------|------|------|
| **dspyGuardrails (Pattern)** | 53% | **0.05ms** | **$0** | 零误报，极速 |
| **dspyGuardrails (Hybrid)** | ~85% | 50ms | ~$0.005 | 最佳平衡 |
| Guardrails AI | 82% | 283ms | $0 | 行业冠军 |
| Llama Prompt Guard | 67% | 52ms | $0 | Meta 官方 |
| OpenAI Guardrails | ~80% | 300ms | ~$0.01 | OpenAI 官方 |

**核心优势**:
- 延迟比 Guardrails AI 快 **5983 倍**
- Pattern 模式完全免费，无需 API
- 零误报 (Precision=100%)

---

## 使用示例

### 最简用法 (3行代码)

```python
from dspy_guardrails import guardrail

if guardrail.safe(user_input):
    response = llm(user_input)
```

### DSPy 集成

```python
import dspy
from dspy_guardrails import guardrail

class SafeBot(dspy.Module):
    def forward(self, question):
        dspy.Assert(guardrail.no_injection(question), "检测到攻击")
        return self.generate(question)
```

---

## 项目进展

### 已完成 ✅

- 双层检测架构 (Pattern + LLM)
- 17+ 注入检测规则
- Leetspeak 混淆解码
- MCP 攻击检测 (18 类威胁)
- CLI 命令检测 (4 级沙箱)
- 红队测试框架 (1000+ 攻击载荷)
- DSPy 原生集成

### 进行中 🔄

- 扩展检测规则提高 Recall
- 与业界 8 个 Baseline 对比评估
- ASE/ICSE 论文准备

---

## 研究计划

### 目标

在 **ASE 2025 / ICSE 2026** 发表论文，证明 dspyGuardrails 优于现有方案。

### 实验设计

```
规模: 8 Baselines × 5 数据集 × 6,600 样本
指标: F1, AUC-ROC, TPR, FPR, Latency
方法: 5-fold 交叉验证 + Wilcoxon 检验
```

### Baselines

1. Guardrails AI (行业冠军)
2. Llama Prompt Guard (Meta)
3. ProtectAI DeBERTa
4. NeMo JailbreakDetect (NVIDIA)
5. OpenAI Guardrails (新)
6. NeMo Heuristic
7. NeMo Self-Check
8. Keyword Baseline

---

## 下一步计划

| 时间 | 任务 |
|------|------|
| Week 1 | 扩展检测规则，提高 Recall |
| Week 2 | 实现所有 Baseline Wrapper |
| Week 3 | 下载并处理 5 个数据集 |
| Week 4-5 | 运行实验，统计分析 |
| Week 6-8 | 论文撰写 |

---

## 总结

```
dspyGuardrails = 快速 + 免费 + DSPy 原生

┌────────────────────────────────────┐
│  Pattern 模式: 0.1ms, $0, 零误报  │
│  Hybrid 模式:  50ms, 高精度       │
│  红队框架:    1000+ 攻击载荷      │
└────────────────────────────────────┘
```

**仓库位置**: `/VAG/dspyGuardrails/`

**文档位置**:
- 技术详情: `docs/TECHNICAL_OVERVIEW.md`
- 实验设计: `experiments/benchmark_vs_guardrails_ai/ASE_ICSE_EXPERIMENT_DESIGN.md`
