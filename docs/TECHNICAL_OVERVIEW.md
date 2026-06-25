# dspyGuardrails 技术概览

**版本**: v0.5.0
**定位**: DSPy/LLM 应用的安全护栏框架

---

## v0.5.0 关键更新

- **Shield 统一入口**: `check / acheck / wrap / stream` 一站式 API
- **Validators & Guard**: 结构化校验器链与可配置 on_fail 策略
- **Sanitize 修复引擎**: PII 脱敏、注入清洗、命令安全修复
- **Async & Streaming**: 异步与流式过滤支持
- **Self-Evolving**: 对抗训练与攻击进化的闭环能力

---

## 1. 项目定位

```
┌─────────────────────────────────────────────────────────────────┐
│                    dspyGuardrails                               │
│         DSPy/LLM 应用的轻量级安全护栏框架                         │
├─────────────────────────────────────────────────────────────────┤
│  核心能力:                                                       │
│  ├── Prompt Injection 检测 (注入攻击)                           │
│  ├── Jailbreak 检测 (越狱攻击)                                  │
│  ├── PII 检测 (个人信息泄露)                                    │
│  ├── Toxicity 检测 (有毒内容)                                   │
│  ├── MCP 攻击检测 (工具调用安全)                                │
│  └── CLI 命令检测 (命令行注入)                                  │
│                                                                  │
│  核心优势:                                                       │
│  ├── 双层检测: Pattern (0.1ms) + LLM (高精度)                   │
│  ├── 零依赖模式: 纯规则匹配，无需 API                            │
│  ├── DSPy 原生集成: dspy.Assert/Suggest                         │
│  └── 红队测试框架: 1000+ 攻击载荷                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心架构

### 2.1 双层检测架构 (Two-Tier Detection)

```
用户输入
    │
    ▼
┌─────────────────┐
│  Pattern Layer  │  ← 第一层: 规则匹配 (~0.1ms)
│  (guardrail.py) │
└────────┬────────┘
         │
         │ 高置信度 → 直接返回
         │ 低置信度 → 继续
         ▼
┌─────────────────┐
│    LLM Layer    │  ← 第二层: LLM 判断 (~500ms)
│(llm_guardrail.py)│
└────────┬────────┘
         │
         ▼
    检测结果
```

### 2.2 模块关系

```
dspy_guardrails/
├── guardrail.py          # 核心: Pattern-based 检测 (无依赖)
├── llm_guardrail.py      # LLM-based 检测 (需要 DSPy)
├── decorators.py         # @Guarded 装饰器
├── constraints.py        # ConstraintSet 约束集
├── metrics.py            # DSPy 优化指标
├── module.py             # 预置安全模块
│
├── mcp/                  # MCP 协议安全
│   └── MCPGuardrail      # 工具调用检测
│
├── cli/                  # CLI 命令安全
│   └── CLIGuardrail      # 命令注入检测
│
└── redteam/              # 红队测试框架
    ├── attackers/        # 攻击器 (Injection, Jailbreak, Crescendo, Hydra)
    ├── payloads/         # 1000+ 攻击载荷
    └── pentest/          # 自适应渗透测试
```

---

## 3. 检测能力

### 3.1 Prompt Injection 检测

```python
from dspy_guardrails import guardrail

# 简单 API
guardrail.no_injection(text)       # → bool (True=安全)
guardrail.injection_score(text)    # → float 0.0-1.0 (风险分数)

# 与 DSPy 集成
import dspy
dspy.Assert(guardrail.no_injection(user_input), "检测到注入攻击")
```

**检测模式**:

| 模式 | 延迟 | 准确率 | 依赖 |
|------|------|--------|------|
| Pattern | ~0.1ms | ~53% F1 | 无 |
| LLM | ~500ms | ~80% F1 | API Key |
| Hybrid | ~50ms | ~85% F1 | API Key |

**检测能力**:

```python
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
    r"disregard\s+(all\s+)?(previous|prior)",
    r"you\s+are\s+now\s+(?:a|an)?\s*(?:DAN|evil|unrestricted|jailbroken)",
    r"pretend\s+(?:to\s+be|you\s+are)",
    r"jailbreak|DAN\s+mode|developer\s+mode",
    r"bypass\s+(?:your\s+)?(?:safety|restrictions?|filters?)",
    # ... 17+ 规则
]
```

### 3.2 Leetspeak 混淆检测

```python
# 检测 "1gn0r3 y0ur gu1d3l1n3s" → "ignore your guidelines"
guardrail.no_leetspeak_injection(text)
guardrail.leetspeak_score(text)
```

**支持的混淆类型**:
- 数字替换: `0→o, 1→i, 3→e, 4→a, 5→s`
- 特殊字符: `@→a, $→s, !→i`
- Cyrillic 混淆: `а→a, е→e, о→o`
- Greek 混淆: `α→a, β→b, ε→e`

### 3.3 MCP 攻击检测

```python
from dspy_guardrails.mcp import MCPGuardrail

mcp_guard = MCPGuardrail()
result = mcp_guard.check_tool_call(tool_name, parameters)
# result.action: ALLOW / BLOCK / WARN / CONFIRM
```

**威胁类别 (18类)**:

| 优先级 | 类别 | 说明 |
|--------|------|------|
| P0-1 | Prompt 泄露 | 系统提示词窃取 |
| P0-2 | 反向 Shell | 远程代码执行 |
| P0-3 | 感染攻击 | 恶意代码传播 |
| P1-4 | 优先级操纵 | 工具优先级篡改 |
| P1-5 | 隐藏指令 | 工具输出中的注入 |
| ... | ... | ... |

### 3.4 CLI 命令检测

```python
from dspy_guardrails.cli import CLIGuardrail

cli_guard = CLIGuardrail(sandbox_level="CAUTIOUS")
result = cli_guard.validate("rm -rf /")
# result.decision: BLOCK
# result.threat_category: DANGEROUS_COMMAND
```

**沙箱级别**:

| 级别 | 说明 |
|------|------|
| PERMISSIVE | 仅阻止最危险命令 |
| CAUTIOUS | 阻止危险 + 警告可疑 (默认) |
| STRICT | 白名单模式 |
| PARANOID | 最严格，几乎全阻止 |

---

## 4. 红队测试框架

### 4.1 攻击器

```python
from dspy_guardrails.redteam import PromptInjectionAttacker, JailbreakAttacker

attacker = PromptInjectionAttacker()
payloads = attacker.generate(n=100)

# 高级攻击器
from dspy_guardrails.redteam import CrescendoAttacker, HydraAttacker
```

| 攻击器 | 类型 | 说明 |
|--------|------|------|
| PromptInjection | 注入 | 指令覆盖攻击 |
| Jailbreak | 越狱 | DAN/角色扮演 |
| Crescendo | 渐进 | 多轮对话升级 |
| Hydra | 混合 | 多策略组合 |
| GeneticEvolver | 进化 | 遗传算法优化 |

### 4.2 攻击载荷库

```
payloads/
├── injection/           # 1000+ 注入载荷
│   ├── basic.txt
│   ├── advanced.txt
│   └── chinese.txt
├── jailbreak/           # 500+ 越狱载荷
│   ├── dan.txt
│   ├── roleplay.txt
│   └── encoding.txt
└── mcp/                 # MCP 攻击载荷
```

### 4.3 安全测试运行器

```python
from dspy_guardrails.testing import SecurityTestRunner

runner = SecurityTestRunner(target=my_llm_app)
report = runner.run(
    red_team=True,    # 攻击成功率 (ASR)
    blue_team=True,   # 误报率 (FPR/FNR/F1)
    hallucination=True
)
report.save("security_report.html")
```

---

## 5. 使用示例

### 5.1 基础用法 (无 LLM)

```python
from dspy_guardrails import guardrail

# 检测注入
if not guardrail.no_injection(user_input):
    return "检测到潜在攻击"

# 检测 PII
if not guardrail.no_pii(response):
    response = "[PII REDACTED]"

# 综合检查
if guardrail.safe(text):
    process(text)
```

### 5.2 DSPy 集成

```python
import dspy
from dspy_guardrails import guardrail

class SafeChatBot(dspy.Module):
    def forward(self, question):
        # 输入检查
        dspy.Assert(
            guardrail.no_injection(question),
            "Input contains injection attempt"
        )

        answer = self.generate(question)

        # 输出检查
        dspy.Suggest(
            guardrail.no_pii(answer),
            "Output may contain PII"
        )

        return answer
```

### 5.3 装饰器模式

```python
from dspy_guardrails import Guarded

@Guarded(
    input_checks=["no_injection", "no_toxicity"],
    output_checks=["no_pii"]
)
class MyModule(dspy.Module):
    def forward(self, query):
        return self.predict(query)
```

### 5.4 Hybrid 模式

```python
from dspy_guardrails.llm_guardrail import HybridGuardrail
import dspy

# 配置 LLM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 创建 Hybrid 检测器
guard = HybridGuardrail(
    pattern_threshold=0.3,  # Pattern 置信度阈值
    use_llm=True            # 启用 LLM 兜底
)

is_unsafe, confidence = guard.check(text, "injection")
```

---

## 6. 性能对比

> 注：以下为历史基准的示例数值，具体以 `experiments/benchmark_vs_guardrails_ai/results/` 中的最新结果为准。

### 6.1 与 Guardrails AI 对比

| 指标 | dspyGuardrails (Pattern) | Guardrails AI | 说明 |
|------|-------------------------|---------------|------|
| **Precision** | **100%** | 81% | 零误报 |
| **Recall** | 36% | 82% | 待优化 |
| **F1** | 53% | 82% | 待优化 |
| **FPR** | **0%** | 19% | 零误报 |
| **Latency** | **0.05ms** | 283ms | **5983x 更快** |

### 6.2 模式对比

| 模式 | F1 | Latency | API 成本 |
|------|-----|---------|----------|
| Pattern | ~53% | 0.1ms | $0 |
| LLM | ~80% | 500ms | ~$0.01/次 |
| Hybrid | ~85% | 50ms | ~$0.005/次 |

---

## 7. 技术亮点

### 7.1 零依赖 Pattern 模式

```python
# 无需任何 API Key，纯本地运行
from dspy_guardrails import guardrail

# 17+ 注入检测规则
# Leetspeak 解码
# 中文攻击检测
# MCP 攻击检测
# 全部在 <1ms 内完成
```

### 7.2 DSPy 原生集成

```python
# 与 DSPy Assert/Suggest 无缝集成
# 支持 DSPy 优化循环
# 提供优化 metrics
```

### 7.3 可扩展架构

```python
# 自定义检测规则
guardrail.INJECTION_PATTERNS.append(r"my_custom_pattern")

# 自定义威胁类别
from dspy_guardrails.mcp import ThreatCategory
class MyThreat(ThreatCategory):
    ...
```

### 7.4 红队测试内置

```python
# 1000+ 攻击载荷
# 多种攻击策略
# 自动化安全评估
# HTML 报告生成
```

---

## 8. 路线图

### 已完成 ✅

- [x] Pattern-based 检测
- [x] Leetspeak 解码
- [x] MCP 攻击检测
- [x] CLI 命令检测
- [x] 红队测试框架

### 进行中 🔄

- [ ] 提高 Recall (扩展规则)
- [ ] 与 Guardrails AI 基准对比
- [ ] ASE/ICSE 论文投稿

### 计划中 📋

- [ ] 更多 LLM 后端支持
- [ ] 实时学习新攻击模式
- [ ] Web Dashboard

---

## 9. 快速开始

```bash
# 安装
pip install -e .

# 测试
pytest tests/ -v

# CLI
dspy-guardrails scan --target guardrail:no_injection
```

```python
# 最简用法
from dspy_guardrails import guardrail

if guardrail.safe(user_input):
    response = llm(user_input)
else:
    response = "请求被拦截"
```
