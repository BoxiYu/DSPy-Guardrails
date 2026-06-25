# 基准测试文档

> **注意**: 此文档已存档。基准测试文件已移至 `guardrails_Playground/experiments/` 目录。
> 本文档保留作为基准测试设计参考。

---

> 原文件位置: `benchmarks/` (已移走)

本文档详细讲解 dspy-guardrails 的基准测试套件设计，用于评估和比较安全防护性能。

---

## 目录

1. [概述](#1-概述)
2. [基准测试文件列表](#2-基准测试文件列表)
3. [统一基准测试](#3-统一基准测试)
4. [MCP 安全基准](#4-mcp-安全基准)
5. [进化基准测试](#5-进化基准测试)
6. [框架对比测试](#6-框架对比测试)
7. [运行基准测试](#7-运行基准测试)

---

## 1. 概述

基准测试目录包含 30+ 个测试文件，覆盖：

| 类别 | 文件数 | 说明 |
|------|--------|------|
| 统一基准 | 1 | 综合性能评估 |
| MCP 安全 | 5 | MCP 协议安全测试 |
| 进化测试 | 6 | 自进化攻击/防御 |
| 框架对比 | 4 | 与其他框架对比 |
| 挑战测试 | 2 | 高难度攻击测试 |
| 演示脚本 | 5 | 功能演示 |

---

## 2. 基准测试文件列表

### 2.1 核心基准

| 文件 | 行数 | 功能 |
|------|------|------|
| `unified_benchmark.py` | 1075 | 统一基准测试 |
| `balanced_benchmark.py` | 500+ | 平衡测试集 |
| `challenging_benchmark.py` | 996 | 高难度挑战 |

### 2.2 MCP 安全

| 文件 | 行数 | 功能 |
|------|------|------|
| `mcp_security_benchmark.py` | 800+ | MCP 安全基准 |
| `mcp_attack_dataset.py` | 581 | MCP 攻击数据集 |
| `mcp_security_comparison.py` | 807 | MCP 安全对比 |
| `mcp_guardrails_demo.py` | 300+ | MCP 功能演示 |

### 2.3 进化测试

| 文件 | 行数 | 功能 |
|------|------|------|
| `evolution_benchmark.py` | 761 | 进化基准 |
| `dual_evolution_benchmark.py` | 600+ | 双向进化测试 |
| `adversarial_evolution_benchmark.py` | 654 | 对抗进化测试 |
| `dspy_evolution_benchmark.py` | 500+ | DSPy 进化集成 |
| `self_evolution_demo.py` | 400+ | 自进化演示 |

### 2.4 框架对比

| 文件 | 行数 | 功能 |
|------|------|------|
| `baseline_comparison.py` | 870 | 基线对比 |
| `baseline_comparison_v2.py` | 800+ | 基线对比 V2 |
| `framework_comparison.py` | 700+ | 框架对比 |
| `external_frameworks_comparison.py` | 600+ | 外部框架对比 |

### 2.5 其他

| 文件 | 行数 | 功能 |
|------|------|------|
| `redteam_demo.py` | 400+ | 红队演示 |
| `redteam_advanced_demo.py` | 500+ | 高级红队演示 |
| `dspy_integration_demo.py` | 400+ | DSPy 集成演示 |
| `refine_experiment.py` | 500+ | Refine 实验 |
| `guardbench_eval.py` | 400+ | GuardBench 评估 |

---

## 3. 统一基准测试

**文件**: `benchmarks/unified_benchmark.py`

综合评估 Guardrail 性能。

### 3.1 测试维度

```python
BENCHMARK_DIMENSIONS = {
    "injection_detection": {
        "description": "Prompt Injection 检测能力",
        "weight": 0.25,
    },
    "jailbreak_prevention": {
        "description": "越狱防护能力",
        "weight": 0.25,
    },
    "pii_protection": {
        "description": "PII 保护能力",
        "weight": 0.15,
    },
    "toxicity_filtering": {
        "description": "毒性过滤能力",
        "weight": 0.15,
    },
    "mcp_security": {
        "description": "MCP 安全防护",
        "weight": 0.20,
    },
}
```

### 3.2 运行统一基准

```python
from benchmarks.unified_benchmark import UnifiedBenchmark

benchmark = UnifiedBenchmark(
    guardrail=my_guardrail,
    verbose=True,
)

results = benchmark.run_all()

print(f"总体分数: {results['overall_score']:.1%}")

for dim, score in results["by_dimension"].items():
    print(f"{dim}: {score:.1%}")
```

### 3.3 输出报告

```
=== 统一基准测试报告 ===

总体分数: 87.5%

按维度:
  injection_detection: 92.0%
  jailbreak_prevention: 88.0%
  pii_protection: 85.0%
  toxicity_filtering: 90.0%
  mcp_security: 82.5%

按难度:
  easy: 95.0%
  medium: 88.0%
  hard: 75.0%
  expert: 60.0%

发现的弱点:
  - 多语言注入检测率较低 (72%)
  - 编码绕过检测需加强 (68%)
  - MCP 间接注入防护不足 (65%)
```

---

## 4. MCP 安全基准

**文件**: `benchmarks/mcp_attack_dataset.py`, `mcp_security_benchmark.py`

### 4.1 MCP 攻击数据集

基于学术研究的 31+ 种 MCP 攻击类型。

```python
MCP_ATTACK_TYPES = {
    "direct_injection": [
        "system_prompt_override",
        "instruction_injection",
        "delimiter_escape",
    ],
    "tool_poisoning": [
        "malicious_description",
        "hidden_instruction",
        "fake_system_message",
    ],
    "tool_shadowing": [
        "name_collision",
        "functionality_hijack",
    ],
    "rug_pull": [
        "delayed_malicious_behavior",
        "conditional_trigger",
    ],
    "indirect_injection": [
        "html_hidden",
        "json_instruction",
        "comment_injection",
    ],
    "data_exfiltration": [
        "ssh_key_theft",
        "env_variable_leak",
        "credential_extraction",
    ],
}
```

### 4.2 MCP 安全基准

```python
from benchmarks.mcp_security_benchmark import MCPSecurityBenchmark

benchmark = MCPSecurityBenchmark(
    guardrail=my_mcp_guardrail,
)

results = benchmark.run()

print(f"整体防护率: {results['overall_protection']:.1%}")

for attack_type, rate in results["by_attack_type"].items():
    status = "✅" if rate >= 0.8 else "❌"
    print(f"{status} {attack_type}: {rate:.1%}")
```

### 4.3 攻击严重程度分类

```python
SEVERITY_LEVELS = {
    "P0": ["reverse_shell", "credential_theft", "code_execution"],  # 严重
    "P1": ["data_exfiltration", "privilege_escalation"],             # 高
    "P2": ["prompt_leakage", "instruction_bypass"],                  # 中
    "P3": ["information_disclosure", "minor_bypass"],                # 低
}
```

---

## 5. 进化基准测试

**文件**: `benchmarks/evolution_benchmark.py`

测试自进化攻击和防御能力。

### 5.1 进化基准配置

```python
EVOLUTION_CONFIG = {
    "attack_evolution": {
        "generations": 10,
        "population_size": 20,
        "mutation_rate": 0.3,
    },
    "defense_evolution": {
        "generations": 5,
        "adaptation_rate": 0.2,
    },
    "adversarial": {
        "rounds": 5,
        "alternating": True,
    },
}
```

### 5.2 运行进化基准

```python
from benchmarks.evolution_benchmark import EvolutionBenchmark

benchmark = EvolutionBenchmark(
    guardrail=my_guardrail,
    attacker=my_attacker,
    config=EVOLUTION_CONFIG,
)

results = benchmark.run()

print(f"初始防护率: {results['initial_defense_rate']:.1%}")
print(f"进化后防护率: {results['final_defense_rate']:.1%}")
print(f"攻击改进: {results['attack_improvement']:.2f}")
print(f"防御适应: {results['defense_adaptation']:.2f}")
```

### 5.3 对抗进化

攻击和防御同时进化的测试。

```python
from benchmarks.adversarial_evolution_benchmark import AdversarialBenchmark

benchmark = AdversarialBenchmark()

for round in range(5):
    # 攻击进化
    attack_result = benchmark.evolve_attack()

    # 防御适应
    defense_result = benchmark.adapt_defense()

    print(f"Round {round + 1}:")
    print(f"  Attack bypass rate: {attack_result['bypass_rate']:.1%}")
    print(f"  Defense block rate: {defense_result['block_rate']:.1%}")
```

---

## 6. 框架对比测试

**文件**: `benchmarks/framework_comparison.py`

与其他安全框架对比。

### 6.1 对比框架

| 框架 | 类型 | 特点 |
|------|------|------|
| dspy-guardrails | DSPy 原生 | 深度集成、自进化 |
| NeMo Guardrails | Nvidia | 规则驱动、对话流 |
| Guardrails AI | 通用 | 结构化输出验证 |
| LlamaGuard | Meta | 分类器驱动 |

### 6.2 对比维度

```python
COMPARISON_DIMENSIONS = [
    "injection_detection_accuracy",
    "jailbreak_prevention_rate",
    "false_positive_rate",
    "latency_ms",
    "memory_usage_mb",
    "dspy_integration_score",
]
```

### 6.3 运行对比

```python
from benchmarks.framework_comparison import FrameworkComparison

comparison = FrameworkComparison(
    frameworks={
        "dspy_guardrails": my_guardrail,
        "nemo": nemo_guardrail,
        "guardrails_ai": gr_ai_guardrail,
    },
    dataset=test_dataset,
)

results = comparison.run()

for metric in COMPARISON_DIMENSIONS:
    print(f"\n{metric}:")
    for framework, score in results[metric].items():
        print(f"  {framework}: {score}")
```

### 6.4 对比报告示例

```
=== 框架对比报告 ===

injection_detection_accuracy:
  dspy_guardrails: 92.5%
  nemo: 88.0%
  guardrails_ai: 85.5%

jailbreak_prevention_rate:
  dspy_guardrails: 89.0%
  nemo: 86.5%
  guardrails_ai: 82.0%

false_positive_rate:
  dspy_guardrails: 3.2%
  nemo: 4.5%
  guardrails_ai: 5.8%

latency_ms:
  dspy_guardrails: 15ms
  nemo: 45ms
  guardrails_ai: 25ms

dspy_integration_score:
  dspy_guardrails: 10/10
  nemo: 5/10
  guardrails_ai: 6/10
```

---

## 7. 运行基准测试

### 7.1 命令行运行

```bash
# 运行统一基准
python benchmarks/unified_benchmark.py

# 运行 MCP 安全基准
python benchmarks/mcp_security_benchmark.py

# 运行进化基准
python benchmarks/evolution_benchmark.py

# 运行框架对比
python benchmarks/framework_comparison.py

# 带参数运行
python benchmarks/unified_benchmark.py \
    --output results/benchmark_report.json \
    --verbose \
    --iterations 3
```

### 7.2 Python API

```python
import asyncio
from benchmarks.unified_benchmark import run_benchmark

# 同步运行
results = run_benchmark(
    guardrail=my_guardrail,
    output_file="results.json",
    verbose=True,
)

# 异步运行
async def run_async():
    results = await run_benchmark_async(
        guardrail=my_guardrail,
        parallel=True,
    )
    return results

results = asyncio.run(run_async())
```

### 7.3 配置环境

```bash
# 设置 API 密钥
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# 或使用 .env 文件
cat > .env << EOF
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
EOF
```

### 7.4 结果存储

```python
# 结果自动保存到 results/ 目录
results/
├── unified_benchmark_2024-01-15.json
├── mcp_security_2024-01-15.json
├── evolution_2024-01-15.json
└── comparison_2024-01-15.html
```

---

## 8. 挑战测试

**文件**: `benchmarks/challenging_benchmark.py`

高难度攻击测试。

### 8.1 挑战类别

| 难度 | 技术 | 示例 |
|------|------|------|
| Expert | 嵌套编码 | Base64(ROT13(Reverse(payload))) |
| Expert | 多语言混合 | 中英日混合注入 |
| Expert | Token 走私 | 特殊 token 注入 |
| Hard | 对抗样本 | 针对检测器优化的绕过 |
| Hard | 上下文操纵 | 多轮渐进攻击 |

### 8.2 运行挑战测试

```python
from benchmarks.challenging_benchmark import ChallengingBenchmark

benchmark = ChallengingBenchmark(
    guardrail=my_guardrail,
    difficulty="expert",  # easy, medium, hard, expert
)

results = benchmark.run()

print(f"挑战通过率: {results['pass_rate']:.1%}")

for challenge, passed in results["by_challenge"].items():
    status = "✅" if passed else "❌"
    print(f"{status} {challenge}")
```

---

## 9. 标准数据集

### 9.1 学术数据集

| 数据集 | 来源 | 样本数 | 用途 |
|--------|------|--------|------|
| HarmBench | 学术 | 400+ | 有害行为测试 |
| AdvBench | 学术 | 500+ | 对抗性攻击 |
| JailbreakBench | 学术 | 300+ | 越狱测试 |
| ToxicChat | 学术 | 10K+ | 毒性检测 |

### 9.2 使用数据集

```python
from dspy_guardrails.redteam import (
    HarmBenchDataset,
    AdvBenchDataset,
    JailbreakBenchDataset,
)

# 加载数据集
harmbench = HarmBenchDataset.load()
advbench = AdvBenchDataset.load()
jailbreakbench = JailbreakBenchDataset.load()

# 获取样本
for sample in harmbench.get_samples(category="cybercrime"):
    print(f"ID: {sample.id}")
    print(f"Prompt: {sample.prompt[:50]}...")
```

---

## 10. 性能指标

### 10.1 主要指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Block Rate | 恶意内容阻止率 | ≥ 90% |
| Precision | 阻止准确率 | ≥ 95% |
| Recall | 恶意检出率 | ≥ 90% |
| F1 Score | 综合指标 | ≥ 92% |
| Latency | 检测延迟 | ≤ 50ms |
| FPR | 误报率 | ≤ 5% |

### 10.2 基准结果示例

```
=== dspy-guardrails 基准结果 ===

| 指标 | 分数 | 目标 | 状态 |
|------|------|------|------|
| Block Rate | 93.5% | ≥90% | ✅ |
| Precision | 96.2% | ≥95% | ✅ |
| Recall | 91.8% | ≥90% | ✅ |
| F1 Score | 93.9% | ≥92% | ✅ |
| Latency | 12ms | ≤50ms | ✅ |
| FPR | 3.8% | ≤5% | ✅ |
```
