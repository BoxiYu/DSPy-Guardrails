# dspy-guardrails 项目文档

> 一个为 DSPy 应用提供全面安全防护的 Python 框架

---

## 项目概述

**dspy-guardrails** 是一个生产级的 AI 安全框架，提供：

- **输入验证**：Prompt 注入检测、PII 检测、内容过滤
- **输出验证**：毒性检测、PII 过滤
- **MCP 安全**：Model Context Protocol 的完整安全防护
- **红队测试**：自动化攻击生成、进化攻击、多轮对话攻击
- **DSPy 深度集成**：支持 `dspy.Assert`、`dspy.Suggest`、装饰器

---

## 文档索引

| 文档 | 描述 |
|------|------|
| [01-core.md](./01-core.md) | 核心模块 - 基础 Guardrail、约束系统、装饰器 |
| [02-mcp.md](./02-mcp.md) | MCP 安全模块 - 工具防护、策略、审计 |
| [03-redteam.md](./03-redteam.md) | 红队攻击工具 - 攻击生成器、进化引擎、多轮攻击 |
| [04-security-testing.md](./04-security-testing.md) | 安全测试框架 - RedTeam/BlueTeam 评估器 |
| [05-benchmarks.md](./05-benchmarks.md) | 基准测试 - 标准数据集、性能评估 (存档) |
| [06-integration.md](./06-integration.md) | 集成指南 - DSPy 集成、快速开始 |
| [07-testing.md](./07-testing.md) | 测试文档 - 测试套件、运行方法、覆盖范围 |

---

## 项目结构

```
dspyGuardrails/
│
├── src/dspy_guardrails/          # 核心源代码
│   ├── __init__.py               # 包初始化和导出
│   ├── guardrail.py              # 核心 Guardrail 函数
│   ├── constraints.py            # 声明式约束系统
│   ├── decorators.py             # @Guarded 装饰器
│   ├── module.py                 # GuardedModule 基类
│   ├── llm_guardrail.py          # LLM 驱动的 Guardrail
│   ├── metrics.py                # DSPy 评估指标
│   ├── optimizer.py              # GEPA 自进化优化器
│   │
│   ├── mcp/                      # MCP 安全模块
│   │   ├── core.py               # 核心框架和配置
│   │   ├── input_guards.py       # 输入防护
│   │   ├── output_guards.py      # 输出过滤
│   │   ├── policies.py           # 访问策略
│   │   ├── auditor.py            # 安全审计
│   │   └── wrapper.py            # 便捷封装
│   │
│   └── redteam/                  # 红队攻击框架
│       ├── attackers.py          # 攻击生成器
│       ├── evolution.py          # 进化引擎
│       ├── multi_turn.py         # 多轮攻击
│       ├── mcp_attacks.py        # MCP 专用攻击
│       ├── benchmarks.py         # 标准数据集
│       ├── evaluator.py          # 评估器
│       ├── patterns.py           # 攻击模式库
│       └── payload_validator.py  # 载荷验证
│
├── tests/                        # 测试套件
│   ├── test_guardrails.py        # 核心功能测试
│   ├── test_dspy_integration.py  # DSPy 集成测试
│   ├── test_llm_guardrail.py     # LLM 检测测试
│   ├── test_redteam.py           # Red Team 模块测试
│   └── security/                 # 安全测试框架
│       ├── evaluators/           # 评估器
│       ├── datasets/             # 测试数据集
│       ├── targets/              # 测试目标
│       └── runner.py             # 测试运行器
│
├── examples/                     # 示例代码
│   ├── basic_usage.py            # 基础用法
│   ├── dspy_integration_demo.py  # DSPy 集成示例
│   ├── llama_guard_demo.py       # LLM 检测示例
│   └── moonshot_setup.py         # Moonshot API 配置
│
├── docs/                         # 文档目录
│
├── README.md                     # 项目说明
├── CHANGELOG.md                  # 版本历史
├── LICENSE                       # MIT 许可证
└── pyproject.toml                # 项目配置
```

---

## 模块概览

### 1. 核心层 (Core Layer)

提供基础的安全检测能力。

| 模块 | 文件 | 功能 |
|------|------|------|
| **Guardrail 函数** | `guardrail.py` | 模式匹配检测（注入、PII、毒性） |
| **约束系统** | `constraints.py` | 声明式输入/输出约束 |
| **装饰器** | `decorators.py` | `@Guarded` 类装饰器 |
| **模块基类** | `module.py` | `GuardedModule` DSPy 模块包装 |
| **LLM 检测** | `llm_guardrail.py` | 基于 LLM 的高精度检测 |
| **评估指标** | `metrics.py` | DSPy 原生指标支持 |
| **自进化优化** | `optimizer.py` | GEPA 自动优化 |

### 2. MCP 安全层

完整的 Model Context Protocol 安全框架。

| 模块 | 文件 | 功能 |
|------|------|------|
| **核心框架** | `mcp/core.py` | `MCPGuardrail`、威胁分类、配置 |
| **输入防护** | `mcp/input_guards.py` | 注入检测、路径遍历、命令注入 |
| **输出过滤** | `mcp/output_guards.py` | 间接注入、PII 过滤、大小限制 |
| **访问策略** | `mcp/policies.py` | 速率限制、访问控制、确认要求 |
| **安全审计** | `mcp/auditor.py` | 事件日志、威胁分类、审计追踪 |
| **便捷封装** | `mcp/wrapper.py` | `SecureMCPServer`、`@secure_tool` |

### 3. 红队攻击层

自动化安全测试和攻击模拟。

| 模块 | 文件 | 功能 |
|------|------|------|
| **攻击生成器** | `redteam/attackers.py` | Prompt 注入、越狱、绕过攻击 |
| **进化引擎** | `redteam/evolution.py` | 遗传算法、`dspy.Refine` 自进化 |
| **多轮攻击** | `redteam/multi_turn.py` | 5 种渐进策略、对话管理 |
| **MCP 攻击** | `redteam/mcp_attacks.py` | 工具投毒、遮蔽、地毯式攻击 |
| **基准数据集** | `redteam/benchmarks.py` | HarmBench、AdvBench、JailbreakBench |
| **评估器** | `redteam/evaluator.py` | 漏洞报告、成功率统计 |

---

## 快速开始

### 安装

```bash
# 基础安装
pip install -e .

# 完整安装（所有功能）
pip install -e ".[all]"

# 按需安装
pip install -e ".[pii]"        # PII 检测
pip install -e ".[toxicity]"   # 毒性检测
pip install -e ".[dev]"        # 开发工具
```

### 基础使用

```python
from dspy_guardrails import guardrail

# 布尔检查
guardrail.no_injection(text)   # True = 安全
guardrail.no_pii(text)         # True = 无 PII
guardrail.no_toxicity(text)    # True = 无毒性
guardrail.safe(text)           # 综合检查

# 分数检查 (0.0 = 安全, 1.0 = 危险)
guardrail.injection_score(text)
guardrail.toxicity(text)
guardrail.mcp_security_score(text)
```

### DSPy 集成

```python
import dspy
from dspy_guardrails import guardrail, Guarded

# 使用 dspy.Assert
class SafeQA(dspy.Module):
    def forward(self, question):
        dspy.Assert(guardrail.no_injection(question), "Injection detected")
        result = self.generate(question=question)
        dspy.Assert(guardrail.no_toxicity(result.answer), "Toxic output")
        return result

# 使用 @Guarded 装饰器
@Guarded(
    input_checks=["no_injection", "no_pii"],
    output_checks=["no_toxicity"],
)
class DecoratedQA(dspy.Module):
    def forward(self, question):
        return self.generate(question=question)
```

---

## 依赖关系

```
核心依赖:
├── dspy-ai>=2.6.0         # DSPy 框架
├── pydantic>=2.0.0        # 数据验证
└── python-dotenv>=1.0.0   # 环境配置

可选依赖:
├── [pii]                  # Presidio, spaCy
├── [toxicity]             # Detoxify, PyTorch
├── [hallucination]        # sentence-transformers
└── [all]                  # 所有可选依赖
```

---

## 环境要求

- Python 3.10+
- DSPy 2.6.0+
- 支持的 LLM 提供商：OpenAI、Anthropic、Moonshot

---

## 相关资源

- **设计文档**: `docs/MCP_SECURITY_DESIGN.md`, `docs/REDTEAM_DESIGN.md`
- **基准结果**: `docs/MCP_BENCHMARK_RESULTS.md`

---

## 统计信息

| 指标 | 数量 |
|------|------|
| 核心源文件 | 10+ |
| 测试文件 | 15+ |
| 示例文件 | 4 |
| 攻击模式 | 50+ |
| 威胁类别 | 10+ |
