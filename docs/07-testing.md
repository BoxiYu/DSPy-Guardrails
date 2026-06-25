# 测试文档

> 文件位置: `tests/`

本文档详细讲解 dspy-guardrails 的测试套件结构和使用方法。

---

## 目录

1. [概述](#1-概述)
2. [测试文件列表](#2-测试文件列表)
3. [运行测试](#3-运行测试)
4. [测试覆盖范围](#4-测试覆盖范围)
5. [编写新测试](#5-编写新测试)

---

## 1. 概述

测试套件覆盖以下功能：

| 类别 | 文件 | 说明 |
|------|------|------|
| 核心 Guardrail | `test_guardrails.py` | 规则检测功能 |
| DSPy 集成 | `test_dspy_integration.py` | DSPy 模块集成 |
| LLM Guardrail | `test_llm_guardrail.py` | LLM 检测功能 |
| Red Team | `test_redteam.py` | 渗透测试框架 |
| 安全测试 | `tests/security/` | 完整安全评估 |

---

## 2. 测试文件列表

### 2.1 核心测试

```
tests/
├── __init__.py
├── test_guardrails.py        # 核心检测功能
├── test_dspy_integration.py  # DSPy 集成
├── test_llm_guardrail.py     # LLM 检测
├── test_redteam.py           # Red Team 框架
└── security/                  # 安全测试框架
    ├── cli.py
    ├── test_framework.py
    ├── datasets/
    ├── evaluators/
    ├── targets/
    └── reports/
```

### 2.2 测试详情

#### `test_guardrails.py` - 核心检测测试

```python
# 测试类
TestPromptInjection      # 注入检测
TestPIIDetection         # PII 检测
TestToxicityDetection    # 毒性检测
TestMCPSecurity          # MCP 安全
TestCombinedChecks       # 综合检查
TestMCPContextChecks     # MCP 上下文检查
```

#### `test_dspy_integration.py` - DSPy 集成测试

```python
# 测试类
TestDSPyAssert           # dspy.Assert 集成
TestGuardedDecorator     # @Guarded 装饰器
TestGuardedModule        # GuardedModule 基类
TestSafeModule           # SafeModule 预配置
TestQualityModule        # QualityModule
TestConstraints          # 约束系统
TestGuardrailModulePattern  # 使用模式
```

#### `test_llm_guardrail.py` - LLM 检测测试

```python
# 测试类
TestLLMGuardrailBasic     # 基础功能
TestHybridGuardrailBasic  # 混合检测基础
TestLLMGuardrailWithLLM   # 需要 LLM API
TestHybridGuardrailWithLLM # 需要 LLM API
TestRuleVsLLMComparison   # 规则 vs LLM 对比
TestDetectionTypeCoverage # 检测类型覆盖
```

#### `test_redteam.py` - Red Team 测试

```python
# 测试类
TestAttackPatterns           # 攻击模式库
TestPromptInjectionAttacker  # 注入攻击器
TestJailbreakAttacker        # 越狱攻击器
TestGuardrailBypassAttacker  # 绕过攻击器
TestAttackEvolver            # 攻击进化
TestRedTeamEvaluator         # 评估器
TestSeverityClassification   # 严重程度分类
TestMultiTurnAttacker        # 多轮攻击
TestMCPAttacker              # MCP 攻击
TestPayloadValidator         # Payload 验证
TestRedTeamIntegration       # 集成测试
```

---

## 3. 运行测试

### 3.1 使用 pytest

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_guardrails.py -v
pytest tests/test_dspy_integration.py -v
pytest tests/test_redteam.py -v
pytest tests/test_llm_guardrail.py -v

# 运行特定测试类
pytest tests/test_guardrails.py::TestPromptInjection -v

# 运行特定测试方法
pytest tests/test_guardrails.py::TestPromptInjection::test_safe_inputs -v

# 显示详细输出
pytest tests/ -v --tb=short

# 只运行不需要 LLM 的测试
pytest tests/ -v -m "not requires_llm"
```

### 3.2 直接运行

```bash
# 运行核心测试
python tests/test_guardrails.py

# 运行 DSPy 集成测试
python tests/test_dspy_integration.py

# 运行 Red Team 测试
python tests/test_redteam.py

# 运行 LLM 测试
python tests/test_llm_guardrail.py
```

### 3.3 环境变量

```bash
# 设置 API 密钥以运行 LLM 测试
export OPENAI_API_KEY="sk-..."
# 或
export MOONSHOT_API_KEY="sk-..."
# 或
export ANTHROPIC_API_KEY="sk-..."
```

---

## 4. 测试覆盖范围

### 4.1 核心功能覆盖

| 功能 | 测试文件 | 覆盖状态 |
|------|----------|----------|
| `guardrail.no_injection()` | `test_guardrails.py` | ✅ |
| `guardrail.no_pii()` | `test_guardrails.py` | ✅ |
| `guardrail.no_toxicity()` | `test_guardrails.py` | ✅ |
| `guardrail.no_mcp_attack()` | `test_guardrails.py` | ✅ |
| `guardrail.safe()` | `test_guardrails.py` | ✅ |
| `guardrail.mcp_safe_input()` | `test_guardrails.py` | ✅ |

### 4.2 DSPy 集成覆盖

| 功能 | 测试文件 | 覆盖状态 |
|------|----------|----------|
| `dspy.Assert` 集成 | `test_dspy_integration.py` | ✅ |
| `dspy.Suggest` 集成 | `test_dspy_integration.py` | ✅ |
| `@Guarded` 装饰器 | `test_dspy_integration.py` | ✅ |
| `GuardedModule` | `test_dspy_integration.py` | ✅ |
| `SafeModule` | `test_dspy_integration.py` | ✅ |
| `QualityModule` | `test_dspy_integration.py` | ✅ |
| `Constraint` 系统 | `test_dspy_integration.py` | ✅ |
| `ConstraintSet` | `test_dspy_integration.py` | ✅ |

### 4.3 LLM Guardrail 覆盖

| 功能 | 测试文件 | 覆盖状态 |
|------|----------|----------|
| `LLMGuardrail` 实例化 | `test_llm_guardrail.py` | ✅ |
| `LLMGuardrail.check()` | `test_llm_guardrail.py` | ✅ |
| `HybridGuardrail` 实例化 | `test_llm_guardrail.py` | ✅ |
| `HybridGuardrail.check()` | `test_llm_guardrail.py` | ✅ |
| 检测类型覆盖 | `test_llm_guardrail.py` | ✅ |

### 4.4 Red Team 覆盖

| 功能 | 测试文件 | 覆盖状态 |
|------|----------|----------|
| `PromptInjectionAttacker` | `test_redteam.py` | ✅ |
| `JailbreakAttacker` | `test_redteam.py` | ✅ |
| `GuardrailBypassAttacker` | `test_redteam.py` | ✅ |
| `AttackEvolver` | `test_redteam.py` | ✅ |
| `RedTeamEvaluator` | `test_redteam.py` | ✅ |
| `MultiTurnAttacker` | `test_redteam.py` | ✅ |
| `MCPAttacker` | `test_redteam.py` | ✅ |
| `PayloadValidator` | `test_redteam.py` | ✅ |
| `AttackPatterns` | `test_redteam.py` | ✅ |
| `SeverityClassifier` | `test_redteam.py` | ✅ |

---

## 5. 编写新测试

### 5.1 测试模板

```python
"""
Test module description.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check dependencies
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from dspy_guardrails import guardrail

# Skip markers
requires_dspy = pytest.mark.skipif(
    not DSPY_AVAILABLE,
    reason="dspy not installed"
)

requires_llm = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="No LLM API key"
)


class TestMyFeature:
    """Test my feature."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        assert True

    @requires_dspy
    def test_dspy_integration(self):
        """Test DSPy integration."""
        import dspy
        # Test code

    @requires_llm
    def test_llm_feature(self):
        """Test LLM-dependent feature."""
        # Test code


# Run without pytest
def run_tests():
    test_classes = [TestMyFeature]
    # ... run logic


if __name__ == "__main__":
    run_tests()
```

### 5.2 测试命名规范

```python
# 类名: Test + 功能名
class TestPromptInjection:
    pass

# 方法名: test_ + 具体测试内容
def test_safe_inputs(self):
    pass

def test_detects_injection_attack(self):
    pass

def test_returns_correct_score(self):
    pass
```

### 5.3 断言模式

```python
# 检测应该通过
assert guardrail.no_injection(safe_text), f"False positive: {safe_text}"

# 检测应该拦截
assert not guardrail.no_injection(attack), f"Missed attack: {attack}"

# DSPy Assert 应该失败
with pytest.raises(dspy.AssertionError):
    dspy.Assert(guardrail.no_injection(attack), "Should fail")

# 检查属性
assert hasattr(result, 'confidence')
assert 0.0 <= result.confidence <= 1.0
```

---

## 6. 测试标记

### 6.1 跳过标记

```python
# 需要 DSPy
@pytest.mark.skipif(not DSPY_AVAILABLE, reason="dspy not installed")
def test_dspy_feature():
    pass

# 需要 LLM
@pytest.mark.skipif(not LLM_CONFIGURED, reason="No LLM API key")
def test_llm_feature():
    pass

# 慢速测试
@pytest.mark.slow
def test_slow_operation():
    pass
```

### 6.2 运行特定标记

```bash
# 跳过慢速测试
pytest tests/ -v -m "not slow"

# 只运行 DSPy 测试
pytest tests/ -v -m "requires_dspy"

# 跳过需要 LLM 的测试
pytest tests/ -v -m "not requires_llm"
```

---

## 7. CI/CD 集成

### 7.1 GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest

      - name: Run tests (no LLM)
        run: pytest tests/ -v -m "not requires_llm"

      - name: Run LLM tests
        if: secrets.OPENAI_API_KEY != ''
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/ -v
```

### 7.2 本地 CI 模拟

```bash
# 运行完整测试套件
./scripts/run_tests.sh

# 或手动
pip install -e .
pytest tests/ -v --tb=short
```

---

## 8. 故障排除

### 8.1 常见问题

**问题**: `ModuleNotFoundError: No module named 'dspy'`
```bash
pip install dspy-ai
```

**问题**: `AssertionError: dspy not configured`
```python
import dspy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

**问题**: LLM 测试超时
```bash
# 增加超时时间
pytest tests/ -v --timeout=120
```

### 8.2 调试测试

```bash
# 显示 print 输出
pytest tests/ -v -s

# 显示完整 traceback
pytest tests/ -v --tb=long

# 在第一个失败时停止
pytest tests/ -v -x

# 运行上次失败的测试
pytest tests/ -v --lf
```
