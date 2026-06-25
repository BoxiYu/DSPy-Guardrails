# MCP 安全模块文档

> 文件位置: `src/dspy_guardrails/mcp/`

本文档详细讲解 Model Context Protocol (MCP) 安全防护模块。

---

## 目录

1. [概述](#1-概述)
2. [核心框架 (core.py)](#2-核心框架)
3. [输入防护 (input_guards.py)](#3-输入防护)
4. [输出过滤 (output_guards.py)](#4-输出过滤)
5. [访问策略 (policies.py)](#5-访问策略)
6. [安全审计 (auditor.py)](#6-安全审计)
7. [便捷封装 (wrapper.py)](#7-便捷封装)

---

## 1. 概述

MCP 安全模块提供完整的 Model Context Protocol 安全防护框架，包括：

- **输入验证**: 检测工具参数中的恶意内容
- **输出过滤**: 过滤工具输出中的敏感信息
- **策略执行**: 速率限制、访问控制、确认要求
- **安全审计**: 事件日志和威胁追踪

### 威胁模型

| 威胁类别 | 说明 | 严重程度 |
|----------|------|----------|
| `INJECTION` | 直接注入攻击 | 高 |
| `INDIRECT_INJECTION` | 间接注入（通过工具输出） | 高 |
| `PATH_TRAVERSAL` | 路径遍历攻击 | 高 |
| `COMMAND_INJECTION` | 命令注入 | 严重 |
| `DATA_EXFILTRATION` | 数据外泄 | 高 |
| `PRIVILEGE_ESCALATION` | 权限提升 | 高 |
| `RESOURCE_ABUSE` | 资源滥用 | 中 |
| `PII_EXPOSURE` | PII 暴露 | 高 |
| `RUG_PULL` | 工具行为篡改 | 高 |

---

## 2. 核心框架

**文件**: `src/dspy_guardrails/mcp/core.py`

### 2.1 GuardAction 枚举

```python
class GuardAction(Enum):
    ALLOW = "allow"       # 允许通过
    BLOCK = "block"       # 阻止
    MODIFY = "modify"     # 修改后通过
    WARN = "warn"         # 警告但通过
    CONFIRM = "confirm"   # 需要用户确认
    AUDIT = "audit"       # 仅审计记录
```

### 2.2 ThreatCategory 枚举

```python
class ThreatCategory(Enum):
    INJECTION = "injection"
    INDIRECT_INJECTION = "indirect_injection"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_ABUSE = "resource_abuse"
    PII_EXPOSURE = "pii_exposure"
    SENSITIVE_DATA = "sensitive_data"
    DANGEROUS_OPERATION = "dangerous_operation"
    RATE_LIMIT = "rate_limit"
    RUG_PULL = "rug_pull"
```

### 2.3 GuardResult 类

```python
@dataclass
class GuardResult:
    action: GuardAction                    # 执行的动作
    passed: bool                           # 是否通过
    threat_category: Optional[ThreatCategory]  # 威胁类别
    threat_score: float                    # 威胁分数 0-1
    message: str                           # 描述信息
    details: Dict[str, Any]                # 详细信息
    modified_value: Any                    # 修改后的值
    timestamp: datetime                    # 时间戳
```

### 2.4 MCPSecurityConfig 配置

```python
config = MCPSecurityConfig(
    # 阻止阈值
    block_threshold=0.8,

    # 危险工具列表
    dangerous_tools=[
        "execute_command",
        "run_shell",
        "delete_file",
        "write_file",
        "send_email",
        "http_request",
        "database_query",
    ],

    # 只读工具（安全）
    readonly_tools=[
        "read_file",
        "list_directory",
        "search",
    ],

    # 敏感参数名
    sensitive_params=[
        "password",
        "secret",
        "token",
        "api_key",
    ],

    # 速率限制
    rate_limit_enabled=True,
    rate_limit_max_calls=100,
    rate_limit_window_seconds=60,

    # 审计
    audit_enabled=True,
    audit_log_path="/var/log/mcp_security.log",

    # 需要确认的工具
    confirmation_required_tools=["delete_file"],

    # 白名单
    allowed_paths=["/home/user/safe/"],
    allowed_hosts=["api.example.com"],
)
```

### 2.5 ToolCallContext 上下文

```python
context = ToolCallContext(
    tool_name="read_file",
    parameters={"path": "/etc/passwd"},
    caller_id="user123",
    session_id="session456",
    metadata={"source": "api"},
)

# 生成调用哈希
hash_id = context.call_hash  # "a1b2c3d4..."
```

### 2.6 MCPGuardrail 主类

```python
from dspy_guardrails.mcp import MCPGuardrail, MCPSecurityConfig

# 创建 Guardrail
guardrail = MCPGuardrail(config=MCPSecurityConfig())

# 检查工具输入
context = ToolCallContext(
    tool_name="execute_command",
    parameters={"command": "rm -rf /"},
)

result = guardrail.check_input(context)
if not result.passed:
    print(f"阻止: {result.message}")
    # 阻止: Dangerous command detected

# 过滤输出
output = "User email: test@example.com"
filtered_output, result = guardrail.filter_output(output, context)
# filtered_output: "User email: [REDACTED]"

# 装饰器用法
@guardrail.protect
async def my_tool(params):
    return result
```

---

## 3. 输入防护

**文件**: `src/dspy_guardrails/mcp/input_guards.py`

### 3.1 可用防护类型

| 防护类 | 创建方法 | 检测内容 |
|--------|----------|----------|
| `InjectionInputGuard` | `injection_check()` | Prompt 注入 |
| `PathTraversalGuard` | `path_traversal_check()` | 路径遍历 |
| `CommandInjectionGuard` | `command_injection_check()` | 命令注入 |
| `SQLInjectionGuard` | `sql_injection_check()` | SQL 注入 |
| `SensitiveParamGuard` | `sensitive_param_check()` | 敏感参数 |
| `RugPullGuard` | `rug_pull_check()` | Rug Pull 攻击 |
| `PrivilegeEscalationGuard` | `privilege_escalation_check()` | 权限提升 |
| `CustomInputGuard` | `custom(fn, name)` | 自定义检查 |

### 3.2 InjectionInputGuard 详解

检测工具参数中的 Prompt Injection。

**敏感度级别**:
| 级别 | 模式 | 说明 |
|------|------|------|
| `low` | 基础模式 | 仅检测明显的注入 |
| `medium` | + 高级模式 | 检测绕过技术 |
| `high` | + 语义模式 | 检测间接请求 |

**基础模式** (第98-126行):
```python
BASIC_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"you\s+are\s+now\s+DAN",
    r"bypass\s+safety",
    r"enter\s+developer\s+mode",
]
```

**高级模式** (第129-161行):
```python
ADVANCED_PATTERNS = [
    r"\[END\s*OF\s*PROMPT\]",           # 分隔符逃逸
    r"I\s+am\s+an\s+admin",              # 权威声明
    r"\[HIDDEN\]",                        # 隐藏标记
    r"[\u200b-\u200f]",                   # Unicode 混淆
]
```

**使用示例**:
```python
from dspy_guardrails.mcp import ToolInputGuard

# 创建注入防护
guard = ToolInputGuard.injection_check()

# 或指定敏感度
guard = InjectionInputGuard(sensitivity="high")

# 添加自定义模式
guard = InjectionInputGuard(
    additional_patterns=[r"my_custom_pattern"],
    sensitivity="medium"
)

# 检查
result = guard.check(context)
```

### 3.3 PathTraversalGuard

```python
# 检测模式
PATTERNS = [
    r"\.\./",                    # ../
    r"\.\.\\",                   # ..\
    r"%2e%2e[%/\\]",             # URL 编码
    r"/etc/passwd",              # 敏感文件
]

# 使用
guard = ToolInputGuard.path_traversal_check()
result = guard.check(context)
```

### 3.4 CommandInjectionGuard

```python
# 检测模式
PATTERNS = [
    r";\s*\w+",                  # 命令分隔
    r"\|\s*\w+",                 # 管道
    r"&&\s*\w+",                 # 命令链
    r"\$\([^)]+\)",              # 命令替换
    r"`[^`]+`",                  # 反引号执行
]

# 使用
guard = ToolInputGuard.command_injection_check()
```

### 3.5 自定义防护

```python
def my_check(context: ToolCallContext) -> GuardResult:
    if "危险内容" in str(context.parameters):
        return GuardResult(
            action=GuardAction.BLOCK,
            passed=False,
            message="检测到危险内容",
        )
    return GuardResult(action=GuardAction.ALLOW, passed=True)

guard = ToolInputGuard.custom(my_check, name="my_guard")
```

---

## 4. 输出过滤

**文件**: `src/dspy_guardrails/mcp/output_guards.py`

### 4.1 可用过滤器

| 过滤器类 | 创建方法 | 功能 |
|----------|----------|------|
| `IndirectInjectionFilter` | `indirect_injection_check()` | 检测间接注入 |
| `PIIFilter` | `pii_filter()` | 过滤 PII |
| `SensitiveDataFilter` | `sensitive_data_filter()` | 过滤敏感数据 |
| `SizeLimitFilter` | `size_limit(max_size)` | 限制输出大小 |

### 4.2 IndirectInjectionFilter

检测工具输出中的间接注入尝试。

```python
# 检测模式
PATTERNS = [
    r"<!--.*ignore.*-->",          # HTML 注释隐藏
    r"display:\s*none",            # CSS 隐藏
    r"\[SYSTEM\]",                 # 伪系统消息
    r"NEW INSTRUCTION:",           # 指令劫持
]

# 使用
from dspy_guardrails.mcp import ToolOutputGuard

filter = ToolOutputGuard.indirect_injection_check()
result = filter.filter(output, context)
```

### 4.3 PIIFilter

自动脱敏 PII 信息。

```python
filter = ToolOutputGuard.pii_filter()

output = "Contact: john@example.com, Phone: 123-456-7890"
filtered, result = filter.filter(output, context)
# filtered: "Contact: [EMAIL REDACTED], Phone: [PHONE REDACTED]"
```

### 4.4 SizeLimitFilter

```python
filter = ToolOutputGuard.size_limit(max_size=10000)  # 10KB

large_output = "..." * 100000
filtered, result = filter.filter(large_output, context)
# 输出被截断
```

---

## 5. 访问策略

**文件**: `src/dspy_guardrails/mcp/policies.py`

### 5.1 可用策略

| 策略类 | 创建方法 | 功能 |
|--------|----------|------|
| `RateLimitPolicy` | `rate_limit(max, window)` | 速率限制 |
| `DangerousToolPolicy` | `dangerous_tool_check(tools)` | 危险工具检查 |
| `AccessControlPolicy` | `access_control(rules)` | 访问控制 |
| `ConfirmationPolicy` | `confirmation_required(tools)` | 确认要求 |

### 5.2 RateLimitPolicy

```python
from dspy_guardrails.mcp import ToolCallPolicy

policy = ToolCallPolicy.rate_limit(
    max_calls=100,
    window_seconds=60,
)

# 检查
result = policy.evaluate(context)
if result.action == GuardAction.BLOCK:
    print("速率限制已触发")
```

### 5.3 DangerousToolPolicy

```python
policy = ToolCallPolicy.dangerous_tool_check([
    "execute_command",
    "delete_file",
    "send_email",
])

context = ToolCallContext(tool_name="delete_file", ...)
result = policy.evaluate(context)
# result.action == GuardAction.WARN 或 CONFIRM
```

### 5.4 AccessControlPolicy

```python
rules = {
    "admin": ["*"],                    # 管理员可访问所有
    "user": ["read_*", "list_*"],      # 普通用户只能读
    "guest": ["search"],               # 访客只能搜索
}

policy = ToolCallPolicy.access_control(rules)

context = ToolCallContext(
    tool_name="delete_file",
    metadata={"role": "user"},
)
result = policy.evaluate(context)
# result.action == GuardAction.BLOCK
```

### 5.5 ConfirmationPolicy

```python
policy = ToolCallPolicy.confirmation_required([
    "delete_*",
    "send_*",
    "execute_*",
])

result = policy.evaluate(context)
if result.action == GuardAction.CONFIRM:
    # 向用户请求确认
    if user_confirms():
        proceed()
```

---

## 6. 安全审计

**文件**: `src/dspy_guardrails/mcp/auditor.py`

### 6.1 MCPSecurityAuditor

```python
from dspy_guardrails.mcp import MCPSecurityAuditor

auditor = MCPSecurityAuditor(
    log_path="/var/log/mcp_security.log",
    log_level="INFO",
)

# 记录事件
auditor.log_event(
    event_type="tool_blocked",
    context=context,
    result=result,
    details={"reason": "injection_detected"},
)

# 获取审计日志
logs = auditor.get_logs(
    start_time=datetime.now() - timedelta(hours=1),
    event_types=["tool_blocked"],
)
```

### 6.2 威胁级别分类

```python
class ThreatLevel(Enum):
    LOW = "low"           # 低风险
    MEDIUM = "medium"     # 中等风险
    HIGH = "high"         # 高风险
    CRITICAL = "critical" # 严重
```

### 6.3 审计报告

```python
report = auditor.generate_report(
    start_time=start,
    end_time=end,
)

print(f"总事件数: {report.total_events}")
print(f"阻止数: {report.blocked_count}")
print(f"威胁分布: {report.threat_distribution}")
```

---

## 7. 便捷封装

**文件**: `src/dspy_guardrails/mcp/wrapper.py`

### 7.1 SecureMCPServer

```python
from dspy_guardrails.mcp import SecureMCPServer

server = SecureMCPServer(
    config=MCPSecurityConfig(),
    name="my_secure_server",
)

# 注册工具
@server.tool("read_file")
async def read_file(path: str) -> str:
    return open(path).read()

# 启动服务器
await server.start()
```

### 7.2 @secure_tool 装饰器

```python
from dspy_guardrails.mcp import secure_tool

@secure_tool(
    input_checks=["injection", "path_traversal"],
    output_checks=["pii", "size_limit"],
    rate_limit=100,
)
async def my_tool(params: dict) -> str:
    return process(params)
```

### 7.3 工具注册表

```python
from dspy_guardrails.mcp import ToolRegistry

registry = ToolRegistry()

# 注册工具
registry.register(
    name="execute_command",
    handler=execute_fn,
    dangerous=True,
    requires_confirmation=True,
)

# 获取工具
tool = registry.get("execute_command")

# 列出所有工具
all_tools = registry.list_tools()
dangerous = registry.list_dangerous_tools()
```

---

## 完整示例

```python
from dspy_guardrails.mcp import (
    MCPGuardrail,
    MCPSecurityConfig,
    ToolCallContext,
    ToolInputGuard,
    ToolOutputGuard,
    ToolCallPolicy,
    MCPSecurityAuditor,
)

# 1. 配置
config = MCPSecurityConfig(
    block_threshold=0.7,
    dangerous_tools=["execute_command", "delete_file"],
    rate_limit_enabled=True,
    rate_limit_max_calls=50,
    audit_enabled=True,
)

# 2. 创建组件
input_guards = [
    ToolInputGuard.injection_check(),
    ToolInputGuard.path_traversal_check(),
    ToolInputGuard.command_injection_check(),
]

output_guards = [
    ToolOutputGuard.indirect_injection_check(),
    ToolOutputGuard.pii_filter(),
]

policies = [
    ToolCallPolicy.rate_limit(50, 60),
    ToolCallPolicy.dangerous_tool_check(config.dangerous_tools),
]

auditor = MCPSecurityAuditor(log_path="./security.log")

# 3. 创建 Guardrail
guardrail = MCPGuardrail(
    config=config,
    input_guards=input_guards,
    output_guards=output_guards,
    policies=policies,
    auditor=auditor,
)

# 4. 使用
async def handle_tool_call(tool_name: str, params: dict):
    context = ToolCallContext(
        tool_name=tool_name,
        parameters=params,
    )

    # 输入检查
    result = guardrail.check_input(context)
    if not result.passed:
        return {"error": result.message}

    # 执行工具
    output = await execute_tool(tool_name, params)

    # 输出过滤
    filtered_output, result = guardrail.filter_output(output, context)

    return {"result": filtered_output}
```

---

## 与 MCP 攻击模块配合测试

```python
from dspy_guardrails.mcp import MCPGuardrail
from dspy_guardrails.redteam import MCPAttackEvaluator

# 创建 Guardrail
guardrail = MCPGuardrail()

# 创建评估器
evaluator = MCPAttackEvaluator(guardrail=guardrail)

# 评估所有攻击
results = evaluator.evaluate_all_attacks()
print(f"防护率: {results['protection_rate']:.1%}")

# 生成报告
report = evaluator.generate_report()
print(report)
```
