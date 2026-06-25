# dspyGuardrails 框架适配指南

**适配其他 LLM 和 Agentic 框架**

---

## 📊 当前适配能力总览

### ✅ 已支持的框架/模式

| 框架/模式 | 适配方式 | 实现文件 | 难度 |
|-----------|----------|----------|------|
| **DSPy** | 原生集成 | `guardrail.py`, `decorators.py` | ⭐ (内置) |
| **HTTP Agent** | HTTP Target | `platform/targets/http.py` | ⭐⭐ (简单) |
| **MCP Server** | MCP Target | `platform/targets/mcp.py`, `mcp/` | ⭐⭐⭐ (中等) |
| **OpenAI Agents SDK** | 已测试 | `../openai-cs-agents-demo/` | ⭐⭐ (简单) |
| **LitingAgent** | 已测试 | `testing/mcp/liting_agent_adapter.py` | ⭐⭐ (简单) |
| **任意 Python 函数** | Guardrail Target | `platform/targets/guardrail.py` | ⭐ (简单) |

---

## 🎯 三种适配层级

### 层级 1: 函数级适配 (最简单) ⭐

**适用场景**: 你有一个返回文本的函数/模型

**只需要**: 直接使用核心 guardrail 函数

```python
from dspy_guardrails import guardrail

# 任意 LLM 框架
def my_llm_call(prompt: str) -> str:
    # 你的 LLM 调用 (OpenAI, Anthropic, Cohere, etc.)
    response = your_llm.generate(prompt)
    return response

# 直接应用 guardrails
user_input = "..."
if not guardrail.no_injection(user_input):
    raise ValueError("Injection detected!")

response = my_llm_call(user_input)

if not guardrail.no_toxicity(response):
    response = "I cannot provide that information."
```

**优点**:
- ✅ 零适配成本
- ✅ 适用于任何返回字符串的函数
- ✅ 支持所有 guardrail 功能

**限制**:
- ⚠️ 需要手动调用
- ⚠️ 无法使用红队测试框架

---

### 层级 2: UnifiedTarget 适配 (推荐) ⭐⭐⭐

**适用场景**: 你想使用完整的红队测试和安全平台

**需要**: 实现 `UnifiedTarget` 接口

#### 步骤 1: 创建 Target 适配器

```python
from dspy_guardrails.platform.targets import UnifiedTarget, TargetResponse, TargetType, TargetCapability

class MyFrameworkTarget(UnifiedTarget):
    """你的框架适配器"""

    def __init__(self, model_config: dict):
        self.target_type = TargetType.HTTP_AGENT  # 或其他类型
        self.capabilities = [
            TargetCapability.SINGLE_TURN,
            TargetCapability.MULTI_TURN,
        ]

        # 初始化你的模型/客户端
        self.client = YourFramework(**model_config)
        self.conversation_history = []

    def invoke(self, prompt: str) -> TargetResponse:
        """单轮调用"""
        try:
            # 调用你的框架
            response = self.client.chat(prompt)

            # 转换为统一格式
            return TargetResponse(
                response=response.text,
                was_blocked=response.blocked if hasattr(response, 'blocked') else False,
                latency_ms=response.latency if hasattr(response, 'latency') else 0.0,
                metadata={"model": response.model}
            )
        except Exception as e:
            return TargetResponse(
                response="",
                was_blocked=True,
                block_reason=str(e)
            )

    def invoke_multi_turn(self, messages: List[Dict[str, str]]) -> TargetResponse:
        """多轮调用"""
        # 将消息格式转换为你的框架格式
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in messages
        ]

        response = self.client.chat_multi_turn(formatted_messages)

        return TargetResponse(
            response=response.text,
            was_blocked=False,
            metadata={"turns": len(messages)}
        )

    def reset_session(self) -> None:
        """重置会话"""
        self.conversation_history = []
```

#### 步骤 2: 使用安全平台测试

```python
from dspy_guardrails import SecurityPlatform

# 创建目标
target = MyFrameworkTarget(model_config={
    "api_key": "...",
    "model": "your-model"
})

# 使用统一平台进行安全测试
platform = (
    SecurityPlatform(target)
    .with_attacks("injection", "jailbreak", "crescendo")
    .with_scanners("quick_scan")
    .with_reports("html", "sarif")
)

results = platform.run_all()
```

**优点**:
- ✅ 使用全部红队测试功能
- ✅ Crescendo/Hydra 攻击器
- ✅ 自动生成 SARIF/HTML 报告
- ✅ CLI 工具支持

---

### 层级 3: 深度集成 (可选) ⭐⭐⭐⭐⭐

**适用场景**: 你想要框架原生的 guardrails 支持

**需要**: 在框架中嵌入 guardrails

```python
# 示例: LangChain 集成
from langchain.callbacks.base import BaseCallbackHandler
from dspy_guardrails import guardrail

class GuardrailCallback(BaseCallbackHandler):
    """LangChain Callback Handler"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        """检查输入"""
        for prompt in prompts:
            if not guardrail.safe(prompt):
                raise ValueError("Unsafe input detected")

    def on_llm_end(self, response, **kwargs):
        """检查输出"""
        for generation in response.generations:
            text = generation[0].text
            if not guardrail.safe(text):
                generation[0].text = "[Response blocked due to safety concerns]"

# 使用
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(callbacks=[GuardrailCallback()])
```

```python
# 示例: LlamaIndex 集成
from llama_index.core.callbacks import BaseCallbackHandler
from dspy_guardrails import guardrail

class GuardrailHandler(BaseCallbackHandler):
    def on_event_start(self, event_type, payload=None, **kwargs):
        if event_type == "llm" and payload:
            prompt = payload.get("prompt", "")
            if not guardrail.no_injection(prompt):
                raise ValueError("Injection detected")
```

---

## 🔧 实战案例

### 案例 1: Anthropic Claude (简单)

```python
from anthropic import Anthropic
from dspy_guardrails import guardrail

client = Anthropic(api_key="...")

def safe_chat(prompt: str) -> str:
    # 输入检查
    if not guardrail.safe(prompt):
        return "I cannot process this request."

    # 调用 Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    result = response.content[0].text

    # 输出检查
    if not guardrail.no_toxicity(result):
        return "I apologize, I cannot provide that response."

    return result
```

### 案例 2: LangChain Agent (UnifiedTarget)

```python
from dspy_guardrails.platform.targets import UnifiedTarget, TargetResponse
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class LangChainTarget(UnifiedTarget):
    def __init__(self, agent_executor: AgentExecutor):
        self.target_type = TargetType.HTTP_AGENT
        self.capabilities = [TargetCapability.SINGLE_TURN, TargetCapability.TOOL_USE]
        self.agent = agent_executor

    def invoke(self, prompt: str) -> TargetResponse:
        result = self.agent.invoke({"input": prompt})
        return TargetResponse(
            response=result["output"],
            tool_calls=[{"name": t.tool, "args": t.tool_input} for t in result.get("intermediate_steps", [])]
        )

    def invoke_multi_turn(self, messages):
        # LangChain agent 自动处理多轮
        last_msg = messages[-1]["content"]
        return self.invoke(last_msg)

    def reset_session(self):
        pass

# 使用
llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

target = LangChainTarget(agent_executor)

# 红队测试
from dspy_guardrails import CrescendoAttacker

attacker = CrescendoAttacker(max_turns=10)
result = attacker.attack(
    target_llm=lambda p: target.invoke(p).response,
    target_behavior="bypass safety filters"
)
```

### 案例 3: AutoGen (多 Agent 系统)

```python
from autogen import ConversableAgent
from dspy_guardrails.platform.targets import UnifiedTarget, TargetResponse

class AutoGenTarget(UnifiedTarget):
    def __init__(self, assistant: ConversableAgent, user_proxy):
        self.target_type = TargetType.HTTP_AGENT
        self.capabilities = [TargetCapability.MULTI_TURN, TargetCapability.TOOL_USE]
        self.assistant = assistant
        self.user_proxy = user_proxy

    def invoke(self, prompt: str) -> TargetResponse:
        # 启动对话
        self.user_proxy.initiate_chat(
            self.assistant,
            message=prompt,
            max_turns=1
        )

        # 获取最后一条助手消息
        last_msg = self.assistant.last_message()

        return TargetResponse(
            response=last_msg["content"],
            metadata={"agent": "assistant"}
        )

    def invoke_multi_turn(self, messages):
        # 重置并运行多轮对话
        for msg in messages:
            if msg["role"] == "user":
                self.user_proxy.send(msg["content"], self.assistant)

        return TargetResponse(
            response=self.assistant.last_message()["content"]
        )

    def reset_session(self):
        self.assistant.clear_history()
        self.user_proxy.clear_history()
```

### 案例 4: CrewAI (任务编排)

```python
from crewai import Agent, Task, Crew
from dspy_guardrails import guardrail

# 为每个 Agent 添加 guardrails
class SafeAgent(Agent):
    def execute_task(self, task):
        # 输入检查
        if not guardrail.no_injection(task.description):
            raise ValueError("Unsafe task detected")

        # 执行原始任务
        result = super().execute_task(task)

        # 输出检查
        if not guardrail.safe(result):
            return "[Blocked] Task output violated safety policies"

        return result

# 使用
researcher = SafeAgent(
    role='Security Researcher',
    goal='Analyze security vulnerabilities',
    backstory='...',
    verbose=True
)
```

---

## 📋 适配 Checklist

### 基础适配 (所有框架必需)

- [ ] 确定你的框架类型 (单轮/多轮/Agent)
- [ ] 决定适配层级 (函数级/UnifiedTarget/深度集成)
- [ ] 测试基础 guardrail 函数是否工作

### UnifiedTarget 适配 (推荐)

- [ ] 实现 `invoke()` 方法
- [ ] 实现 `invoke_multi_turn()` 方法 (如果支持)
- [ ] 实现 `reset_session()` 方法
- [ ] 设置正确的 `target_type` 和 `capabilities`
- [ ] 转换响应格式为 `TargetResponse`
- [ ] 处理异常和错误情况

### 测试验证

- [ ] 单轮调用测试
- [ ] 多轮调用测试 (如果支持)
- [ ] 红队攻击测试 (Crescendo/Hydra)
- [ ] 边界情况测试 (超时、错误、异常)

---

## 🎯 支持的 LLM 提供商

### 开箱即用 (只需函数级适配)

| 提供商 | API 库 | 难度 | 示例 |
|--------|--------|------|------|
| **OpenAI** | `openai` | ⭐ | `client.chat.completions.create()` |
| **Anthropic** | `anthropic` | ⭐ | `client.messages.create()` |
| **Google Gemini** | `google-generativeai` | ⭐ | `model.generate_content()` |
| **Cohere** | `cohere` | ⭐ | `co.chat()` |
| **Mistral** | `mistralai` | ⭐ | `client.chat()` |
| **Moonshot (Kimi)** | OpenAI 兼容 | ⭐ | 已测试 ✓ |
| **DeepSeek** | OpenAI 兼容 | ⭐ | 兼容 |
| **智谱 (GLM)** | `zhipuai` | ⭐ | `client.chat.completions.create()` |
| **百川** | `baidubce` | ⭐⭐ | 需要转换格式 |

### 需要 UnifiedTarget 适配

| 框架 | 类型 | 难度 | 推荐方案 |
|------|------|------|----------|
| **LangChain** | Agent 框架 | ⭐⭐ | Callback Handler |
| **LlamaIndex** | RAG 框架 | ⭐⭐ | Callback Handler |
| **AutoGen** | 多 Agent | ⭐⭐⭐ | UnifiedTarget |
| **CrewAI** | 任务编排 | ⭐⭐⭐ | Agent 继承 |
| **OpenAI Agents SDK** | 原生 Agent | ⭐⭐ | 已有示例 ✓ |
| **Amazon Bedrock** | 托管服务 | ⭐⭐ | HTTP Target |
| **Azure OpenAI** | 托管服务 | ⭐ | OpenAI 兼容 |

---

## 🚀 快速开始模板

### 模板 1: 函数级适配 (1 分钟)

```python
from dspy_guardrails import guardrail

def your_llm_function(prompt: str) -> str:
    # Step 1: 输入检查
    if not guardrail.safe(prompt):
        return "Request blocked for safety reasons"

    # Step 2: 调用你的 LLM
    response = your_framework.generate(prompt)

    # Step 3: 输出检查
    if not guardrail.safe(response):
        return "Response blocked for safety reasons"

    return response
```

### 模板 2: UnifiedTarget 适配 (10 分钟)

```python
from dspy_guardrails.platform.targets import UnifiedTarget, TargetResponse, TargetType, TargetCapability
from typing import List, Dict

class MyTarget(UnifiedTarget):
    def __init__(self, **config):
        self.target_type = TargetType.HTTP_AGENT
        self.capabilities = [TargetCapability.SINGLE_TURN]
        # 初始化你的客户端
        self.client = YourClient(**config)

    def invoke(self, prompt: str) -> TargetResponse:
        response = self.client.call(prompt)
        return TargetResponse(response=response)

    def invoke_multi_turn(self, messages: List[Dict[str, str]]) -> TargetResponse:
        # 如果不支持多轮，只处理最后一条
        return self.invoke(messages[-1]["content"])

    def reset_session(self) -> None:
        pass

# 使用
from dspy_guardrails import SecurityPlatform

target = MyTarget(api_key="...")
platform = SecurityPlatform(target).with_attacks("injection", "jailbreak")
results = platform.run_all()
```

---

## 💡 最佳实践

### 1. 选择合适的适配层级

```
简单 LLM 调用          → 函数级适配
需要红队测试           → UnifiedTarget 适配
框架原生集成           → 深度集成
```

### 2. 性能优化

```python
# 对于高频调用，使用批量检查
texts = [msg1, msg2, msg3, ...]
results = [guardrail.safe(t) for t in texts]  # 可以并行
```

### 3. 错误处理

```python
def invoke(self, prompt: str) -> TargetResponse:
    try:
        response = self.client.call(prompt)
        return TargetResponse(response=response)
    except TimeoutError:
        return TargetResponse(response="", was_blocked=True, block_reason="timeout")
    except Exception as e:
        return TargetResponse(response="", was_blocked=True, block_reason=str(e))
```

### 4. 日志和监控

```python
import logging

class MyTarget(UnifiedTarget):
    def invoke(self, prompt: str) -> TargetResponse:
        logging.info(f"Invoking target with prompt length: {len(prompt)}")
        response = self.client.call(prompt)
        logging.info(f"Response length: {len(response)}")
        return TargetResponse(response=response)
```

---

## 📚 参考资源

- **完整示例**: `../openai-cs-agents-demo/` - OpenAI Agents SDK 集成
- **测试适配器**: `tests/security/targets/` - 多个适配器实现
- **UnifiedTarget 协议**: `src/dspy_guardrails/platform/targets/protocol.py`
- **HTTP Target 实现**: `src/dspy_guardrails/platform/targets/http.py`

---

## 🤝 社区贡献

欢迎贡献新的框架适配器！

**提交适配器**:
1. Fork 项目
2. 创建适配器: `src/dspy_guardrails/platform/targets/your_framework.py`
3. 添加测试: `tests/platform/test_your_framework_target.py`
4. 提交 PR

---

## ❓ 常见问题

**Q: 我的框架是异步的怎么办？**
A: 可以在 `invoke()` 中使用 `asyncio.run()`，或者创建异步版本的 UnifiedTarget

**Q: 我的框架不返回文本，而是结构化数据？**
A: 在 `invoke()` 中将结构化数据转换为文本，或使用 `metadata` 字段存储原始数据

**Q: 适配后性能会下降吗？**
A: 函数级适配几乎零开销 (<1ms)，UnifiedTarget 适配开销也很小 (<5ms)

**Q: 能否适配本地模型 (Ollama, LM Studio)?**
A: 可以！只要提供 HTTP API 或 Python 接口即可适配

---

## 🎉 总结

dspyGuardrails **已经可以方便地适配几乎所有 LLM 和 Agentic 框架**：

- ✅ **零适配成本**: 函数级使用 (1 行代码)
- ✅ **快速适配**: UnifiedTarget 实现 (< 50 行代码)
- ✅ **深度集成**: Callback/Handler 机制 (可选)
- ✅ **已验证**: OpenAI Agents, LitingAgent, HTTP API

**下一步**: 选择你的框架，按照本指南 10 分钟完成适配！🚀
