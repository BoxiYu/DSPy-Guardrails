"""
测试 Platform 的目标解析功能
"""

import pytest
from dspy_guardrails.platform import SecurityPlatform
from dspy_guardrails.platform.targets import (
    GuardrailTarget,
    HTTPTarget,
    DSPyModuleTarget,
)
from dspy_guardrails import guardrail


class TestTargetResolution:
    """测试目标解析"""

    def test_string_http_target(self):
        """测试 HTTP 字符串目标"""
        platform = SecurityPlatform("http://localhost:8000/chat")

        assert isinstance(platform.target, HTTPTarget)
        assert platform.target._base_url == "http://localhost:8000/chat"

    def test_string_https_target(self):
        """测试 HTTPS 字符串目标"""
        platform = SecurityPlatform("https://api.example.com/v1/chat")

        assert isinstance(platform.target, HTTPTarget)
        assert platform.target._base_url == "https://api.example.com/v1/chat"

    def test_string_guardrail_target(self):
        """测试 Guardrail 字符串目标"""
        platform = SecurityPlatform("guardrail:no_injection")

        assert isinstance(platform.target, GuardrailTarget)
        assert platform.target.name == "no_injection"
        assert callable(platform.target._guardrail_fn)

    def test_string_multiple_guardrail_functions(self):
        """测试多个 Guardrail 函数"""
        test_cases = [
            "guardrail:no_injection",
            "guardrail:no_toxicity",
            "guardrail:no_pii",
            "guardrail:safe",
        ]

        for target_str in test_cases:
            platform = SecurityPlatform(target_str)
            assert isinstance(platform.target, GuardrailTarget)
            func_name = target_str.split(":")[1]
            assert platform.target.name == func_name

    def test_dict_http_target(self):
        """测试 HTTP 字典目标"""
        config = {
            "type": "http",
            "url": "http://localhost:8000/chat",
            "headers": {"Authorization": "Bearer token"},
            "timeout": 60,
        }

        platform = SecurityPlatform(config)

        assert isinstance(platform.target, HTTPTarget)
        assert platform.target._base_url == "http://localhost:8000/chat"
        assert platform.target._timeout == 60

    def test_dict_guardrail_target(self):
        """测试 Guardrail 字典目标"""
        config = {
            "type": "guardrail",
            "function": "no_injection",
        }

        platform = SecurityPlatform(config)

        assert isinstance(platform.target, GuardrailTarget)
        assert platform.target.name == "no_injection"

    def test_invalid_string_format(self):
        """测试无效的字符串格式"""
        with pytest.raises(ValueError, match="Invalid target string"):
            SecurityPlatform("invalid_target_format")

    def test_invalid_dict_no_type(self):
        """测试无 type 字段的字典"""
        with pytest.raises(ValueError, match="must include 'type' field"):
            SecurityPlatform({"url": "http://localhost"})

    def test_invalid_dict_unknown_type(self):
        """测试未知的目标类型"""
        with pytest.raises(ValueError, match="Unknown target type"):
            SecurityPlatform({"type": "unknown_type"})

    def test_http_dict_missing_url(self):
        """测试缺少 URL 的 HTTP 配置"""
        with pytest.raises(ValueError, match="requires 'base_url' or 'url' field"):
            SecurityPlatform({"type": "http"})

    def test_guardrail_dict_missing_function(self):
        """测试缺少 function 的 Guardrail 配置"""
        with pytest.raises(ValueError, match="requires 'function' field"):
            SecurityPlatform({"type": "guardrail"})

    def test_unified_target_passthrough(self):
        """测试直接传入 UnifiedTarget 实例"""
        target = GuardrailTarget(guardrail_fn=guardrail.no_injection, name="test")
        platform = SecurityPlatform(target)

        assert platform.target is target

    def test_fluent_api_with_string_target(self):
        """测试流式 API 与字符串目标的组合"""
        platform = (
            SecurityPlatform("guardrail:no_injection")
            .with_attacks("injection", "jailbreak")
            .with_reports("json")
        )

        assert isinstance(platform.target, GuardrailTarget)
        assert platform.config.attacks == ["injection", "jailbreak"]
        assert platform.config.report.formats == ["json"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
