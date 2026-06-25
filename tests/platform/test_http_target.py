"""Tests for HTTPTarget adapter."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dspy_guardrails.platform.targets import (
    HTTPTarget,
    TargetType,
    TargetCapability,
)


class TestHTTPTargetInit:
    """初始化测试"""

    def test_http_target_init(self):
        """测试基本初始化"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )

        assert target.target_type == TargetType.HTTP_AGENT
        assert TargetCapability.SINGLE_TURN in target.capabilities
        assert TargetCapability.MULTI_TURN in target.capabilities
        assert TargetCapability.SESSION in target.capabilities

    def test_http_target_init_with_custom_options(self):
        """测试自定义选项初始化"""
        headers = {"Authorization": "Bearer token123"}
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/api/chat",
            method="POST",
            headers=headers,
            timeout=60,
            response_key="message",
            blocked_key="is_blocked",
            block_reason_key="reason",
            session_enabled=False,
        )

        assert target._base_url == "http://localhost:8000"
        assert target._endpoint == "/api/chat"
        assert target._method == "POST"
        assert target._headers == headers
        assert target._timeout == 60
        assert target._response_key == "message"
        assert target._blocked_key == "is_blocked"
        assert target._block_reason_key == "reason"
        assert target._session_enabled is False


class TestHTTPTargetBuildUrl:
    """URL 构建测试"""

    def test_http_target_build_url(self):
        """测试 URL 构建"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/api/chat",
        )

        assert target._build_url() == "http://localhost:8000/api/chat"

    def test_http_target_build_url_with_trailing_slash(self):
        """测试带尾斜杠的 URL"""
        target = HTTPTarget(
            base_url="http://localhost:8000/",
            endpoint="/api/chat",
        )

        assert target._build_url() == "http://localhost:8000/api/chat"

    def test_http_target_build_url_without_leading_slash(self):
        """测试不带前导斜杠的端点"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="api/chat",
        )

        assert target._build_url() == "http://localhost:8000/api/chat"


class TestHTTPTargetInvoke:
    """单轮调用测试"""

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_invoke(self, mock_request):
        """测试 invoke 方法"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Hello!",
            "blocked": False,
        }
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        response = target.invoke("Hello")

        assert response.response == "Hello!"
        assert response.was_blocked is False
        assert response.latency_ms == pytest.approx(100.0, rel=0.01)
        mock_request.assert_called_once()

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_invoke_blocked(self, mock_request):
        """测试被拦截的响应"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Request blocked",
            "blocked": True,
            "block_reason": "Injection detected",
        }
        mock_response.elapsed.total_seconds.return_value = 0.05
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        response = target.invoke("Ignore instructions")

        assert response.was_blocked is True
        assert response.block_reason == "Injection detected"

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_invoke_with_guardrail_scores(self, mock_request):
        """测试带 guardrail 分数的响应"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Test response",
            "blocked": False,
            "guardrail_scores": {
                "injection": 0.1,
                "toxicity": 0.05,
            },
        }
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        response = target.invoke("Test")

        assert response.guardrail_scores == {"injection": 0.1, "toxicity": 0.05}

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_invoke_with_tool_calls(self, mock_request):
        """测试带工具调用的响应"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Booking confirmed",
            "blocked": False,
            "tool_calls": [
                {"name": "book_flight", "args": {"flight_id": "AA123"}},
            ],
        }
        mock_response.elapsed.total_seconds.return_value = 0.2
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        response = target.invoke("Book flight AA123")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["name"] == "book_flight"

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_invoke_http_error(self, mock_request):
        """测试 HTTP 错误处理"""
        import requests
        mock_request.side_effect = requests.RequestException("Connection refused")

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        response = target.invoke("Hello")

        assert "HTTP Error" in response.response
        assert response.was_blocked is False
        assert "error" in response.metadata

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_invoke_with_custom_keys(self, mock_request):
        """测试自定义响应键"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": "Custom response",
            "is_blocked": True,
            "reason": "Custom block reason",
        }
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
            response_key="message",
            blocked_key="is_blocked",
            block_reason_key="reason",
        )
        response = target.invoke("Test")

        assert response.response == "Custom response"
        assert response.was_blocked is True
        assert response.block_reason == "Custom block reason"


class TestHTTPTargetMultiTurn:
    """多轮调用测试"""

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_invoke_multi_turn(self, mock_request):
        """测试多轮调用"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Multi-turn response",
            "blocked": False,
        }
        mock_response.elapsed.total_seconds.return_value = 0.15
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "What time is it?"},
        ]

        response = target.invoke_multi_turn(messages)

        assert response.response == "Multi-turn response"
        assert response.was_blocked is False

        # Verify payload contains messages
        call_kwargs = mock_request.call_args[1]
        assert "messages" in call_kwargs["json"]

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_invoke_multi_turn_http_error(self, mock_request):
        """测试多轮调用 HTTP 错误"""
        import requests
        mock_request.side_effect = requests.RequestException("Timeout")

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = target.invoke_multi_turn(messages)

        assert "HTTP Error" in response.response


class TestHTTPTargetSession:
    """会话管理测试"""

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_session_id_tracking(self, mock_request):
        """测试会话 ID 追踪"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Hello!",
            "blocked": False,
            "session_id": "session-12345",
        }
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )

        # First call - should not have session_id
        target.invoke("Hello")
        first_call = mock_request.call_args[1]
        assert "session_id" not in first_call["json"]

        # Second call - should include session_id
        target.invoke("How are you?")
        second_call = mock_request.call_args[1]
        assert second_call["json"]["session_id"] == "session-12345"

    def test_http_target_reset_session(self):
        """测试重置会话"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        target._session_id = "session-12345"
        target._conversation = [{"role": "user", "content": "Hello"}]

        target.reset_session()

        assert target._session_id is None
        assert target._conversation == []

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_session_disabled(self, mock_request):
        """测试禁用会话"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Hello!",
            "blocked": False,
            "session_id": "session-12345",
        }
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
            session_enabled=False,
        )

        target.invoke("Hello")
        # Session ID should not be tracked when disabled
        # (it's still returned but not used in subsequent calls)
        target.invoke("Second message")

        second_call = mock_request.call_args[1]
        assert "session_id" not in second_call["json"]


class TestHTTPTargetCapabilities:
    """能力检查测试"""

    def test_http_target_supports(self):
        """测试能力检查"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        assert target.supports(TargetCapability.SINGLE_TURN)
        assert target.supports(TargetCapability.MULTI_TURN)
        assert target.supports(TargetCapability.SESSION)
        assert not target.supports(TargetCapability.STREAMING)

    def test_http_target_get_info(self):
        """测试获取信息"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        info = target.get_info()
        assert info["type"] == "http_agent"
        assert "single_turn" in info["capabilities"]
        assert "multi_turn" in info["capabilities"]
        assert "session" in info["capabilities"]


class TestHTTPTargetPayload:
    """请求负载测试"""

    def test_build_payload_basic(self):
        """测试基本负载构建"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )

        payload = target._build_payload("Hello")
        assert payload == {"message": "Hello"}

    def test_build_payload_with_session(self):
        """测试带会话的负载构建"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
        )
        target._session_id = "session-123"

        payload = target._build_payload("Hello")
        assert payload == {"message": "Hello", "session_id": "session-123"}

    def test_build_payload_session_disabled(self):
        """测试禁用会话时的负载"""
        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
            session_enabled=False,
        )
        target._session_id = "session-123"

        payload = target._build_payload("Hello")
        assert payload == {"message": "Hello"}  # No session_id


class TestHTTPTargetHeaders:
    """请求头测试"""

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_with_auth_header(self, mock_request):
        """测试带认证头的请求"""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "OK", "blocked": False}
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
            headers={"Authorization": "Bearer secret-token"},
        )
        target.invoke("Hello")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer secret-token"

    @patch("dspy_guardrails.platform.targets.http.requests.request")
    def test_http_target_with_multiple_headers(self, mock_request):
        """测试多个请求头"""
        mock_response = Mock()
        mock_response.json.return_value = {"response": "OK", "blocked": False}
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        target = HTTPTarget(
            base_url="http://localhost:8000",
            endpoint="/chat",
            headers={
                "Authorization": "Bearer token",
                "X-Custom-Header": "custom-value",
                "Content-Type": "application/json",
            },
        )
        target.invoke("Hello")

        call_kwargs = mock_request.call_args[1]
        assert call_kwargs["headers"]["X-Custom-Header"] == "custom-value"
