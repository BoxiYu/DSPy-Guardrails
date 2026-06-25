"""
from pathlib import Path
OpenAI CS Agents Demo Target Adapter

Provides interface to test openai-cs-agents-demo via API or module import.
"""

import json
import time
import asyncio
import httpx
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator

from .base import BaseTarget, TargetResponse, ConversationTurn


@dataclass
class GuardrailCheck:
    """Guardrail check result from the API."""

    name: str
    passed: bool
    reasoning: str
    input_text: str


class OpenAICSAgentTarget(BaseTarget):
    """
    Target adapter for openai-cs-agents-demo.

    Supports two modes:
    1. API mode - HTTP calls to /chatkit endpoint
    2. Module mode - Direct import for faster testing

    Example:
        target = OpenAICSAgentTarget(base_url="http://localhost:8000")
        response = target.invoke("What is my flight status?")
        print(response.response)
        print(response.guardrail_status)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        mode: str = "api",
        timeout: float = 30.0,
    ):
        """
        Initialize the target.

        Args:
            base_url: Base URL for the API
            mode: "api" for HTTP calls, "module" for direct import
            timeout: Request timeout in seconds
        """
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.thread_id: Optional[str] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def _ensure_thread(self) -> str:
        """Ensure we have a valid thread ID."""
        if self.thread_id is None:
            client = await self._get_client()
            response = await client.get("/chatkit/state")
            if response.status_code == 200:
                data = response.json()
                self.thread_id = data.get("thread_id")
            else:
                # Create new thread by making initial request
                self.thread_id = None
        return self.thread_id

    async def _invoke_api_async(self, prompt: str) -> TargetResponse:
        """Invoke target via API (async)."""
        client = await self._get_client()
        start_time = time.time()

        # Prepare request - ChatKit requires specific format
        # Note: content type must be "input_text" not "text"
        user_input = {
            "content": [{"type": "input_text", "text": prompt}],
            "attachments": [],
            "inference_options": {},
        }

        if self.thread_id:
            # Add message to existing thread
            request_data = {
                "type": "threads.add_user_message",
                "thread_id": self.thread_id,
                "params": {"input": user_input},
            }
        else:
            # Create new thread with first message
            request_data = {
                "type": "threads.create",
                "params": {"input": user_input},
            }

        # Send message
        response_text = ""
        guardrail_status = {}

        try:
            # POST to /chatkit
            async with client.stream(
                "POST",
                "/chatkit",
                json=request_data,
                headers={"Accept": "text/event-stream"},
            ) as response:
                if response.status_code != 200:
                    return TargetResponse(
                        response=f"Error: HTTP {response.status_code}",
                        guardrail_status={},
                        latency_ms=(time.time() - start_time) * 1000,
                    )

                # Parse SSE stream
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())
                            # Extract response content
                            if "item" in data:
                                item = data["item"]
                                if "content" in item:
                                    for content in item.get("content", []):
                                        if "text" in content:
                                            response_text += content["text"]
                            # Extract thread_id
                            if "thread_id" in data:
                                self.thread_id = data["thread_id"]
                        except json.JSONDecodeError:
                            continue

            # Get guardrail status from state endpoint
            state_response = await client.get(
                "/chatkit/state",
                params={"thread_id": self.thread_id} if self.thread_id else {},
            )
            if state_response.status_code == 200:
                state_data = state_response.json()
                self.thread_id = state_data.get("thread_id", self.thread_id)

                # Parse guardrails
                for g in state_data.get("guardrails", []):
                    guardrail_status[g["name"]] = {
                        "passed": g.get("passed", True),
                        "reasoning": g.get("reasoning", ""),
                        "input": g.get("input", ""),
                    }

        except httpx.RequestError as e:
            return TargetResponse(
                response=f"Request error: {str(e)}",
                guardrail_status={},
                latency_ms=(time.time() - start_time) * 1000,
            )

        latency_ms = (time.time() - start_time) * 1000

        # Record conversation turn
        self.conversation_history.append(
            ConversationTurn(role="user", content=prompt)
        )
        self.conversation_history.append(
            ConversationTurn(
                role="assistant",
                content=response_text,
                guardrail_status=guardrail_status,
            )
        )

        return TargetResponse(
            response=response_text,
            guardrail_status=guardrail_status,
            latency_ms=latency_ms,
            metadata={"thread_id": self.thread_id},
        )

    def invoke(self, prompt: str) -> TargetResponse:
        """
        Invoke the target with a prompt.

        Args:
            prompt: The user message to send

        Returns:
            TargetResponse containing the response and guardrail status
        """
        if self.mode == "api":
            return asyncio.run(self._invoke_api_async(prompt))
        else:
            return self._invoke_module(prompt)

    async def invoke_async(self, prompt: str) -> TargetResponse:
        """Async version of invoke."""
        if self.mode == "api":
            return await self._invoke_api_async(prompt)
        else:
            return self._invoke_module(prompt)

    def _invoke_module(self, prompt: str) -> TargetResponse:
        """
        Invoke target by directly importing modules.

        This is faster but requires the target code to be importable.
        """
        try:
            # Import guardrails directly
            import sys
            sys.path.insert(
                0, str(Path(__file__).resolve().parents[4] / "openai-cs-agents-demo/python-backend")
            )
            from airline.guardrails import relevance_guardrail, jailbreak_guardrail

            start_time = time.time()
            guardrail_status = {}

            # Test relevance guardrail
            # Note: This is a simplified version - actual implementation
            # would need to set up proper context
            # For now, we'll use the API mode for full testing

            latency_ms = (time.time() - start_time) * 1000

            return TargetResponse(
                response="Module mode not fully implemented - use API mode",
                guardrail_status=guardrail_status,
                latency_ms=latency_ms,
            )

        except ImportError as e:
            return TargetResponse(
                response=f"Import error: {str(e)}",
                guardrail_status={},
            )

    def reset_session(self) -> None:
        """Reset the conversation session."""
        self.thread_id = None
        self.clear_history()

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def __del__(self):
        """Cleanup on deletion."""
        if self._client:
            try:
                asyncio.run(self.close())
            except RuntimeError:
                pass


class OpenAICSAgentTargetSync:
    """
    Synchronous wrapper for OpenAICSAgentTarget.

    Simpler interface for scripts that don't use async.
    """

    def __init__(self, **kwargs):
        self._target = OpenAICSAgentTarget(**kwargs)

    def invoke(self, prompt: str) -> TargetResponse:
        """Invoke synchronously."""
        return self._target.invoke(prompt)

    def reset_session(self) -> None:
        """Reset session."""
        self._target.reset_session()

    def get_conversation_history(self) -> List[ConversationTurn]:
        """Get conversation history."""
        return self._target.get_conversation_history()
