"""SecurityPlatform - Unified security testing entry point.

This module provides the main SecurityPlatform class that serves as the
unified entry point for all security testing operations. It supports:

- Fluent API for configuration
- Multiple target types (guardrails, HTTP agents, MCP servers, DSPy modules)
- Pluggable scanners, attackers, trainers, and reporters
- YAML-based configuration loading

Example:
    >>> from dspy_guardrails.platform import SecurityPlatform
    >>> platform = (
    ...     SecurityPlatform(my_target)
    ...     .with_attacks("injection", "jailbreak")
    ...     .with_scanners("quick_scan")
    ...     .with_reports("html", "json")
    ... )
    >>> results = platform.run_all()
"""

from pathlib import Path
from typing import Any

from .config import PlatformConfig, TrainingConfig
from .plugins import PluginRegistry
from .targets import UnifiedTarget


class SecurityPlatform:
    """统一安全测试平台

    提供安全测试的统一入口，支持流式 API 配置、多种目标类型和插件系统。

    Attributes:
        target: 测试目标（UnifiedTarget 实例）
        config: 平台配置（PlatformConfig 实例）
        registry: 插件注册表（PluginRegistry 实例）

    Example:
        >>> target = MyGuardrail()
        >>> platform = SecurityPlatform(target)
        >>> platform.with_attacks("injection").with_reports("html")
        >>> results = platform.run_all()
    """

    def __init__(
        self,
        target: UnifiedTarget | str | dict[str, Any],
        config: PlatformConfig | None = None,
    ):
        """初始化 SecurityPlatform

        Args:
            target: 测试目标，支持以下类型：
                - UnifiedTarget: 已实例化的目标对象
                - str: 目标 URL（未实现）
                - Dict: 目标配置字典（未实现）
            config: 平台配置，默认使用 PlatformConfig()
        """
        self.target = self._resolve_target(target)
        self.config = config or PlatformConfig()
        self.registry = PluginRegistry()
        self._results: list[Any] = []

    def _resolve_target(
        self, target: UnifiedTarget | str | dict[str, Any]
    ) -> UnifiedTarget:
        """解析目标参数为 UnifiedTarget 实例

        Args:
            target: 原始目标参数

        Returns:
            UnifiedTarget: 解析后的目标实例

        Raises:
            ValueError: 如果目标参数无效
            TypeError: 如果目标类型不受支持
        """
        if isinstance(target, UnifiedTarget):
            return target

        elif isinstance(target, str):
            # String-based target resolution
            return self._resolve_string_target(target)

        elif isinstance(target, dict):
            # Dict-based target resolution
            return self._resolve_dict_target(target)

        else:
            raise TypeError(f"Unsupported target type: {type(target)}")

    def _resolve_string_target(self, target: str) -> UnifiedTarget:
        """从字符串解析目标

        Args:
            target: 目标字符串，支持以下格式：
                - "http://..." 或 "https://..." - HTTP 目标
                - "guardrail:function_name" - Guardrail 函数
                - "module:class.method" - Python 模块路径

        Returns:
            UnifiedTarget: 解析后的目标实例

        Raises:
            ValueError: 如果字符串格式无效
        """
        from .targets import DSPyModuleTarget, GuardrailTarget, HTTPTarget

        # HTTP/HTTPS URL
        if target.startswith(("http://", "https://")):
            return HTTPTarget(base_url=target)

        # Guardrail function: "guardrail:no_injection"
        elif ":" in target:
            parts = target.split(":", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid target format: {target}")

            module_path, func_name = parts

            # Import and get the function
            try:
                if module_path == "guardrail":
                    # Built-in guardrail functions
                    from dspy_guardrails import guardrail
                    func = getattr(guardrail, func_name)
                    return GuardrailTarget(guardrail_fn=func, name=func_name)
                else:
                    # Custom module
                    import importlib
                    module = importlib.import_module(module_path)

                    # Handle nested attributes (e.g., "module:Class.method")
                    obj = module
                    for attr in func_name.split('.'):
                        obj = getattr(obj, attr)

                    # Determine target type
                    if callable(obj) and not hasattr(obj, 'forward'):
                        return GuardrailTarget(guardrail_fn=obj, name=func_name)
                    else:
                        return DSPyModuleTarget(module=obj)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Cannot resolve target '{target}': {e}") from e

        else:
            raise ValueError(
                f"Invalid target string: {target}. "
                "Expected format: 'http://...', 'guardrail:func_name', or 'module:path'"
            )

    def _resolve_dict_target(self, target: dict[str, Any]) -> UnifiedTarget:
        """从字典配置解析目标

        Args:
            target: 目标配置字典，必须包含 'type' 字段

        Returns:
            UnifiedTarget: 解析后的目标实例

        Raises:
            ValueError: 如果字典格式无效
        """
        from .targets import DSPyModuleTarget, GuardrailTarget, HTTPTarget, MCPTarget

        target_type = target.get("type")
        if not target_type:
            raise ValueError("Target dict must include 'type' field")

        if target_type == "http":
            # HTTP target: {"type": "http", "base_url": "...", "headers": {...}}
            base_url = target.get("base_url") or target.get("url")  # Support both for backward compat
            if not base_url:
                raise ValueError("HTTP target requires 'base_url' or 'url' field")
            return HTTPTarget(
                base_url=base_url,
                endpoint=target.get("endpoint", "/chat"),
                headers=target.get("headers"),
                timeout=target.get("timeout", 30),
            )

        elif target_type == "guardrail":
            # Guardrail target: {"type": "guardrail", "function": "no_injection"}
            func_name = target.get("function")
            if not func_name:
                raise ValueError("Guardrail target requires 'function' field")

            from dspy_guardrails import guardrail
            func = getattr(guardrail, func_name)
            return GuardrailTarget(guardrail_fn=func, name=func_name)

        elif target_type == "dspy_module":
            # DSPy module: {"type": "dspy_module", "module": <instance>}
            module = target.get("module")
            if not module:
                raise ValueError("DSPy module target requires 'module' field")
            return DSPyModuleTarget(module=module)

        elif target_type == "mcp":
            # MCP target: {"type": "mcp", "server_url": "..."}
            server_url = target.get("server_url")
            if not server_url:
                raise ValueError("MCP target requires 'server_url' field")
            return MCPTarget(server_url=server_url)

        else:
            raise ValueError(f"Unknown target type: {target_type}")

    # ==================== Fluent API Methods ====================

    def with_attacks(self, *attack_types: str) -> "SecurityPlatform":
        """配置攻击类型

        Args:
            *attack_types: 攻击类型名称（如 "injection", "jailbreak", "bypass"）

        Returns:
            SecurityPlatform: 返回自身以支持链式调用

        Example:
            >>> platform.with_attacks("injection", "jailbreak")
        """
        self.config.attacks = list(attack_types)
        return self

    def with_scanners(self, *scanner_names: str) -> "SecurityPlatform":
        """配置扫描器

        Args:
            *scanner_names: 扫描器名称（如 "quick_scan", "full_scan"）

        Returns:
            SecurityPlatform: 返回自身以支持链式调用

        Example:
            >>> platform.with_scanners("quick_scan", "deep_scan")
        """
        self.config.scanners = list(scanner_names)
        return self

    def with_training(self, **train_config) -> "SecurityPlatform":
        """配置对抗训练参数

        Args:
            **train_config: 训练配置参数，支持以下选项：
                - enabled (bool): 是否启用训练
                - mode (str): 训练模式 ("redblue", "adversarial")
                - epochs (int): 训练轮数
                - early_stop_threshold (float): 早停阈值
                - attack_budget_per_epoch (int): 每轮攻击预算

        Returns:
            SecurityPlatform: 返回自身以支持链式调用

        Example:
            >>> platform.with_training(enabled=True, mode="redblue", epochs=20)
        """
        self.config.training = TrainingConfig(**train_config)
        return self

    def with_reports(self, *formats: str) -> "SecurityPlatform":
        """配置报告格式

        Args:
            *formats: 报告格式（如 "html", "json", "console"）

        Returns:
            SecurityPlatform: 返回自身以支持链式调用

        Example:
            >>> platform.with_reports("html", "json")
        """
        self.config.report.formats = list(formats)
        return self

    def with_output_dir(self, path: str) -> "SecurityPlatform":
        """配置报告输出目录

        Args:
            path: 输出目录路径

        Returns:
            SecurityPlatform: 返回自身以支持链式调用

        Example:
            >>> platform.with_output_dir("./reports")
        """
        self.config.report.output_dir = Path(path)
        return self

    # ==================== Execution Methods ====================

    def scan(self, mode: str = "quick") -> dict[str, Any]:
        """执行安全扫描

        Args:
            mode: 扫描模式 ("quick", "full")

        Returns:
            Dict with scan results
        """
        from .plugins import PluginConfig
        from .scanners import QuickScanner

        scanner = QuickScanner()
        scanner.configure(PluginConfig(options={
            "max_payloads": self.config.attack_budget if mode == "full" else 20,
            "categories": self.config.attacks or ["injection", "jailbreak"],
            "severity_filter": "medium" if mode == "quick" else "low",
        }))

        result = scanner.execute({"target": self.target})
        self._results.append(("scan", result))

        return {
            "success": result.success,
            "vulnerabilities": result.data.get("vulnerabilities", []),
            "metrics": result.metrics,
        }

    def attack(
        self,
        budget: int | None = None,
        categories: list[str] | None = None,
        use_llm: bool = False,
    ) -> dict[str, Any]:
        """执行安全攻击测试

        Args:
            budget: 攻击预算
            categories: 攻击类别
            use_llm: 是否使用 LLM 生成攻击

        Returns:
            Dict with attack results
        """
        from .attackers import LLMAttacker, StaticAttacker
        from .plugins import PluginConfig

        attack_budget = budget or self.config.attack_budget
        attack_categories = categories or self.config.attacks or ["injection", "jailbreak", "bypass"]

        if use_llm or self.config.use_llm_generation:
            attacker = LLMAttacker()
            attacker.configure(PluginConfig(options={
                "attack_types": attack_categories,
                "num_attacks": attack_budget,
                "fallback_to_static": True,
            }))
        else:
            attacker = StaticAttacker()
            attacker.configure(PluginConfig(options={
                "categories": attack_categories,
                "attack_budget": attack_budget,
            }))

        result = attacker.execute({"target": self.target})
        self._results.append(("attack", result))

        return {
            "success": result.success,
            "attack_results": result.data.get("attack_results", result.data.get("attacks", [])),
            "metrics": result.metrics,
        }

    def train(self, mode: str | None = None) -> dict[str, Any]:
        """执行对抗训练

        Args:
            mode: 训练模式 (reserved for future)

        Returns:
            Dict with training results
        """
        if not self.config.training.enabled:
            return {"success": False, "message": "Training not enabled"}

        # Placeholder for future trainer implementation
        return {
            "success": True,
            "message": "Training placeholder - trainers not yet implemented",
            "config": self.config.training.model_dump(),
        }

    def report(self, results: Any = None) -> list[Path]:
        """生成安全报告

        Args:
            results: 测试结果

        Returns:
            List of generated report file paths
        """
        import json
        from datetime import datetime

        results_to_report = results or self._results
        output_dir = self.config.report.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare report data
        report_data = {
            "timestamp": timestamp,
            "target": str(self.target.target_type.value) if self.target else "unknown",
            "config": {
                "attacks": self.config.attacks,
                "attack_budget": self.config.attack_budget,
            },
            "results": [],
        }

        for result_type, result in results_to_report:
            report_data["results"].append({
                "type": result_type,
                "success": result.success,
                "data": result.data,
                "metrics": result.metrics,
            })

        # Generate reports in requested formats
        for fmt in self.config.report.formats:
            if fmt == "json":
                path = output_dir / f"security_report_{timestamp}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=2, default=str)
                generated_files.append(path)

            elif fmt == "html":
                path = output_dir / f"security_report_{timestamp}.html"
                html = self._generate_html_report(report_data)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
                generated_files.append(path)

            elif fmt == "console":
                self._print_console_report(report_data)

        return generated_files

    def _generate_html_report(self, data: dict[str, Any]) -> str:
        """Generate HTML report."""
        results_html = ""
        for r in data.get("results", []):
            metrics = r.get("metrics", {})
            metrics_items = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    metrics_items.append(f"<li>{k}: {v:.2%}</li>")
                else:
                    metrics_items.append(f"<li>{k}: {v}</li>")
            metrics_str = "".join(metrics_items)

            results_html += f"""
        <div class="result-section">
            <h3>{r['type'].upper()}</h3>
            <p>Success: {'&#10004;' if r['success'] else '&#10008;'}</p>
            <ul>
                {metrics_str}
            </ul>
        </div>
        """

        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Security Report - {data['timestamp']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .result-section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .result-section h3 {{ margin-top: 0; color: #0066cc; }}
    </style>
</head>
<body>
    <h1>Security Report</h1>
    <p>Generated: {data['timestamp']}</p>
    <p>Target: {data['target']}</p>
    {results_html}
</body>
</html>"""

    def _print_console_report(self, data: dict[str, Any]) -> None:
        """Print console report."""
        print("=" * 60)
        print("SECURITY REPORT")
        print("=" * 60)
        print(f"Timestamp: {data['timestamp']}")
        print(f"Target: {data['target']}")
        for r in data.get("results", []):
            print(f"\n--- {r['type'].upper()} ---")
            print(f"Success: {r['success']}")
            for k, v in r.get("metrics", {}).items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.2%}")
                else:
                    print(f"  {k}: {v}")
        print("=" * 60)

    def run_all(self) -> dict[str, Any]:
        """执行完整安全测试流程

        Returns:
            Dict with complete test results
        """
        results = {
            "scan": None,
            "attack": None,
            "train": None,
            "reports": [],
        }

        # Run scan
        results["scan"] = self.scan(mode="quick")

        # Run attack
        results["attack"] = self.attack()

        # Run training if enabled
        if self.config.training.enabled:
            results["train"] = self.train()

        # Generate reports
        if self.config.report.formats:
            results["reports"] = [str(p) for p in self.report()]

        return results

    # ==================== Class Methods ====================

    @classmethod
    def from_yaml(cls, path: str) -> "SecurityPlatform":
        """从 YAML 配置文件创建 SecurityPlatform

        Args:
            path: YAML 配置文件路径

        Returns:
            SecurityPlatform instance

        Example:
            >>> platform = SecurityPlatform.from_yaml("security_config.yaml")
        """
        from .cli.utils import parse_target
        from .cli.yaml_config import SecurityConfig

        config = SecurityConfig.from_yaml(path)

        # Build target string
        target_str = config.target.to_target_string()
        target = parse_target(target_str)

        # Build platform config
        platform_config = PlatformConfig(
            attacks=config.scan.categories if config.scan.enabled else config.attack.types,
            attack_budget=config.attack.budget,
            use_llm_generation=config.attack.use_llm,
        )
        platform_config.report.formats = config.report.formats
        platform_config.report.output_dir = Path(config.report.output_dir)

        return cls(target=target, config=platform_config)

    # ==================== Magic Methods ====================

    def __repr__(self) -> str:
        """返回平台的字符串表示"""
        target_type = self.target.target_type.value if self.target else "none"
        return (
            f"<SecurityPlatform target={target_type} "
            f"attacks={self.config.attacks} "
            f"scanners={self.config.scanners}>"
        )
