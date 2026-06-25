"""
Tool Input Guards - 工具输入防护

检测和阻止恶意的工具输入参数。
"""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .core import GuardAction, GuardResult, ThreatCategory, ToolCallContext


@dataclass
class InputValidationResult:
    """输入验证结果"""
    valid: bool
    threat_detected: bool
    threat_type: str | None = None
    matched_pattern: str | None = None
    suspicious_value: str | None = None
    confidence: float = 0.0


class ToolInputGuard(ABC):
    """工具输入防护基类"""

    name: str = "base_guard"
    description: str = "Base input guard"

    @abstractmethod
    def check(self, context: ToolCallContext) -> GuardResult:
        """检查工具输入"""
        pass

    @classmethod
    def injection_check(cls) -> "InjectionInputGuard":
        """创建注入检测防护"""
        return InjectionInputGuard()

    @classmethod
    def path_traversal_check(cls) -> "PathTraversalGuard":
        """创建路径遍历防护"""
        return PathTraversalGuard()

    @classmethod
    def command_injection_check(cls) -> "CommandInjectionGuard":
        """创建命令注入防护"""
        return CommandInjectionGuard()

    @classmethod
    def sql_injection_check(cls) -> "SQLInjectionGuard":
        """创建 SQL 注入防护"""
        return SQLInjectionGuard()

    @classmethod
    def sensitive_param_check(
        cls,
        sensitive_params: list[str] = None,
    ) -> "SensitiveParamGuard":
        """创建敏感参数防护"""
        return SensitiveParamGuard(sensitive_params)

    @classmethod
    def rug_pull_check(cls) -> "RugPullGuard":
        """创建 Rug Pull 防护"""
        return RugPullGuard()

    @classmethod
    def privilege_escalation_check(cls) -> "PrivilegeEscalationGuard":
        """创建权限提升防护"""
        return PrivilegeEscalationGuard()

    @classmethod
    def custom(
        cls,
        check_fn: Callable[[ToolCallContext], GuardResult],
        name: str = "custom",
    ) -> "CustomInputGuard":
        """创建自定义防护"""
        return CustomInputGuard(check_fn, name)


class InjectionInputGuard(ToolInputGuard):
    """
    Prompt Injection 输入防护 (增强版)

    检测工具参数中的 prompt injection 尝试。
    支持多种高级绕过技术检测。
    """

    name = "injection_guard"
    description = "Detect prompt injection in tool parameters (enhanced)"

    # === 基础注入模式 ===
    BASIC_PATTERNS = [
        # 直接覆盖指令
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|commands|rules|prompts?)",
        r"disregard\s+(all\s+)?(previous|prior|above)",
        r"forget\s+(everything|all|your\s+instructions|what\s+I\s+said)",
        r"new\s+(instructions?|task|directive|goal)\s*[:=]",
        r"override\s+(previous|all|safety|your)",

        # 系统提示伪装
        r"\]\s*\n*\s*system\s*[:=]",
        r"<\|?(system|im_start|im_end|endoftext)\|?>",
        r"---\s*end\s*(of\s*)?(prompt|instructions|context)",
        r"\[INST\]|\[/INST\]|\[SYS\]",
        r"<(system|assistant|user)>",
        r"\*\*\s*SYSTEM\s*(MESSAGE|PROMPT|INSTRUCTION)",

        # 角色切换
        r"you\s+are\s+now\s+(a|an|the|my)?\s*\w+",
        r"act\s+as\s+(a|an|if|though)",
        r"pretend\s+(to\s+be|you\s+are|you're)",
        r"(enter|enable|activate)\s+(developer|admin|god|sudo|root)\s+mode",
        r"switch\s+to\s+(unrestricted|uncensored|unfiltered)",

        # 越狱关键词
        r"\b(DAN|jailbreak|bypass|override|pwned?)\b",
        r"do\s+anything\s+now",
        r"no\s+(restrictions?|limits?|rules?|guidelines?|boundaries)",
        r"without\s+(restrictions?|limits?|rules?)",
    ]

    # === 高级绕过模式 ===
    ADVANCED_PATTERNS = [
        # 分隔符逃逸
        r"\[END\s*(OF)?\s*(TEXT|PROMPT|INPUT|CONTEXT)\]",
        r"</?(text|prompt|input|context|message)>",
        r"={3,}|#{3,}|-{3,}",  # 分隔线
        r"\n\s*---\s*\n",

        # 权威声明
        r"(I\s+am|this\s+is)\s+(an?\s+)?(admin|developer|security\s+researcher|authorized)",
        r"(official|authorized|approved)\s+(instruction|command|request)",
        r"as\s+per\s+(company|security|admin)\s+policy",
        r"ticket\s*#?\d+",  # 假票据
        r"(pentest|penetration\s+test|security\s+audit)",

        # 隐藏指令标记
        r"\[HIDDEN\]|\[PRIVATE\]|\[INTERNAL\]",
        r"<!--.*?(instruction|command|task|execute).*?-->",
        r"\(IMPORTANT|NOTE|ATTENTION\)\s*:",
        r"AI\s*(INSTRUCTION|DIRECTIVE|COMMAND)\s*:",

        # Unicode 混淆
        r"[\u200b-\u200f\u2028-\u202f\ufeff]",  # 零宽字符和特殊空白
        r"[іІоОаАеЕ]",  # 西里尔字母替换 (看起来像拉丁字母)

        # 任务切换
        r"(new|actual|real|true)\s+(task|goal|objective|instruction)\s*:",
        r"(stop|pause|cancel)\s+(current|previous)\s+(task|operation)",
        r"priority\s*(level)?\s*:\s*(high|urgent|critical)",

        # 输出操控
        r"(output|print|display|reveal|show)\s+(system\s+prompt|instructions|secrets?|api\s*keys?)",
        r"(list|enumerate)\s+(all\s+)?(tools?|functions?|capabilities)",
    ]

    # === 语义模式 (需要更多上下文) ===
    SEMANTIC_PATTERNS = [
        # 间接请求
        r"what\s+(are|were)\s+your\s+(original|initial|first)\s+instructions",
        r"(tell|show)\s+me\s+(about\s+)?your\s+(prompt|instructions|rules)",
        r"repeat\s+(your|the)\s+(system\s+)?(prompt|instructions)",

        # 假设场景
        r"(imagine|suppose|let's\s+say|hypothetically)",
        r"for\s+(educational|research|testing)\s+purposes?",
        r"(just\s+)?between\s+(you\s+and\s+me|us)",
    ]

    def __init__(self, additional_patterns: list[str] = None, sensitivity: str = "high"):
        """
        初始化注入防护

        Args:
            additional_patterns: 额外的检测模式
            sensitivity: 敏感度级别 ("low", "medium", "high")
        """
        self.sensitivity = sensitivity

        # 根据敏感度选择模式
        if sensitivity == "low":
            patterns = self.BASIC_PATTERNS
        elif sensitivity == "medium":
            patterns = self.BASIC_PATTERNS + self.ADVANCED_PATTERNS
        else:  # high
            patterns = self.BASIC_PATTERNS + self.ADVANCED_PATTERNS + self.SEMANTIC_PATTERNS

        self.patterns = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]

        if additional_patterns:
            self.patterns.extend([
                re.compile(p, re.IGNORECASE) for p in additional_patterns
            ])

        # Unicode 正规化
        self._unicode_confusables = {
            'а': 'a', 'е': 'e', 'і': 'i', 'о': 'o', 'р': 'p',
            'с': 'c', 'х': 'x', 'у': 'y', 'А': 'A', 'Е': 'E',
            'І': 'I', 'О': 'O', 'Р': 'P', 'С': 'C', 'Х': 'X',
            '．': '.', '／': '/', '＼': '\\',
        }

    def check(self, context: ToolCallContext) -> GuardResult:
        """检查注入攻击"""
        # 将所有参数转为字符串检查
        params_str = self._flatten_params(context.parameters)

        matches = []
        total_score = 0.0

        for text in params_str:
            # Unicode 正规化
            normalized_text = self._normalize_unicode(text)

            for pattern in self.patterns:
                # 检查原始文本
                match = pattern.search(text)
                if match:
                    matches.append({
                        "pattern": pattern.pattern[:50],
                        "matched": match.group()[:50],
                        "type": "direct",
                    })
                    total_score += 0.35

                # 检查正规化后的文本 (如果不同)
                elif normalized_text != text:
                    match = pattern.search(normalized_text)
                    if match:
                        matches.append({
                            "pattern": pattern.pattern[:50],
                            "matched": match.group()[:50],
                            "type": "unicode_normalized",
                        })
                        total_score += 0.4  # Unicode 绕过更危险

        # 额外检查：多行分隔符攻击
        for text in params_str:
            if self._check_delimiter_attack(text):
                matches.append({"type": "delimiter_attack", "pattern": "multi-line delimiter"})
                total_score += 0.5

        if matches:
            threat_score = min(1.0, total_score)
            action = GuardAction.BLOCK if threat_score >= 0.4 else GuardAction.WARN

            return GuardResult(
                action=action,
                passed=action != GuardAction.BLOCK,
                threat_category=ThreatCategory.INJECTION,
                threat_score=threat_score,
                message=f"Detected {len(matches)} potential injection patterns",
                details={"matches": matches[:5], "sensitivity": self.sensitivity},
            )

        return GuardResult(
            action=GuardAction.ALLOW,
            passed=True,
            threat_score=0.0,
        )

    def _normalize_unicode(self, text: str) -> str:
        """Unicode 正规化 - 将混淆字符转换为标准字符"""
        result = text
        for confusable, replacement in self._unicode_confusables.items():
            result = result.replace(confusable, replacement)
        # 移除零宽字符
        result = re.sub(r'[\u200b-\u200f\u2028-\u202f\ufeff]', '', result)
        return result

    def _check_delimiter_attack(self, text: str) -> bool:
        """检查分隔符攻击"""
        # 检查是否有多行结构试图切换上下文
        lines = text.split('\n')
        if len(lines) < 2:
            return False

        delimiter_patterns = [
            r'^-{3,}$', r'^={3,}$', r'^#{3,}$',
            r'^\[END', r'^</\w+>', r'^\*{3,}$',
        ]

        for i, line in enumerate(lines):
            line = line.strip()
            for pattern in delimiter_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # 检查分隔符后面是否有可疑内容
                    remaining = '\n'.join(lines[i+1:]).lower()
                    suspicious_after = [
                        'instruction', 'task', 'ignore', 'override',
                        'system', 'important', 'priority',
                    ]
                    if any(s in remaining for s in suspicious_after):
                        return True
        return False

    def _flatten_params(self, params: dict[str, Any]) -> list[str]:
        """将参数展平为字符串列表"""
        result = []

        def extract_strings(obj, path=""):
            if isinstance(obj, str):
                result.append(obj)
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    # 同时检查键名 (工具描述可能在键中)
                    if isinstance(k, str) and len(k) > 20:
                        result.append(k)
                    extract_strings(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract_strings(item, f"{path}[{i}]")

        extract_strings(params)
        return result


class PathTraversalGuard(ToolInputGuard):
    """
    路径遍历防护

    检测和阻止路径遍历攻击。
    """

    name = "path_traversal_guard"
    description = "Detect path traversal attacks"

    PATH_PATTERNS = [
        r"\.\./",           # ../
        r"\.\.\\",          # ..\
        r"%2e%2e[%/\\]",    # URL encoded ../
        r"\.\.%2f",         # mixed encoding
        r"\.\.%5c",         # mixed encoding
        r"/etc/passwd",
        r"/etc/shadow",
        r"C:\\Windows",
        r"\\\\",            # UNC paths
        r"~/.+",            # Home directory
        r"\$HOME",
        r"\$\{.+\}",        # Variable expansion
    ]

    # 危险路径
    DANGEROUS_PATHS = [
        "/etc/", "/root/", "/var/log/", "/proc/", "/sys/",
        "C:\\Windows\\", "C:\\System32\\",
        "~/.ssh/", "~/.aws/", "~/.config/",
    ]

    def __init__(
        self,
        allowed_paths: list[str] = None,
        additional_patterns: list[str] = None,
    ):
        self.allowed_paths = allowed_paths or []
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.PATH_PATTERNS]
        if additional_patterns:
            self.patterns.extend([
                re.compile(p, re.IGNORECASE) for p in additional_patterns
            ])

    def check(self, context: ToolCallContext) -> GuardResult:
        """检查路径遍历"""
        path_params = self._extract_path_params(context.parameters)

        for param_name, path_value in path_params:
            # 检查路径模式
            for pattern in self.patterns:
                if pattern.search(path_value):
                    return GuardResult(
                        action=GuardAction.BLOCK,
                        passed=False,
                        threat_category=ThreatCategory.PATH_TRAVERSAL,
                        threat_score=0.9,
                        message=f"Path traversal pattern detected in '{param_name}'",
                        details={
                            "param": param_name,
                            "value": path_value[:100],
                            "pattern": pattern.pattern,
                        },
                    )

            # 检查危险路径
            for dangerous in self.DANGEROUS_PATHS:
                if dangerous.lower() in path_value.lower():
                    # 检查是否在白名单
                    if not self._is_allowed(path_value):
                        return GuardResult(
                            action=GuardAction.BLOCK,
                            passed=False,
                            threat_category=ThreatCategory.PATH_TRAVERSAL,
                            threat_score=0.85,
                            message="Access to dangerous path detected",
                            details={
                                "param": param_name,
                                "dangerous_path": dangerous,
                            },
                        )

        return GuardResult(action=GuardAction.ALLOW, passed=True)

    def _extract_path_params(self, params: dict[str, Any]) -> list[tuple]:
        """提取可能是路径的参数"""
        path_keywords = ["path", "file", "directory", "dir", "folder", "location", "url", "uri"]
        result = []

        def check_param(key: str, value: Any, prefix: str = ""):
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, str):
                # 检查参数名是否包含路径关键词
                if any(kw in key.lower() for kw in path_keywords):
                    result.append((full_key, value))
                # 或者值看起来像路径
                elif "/" in value or "\\" in value:
                    result.append((full_key, value))
            elif isinstance(value, dict):
                for k, v in value.items():
                    check_param(k, v, full_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    check_param(f"[{i}]", item, full_key)

        for k, v in params.items():
            check_param(k, v)

        return result

    def _is_allowed(self, path: str) -> bool:
        """检查路径是否在白名单"""
        for allowed in self.allowed_paths:
            if path.startswith(allowed):
                return True
        return False


class CommandInjectionGuard(ToolInputGuard):
    """
    命令注入防护

    检测 shell 命令注入尝试。
    """

    name = "command_injection_guard"
    description = "Detect command injection attacks"

    # 命令注入模式
    COMMAND_PATTERNS = [
        # Shell 元字符
        r"[;&|`$]",
        r"\$\(.*\)",        # $(command)
        r"`.*`",            # `command`
        r"\|\|",            # ||
        r"&&",              # &&
        r"\n",              # newline injection
        r"\r",              # carriage return

        # 危险命令
        r"\b(rm|del|rmdir|format|mkfs)\s+(-rf?|/[sq])?",
        r"\b(curl|wget|nc|netcat)\b",
        r"\b(chmod|chown|sudo|su)\b",
        r"\b(eval|exec|system|popen|spawn)\b",
        r"\b(python|perl|ruby|bash|sh|cmd|powershell)\s+-[ce]",

        # 重定向
        r">\s*/",           # > /path
        r">>\s*",           # >>
        r"<\s*/",           # < /path
    ]

    def __init__(self, additional_patterns: list[str] = None):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.COMMAND_PATTERNS]
        if additional_patterns:
            self.patterns.extend([
                re.compile(p, re.IGNORECASE) for p in additional_patterns
            ])

    def check(self, context: ToolCallContext) -> GuardResult:
        """检查命令注入"""
        # 检查工具名是否涉及命令执行
        command_tools = ["execute", "run", "shell", "command", "exec", "system"]
        is_command_tool = any(t in context.tool_name.lower() for t in command_tools)

        params_str = self._get_string_params(context.parameters)
        matches = []

        for param_name, value in params_str:
            for pattern in self.patterns:
                match = pattern.search(value)
                if match:
                    matches.append({
                        "param": param_name,
                        "pattern": pattern.pattern,
                        "matched": match.group(),
                    })

        if matches:
            # 命令工具中的注入更危险
            threat_score = min(1.0, len(matches) * 0.25 + (0.3 if is_command_tool else 0))
            action = GuardAction.BLOCK if threat_score >= 0.5 else GuardAction.WARN

            return GuardResult(
                action=action,
                passed=action != GuardAction.BLOCK,
                threat_category=ThreatCategory.COMMAND_INJECTION,
                threat_score=threat_score,
                message=f"Detected {len(matches)} potential command injection patterns",
                details={"matches": matches[:5], "is_command_tool": is_command_tool},
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)

    def _get_string_params(self, params: dict[str, Any]) -> list[tuple]:
        """获取所有字符串参数"""
        result = []

        def extract(obj, path=""):
            if isinstance(obj, str):
                result.append((path, obj))
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    extract(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    extract(item, f"{path}[{i}]")

        extract(params)
        return result


class SQLInjectionGuard(ToolInputGuard):
    """
    SQL 注入防护

    检测 SQL 注入尝试。
    """

    name = "sql_injection_guard"
    description = "Detect SQL injection attacks"

    SQL_PATTERNS = [
        r"'\s*or\s+'?1'?\s*=\s*'?1",           # ' or '1'='1
        r"'\s*or\s+''='",                       # ' or ''='
        r";\s*(drop|delete|truncate|update|insert)\s",
        r"union\s+(all\s+)?select",
        r"--\s*$",                              # SQL comment
        r"/\*.*\*/",                            # Block comment
        r"'\s*;\s*",                            # String termination
        r"xp_cmdshell",
        r"exec\s*\(",
        r"char\s*\(\s*\d+\s*\)",
        r"convert\s*\(",
        r"cast\s*\(",
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_PATTERNS]

    def check(self, context: ToolCallContext) -> GuardResult:
        """检查 SQL 注入"""
        # 检查是否是数据库相关工具
        db_tools = ["query", "sql", "database", "db", "select", "execute"]
        is_db_tool = any(t in context.tool_name.lower() for t in db_tools)

        params_str = self._flatten_to_strings(context.parameters)
        matches = []

        for text in params_str:
            for pattern in self.patterns:
                match = pattern.search(text)
                if match:
                    matches.append({
                        "pattern": pattern.pattern,
                        "matched": match.group(),
                    })

        if matches:
            threat_score = min(1.0, len(matches) * 0.3 + (0.2 if is_db_tool else 0))

            return GuardResult(
                action=GuardAction.BLOCK,
                passed=False,
                threat_category=ThreatCategory.INJECTION,
                threat_score=threat_score,
                message=f"Detected {len(matches)} SQL injection patterns",
                details={"matches": matches[:5], "is_db_tool": is_db_tool},
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)

    def _flatten_to_strings(self, obj: Any) -> list[str]:
        result = []
        if isinstance(obj, str):
            result.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                result.extend(self._flatten_to_strings(v))
        elif isinstance(obj, list):
            for item in obj:
                result.extend(self._flatten_to_strings(item))
        return result


class SensitiveParamGuard(ToolInputGuard):
    """
    敏感参数防护

    检测并保护敏感参数。
    """

    name = "sensitive_param_guard"
    description = "Protect sensitive parameters"

    DEFAULT_SENSITIVE = [
        "password", "passwd", "pwd", "secret", "token",
        "api_key", "apikey", "access_key", "private_key",
        "credential", "auth", "authorization",
    ]

    def __init__(self, sensitive_params: list[str] = None):
        self.sensitive_params = sensitive_params or self.DEFAULT_SENSITIVE

    def check(self, context: ToolCallContext) -> GuardResult:
        """检查敏感参数"""
        found_sensitive = []

        def check_keys(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_path = f"{path}.{key}" if path else key
                    key_lower = key.lower()

                    # 检查是否是敏感参数
                    if any(s in key_lower for s in self.sensitive_params):
                        found_sensitive.append({
                            "path": full_path,
                            "key": key,
                            "value_type": type(value).__name__,
                        })

                    check_keys(value, full_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_keys(item, f"{path}[{i}]")

        check_keys(context.parameters)

        if found_sensitive:
            return GuardResult(
                action=GuardAction.WARN,
                passed=True,  # 警告但通过
                threat_category=ThreatCategory.SENSITIVE_DATA,
                threat_score=0.4,
                message=f"Found {len(found_sensitive)} sensitive parameters",
                details={"sensitive_params": found_sensitive},
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)


class RugPullGuard(ToolInputGuard):
    """
    Rug Pull 防护

    检测工具定义在审批后被篡改的攻击。

    Rug Pull 攻击:
    1. 工具首次使用时表现正常
    2. 获得用户信任后，工具行为改变
    3. 可能通过版本更新、checksum 变化等方式实现

    防护策略:
    1. 工具描述/版本锁定 (pinning)
    2. Checksum 验证
    3. 行为变化检测
    """

    name = "rug_pull_guard"
    description = "Detect rug pull attacks via tool tampering"

    def __init__(self):
        # 存储已信任工具的指纹
        self._trusted_tools: dict[str, dict[str, Any]] = {}
        self._call_history: dict[str, list[dict]] = {}

    def check(self, context: ToolCallContext) -> GuardResult:
        """检查 Rug Pull 攻击"""
        tool_name = context.tool_name
        params = context.parameters

        # 检查工具元数据
        version = params.get("version") or params.get("_version")
        checksum = params.get("checksum") or params.get("_checksum")
        description = params.get("description") or params.get("_description", "")

        # 计算当前工具指纹
        current_fingerprint = self._compute_fingerprint(tool_name, version, description)

        # 如果是首次见到的工具，记录并通过
        if tool_name not in self._trusted_tools:
            self._trusted_tools[tool_name] = {
                "fingerprint": current_fingerprint,
                "version": version,
                "checksum": checksum,
                "description_hash": hash(description) if description else None,
                "first_seen": True,
            }
            return GuardResult(
                action=GuardAction.ALLOW,
                passed=True,
                message="New tool registered for monitoring",
                details={"tool": tool_name, "status": "trusted"},
            )

        # 检查指纹变化
        trusted = self._trusted_tools[tool_name]
        violations = []

        # 1. Checksum 变化
        if checksum and trusted.get("checksum") and checksum != trusted["checksum"]:
            violations.append({
                "type": "checksum_mismatch",
                "expected": trusted["checksum"],
                "actual": checksum,
            })

        # 2. 版本变化 (可能是恶意更新)
        if version and trusted.get("version") and version != trusted["version"]:
            violations.append({
                "type": "version_change",
                "expected": trusted["version"],
                "actual": version,
            })

        # 3. 描述变化 (可能注入恶意指令)
        if description:
            current_desc_hash = hash(description)
            if trusted.get("description_hash") and current_desc_hash != trusted["description_hash"]:
                # 检查描述中是否有注入
                injection_guard = InjectionInputGuard(sensitivity="high")
                injection_result = injection_guard.check(ToolCallContext(
                    tool_name=tool_name,
                    parameters={"description": description},
                ))
                if not injection_result.passed:
                    violations.append({
                        "type": "description_tampered",
                        "injection_detected": True,
                    })
                else:
                    violations.append({
                        "type": "description_changed",
                        "warning": "Tool description modified",
                    })

        # 4. 指纹不匹配
        if current_fingerprint != trusted["fingerprint"]:
            violations.append({
                "type": "fingerprint_mismatch",
                "detail": "Tool signature has changed",
            })

        if violations:
            threat_score = min(1.0, len(violations) * 0.4)
            return GuardResult(
                action=GuardAction.BLOCK,
                passed=False,
                threat_category=ThreatCategory.RUG_PULL,
                threat_score=threat_score,
                message=f"Rug pull attack detected: {len(violations)} violations",
                details={"violations": violations, "tool": tool_name},
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)

    def _compute_fingerprint(self, tool_name: str, version: Any, description: str) -> str:
        """计算工具指纹"""
        import hashlib
        content = f"{tool_name}:{version}:{description[:500] if description else ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def trust_tool(self, tool_name: str, version: str = None, checksum: str = None, description: str = None):
        """手动信任工具"""
        self._trusted_tools[tool_name] = {
            "fingerprint": self._compute_fingerprint(tool_name, version, description),
            "version": version,
            "checksum": checksum,
            "description_hash": hash(description) if description else None,
            "first_seen": False,
        }

    def reset_tool(self, tool_name: str = None):
        """重置工具信任"""
        if tool_name:
            self._trusted_tools.pop(tool_name, None)
        else:
            self._trusted_tools.clear()


class PrivilegeEscalationGuard(ToolInputGuard):
    """
    权限提升防护

    检测尝试提升权限的攻击。
    """

    name = "privilege_escalation_guard"
    description = "Detect privilege escalation attempts"

    # 权限相关模式
    PRIVILEGE_PATTERNS = [
        # 角色注入
        r'"?role"?\s*[:=]\s*"?(admin|root|superuser|system)",',
        r'"?user_role"?\s*[:=]\s*"?(admin|root|superuser)",',
        r'"?is_admin"?\s*[:=]\s*(true|1)',
        r'"?privilege"?\s*[:=]\s*"?(elevated|high|admin)",',

        # JWT 攻击
        r'"?alg"?\s*[:=]\s*"?none"?',  # Algorithm none attack
        r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.',  # JWT pattern

        # Sudo/权限命令
        r'\bsudo\b',
        r'\bsu\s+-',
        r'\brunas\b',
        r'\bdoas\b',

        # 权限修改
        r'chmod\s+[0-7]*7[0-7]*',  # World writable
        r'chown\s+root',
        r'setuid|setgid|setcap',
    ]

    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.PRIVILEGE_PATTERNS]

    def check(self, context: ToolCallContext) -> GuardResult:
        """检查权限提升"""
        params_str = json.dumps(context.parameters)
        matches = []

        for pattern in self.patterns:
            match = pattern.search(params_str)
            if match:
                matches.append({
                    "pattern": pattern.pattern[:40],
                    "matched": match.group()[:50],
                })

        # 检查 JWT none 攻击
        jwt_match = re.search(r'eyJ[A-Za-z0-9_-]*\.eyJ[A-Za-z0-9_-]*\.', params_str)
        if jwt_match:
            # 尝试解码 header
            try:
                import base64
                header = jwt_match.group().split('.')[0]
                # 添加 padding
                header += '=' * (4 - len(header) % 4)
                decoded = base64.urlsafe_b64decode(header).decode()
                if '"alg"' in decoded and ('none' in decoded.lower() or '"HS256"' not in decoded):
                    matches.append({
                        "pattern": "jwt_algorithm_attack",
                        "matched": decoded[:50],
                    })
            except Exception:
                pass

        if matches:
            return GuardResult(
                action=GuardAction.BLOCK,
                passed=False,
                threat_category=ThreatCategory.PRIVILEGE_ESCALATION,
                threat_score=0.9,
                message="Privilege escalation attempt detected",
                details={"matches": matches[:5]},
            )

        return GuardResult(action=GuardAction.ALLOW, passed=True)


class CustomInputGuard(ToolInputGuard):
    """自定义输入防护"""

    def __init__(
        self,
        check_fn: Callable[[ToolCallContext], GuardResult],
        name: str = "custom",
    ):
        self.check_fn = check_fn
        self.name = name

    def check(self, context: ToolCallContext) -> GuardResult:
        return self.check_fn(context)
