"""
CLI Guardrail Core - Main security check implementation

Provides the core CLIGuardrail class for detecting and blocking
dangerous command line operations.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto

from .blocklist import DangerousCommands, DangerousPatterns, SensitivePaths
from .parser import CommandParser, ParsedCommand
from .policies import CLISecurityConfig, SandboxLevel


class CLIThreatCategory(Enum):
    """CLI threat categories"""

    # Critical threats
    DESTRUCTIVE_COMMAND = auto()      # rm -rf /, mkfs, dd
    COMMAND_INJECTION = auto()        # ; && || backticks $()
    PRIVILEGE_ESCALATION = auto()     # sudo, su, chmod 777

    # High threats
    DATA_EXFILTRATION = auto()        # curl upload, nc, etc
    CREDENTIAL_ACCESS = auto()        # cat ~/.ssh/id_rsa
    REMOTE_CODE_EXECUTION = auto()    # curl | bash, wget && sh

    # Medium threats
    NETWORK_ACCESS = auto()           # curl, wget, nc without explicit allow
    RESOURCE_EXHAUSTION = auto()      # fork bomb, infinite loops
    PERSISTENCE = auto()              # crontab, .bashrc modification

    # Low threats
    INFORMATION_DISCLOSURE = auto()   # env, printenv, whoami
    PATH_TRAVERSAL = auto()           # ../../../etc/passwd
    SUSPICIOUS_PATTERN = auto()       # Obfuscation, encoding


class CLIGuardAction(Enum):
    """Actions to take when threat detected"""
    ALLOW = "allow"
    BLOCK = "block"
    SANITIZE = "sanitize"
    WARN = "warn"
    AUDIT = "audit"


@dataclass
class CLIGuardResult:
    """Result of CLI security check"""

    is_safe: bool
    action: CLIGuardAction
    original_command: str
    sanitized_command: str | None = None
    threat_type: CLIThreatCategory | None = None
    threat_details: str = ""
    confidence: float = 1.0
    matched_patterns: list[str] = field(default_factory=list)
    parsed_command: ParsedCommand | None = None

    def __bool__(self) -> bool:
        return self.is_safe


class CLIGuardrail:
    """
    CLI Security Guardrail for LLM Agents

    Detects and blocks dangerous command line operations including:
    - Destructive commands (rm -rf, mkfs, dd)
    - Command injection (; && || $() ``)
    - Privilege escalation (sudo, su)
    - Data exfiltration (curl upload, nc)
    - Remote code execution (curl | bash)
    - Credential access (~/.ssh, .env)

    Examples:
        # Basic usage
        guard = CLIGuardrail()

        result = guard.check("ls -la")
        print(result.is_safe)  # True

        result = guard.check("rm -rf /")
        print(result.is_safe)  # False
        print(result.threat_type)  # CLIThreatCategory.DESTRUCTIVE_COMMAND

        # With custom config
        from dspy_guardrails.cli import CLISecurityConfig, SandboxLevel

        config = CLISecurityConfig(
            sandbox_level=SandboxLevel.STRICT,
            allow_network=False,
            allow_sudo=False,
            protected_paths=["~/.ssh", "/etc/passwd"],
            allowed_commands=["ls", "cat", "echo", "pwd"],
        )
        guard = CLIGuardrail(config=config)

        # Boolean check for DSPy Assert
        from dspy_guardrails.cli import cli_guardrail

        dspy.Assert(
            cli_guardrail.is_safe(command),
            "Command failed security check"
        )
    """

    def __init__(self, config: CLISecurityConfig | None = None):
        self.config = config or CLISecurityConfig()
        self.parser = CommandParser()
        self.dangerous_commands = DangerousCommands()
        self.dangerous_patterns = DangerousPatterns()
        self.sensitive_paths = SensitivePaths()

    def check(self, command: str) -> CLIGuardResult:
        """
        Check if a command is safe to execute

        Args:
            command: Shell command string to check

        Returns:
            CLIGuardResult with safety assessment
        """
        if not command or not command.strip():
            return CLIGuardResult(
                is_safe=True,
                action=CLIGuardAction.ALLOW,
                original_command=command,
            )

        # Parse command
        parsed = self.parser.parse(command)

        # Run all checks
        checks = [
            self._check_destructive_commands,
            self._check_command_injection,
            self._check_privilege_escalation,
            self._check_remote_code_execution,
            self._check_data_exfiltration,
            self._check_credential_access,
            self._check_network_access,
            self._check_resource_exhaustion,
            self._check_persistence,
            self._check_path_traversal,
            self._check_obfuscation,
            self._check_allowlist,
        ]

        for check_fn in checks:
            result = check_fn(command, parsed)
            if result is not None and not result.is_safe:
                return result

        # All checks passed
        return CLIGuardResult(
            is_safe=True,
            action=CLIGuardAction.ALLOW,
            original_command=command,
            parsed_command=parsed,
        )

    def is_safe(self, command: str) -> bool:
        """Simple boolean check for DSPy Assert compatibility"""
        return self.check(command).is_safe

    def _check_destructive_commands(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for destructive commands like rm -rf /"""

        matched = self.dangerous_commands.check_destructive(command)
        if matched:
            return CLIGuardResult(
                is_safe=False,
                action=CLIGuardAction.BLOCK,
                original_command=command,
                threat_type=CLIThreatCategory.DESTRUCTIVE_COMMAND,
                threat_details=f"Destructive command detected: {matched}",
                matched_patterns=[matched],
                parsed_command=parsed,
            )
        return None

    def _check_command_injection(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for command injection attempts"""

        # Check for injection patterns
        injection_patterns = [
            (r';\s*\w+', "semicolon injection"),
            (r'\|\|', "OR operator"),
            (r'&&', "AND operator"),
            (r'\$\([^)]+\)', "command substitution $()"),
            (r'`[^`]+`', "backtick substitution"),
            (r'\$\{[^}]+\}', "variable expansion"),
        ]

        for pattern, desc in injection_patterns:
            if re.search(pattern, command):
                # Allow safe uses in strict mode
                if self.config.sandbox_level != SandboxLevel.STRICT:
                    # Check if it's a simple compound command
                    if self._is_safe_compound(command):
                        continue

                return CLIGuardResult(
                    is_safe=False,
                    action=CLIGuardAction.BLOCK,
                    original_command=command,
                    threat_type=CLIThreatCategory.COMMAND_INJECTION,
                    threat_details=f"Command injection detected: {desc}",
                    matched_patterns=[pattern],
                    parsed_command=parsed,
                )

        return None

    def _check_privilege_escalation(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for privilege escalation attempts"""

        if not self.config.allow_sudo:
            priv_patterns = [
                (r'\bsudo\b', "sudo command"),
                (r'\bsu\b\s+', "su command"),
                (r'\bdoas\b', "doas command"),
                (r'chmod\s+[0-7]*7[0-7]*\s', "world-writable permission"),
                (r'chmod\s+\+s\b', "setuid bit"),
                (r'chown\s+root\b', "change owner to root"),
            ]

            for pattern, desc in priv_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return CLIGuardResult(
                        is_safe=False,
                        action=CLIGuardAction.BLOCK,
                        original_command=command,
                        threat_type=CLIThreatCategory.PRIVILEGE_ESCALATION,
                        threat_details=f"Privilege escalation detected: {desc}",
                        matched_patterns=[pattern],
                        parsed_command=parsed,
                    )

        return None

    def _check_remote_code_execution(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for remote code execution patterns"""

        rce_patterns = [
            (r'curl\s+[^\|]+\|\s*(ba)?sh', "curl pipe to shell"),
            (r'wget\s+[^;]+;\s*(ba)?sh', "wget then shell"),
            (r'wget\s+-O\s*-[^\|]+\|\s*(ba)?sh', "wget -O - pipe to shell"),
            (r'curl\s+[^>]+>\s*[^;]+;\s*(ba)?sh', "curl redirect then shell"),
            (r'python\s+-c\s*["\'].*urllib', "python URL execution"),
            (r'python\s+-c\s*["\'].*requests', "python requests execution"),
            (r'eval\s+"\$\(curl', "eval curl"),
            (r'source\s+<\(curl', "source curl"),
        ]

        for pattern, desc in rce_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return CLIGuardResult(
                    is_safe=False,
                    action=CLIGuardAction.BLOCK,
                    original_command=command,
                    threat_type=CLIThreatCategory.REMOTE_CODE_EXECUTION,
                    threat_details=f"Remote code execution detected: {desc}",
                    matched_patterns=[pattern],
                    parsed_command=parsed,
                )

        return None

    def _check_data_exfiltration(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for data exfiltration attempts"""

        exfil_patterns = [
            (r'curl\s+.*-[dX]\s*(POST|PUT)', "curl POST/PUT data"),
            (r'curl\s+.*--data', "curl data upload"),
            (r'curl\s+.*-F\s', "curl form upload"),
            (r'nc\s+-[^l]', "netcat outbound"),
            (r'scp\s+.*@.*:', "scp outbound"),
            (r'rsync\s+.*@.*:', "rsync outbound"),
            (r'ftp\s+-n', "ftp automated"),
            (r'tftp\s+', "tftp transfer"),
        ]

        for pattern, desc in exfil_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return CLIGuardResult(
                    is_safe=False,
                    action=CLIGuardAction.BLOCK,
                    original_command=command,
                    threat_type=CLIThreatCategory.DATA_EXFILTRATION,
                    threat_details=f"Data exfiltration detected: {desc}",
                    matched_patterns=[pattern],
                    parsed_command=parsed,
                )

        return None

    def _check_credential_access(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for credential/sensitive file access"""

        matched = self.sensitive_paths.check(command)
        if matched:
            return CLIGuardResult(
                is_safe=False,
                action=CLIGuardAction.BLOCK,
                original_command=command,
                threat_type=CLIThreatCategory.CREDENTIAL_ACCESS,
                threat_details=f"Sensitive path access detected: {matched}",
                matched_patterns=[matched],
                parsed_command=parsed,
            )

        # Check for additional config protected paths
        for path in self.config.protected_paths:
            if path in command:
                return CLIGuardResult(
                    is_safe=False,
                    action=CLIGuardAction.BLOCK,
                    original_command=command,
                    threat_type=CLIThreatCategory.CREDENTIAL_ACCESS,
                    threat_details=f"Protected path access: {path}",
                    matched_patterns=[path],
                    parsed_command=parsed,
                )

        return None

    def _check_network_access(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for network access when not allowed"""

        if not self.config.allow_network:
            network_commands = [
                "curl", "wget", "nc", "netcat", "ncat",
                "ssh", "scp", "sftp", "rsync", "ftp",
                "telnet", "ping", "traceroute", "nmap",
            ]

            base_cmd = parsed.base_command.lower() if parsed else command.split()[0].lower()

            if base_cmd in network_commands:
                return CLIGuardResult(
                    is_safe=False,
                    action=CLIGuardAction.BLOCK,
                    original_command=command,
                    threat_type=CLIThreatCategory.NETWORK_ACCESS,
                    threat_details=f"Network access not allowed: {base_cmd}",
                    matched_patterns=[base_cmd],
                    parsed_command=parsed,
                )

        return None

    def _check_resource_exhaustion(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for resource exhaustion attacks"""

        exhaustion_patterns = [
            (r':\(\)\{\s*:\|:&\s*\};:', "fork bomb"),
            (r'while\s+true\s*;?\s*do', "infinite loop"),
            (r'yes\s*\|', "yes pipe"),
            (r'/dev/zero', "zero device"),
            (r'/dev/urandom.*dd', "random data generation"),
            (r'cat\s+/dev/zero', "cat zero device"),
        ]

        for pattern, desc in exhaustion_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return CLIGuardResult(
                    is_safe=False,
                    action=CLIGuardAction.BLOCK,
                    original_command=command,
                    threat_type=CLIThreatCategory.RESOURCE_EXHAUSTION,
                    threat_details=f"Resource exhaustion detected: {desc}",
                    matched_patterns=[pattern],
                    parsed_command=parsed,
                )

        return None

    def _check_persistence(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for persistence mechanisms"""

        if not self.config.allow_persistence:
            persistence_patterns = [
                (r'crontab\s+', "crontab modification"),
                (r'>>\s*~/?\.\w+rc', "rc file append"),
                (r'>>\s*~/?\.profile', "profile append"),
                (r'>>\s*~/?\.bash', "bash config append"),
                (r'/etc/init\.d/', "init.d modification"),
                (r'systemctl\s+(enable|start)', "systemd service"),
                (r'launchctl\s+load', "launchd load"),
            ]

            for pattern, desc in persistence_patterns:
                if re.search(pattern, command, re.IGNORECASE):
                    return CLIGuardResult(
                        is_safe=False,
                        action=CLIGuardAction.BLOCK,
                        original_command=command,
                        threat_type=CLIThreatCategory.PERSISTENCE,
                        threat_details=f"Persistence mechanism detected: {desc}",
                        matched_patterns=[pattern],
                        parsed_command=parsed,
                    )

        return None

    def _check_path_traversal(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for path traversal attempts"""

        traversal_patterns = [
            (r'\.\./', "path traversal ../"),
            (r'\.\.\\', "path traversal ..\\"),
            (r'/\.\.', "path traversal /.."),
        ]

        # Count traversal depth
        traversal_count = command.count('../') + command.count('..\\')

        if traversal_count >= self.config.max_path_traversal:
            return CLIGuardResult(
                is_safe=False,
                action=CLIGuardAction.BLOCK,
                original_command=command,
                threat_type=CLIThreatCategory.PATH_TRAVERSAL,
                threat_details=f"Excessive path traversal: {traversal_count} levels",
                matched_patterns=["../"],
                parsed_command=parsed,
            )

        return None

    def _check_obfuscation(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check for obfuscation attempts"""

        obfuscation_patterns = [
            (r'base64\s+-d', "base64 decode"),
            (r'\\x[0-9a-fA-F]{2}', "hex encoding"),
            (r'\$\'\\', "ANSI-C quoting"),
            (r'printf\s+["\']%s', "printf formatting"),
            (r'echo\s+-e\s+["\']\\', "echo escape sequences"),
            (r'xxd\s+-r', "xxd reverse"),
            (r'rev\s*\|', "reverse pipe"),
        ]

        for pattern, desc in obfuscation_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return CLIGuardResult(
                    is_safe=False,
                    action=CLIGuardAction.WARN,
                    original_command=command,
                    threat_type=CLIThreatCategory.SUSPICIOUS_PATTERN,
                    threat_details=f"Obfuscation detected: {desc}",
                    matched_patterns=[pattern],
                    parsed_command=parsed,
                    confidence=0.7,
                )

        return None

    def _check_allowlist(
        self, command: str, parsed: ParsedCommand
    ) -> CLIGuardResult | None:
        """Check against allowlist if configured"""

        if self.config.allowed_commands:
            base_cmd = parsed.base_command if parsed else command.split()[0]

            if base_cmd not in self.config.allowed_commands:
                return CLIGuardResult(
                    is_safe=False,
                    action=CLIGuardAction.BLOCK,
                    original_command=command,
                    threat_type=CLIThreatCategory.SUSPICIOUS_PATTERN,
                    threat_details=f"Command not in allowlist: {base_cmd}",
                    matched_patterns=[base_cmd],
                    parsed_command=parsed,
                )

        return None

    def _is_safe_compound(self, command: str) -> bool:
        """Check if compound command is safe (e.g., cd && ls)"""
        # Simple heuristic: safe if both sides are safe commands
        safe_commands = {"ls", "pwd", "cd", "echo", "cat", "head", "tail", "grep", "find", "wc"}

        parts = re.split(r'[;&|]+', command)
        for part in parts:
            part = part.strip()
            if part:
                base = part.split()[0] if part.split() else ""
                if base not in safe_commands:
                    return False
        return True


# Singleton instance for simple usage
cli_guardrail = CLIGuardrail()
