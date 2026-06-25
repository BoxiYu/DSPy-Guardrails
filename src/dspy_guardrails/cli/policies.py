"""
CLI Security Policies - Configuration and execution policies

Provides configurable security policies for CLI guardrails including
sandbox levels, execution policies, and custom rules.
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class SandboxLevel(Enum):
    """
    Sandbox security levels

    PERMISSIVE: Allow most commands, block only critical threats
    STANDARD: Block dangerous commands and patterns
    STRICT: Only allow explicitly whitelisted commands
    PARANOID: Block everything suspicious, require explicit approval
    """
    PERMISSIVE = auto()
    STANDARD = auto()
    STRICT = auto()
    PARANOID = auto()


class ExecutionPolicy(Enum):
    """
    Execution policies for detected threats

    BLOCK: Stop execution completely
    SANITIZE: Attempt to sanitize and execute safe version
    WARN: Log warning but allow execution
    AUDIT: Allow execution, log for review
    PROMPT: Ask for explicit confirmation
    """
    BLOCK = "block"
    SANITIZE = "sanitize"
    WARN = "warn"
    AUDIT = "audit"
    PROMPT = "prompt"


@dataclass
class CLISecurityConfig:
    """
    CLI Security Configuration

    Configures the behavior of CLI guardrails.

    Examples:
        # Default configuration
        config = CLISecurityConfig()

        # Strict mode
        config = CLISecurityConfig(
            sandbox_level=SandboxLevel.STRICT,
            allow_network=False,
            allow_sudo=False,
            allowed_commands=["ls", "cat", "echo", "pwd", "cd"],
        )

        # Custom protected paths
        config = CLISecurityConfig(
            protected_paths=[
                "~/.ssh",
                "~/.aws",
                "/etc/passwd",
                ".env",
            ],
        )

        # Enterprise configuration
        config = CLISecurityConfig(
            sandbox_level=SandboxLevel.STANDARD,
            execution_policy=ExecutionPolicy.AUDIT,
            audit_log_path="/var/log/cli-guardrail.log",
            max_command_length=1000,
        )
    """

    # Core settings
    sandbox_level: SandboxLevel = SandboxLevel.STANDARD
    execution_policy: ExecutionPolicy = ExecutionPolicy.BLOCK

    # Permission settings
    allow_network: bool = True          # Allow curl, wget, etc
    allow_sudo: bool = False            # Allow sudo commands
    allow_persistence: bool = False     # Allow crontab, rc file edits
    allow_background: bool = True       # Allow background processes (&)
    allow_redirects: bool = True        # Allow > >> < redirects
    allow_pipes: bool = True            # Allow | pipes
    allow_subshells: bool = True        # Allow $() and ``
    allow_variables: bool = True        # Allow $VAR references

    # Path restrictions
    protected_paths: list[str] = field(default_factory=lambda: [
        "~/.ssh",
        "~/.aws",
        "~/.kube",
        ".env",
        "/etc/passwd",
        "/etc/shadow",
    ])

    # Working directory restrictions
    allowed_directories: list[str] = field(default_factory=list)  # Empty = all allowed
    blocked_directories: list[str] = field(default_factory=lambda: [
        "/",
        "/etc",
        "/var",
        "/usr",
        "/bin",
        "/sbin",
        "/root",
    ])

    # Command restrictions
    allowed_commands: list[str] = field(default_factory=list)  # Empty = all allowed (with blocklist)
    blocked_commands: list[str] = field(default_factory=lambda: [
        "rm", "dd", "mkfs", "fdisk", "parted",
        "sudo", "su", "doas",
        "chmod", "chown", "chgrp",
        "iptables", "ufw", "firewall-cmd",
        "shutdown", "reboot", "init", "systemctl",
    ])

    # Argument restrictions
    blocked_arguments: list[str] = field(default_factory=lambda: [
        "-rf", "-fr", "--force",
        "--no-preserve-root",
        "-x", "--execute",
    ])

    # Limits
    max_command_length: int = 2000      # Maximum command string length
    max_arguments: int = 50             # Maximum number of arguments
    max_pipe_depth: int = 5             # Maximum pipe chain depth
    max_path_traversal: int = 3         # Maximum ../ levels allowed

    # Logging
    audit_log_path: str | None = None
    log_blocked_commands: bool = True
    log_allowed_commands: bool = False

    # Advanced
    custom_blocklist_patterns: list[str] = field(default_factory=list)
    custom_allowlist_patterns: list[str] = field(default_factory=list)

    def is_command_allowed(self, command: str) -> bool:
        """Check if a base command is allowed by policy"""

        # Check blocked commands
        if command in self.blocked_commands:
            return False

        # Check allowlist if specified
        if self.allowed_commands and command not in self.allowed_commands:
            return False

        return True


# Predefined configurations
class PredefinedPolicies:
    """Predefined security policies for common use cases"""

    @staticmethod
    def development() -> CLISecurityConfig:
        """Development mode - permissive but safe"""
        return CLISecurityConfig(
            sandbox_level=SandboxLevel.PERMISSIVE,
            allow_network=True,
            allow_sudo=False,
            execution_policy=ExecutionPolicy.WARN,
        )

    @staticmethod
    def production() -> CLISecurityConfig:
        """Production mode - standard security"""
        return CLISecurityConfig(
            sandbox_level=SandboxLevel.STANDARD,
            allow_network=True,
            allow_sudo=False,
            execution_policy=ExecutionPolicy.BLOCK,
            log_blocked_commands=True,
        )

    @staticmethod
    def strict() -> CLISecurityConfig:
        """Strict mode - high security"""
        return CLISecurityConfig(
            sandbox_level=SandboxLevel.STRICT,
            allow_network=False,
            allow_sudo=False,
            allow_persistence=False,
            allow_subshells=False,
            execution_policy=ExecutionPolicy.BLOCK,
            log_blocked_commands=True,
            log_allowed_commands=True,
        )

    @staticmethod
    def read_only() -> CLISecurityConfig:
        """Read-only mode - only allow read operations"""
        return CLISecurityConfig(
            sandbox_level=SandboxLevel.STRICT,
            allow_network=False,
            allow_sudo=False,
            allow_redirects=False,
            allowed_commands=[
                "ls", "cat", "head", "tail", "less", "more",
                "grep", "find", "wc", "du", "df",
                "pwd", "whoami", "hostname", "date",
                "echo", "printf",
            ],
            execution_policy=ExecutionPolicy.BLOCK,
        )

    @staticmethod
    def minimal() -> CLISecurityConfig:
        """Minimal mode - extremely restricted"""
        return CLISecurityConfig(
            sandbox_level=SandboxLevel.PARANOID,
            allow_network=False,
            allow_sudo=False,
            allow_persistence=False,
            allow_background=False,
            allow_redirects=False,
            allow_pipes=False,
            allow_subshells=False,
            allow_variables=False,
            allowed_commands=["ls", "pwd", "echo"],
            execution_policy=ExecutionPolicy.BLOCK,
        )

    @staticmethod
    def ci_cd() -> CLISecurityConfig:
        """CI/CD mode - suitable for build pipelines"""
        return CLISecurityConfig(
            sandbox_level=SandboxLevel.STANDARD,
            allow_network=True,  # Need for package downloads
            allow_sudo=False,
            allow_persistence=False,
            blocked_commands=[
                "rm",  # Careful with cleanup
                "sudo", "su",
                "chmod", "chown",
                "shutdown", "reboot",
            ],
            execution_policy=ExecutionPolicy.BLOCK,
        )

    @staticmethod
    def data_science() -> CLISecurityConfig:
        """Data science mode - allow common DS tools"""
        return CLISecurityConfig(
            sandbox_level=SandboxLevel.STANDARD,
            allow_network=True,  # Need for data downloads
            allow_sudo=False,
            allowed_commands=[
                # Basic shell
                "ls", "cat", "head", "tail", "grep", "find", "wc",
                "pwd", "cd", "mkdir", "cp", "mv",
                # Data tools
                "python", "python3", "pip", "conda",
                "jupyter", "ipython",
                "curl", "wget",  # Data download
                # Version control
                "git",
            ],
            execution_policy=ExecutionPolicy.BLOCK,
        )


# Default policy based on sandbox level
DEFAULT_POLICIES = {
    SandboxLevel.PERMISSIVE: PredefinedPolicies.development(),
    SandboxLevel.STANDARD: PredefinedPolicies.production(),
    SandboxLevel.STRICT: PredefinedPolicies.strict(),
    SandboxLevel.PARANOID: PredefinedPolicies.minimal(),
}
