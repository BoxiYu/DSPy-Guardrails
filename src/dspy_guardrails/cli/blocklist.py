"""
Blocklist Module - Dangerous commands, patterns, and sensitive paths

Contains comprehensive lists of:
- Destructive commands
- Dangerous patterns
- Sensitive file paths
- Suspicious keywords
"""

import re
from dataclasses import dataclass


@dataclass
class BlocklistMatch:
    """Result of a blocklist check"""
    matched: bool
    pattern: str
    category: str
    severity: str  # critical, high, medium, low
    description: str


class DangerousCommands:
    """
    Dangerous command detection

    Maintains lists of dangerous commands categorized by severity and type.
    """

    # Critical: Can destroy system or cause irreversible damage
    CRITICAL_DESTRUCTIVE = [
        # File system destruction
        r'rm\s+(-[rf]+\s+)*/',                           # rm -rf /
        r'rm\s+(-[rf]+\s+)*~',                           # rm -rf ~
        r'rm\s+-[rf]*\s+\*',                             # rm -rf *
        r'rm\s+--no-preserve-root',                      # rm --no-preserve-root
        r'>\s*/dev/sd[a-z]',                             # > /dev/sda
        r'dd\s+.*of=/dev/sd',                            # dd of=/dev/sda
        r'dd\s+.*of=/dev/nvme',                          # dd of=/dev/nvme
        r'mkfs\.',                                        # mkfs.ext4, etc
        r'wipefs\s',                                      # wipefs
        r'shred\s',                                       # shred

        # Fork bomb and resource exhaustion
        r':\(\)\{\s*:\|:&\s*\};:',                       # :(){ :|:& };:
        r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:',    # Variations

        # System damage
        r'mv\s+/\s',                                      # mv / somewhere
        r'chmod\s+-R\s+777\s+/',                         # chmod -R 777 /
        r'chown\s+-R\s+.*\s+/',                          # chown -R user /
    ]

    # High: Can cause significant damage or security issues
    HIGH_RISK = [
        # Dangerous rm patterns
        r'rm\s+-[rf]+',                                   # rm with force/recursive
        r'rm\s+\*',                                       # rm * (current dir)

        # Privilege escalation
        r'sudo\s+rm',                                     # sudo rm
        r'sudo\s+dd',                                     # sudo dd
        r'sudo\s+chmod',                                  # sudo chmod
        r'sudo\s+chown',                                  # sudo chown
        r'sudo\s+su',                                     # sudo su

        # Dangerous operations
        r'dd\s+if=/dev/zero',                            # Write zeros
        r'dd\s+if=/dev/urandom',                         # Write random
        r'>\s*/etc/',                                     # Overwrite /etc files
        r'>\s*/var/log/',                                # Clear logs

        # Network dangers
        r'iptables\s+-F',                                # Flush firewall
        r'ufw\s+disable',                                # Disable firewall
    ]

    # Medium: Potentially dangerous depending on context
    MEDIUM_RISK = [
        r'chmod\s+777',                                   # World writable
        r'chmod\s+\+s',                                   # Setuid
        r'kill\s+-9\s+-1',                               # Kill all processes
        r'killall\s',                                     # Kill by name
        r'pkill\s',                                       # Kill by pattern
        r'shutdown',                                      # System shutdown
        r'reboot',                                        # System reboot
        r'init\s+[0-6]',                                 # Change runlevel
        r'systemctl\s+stop',                             # Stop service
        r'service\s+\w+\s+stop',                         # Stop service
    ]

    def __init__(self):
        # Compile all patterns
        self._critical = [re.compile(p, re.IGNORECASE) for p in self.CRITICAL_DESTRUCTIVE]
        self._high = [re.compile(p, re.IGNORECASE) for p in self.HIGH_RISK]
        self._medium = [re.compile(p, re.IGNORECASE) for p in self.MEDIUM_RISK]

    def check_destructive(self, command: str) -> str | None:
        """Check for destructive commands, return matched pattern or None"""

        for pattern in self._critical:
            if pattern.search(command):
                return pattern.pattern

        return None

    def check_high_risk(self, command: str) -> str | None:
        """Check for high risk commands"""

        for pattern in self._high:
            if pattern.search(command):
                return pattern.pattern

        return None

    def check_medium_risk(self, command: str) -> str | None:
        """Check for medium risk commands"""

        for pattern in self._medium:
            if pattern.search(command):
                return pattern.pattern

        return None

    def check(self, command: str) -> BlocklistMatch:
        """Full check against all levels"""

        # Check critical first
        if matched := self.check_destructive(command):
            return BlocklistMatch(
                matched=True,
                pattern=matched,
                category="destructive",
                severity="critical",
                description="Destructive command that could cause irreversible damage",
            )

        # Check high risk
        if matched := self.check_high_risk(command):
            return BlocklistMatch(
                matched=True,
                pattern=matched,
                category="high_risk",
                severity="high",
                description="High risk command that could cause significant damage",
            )

        # Check medium risk
        if matched := self.check_medium_risk(command):
            return BlocklistMatch(
                matched=True,
                pattern=matched,
                category="medium_risk",
                severity="medium",
                description="Potentially dangerous command depending on context",
            )

        return BlocklistMatch(
            matched=False,
            pattern="",
            category="",
            severity="",
            description="",
        )


class DangerousPatterns:
    """
    Dangerous command patterns detection

    Detects patterns like command injection, encoding tricks, etc.
    """

    # Command injection patterns
    INJECTION_PATTERNS = [
        # Basic injection
        (r';\s*\w+', "semicolon_injection", "Command chaining with semicolon"),
        (r'\|\|', "or_injection", "Logical OR command chaining"),
        (r'&&', "and_injection", "Logical AND command chaining"),

        # Subshell injection
        (r'\$\([^)]+\)', "subshell_dollar", "Command substitution with $()"),
        (r'`[^`]+`', "subshell_backtick", "Command substitution with backticks"),

        # Variable injection
        (r'\$\{[^}]+\}', "variable_expansion", "Variable expansion"),
        (r'\$\w+', "variable_reference", "Variable reference"),

        # Redirect injection
        (r'>\s*/dev/tcp/', "tcp_redirect", "TCP redirect (bash)"),
        (r'>\s*/dev/udp/', "udp_redirect", "UDP redirect (bash)"),
    ]

    # Encoding/Obfuscation patterns
    OBFUSCATION_PATTERNS = [
        (r'base64\s+-d', "base64_decode", "Base64 decode execution"),
        (r'base64\s+--decode', "base64_decode_long", "Base64 decode (long form)"),
        (r'\\x[0-9a-fA-F]{2}', "hex_encoding", "Hex character encoding"),
        (r'\\[0-7]{3}', "octal_encoding", "Octal character encoding"),
        (r'\$\'\\', "ansi_c_quoting", "ANSI-C quoting"),
        (r'printf\s+.*%s', "printf_format", "Printf format string"),
        (r'xxd\s+-r', "xxd_reverse", "xxd hex reverse"),
        (r'od\s+-A', "od_dump", "od octal dump"),
        (r'rev\s*\|', "reverse_pipe", "String reversal in pipe"),
    ]

    # Remote execution patterns
    REMOTE_EXEC_PATTERNS = [
        (r'curl\s+[^\|]+\|\s*(ba)?sh', "curl_pipe_sh", "Curl pipe to shell"),
        (r'wget\s+[^;]+;\s*(ba)?sh', "wget_then_sh", "Wget then shell"),
        (r'wget\s+-O\s*-[^\|]+\|\s*(ba)?sh', "wget_stdout_sh", "Wget stdout to shell"),
        (r'curl\s+-s[^\|]+\|\s*python', "curl_pipe_python", "Curl pipe to python"),
        (r'eval\s+"\$\(curl', "eval_curl", "Eval curl output"),
        (r'source\s+<\(curl', "source_curl", "Source curl process substitution"),
        (r'python\s+-c\s*["\']import\s+urllib', "python_urllib", "Python urllib execution"),
        (r'python\s+-c\s*["\']import\s+requests', "python_requests", "Python requests execution"),
    ]

    def __init__(self):
        self._injection = [(re.compile(p, re.IGNORECASE), n, d) for p, n, d in self.INJECTION_PATTERNS]
        self._obfuscation = [(re.compile(p, re.IGNORECASE), n, d) for p, n, d in self.OBFUSCATION_PATTERNS]
        self._remote = [(re.compile(p, re.IGNORECASE), n, d) for p, n, d in self.REMOTE_EXEC_PATTERNS]

    def check_injection(self, command: str) -> list[tuple[str, str]]:
        """Check for injection patterns, return list of (name, description)"""
        matches = []
        for pattern, name, desc in self._injection:
            if pattern.search(command):
                matches.append((name, desc))
        return matches

    def check_obfuscation(self, command: str) -> list[tuple[str, str]]:
        """Check for obfuscation patterns"""
        matches = []
        for pattern, name, desc in self._obfuscation:
            if pattern.search(command):
                matches.append((name, desc))
        return matches

    def check_remote_exec(self, command: str) -> list[tuple[str, str]]:
        """Check for remote execution patterns"""
        matches = []
        for pattern, name, desc in self._remote:
            if pattern.search(command):
                matches.append((name, desc))
        return matches

    def check_all(self, command: str) -> dict[str, list[tuple[str, str]]]:
        """Check all pattern categories"""
        return {
            "injection": self.check_injection(command),
            "obfuscation": self.check_obfuscation(command),
            "remote_exec": self.check_remote_exec(command),
        }


class SensitivePaths:
    """
    Sensitive file path detection

    Detects access to sensitive system files, credentials, and configs.
    """

    # Credential files
    CREDENTIAL_PATHS = [
        # SSH
        r'~?/?\.ssh/id_',                    # SSH private keys
        r'~?/?\.ssh/authorized_keys',        # SSH authorized keys
        r'~?/?\.ssh/known_hosts',            # SSH known hosts
        r'~?/?\.ssh/config',                 # SSH config

        # AWS
        r'~?/?\.aws/credentials',            # AWS credentials
        r'~?/?\.aws/config',                 # AWS config

        # GCP
        r'~?/?\.config/gcloud/',             # GCloud config
        r'application_default_credentials',  # GCP ADC

        # Azure
        r'~?/?\.azure/',                     # Azure config

        # Kubernetes
        r'~?/?\.kube/config',                # Kubeconfig

        # Generic
        r'\.env$',                           # Environment files
        r'\.env\.',                          # .env.local, .env.production
        r'credentials\.json',                # Credentials file
        r'secrets\.ya?ml',                   # Secrets file
        r'\.pem$',                           # PEM certificates
        r'\.key$',                           # Key files
        r'\.p12$',                           # PKCS12 files
        r'\.pfx$',                           # PFX certificates
    ]

    # System files
    SYSTEM_PATHS = [
        r'/etc/passwd',                      # User database
        r'/etc/shadow',                      # Password hashes
        r'/etc/sudoers',                     # Sudo configuration
        r'/etc/hosts',                       # Host mappings
        r'/etc/ssh/',                        # SSH server config
        r'/etc/ssl/',                        # SSL certificates
        r'/etc/pam\.d/',                     # PAM configuration

        # Logs (potential info disclosure)
        r'/var/log/auth',                    # Auth logs
        r'/var/log/secure',                  # Security logs
        r'/var/log/messages',                # System messages
    ]

    # History files (command history)
    HISTORY_PATHS = [
        r'~?/?\.bash_history',
        r'~?/?\.zsh_history',
        r'~?/?\.history',
        r'~?/?\.mysql_history',
        r'~?/?\.psql_history',
        r'~?/?\.python_history',
        r'~?/?\.node_repl_history',
    ]

    # Application configs that may contain secrets
    APP_CONFIG_PATHS = [
        r'~?/?\.npmrc',                      # NPM config (may have tokens)
        r'~?/?\.pypirc',                     # PyPI config
        r'~?/?\.gitconfig',                  # Git config
        r'~?/?\.netrc',                      # Netrc (credentials)
        r'~?/?\.docker/config\.json',        # Docker config
        r'~?/?\.gradle/gradle\.properties',  # Gradle (may have credentials)
        r'~?/?\.m2/settings\.xml',           # Maven (may have credentials)
    ]

    def __init__(self):
        all_patterns = (
            self.CREDENTIAL_PATHS +
            self.SYSTEM_PATHS +
            self.HISTORY_PATHS +
            self.APP_CONFIG_PATHS
        )
        self._patterns = [re.compile(p, re.IGNORECASE) for p in all_patterns]

        # Category mapping
        self._categories = {}
        for p in self.CREDENTIAL_PATHS:
            self._categories[p] = "credential"
        for p in self.SYSTEM_PATHS:
            self._categories[p] = "system"
        for p in self.HISTORY_PATHS:
            self._categories[p] = "history"
        for p in self.APP_CONFIG_PATHS:
            self._categories[p] = "app_config"

    def check(self, command: str) -> str | None:
        """Check if command accesses sensitive paths"""

        for pattern in self._patterns:
            if pattern.search(command):
                return pattern.pattern

        return None

    def check_detailed(self, command: str) -> list[BlocklistMatch]:
        """Check with detailed categorization"""

        matches = []

        for pattern_str in (self.CREDENTIAL_PATHS + self.SYSTEM_PATHS +
                           self.HISTORY_PATHS + self.APP_CONFIG_PATHS):
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(command):
                category = self._categories.get(pattern_str, "unknown")
                severity = "high" if category in ("credential", "system") else "medium"

                matches.append(BlocklistMatch(
                    matched=True,
                    pattern=pattern_str,
                    category=category,
                    severity=severity,
                    description=f"Access to sensitive {category} path",
                ))

        return matches

    def get_all_patterns(self) -> list[str]:
        """Get all sensitive path patterns"""
        return (
            self.CREDENTIAL_PATHS +
            self.SYSTEM_PATHS +
            self.HISTORY_PATHS +
            self.APP_CONFIG_PATHS
        )
