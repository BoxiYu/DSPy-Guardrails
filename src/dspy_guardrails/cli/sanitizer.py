"""
Command Sanitizer - Sanitize dangerous commands

Attempts to sanitize potentially dangerous commands by removing
or replacing dangerous elements while preserving functionality.
"""

import re
from dataclasses import dataclass, field

from .parser import CommandParser


@dataclass
class SanitizeResult:
    """Result of command sanitization"""

    success: bool
    original: str
    sanitized: str
    changes_made: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    removed_elements: list[str] = field(default_factory=list)


class CommandSanitizer:
    """
    Command sanitizer for CLI operations

    Attempts to make dangerous commands safer by:
    - Removing dangerous flags (-rf, --force)
    - Adding safety flags (--interactive, --verbose)
    - Replacing dangerous patterns
    - Limiting scope of operations

    Examples:
        sanitizer = CommandSanitizer()

        # Remove dangerous flags
        result = sanitizer.sanitize("rm -rf /tmp/test")
        print(result.sanitized)  # "rm -i /tmp/test"

        # Add safety flags
        result = sanitizer.sanitize("chmod 777 file.txt")
        print(result.sanitized)  # "chmod 755 file.txt"
        print(result.warnings)   # ["Changed 777 to 755"]
    """

    def __init__(self):
        self.parser = CommandParser()

        # Dangerous flag replacements
        self._flag_replacements = {
            "-rf": "-ri",           # rm: add interactive
            "-fr": "-ri",           # rm: add interactive
            "--force": "-i",        # replace force with interactive
            "--no-preserve-root": "",  # remove entirely
            "-x": "",               # remove execute flag
        }

        # Permission sanitization
        self._permission_map = {
            "777": "755",
            "666": "644",
            "776": "755",
            "667": "644",
        }

        # Command-specific sanitizers
        self._sanitizers = {
            "rm": self._sanitize_rm,
            "chmod": self._sanitize_chmod,
            "chown": self._sanitize_chown,
            "curl": self._sanitize_curl,
            "wget": self._sanitize_wget,
            "dd": self._sanitize_dd,
        }

    def sanitize(self, command: str) -> SanitizeResult:
        """
        Sanitize a command

        Args:
            command: Command to sanitize

        Returns:
            SanitizeResult with sanitized command
        """
        if not command or not command.strip():
            return SanitizeResult(
                success=True,
                original=command,
                sanitized=command,
            )

        result = SanitizeResult(
            success=True,
            original=command,
            sanitized=command,
        )

        # Parse to get base command
        parsed = self.parser.parse(command)
        base_cmd = parsed.base_command.lower()

        # Apply command-specific sanitizer if available
        if base_cmd in self._sanitizers:
            result = self._sanitizers[base_cmd](result)

        # Apply general sanitization
        result = self._sanitize_general(result)

        return result

    def _sanitize_general(self, result: SanitizeResult) -> SanitizeResult:
        """Apply general sanitization rules"""

        sanitized = result.sanitized

        # Remove dangerous flag patterns
        for dangerous, safe in self._flag_replacements.items():
            if dangerous in sanitized:
                sanitized = sanitized.replace(dangerous, safe)
                result.changes_made.append(f"Replaced '{dangerous}' with '{safe}'")
                result.removed_elements.append(dangerous)

        # Remove command injection attempts (be careful not to break valid commands)
        # Only remove if clearly malicious
        injection_patterns = [
            (r';\s*rm\s', "; "),           # ; rm -> ;
            (r';\s*curl\s+[^\|]+\|\s*sh', "; "),  # ; curl | sh
            (r'\|\|\s*rm\s', "|| "),       # || rm
        ]

        for pattern, replacement in injection_patterns:
            if re.search(pattern, sanitized, re.IGNORECASE):
                sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
                result.changes_made.append(f"Removed injection pattern: {pattern}")
                result.warnings.append("Potential command injection was removed")

        result.sanitized = sanitized.strip()
        return result

    def _sanitize_rm(self, result: SanitizeResult) -> SanitizeResult:
        """Sanitize rm commands"""

        sanitized = result.sanitized

        # Never allow rm on root
        if re.search(r'rm\s+.*\s+/$', sanitized) or re.search(r'rm\s+/$', sanitized):
            result.success = False
            result.warnings.append("Cannot sanitize: rm on root directory")
            return result

        # Replace -rf with -ri (interactive)
        sanitized = re.sub(r'\s+-rf\b', ' -ri', sanitized)
        sanitized = re.sub(r'\s+-fr\b', ' -ri', sanitized)

        if sanitized != result.sanitized:
            result.changes_made.append("Changed rm to interactive mode")

        # Add --verbose if not present
        if '-v' not in sanitized and '--verbose' not in sanitized:
            sanitized = sanitized.replace('rm ', 'rm -v ', 1)
            result.changes_made.append("Added verbose flag")

        result.sanitized = sanitized
        return result

    def _sanitize_chmod(self, result: SanitizeResult) -> SanitizeResult:
        """Sanitize chmod commands"""

        sanitized = result.sanitized

        # Replace overly permissive permissions
        for dangerous, safe in self._permission_map.items():
            pattern = rf'\b{dangerous}\b'
            if re.search(pattern, sanitized):
                sanitized = re.sub(pattern, safe, sanitized)
                result.changes_made.append(f"Changed permission {dangerous} to {safe}")
                result.warnings.append(f"Reduced permission from {dangerous} to {safe}")

        # Remove setuid/setgid
        if '+s' in sanitized:
            sanitized = sanitized.replace('+s', '')
            result.changes_made.append("Removed setuid/setgid flag")
            result.warnings.append("Setuid/setgid flag was removed")

        # Remove -R on sensitive directories
        if '-R' in sanitized or '--recursive' in sanitized:
            if any(d in sanitized for d in ['/', '/etc', '/var', '/usr', '/home']):
                sanitized = sanitized.replace('-R', '').replace('--recursive', '')
                result.changes_made.append("Removed recursive flag for sensitive directory")
                result.warnings.append("Recursive chmod on sensitive directory blocked")

        result.sanitized = sanitized
        return result

    def _sanitize_chown(self, result: SanitizeResult) -> SanitizeResult:
        """Sanitize chown commands"""

        sanitized = result.sanitized

        # Don't allow chown to root
        if 'root:' in sanitized or ':root' in sanitized:
            result.success = False
            result.warnings.append("Cannot sanitize: chown to root")
            return result

        # Remove -R on sensitive directories
        if '-R' in sanitized or '--recursive' in sanitized:
            if any(d in sanitized for d in ['/', '/etc', '/var', '/usr']):
                sanitized = sanitized.replace('-R', '').replace('--recursive', '')
                result.changes_made.append("Removed recursive flag")
                result.warnings.append("Recursive chown on sensitive directory blocked")

        result.sanitized = sanitized
        return result

    def _sanitize_curl(self, result: SanitizeResult) -> SanitizeResult:
        """Sanitize curl commands"""

        sanitized = result.sanitized

        # Remove pipe to shell
        pipe_to_shell = re.search(r'\|\s*(ba)?sh', sanitized, re.IGNORECASE)
        if pipe_to_shell:
            sanitized = sanitized[:pipe_to_shell.start()]
            result.changes_made.append("Removed pipe to shell")
            result.warnings.append("Curl pipe to shell was removed - download file instead")

        # Add output to file if downloading
        if '-o' not in sanitized.lower() and '-O' not in sanitized:
            # Check if it looks like a download
            if 'http' in sanitized:
                sanitized = sanitized.rstrip() + ' -O'
                result.changes_made.append("Added -O flag to save to file")

        result.sanitized = sanitized
        return result

    def _sanitize_wget(self, result: SanitizeResult) -> SanitizeResult:
        """Sanitize wget commands"""

        sanitized = result.sanitized

        # Remove execution after download
        exec_pattern = re.search(r';\s*(ba)?sh\s+', sanitized, re.IGNORECASE)
        if exec_pattern:
            sanitized = sanitized[:exec_pattern.start()]
            result.changes_made.append("Removed execution after download")
            result.warnings.append("Script execution after wget was removed")

        # Remove -O - (stdout) if piped to shell
        if '-O -' in sanitized or '-O-' in sanitized:
            if '|' in sanitized:
                sanitized = re.sub(r'-O\s*-', '-O downloaded_file', sanitized)
                result.changes_made.append("Changed output from stdout to file")

        result.sanitized = sanitized
        return result

    def _sanitize_dd(self, result: SanitizeResult) -> SanitizeResult:
        """Sanitize dd commands"""

        sanitized = result.sanitized

        # Block writing to block devices
        if re.search(r'of=/dev/(sd|hd|nvme|vd)', sanitized):
            result.success = False
            result.warnings.append("Cannot sanitize: dd to block device")
            return result

        # Block if=/dev/zero or if=/dev/urandom with of=/
        if re.search(r'if=/dev/(zero|urandom).*of=/', sanitized):
            result.success = False
            result.warnings.append("Cannot sanitize: dd from zero/urandom to system path")
            return result

        # Add status=progress for visibility
        if 'status=' not in sanitized:
            sanitized = sanitized.rstrip() + ' status=progress'
            result.changes_made.append("Added status=progress")

        result.sanitized = sanitized
        return result

    def can_sanitize(self, command: str) -> bool:
        """Check if a command can be sanitized"""
        result = self.sanitize(command)
        return result.success

    def get_safe_alternative(self, command: str) -> str | None:
        """Get a safe alternative for a dangerous command"""

        result = self.sanitize(command)
        if result.success:
            return result.sanitized

        # Provide alternatives for common dangerous commands
        parsed = self.parser.parse(command)
        base_cmd = parsed.base_command.lower()

        alternatives = {
            "rm": "Consider using 'trash' or 'mv' to a temp directory instead",
            "dd": "Consider using 'cp' or a GUI tool for safer file operations",
            "chmod": "Consider using more restrictive permissions (644 for files, 755 for dirs)",
            "sudo": "Consider running as a limited user with specific permissions",
        }

        return alternatives.get(base_cmd)
