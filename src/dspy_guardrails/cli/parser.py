"""
Command Parser - Shell command parsing and analysis

Parses shell commands to extract structure, arguments, redirections,
and piped commands for security analysis.
"""

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum, auto


class CommandType(Enum):
    """Types of shell commands"""
    SIMPLE = auto()          # ls -la
    PIPE = auto()            # cat file | grep pattern
    REDIRECT = auto()        # echo "text" > file
    COMPOUND_AND = auto()    # cmd1 && cmd2
    COMPOUND_OR = auto()     # cmd1 || cmd2
    COMPOUND_SEQ = auto()    # cmd1 ; cmd2
    SUBSHELL = auto()        # $(cmd) or `cmd`
    BACKGROUND = auto()      # cmd &
    COMPLEX = auto()         # Multiple types combined


@dataclass
class ParsedCommand:
    """Parsed command structure"""

    raw: str
    base_command: str
    arguments: list[str] = field(default_factory=list)
    command_type: CommandType = CommandType.SIMPLE

    # Components
    input_redirect: str | None = None    # < file
    output_redirect: str | None = None   # > file
    append_redirect: str | None = None   # >> file
    error_redirect: str | None = None    # 2> file

    piped_commands: list["ParsedCommand"] = field(default_factory=list)
    chained_commands: list["ParsedCommand"] = field(default_factory=list)
    subcommands: list[str] = field(default_factory=list)

    # Flags
    is_backgrounded: bool = False
    has_glob: bool = False
    has_variable: bool = False
    has_env_var: bool = False

    # Risk indicators
    risk_score: float = 0.0
    risk_factors: list[str] = field(default_factory=list)


class CommandParser:
    """
    Shell command parser for security analysis

    Parses shell commands to extract:
    - Base command and arguments
    - Redirections (input, output, append, error)
    - Piped commands
    - Chained commands (&&, ||, ;)
    - Subshells ($(), ``)
    - Environment variables

    Examples:
        parser = CommandParser()

        result = parser.parse("ls -la")
        print(result.base_command)  # "ls"
        print(result.arguments)      # ["-la"]

        result = parser.parse("cat file | grep pattern | wc -l")
        print(result.command_type)   # CommandType.PIPE
        print(len(result.piped_commands))  # 2

        result = parser.parse("echo 'hello' > output.txt")
        print(result.output_redirect)  # "output.txt"
    """

    def __init__(self):
        # Regex patterns for parsing
        self._patterns = {
            "subshell_dollar": re.compile(r'\$\(([^)]+)\)'),
            "subshell_backtick": re.compile(r'`([^`]+)`'),
            "env_var": re.compile(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?'),
            "redirect_out": re.compile(r'(?<!2)>\s*([^\s;&|]+)'),
            "redirect_append": re.compile(r'>>\s*([^\s;&|]+)'),
            "redirect_in": re.compile(r'<\s*([^\s;&|]+)'),
            "redirect_err": re.compile(r'2>\s*([^\s;&|]+)'),
            "glob": re.compile(r'[\*\?\[\]]'),
            "background": re.compile(r'&\s*$'),
        }

    def parse(self, command: str) -> ParsedCommand:
        """
        Parse a shell command string

        Args:
            command: Shell command to parse

        Returns:
            ParsedCommand with extracted structure
        """
        if not command or not command.strip():
            return ParsedCommand(raw=command, base_command="")

        command = command.strip()

        # Determine command type
        cmd_type = self._determine_type(command)

        # Parse based on type
        if cmd_type == CommandType.PIPE:
            return self._parse_pipe(command)
        elif cmd_type in (CommandType.COMPOUND_AND, CommandType.COMPOUND_OR, CommandType.COMPOUND_SEQ):
            return self._parse_compound(command, cmd_type)
        else:
            return self._parse_simple(command)

    def _determine_type(self, command: str) -> CommandType:
        """Determine the type of command"""

        # Check for multiple types (complex)
        has_pipe = '|' in command and '||' not in command
        has_and = '&&' in command
        has_or = '||' in command
        has_seq = ';' in command

        type_count = sum([has_pipe, has_and, has_or, has_seq])

        if type_count > 1:
            return CommandType.COMPLEX
        elif has_pipe:
            return CommandType.PIPE
        elif has_and:
            return CommandType.COMPOUND_AND
        elif has_or:
            return CommandType.COMPOUND_OR
        elif has_seq:
            return CommandType.COMPOUND_SEQ
        else:
            return CommandType.SIMPLE

    def _parse_simple(self, command: str) -> ParsedCommand:
        """Parse a simple command"""

        result = ParsedCommand(raw=command, base_command="")

        # Check for background execution
        if self._patterns["background"].search(command):
            result.is_backgrounded = True
            command = command.rstrip('& ')

        # Extract redirections first
        result.output_redirect = self._extract_match(self._patterns["redirect_out"], command)
        result.append_redirect = self._extract_match(self._patterns["redirect_append"], command)
        result.input_redirect = self._extract_match(self._patterns["redirect_in"], command)
        result.error_redirect = self._extract_match(self._patterns["redirect_err"], command)

        # Remove redirections from command for argument parsing
        clean_cmd = command
        for pattern in ["redirect_append", "redirect_out", "redirect_in", "redirect_err"]:
            clean_cmd = self._patterns[pattern].sub('', clean_cmd)

        # Extract subcommands
        for pattern_name in ["subshell_dollar", "subshell_backtick"]:
            for match in self._patterns[pattern_name].finditer(command):
                result.subcommands.append(match.group(1))

        # Check for variables and globs
        result.has_variable = bool(self._patterns["env_var"].search(command))
        result.has_glob = bool(self._patterns["glob"].search(command))

        # Parse arguments
        try:
            tokens = shlex.split(clean_cmd.strip())
            if tokens:
                result.base_command = tokens[0]
                result.arguments = tokens[1:]
        except ValueError:
            # shlex can fail on malformed commands
            tokens = clean_cmd.split()
            if tokens:
                result.base_command = tokens[0]
                result.arguments = tokens[1:]

        # Calculate risk score
        result.risk_score, result.risk_factors = self._calculate_risk(result)

        return result

    def _parse_pipe(self, command: str) -> ParsedCommand:
        """Parse a piped command"""

        # Split by pipe (but not ||)
        parts = re.split(r'\|(?!\|)', command)

        if len(parts) < 2:
            return self._parse_simple(command)

        # Parse first command as the main one
        main = self._parse_simple(parts[0])
        main.command_type = CommandType.PIPE
        main.raw = command

        # Parse piped commands
        for part in parts[1:]:
            piped = self._parse_simple(part.strip())
            main.piped_commands.append(piped)

        return main

    def _parse_compound(self, command: str, cmd_type: CommandType) -> ParsedCommand:
        """Parse a compound command (&&, ||, ;)"""

        if cmd_type == CommandType.COMPOUND_AND:
            parts = command.split('&&')
        elif cmd_type == CommandType.COMPOUND_OR:
            parts = command.split('||')
        else:  # COMPOUND_SEQ
            parts = command.split(';')

        if len(parts) < 2:
            return self._parse_simple(command)

        # Parse first as main
        main = self._parse_simple(parts[0].strip())
        main.command_type = cmd_type
        main.raw = command

        # Parse chained commands
        for part in parts[1:]:
            part = part.strip()
            if part:
                chained = self.parse(part)  # Recursive to handle nested
                main.chained_commands.append(chained)

        return main

    def _extract_match(self, pattern: re.Pattern, text: str) -> str | None:
        """Extract first match from pattern"""
        match = pattern.search(text)
        return match.group(1) if match else None

    def _calculate_risk(self, parsed: ParsedCommand) -> tuple[float, list[str]]:
        """Calculate risk score based on parsed command"""

        score = 0.0
        factors = []

        # High risk base commands
        high_risk_commands = {
            "rm": 0.6, "dd": 0.7, "mkfs": 0.9, "chmod": 0.5,
            "chown": 0.5, "sudo": 0.7, "su": 0.7, "eval": 0.8,
            "exec": 0.6, "source": 0.5, ".": 0.5,
        }

        # Medium risk commands
        medium_risk_commands = {
            "curl": 0.3, "wget": 0.3, "nc": 0.4, "ssh": 0.3,
            "python": 0.3, "perl": 0.3, "ruby": 0.3, "node": 0.3,
            "bash": 0.4, "sh": 0.4, "zsh": 0.4,
        }

        base = parsed.base_command.lower()

        if base in high_risk_commands:
            score += high_risk_commands[base]
            factors.append(f"high_risk_command:{base}")
        elif base in medium_risk_commands:
            score += medium_risk_commands[base]
            factors.append(f"medium_risk_command:{base}")

        # Risk factors
        if parsed.has_variable:
            score += 0.1
            factors.append("has_variable")

        if parsed.subcommands:
            score += 0.2
            factors.append("has_subcommand")

        if parsed.output_redirect:
            score += 0.1
            factors.append("has_output_redirect")

        if parsed.is_backgrounded:
            score += 0.1
            factors.append("is_backgrounded")

        # Check for dangerous arguments
        dangerous_args = ["-rf", "-fr", "--force", "--recursive", "-x", "--execute"]
        for arg in parsed.arguments:
            if arg.lower() in dangerous_args:
                score += 0.2
                factors.append(f"dangerous_arg:{arg}")

        return min(score, 1.0), factors

    def extract_all_commands(self, command: str) -> list[str]:
        """Extract all individual commands from a complex command string"""

        commands = []

        # Split by all separators
        parts = re.split(r'[;&|]+', command)

        for part in parts:
            part = part.strip()
            if part:
                # Also extract subcommands
                for match in self._patterns["subshell_dollar"].finditer(part):
                    commands.append(match.group(1))
                for match in self._patterns["subshell_backtick"].finditer(part):
                    commands.append(match.group(1))

                commands.append(part)

        return commands
