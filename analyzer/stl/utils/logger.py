"""
Logging infrastructure for STL monitoring and debugging.

Provides configurable logging with different verbosity levels
and structured output.
"""

import logging
import sys
from typing import Optional, TextIO
from enum import IntEnum


class LogLevel(IntEnum):
    """STL logging levels."""
    SILENT = 0      # No output
    ERROR = 1       # Only errors
    WARNING = 2     # Errors and warnings
    INFO = 3        # Errors, warnings, and info
    DEBUG = 4       # Everything including debug details
    TRACE = 5       # Maximum verbosity with trace information


class STLLogger:
    """
    Centralized logger for STL operations.

    Provides structured logging with configurable verbosity levels
    and formatted output for better debugging.
    """

    def __init__(
        self,
        name: str = "STL",
        level: LogLevel = LogLevel.WARNING,
        output: Optional[TextIO] = None
    ):
        """
        Initialize STL logger.

        Args:
            name: Logger name (appears in log messages)
            level: Logging level
            output: Output stream (default: sys.stdout)
        """
        self.name = name
        self.level = level
        self.output = output or sys.stdout
        self._indent_level = 0

    def set_level(self, level: LogLevel):
        """Set logging level."""
        self.level = level

    def indent(self):
        """Increase indentation level."""
        self._indent_level += 1

    def dedent(self):
        """Decrease indentation level."""
        self._indent_level = max(0, self._indent_level - 1)

    def reset_indent(self):
        """Reset indentation to zero."""
        self._indent_level = 0

    def _format_message(self, level_name: str, message: str) -> str:
        """Format log message with level and indentation."""
        indent = "  " * self._indent_level
        return f"[{self.name}] {level_name}: {indent}{message}"

    def error(self, message: str, **kwargs):
        """Log error message."""
        if self.level >= LogLevel.ERROR:
            formatted = self._format_message("ERROR", message)
            print(formatted, file=self.output, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if self.level >= LogLevel.WARNING:
            formatted = self._format_message("WARNING", message)
            print(formatted, file=self.output, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message."""
        if self.level >= LogLevel.INFO:
            formatted = self._format_message("INFO", message)
            print(formatted, file=self.output, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if self.level >= LogLevel.DEBUG:
            formatted = self._format_message("DEBUG", message)
            print(formatted, file=self.output, **kwargs)

    def trace(self, message: str, **kwargs):
        """Log trace message (maximum verbosity)."""
        if self.level >= LogLevel.TRACE:
            formatted = self._format_message("TRACE", message)
            print(formatted, file=self.output, **kwargs)

    def section(self, title: str):
        """Log a section header."""
        if self.level >= LogLevel.INFO:
            separator = "=" * 60
            print(f"\n{separator}", file=self.output)
            print(f"  {title}", file=self.output)
            print(f"{separator}\n", file=self.output)

    def subsection(self, title: str):
        """Log a subsection header."""
        if self.level >= LogLevel.INFO:
            separator = "-" * 60
            print(f"\n{separator}", file=self.output)
            print(f"  {title}", file=self.output)
            print(f"{separator}", file=self.output)


# Global logger instance
_global_logger = STLLogger(level=LogLevel.WARNING)


def get_logger() -> STLLogger:
    """Get the global STL logger instance."""
    return _global_logger


def set_log_level(level: LogLevel):
    """Set global logging level."""
    _global_logger.set_level(level)


def enable_debug_logging():
    """Enable debug-level logging globally."""
    set_log_level(LogLevel.DEBUG)


def enable_trace_logging():
    """Enable trace-level logging globally."""
    set_log_level(LogLevel.TRACE)


def disable_logging():
    """Disable all logging."""
    set_log_level(LogLevel.SILENT)
