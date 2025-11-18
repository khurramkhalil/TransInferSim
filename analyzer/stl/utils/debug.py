"""
Debugging utilities for STL monitoring and analysis.

Provides tools for inspecting STL specifications, signals,
and identifying issues in constraint evaluation.
"""

from typing import Dict, List, Tuple, Any, Optional
import traceback
from .logger import get_logger


class STLDebugger:
    """
    Comprehensive debugging utility for STL operations.

    Helps identify and diagnose issues in:
    - Specification parsing
    - Signal extraction
    - Robustness computation
    - Constraint evaluation
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize debugger.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.logger = get_logger()
        self.errors = []
        self.warnings = []

    def clear(self):
        """Clear accumulated errors and warnings."""
        self.errors = []
        self.warnings = []

    def add_error(self, message: str, context: Optional[Dict] = None):
        """
        Record an error with context.

        Args:
            message: Error message
            context: Additional context information
        """
        error = {
            'message': message,
            'context': context or {},
            'traceback': traceback.format_stack()
        }
        self.errors.append(error)
        self.logger.error(message)
        if context and self.verbose:
            self.logger.debug(f"  Context: {context}")

    def add_warning(self, message: str, context: Optional[Dict] = None):
        """
        Record a warning with context.

        Args:
            message: Warning message
            context: Additional context information
        """
        warning = {
            'message': message,
            'context': context or {}
        }
        self.warnings.append(warning)
        self.logger.warning(message)
        if context and self.verbose:
            self.logger.debug(f"  Context: {context}")

    def validate_specification(self, spec) -> Dict[str, Any]:
        """
        Validate an STL specification.

        Args:
            spec: STLSpecification instance

        Returns:
            Validation report with issues found
        """
        self.logger.debug(f"Validating specification: {spec.name}")

        issues = []
        is_valid = True

        # Check formula
        if not spec.formula:
            issues.append("Formula is empty")
            is_valid = False
        else:
            self.logger.trace(f"  Formula: {spec.formula}")

        # Check signal names
        if not spec.signal_names:
            issues.append("No signal names specified")
            is_valid = False
        else:
            self.logger.trace(f"  Signals: {spec.signal_names}")

        # Check if parsed
        if hasattr(spec, '_parsed') and not spec._parsed:
            issues.append("Specification not parsed (call parse() or evaluate())")
            self.logger.debug("  Specification needs parsing")

        return {
            'valid': is_valid,
            'issues': issues,
            'spec_name': spec.name,
            'formula': spec.formula,
            'signals': spec.signal_names
        }

    def validate_signals(
        self,
        signals: Dict[str, List[Tuple[float, float]]],
        required_signals: List[str]
    ) -> Dict[str, Any]:
        """
        Validate that required signals are present and well-formed.

        Args:
            signals: Dictionary of available signals
            required_signals: List of required signal names

        Returns:
            Validation report
        """
        self.logger.debug("Validating signals...")

        issues = []
        missing_signals = []
        malformed_signals = []

        # Check for missing signals
        for sig_name in required_signals:
            if sig_name not in signals:
                missing_signals.append(sig_name)
                issues.append(f"Missing required signal: {sig_name}")
                self.logger.error(f"  Missing signal: {sig_name}")
            else:
                # Validate signal format
                signal = signals[sig_name]

                if not isinstance(signal, list):
                    malformed_signals.append(sig_name)
                    issues.append(f"Signal {sig_name} is not a list")
                    self.logger.error(f"  Signal {sig_name} has wrong type: {type(signal)}")
                elif len(signal) == 0:
                    malformed_signals.append(sig_name)
                    issues.append(f"Signal {sig_name} is empty")
                    self.logger.warning(f"  Signal {sig_name} is empty")
                else:
                    # Check first element format
                    first_elem = signal[0]
                    if not isinstance(first_elem, tuple) or len(first_elem) != 2:
                        malformed_signals.append(sig_name)
                        issues.append(f"Signal {sig_name} elements not in (time, value) format")
                        self.logger.error(f"  Signal {sig_name} malformed: {first_elem}")
                    else:
                        self.logger.trace(f"  Signal {sig_name}: {len(signal)} points, first={first_elem}")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'missing_signals': missing_signals,
            'malformed_signals': malformed_signals,
            'available_signals': list(signals.keys()),
            'required_signals': required_signals
        }

    def inspect_signal(
        self,
        signal: List[Tuple[float, float]],
        name: str = "signal"
    ) -> Dict[str, Any]:
        """
        Inspect a signal and return detailed statistics.

        Args:
            signal: Time-series signal
            name: Signal name for reporting

        Returns:
            Dictionary with signal statistics
        """
        self.logger.debug(f"Inspecting signal: {name}")

        if not signal:
            self.logger.error(f"  Signal {name} is empty!")
            return {
                'name': name,
                'empty': True,
                'error': 'Signal is empty'
            }

        times = [t for t, _ in signal]
        values = [v for _, v in signal]

        stats = {
            'name': name,
            'empty': False,
            'length': len(signal),
            'time_range': (min(times), max(times)),
            'value_range': (min(values), max(values)),
            'value_mean': sum(values) / len(values),
            'constant': len(set(values)) == 1,
            'first_point': signal[0],
            'last_point': signal[-1]
        }

        self.logger.trace(f"  Length: {stats['length']}")
        self.logger.trace(f"  Time range: {stats['time_range']}")
        self.logger.trace(f"  Value range: {stats['value_range']}")
        self.logger.trace(f"  Mean: {stats['value_mean']:.6f}")
        self.logger.trace(f"  Constant: {stats['constant']}")

        if stats['constant']:
            self.logger.debug(f"  WARNING: Signal {name} is constant (all values = {values[0]})")

        return stats

    def diagnose_evaluation_failure(
        self,
        spec,
        signals: Dict[str, List[Tuple[float, float]]],
        error: Exception
    ) -> Dict[str, Any]:
        """
        Diagnose why an STL evaluation failed.

        Args:
            spec: STL specification that failed
            signals: Signals that were provided
            error: Exception that was raised

        Returns:
            Diagnostic report
        """
        self.logger.section(f"Diagnosing Evaluation Failure: {spec.name}")

        diagnosis = {
            'spec_name': spec.name,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'likely_causes': [],
            'suggested_fixes': []
        }

        # Validate specification
        spec_validation = self.validate_specification(spec)
        if not spec_validation['valid']:
            diagnosis['likely_causes'].append("Invalid specification")
            diagnosis['spec_issues'] = spec_validation['issues']
            self.logger.error("  Specification validation failed")

        # Validate signals
        signal_validation = self.validate_signals(signals, spec.signal_names)
        if not signal_validation['valid']:
            diagnosis['likely_causes'].append("Missing or malformed signals")
            diagnosis['signal_issues'] = signal_validation['issues']
            self.logger.error("  Signal validation failed")

        # Inspect each required signal
        signal_stats = {}
        for sig_name in spec.signal_names:
            if sig_name in signals:
                signal_stats[sig_name] = self.inspect_signal(signals[sig_name], sig_name)

        diagnosis['signal_stats'] = signal_stats

        # Error-specific diagnosis
        error_msg = str(error).lower()

        if 'rtamt' in error_msg or 'import' in error_msg:
            diagnosis['likely_causes'].append("rtamt library not installed or not importable")
            diagnosis['suggested_fixes'].append("Install rtamt: pip install rtamt")

        if 'parse' in error_msg:
            diagnosis['likely_causes'].append("Formula parsing failed")
            diagnosis['suggested_fixes'].append(f"Check formula syntax: {spec.formula}")

        if 'signal' in error_msg or 'key' in error_msg:
            diagnosis['likely_causes'].append("Signal name mismatch")
            diagnosis['suggested_fixes'].append(
                f"Required: {spec.signal_names}, Available: {list(signals.keys())}"
            )

        # Print diagnosis
        self.logger.error("Diagnosis Summary:")
        self.logger.error(f"  Error: {diagnosis['error_type']}: {diagnosis['error_message']}")

        if diagnosis['likely_causes']:
            self.logger.error("  Likely causes:")
            for cause in diagnosis['likely_causes']:
                self.logger.error(f"    - {cause}")

        if diagnosis['suggested_fixes']:
            self.logger.error("  Suggested fixes:")
            for fix in diagnosis['suggested_fixes']:
                self.logger.error(f"    - {fix}")

        return diagnosis

    def print_summary(self):
        """Print summary of all errors and warnings."""
        self.logger.section("Debug Summary")

        print(f"Total Errors: {len(self.errors)}")
        print(f"Total Warnings: {len(self.warnings)}")
        print()

        if self.errors:
            print("ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error['message']}")
                if error['context']:
                    print(f"   Context: {error['context']}")
            print()

        if self.warnings:
            print("WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning['message']}")
                if warning['context']:
                    print(f"   Context: {warning['context']}")
            print()


# Global debugger instance
_global_debugger = STLDebugger(verbose=False)


def get_debugger() -> STLDebugger:
    """Get the global STL debugger instance."""
    return _global_debugger


def enable_debugging():
    """Enable verbose debugging globally."""
    _global_debugger.verbose = True
    from .logger import enable_debug_logging
    enable_debug_logging()


def disable_debugging():
    """Disable verbose debugging."""
    _global_debugger.verbose = False
