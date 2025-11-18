"""
Diagnostic utilities for STL monitoring.

Provides high-level functions for diagnosing common issues
in STL constraint evaluation and DSE.
"""

from typing import Dict, List, Any
from ..core.specification import STLSpecification
from .debug import get_debugger
from .logger import get_logger


def diagnose_monitor_failure(
    monitor,
    stats_dict: Dict,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Diagnose why an STL monitor evaluation failed.

    Args:
        monitor: STL monitor instance
        stats_dict: Statistics dictionary
        verbose: Print detailed diagnostic information

    Returns:
        Diagnostic report
    """
    debugger = get_debugger()
    logger = get_logger()

    if verbose:
        logger.section("Monitor Failure Diagnosis")

    report = {
        'monitor_type': type(monitor).__name__,
        'num_specifications': len(monitor.specifications),
        'issues_found': []
    }

    # Extract signals
    try:
        signals = monitor.extractor.extract_signals(stats_dict)
        report['num_signals_extracted'] = len(signals)
        report['available_signals'] = list(signals.keys())

        if verbose:
            logger.info(f"Extracted {len(signals)} signals")
            logger.debug(f"  Available signals: {', '.join(list(signals.keys())[:10])}...")

    except Exception as e:
        report['signal_extraction_error'] = str(e)
        report['issues_found'].append(f"Signal extraction failed: {e}")
        logger.error(f"Signal extraction failed: {e}")
        return report

    # Validate each specification
    spec_reports = []
    for i, spec in enumerate(monitor.specifications):
        if verbose:
            logger.subsection(f"Specification {i+1}: {spec.name}")

        spec_report = {
            'spec_name': spec.name,
            'spec_formula': spec.formula,
            'required_signals': spec.signal_names,
            'issues': []
        }

        # Validate specification
        spec_validation = debugger.validate_specification(spec)
        if not spec_validation['valid']:
            spec_report['issues'].extend(spec_validation['issues'])

        # Validate required signals are available
        signal_validation = debugger.validate_signals(signals, spec.signal_names)
        if not signal_validation['valid']:
            spec_report['issues'].extend(signal_validation['issues'])
            spec_report['missing_signals'] = signal_validation['missing_signals']

        # Try to evaluate
        try:
            required_signals = {
                name: signals[name]
                for name in spec.signal_names
                if name in signals
            }

            if len(required_signals) == len(spec.signal_names):
                robustness = spec.evaluate(required_signals)
                spec_report['evaluation_success'] = True
                spec_report['robustness'] = robustness

                if verbose:
                    logger.info(f"  ✓ Evaluation succeeded: robustness = {robustness:.6f}")
            else:
                spec_report['evaluation_success'] = False
                spec_report['issues'].append("Missing required signals")

                if verbose:
                    logger.error(f"  ✗ Evaluation failed: missing signals")

        except Exception as e:
            spec_report['evaluation_success'] = False
            spec_report['evaluation_error'] = str(e)
            spec_report['issues'].append(f"Evaluation error: {e}")

            if verbose:
                logger.error(f"  ✗ Evaluation error: {e}")
                diagnosis = debugger.diagnose_evaluation_failure(spec, signals, e)
                spec_report['diagnosis'] = diagnosis

        spec_reports.append(spec_report)

    report['specification_reports'] = spec_reports

    # Summary
    successful = sum(1 for r in spec_reports if r.get('evaluation_success', False))
    failed = len(spec_reports) - successful

    report['summary'] = {
        'total_specs': len(spec_reports),
        'successful': successful,
        'failed': failed
    }

    if verbose:
        logger.section("Diagnosis Summary")
        logger.info(f"Total specifications: {len(spec_reports)}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")

        if failed > 0:
            logger.error("\nFailed specifications:")
            for r in spec_reports:
                if not r.get('evaluation_success', False):
                    logger.error(f"  - {r['spec_name']}")
                    for issue in r['issues']:
                        logger.error(f"      {issue}")

    return report


def check_signal_availability(
    stats_dict: Dict,
    required_signals: List[str],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Check if required signals are available in statistics.

    Args:
        stats_dict: Statistics dictionary
        required_signals: List of required signal names
        verbose: Print detailed information

    Returns:
        Availability report
    """
    from ..signals.signal_extractor import SignalExtractor

    logger = get_logger()

    if verbose:
        logger.section("Signal Availability Check")

    extractor = SignalExtractor()
    available_signals = extractor.get_available_signals(stats_dict)

    missing = []
    available = []

    for sig_name in required_signals:
        if sig_name in available_signals:
            available.append(sig_name)
        else:
            missing.append(sig_name)

    report = {
        'required_signals': required_signals,
        'available_signals': available,
        'missing_signals': missing,
        'all_available': len(missing) == 0,
        'total_signals_in_stats': len(available_signals)
    }

    if verbose:
        logger.info(f"Total signals in statistics: {len(available_signals)}")
        logger.info(f"Required signals: {len(required_signals)}")
        logger.info(f"  Available: {len(available)}")
        logger.info(f"  Missing: {len(missing)}")

        if missing:
            logger.error("\nMissing signals:")
            for sig in missing:
                logger.error(f"  - {sig}")

            logger.info("\nSuggested alternatives:")
            # Suggest similar signal names
            for sig in missing:
                similar = [s for s in available_signals if sig.lower() in s.lower() or s.lower() in sig.lower()]
                if similar:
                    logger.info(f"  {sig} → Maybe: {', '.join(similar[:3])}")

    return report


def validate_dse_configuration(
    dse,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate DSE configuration before running exploration.

    Args:
        dse: ConstraintBasedDSE instance
        verbose: Print detailed information

    Returns:
        Validation report
    """
    logger = get_logger()
    debugger = get_debugger()

    if verbose:
        logger.section("DSE Configuration Validation")

    report = {
        'model': str(dse.model),
        'num_constraints': len(dse.constraints),
        'data_bitwidth': dse.data_bitwidth,
        'issues': []
    }

    # Validate model
    if not hasattr(dse.model, 'plan'):
        report['issues'].append("Model missing 'plan' attribute")
        logger.error("  Model missing 'plan' attribute")

    # Validate constraints
    for i, spec in enumerate(dse.constraints):
        spec_validation = debugger.validate_specification(spec)
        if not spec_validation['valid']:
            report['issues'].append(f"Constraint {i+1} ({spec.name}) is invalid: {spec_validation['issues']}")
            logger.error(f"  Constraint {i+1} invalid: {spec.name}")

    # Summary
    report['valid'] = len(report['issues']) == 0

    if verbose:
        if report['valid']:
            logger.info("✓ DSE configuration is valid")
        else:
            logger.error(f"✗ Found {len(report['issues'])} issues")

    return report


def print_signal_statistics(
    signals: Dict[str, List],
    max_signals: int = 20
):
    """
    Print statistics for all signals.

    Args:
        signals: Dictionary of signals
        max_signals: Maximum number of signals to display
    """
    debugger = get_debugger()
    logger = get_logger()

    logger.section(f"Signal Statistics ({len(signals)} signals)")

    for i, (name, signal) in enumerate(signals.items()):
        if i >= max_signals:
            logger.info(f"\n... and {len(signals) - max_signals} more signals")
            break

        stats = debugger.inspect_signal(signal, name)

        if not stats['empty']:
            logger.info(f"\n{i+1}. {name}")
            logger.info(f"   Length: {stats['length']}")
            logger.info(f"   Value range: [{stats['value_range'][0]:.6e}, {stats['value_range'][1]:.6e}]")
            logger.info(f"   Mean: {stats['value_mean']:.6e}")
            if stats['constant']:
                logger.warning(f"   ⚠ Constant signal (value = {stats['first_point'][1]:.6e})")
