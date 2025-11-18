"""
Offline STL monitor for post-simulation constraint checking.

Evaluates STL specifications after simulation completes, using
the statistics dictionary from accelerator.get_statistics().
"""

from typing import Dict, List
from .base_monitor import BaseSTLMonitor
from ..core.specification import STLSpecification
from ..signals.signal_extractor import SignalExtractor
from ..utils.logger import get_logger
from ..utils.debug import get_debugger


class OfflineSTLMonitor(BaseSTLMonitor):
    """
    Post-simulation STL monitoring.

    Evaluates STL constraints after a simulation run completes, using
    the aggregate statistics dictionary.

    This is the simplest form of monitoring and doesn't require any
    changes to the simulation engine.

    Example:
        # Define constraints
        constraints = [
            PerformanceConstraints.max_latency(10e-3),
            PerformanceConstraints.min_utilization(0.7)
        ]

        # Create monitor
        monitor = OfflineSTLMonitor(constraints, hw_arch=accelerator)

        # Run simulation
        analyzer.run_simulation_analysis()
        stats = accelerator.get_statistics()

        # Evaluate constraints
        results = monitor.evaluate(stats)
    """

    def __init__(
        self,
        specifications: List[STLSpecification],
        hw_arch=None
    ):
        """
        Initialize offline STL monitor.

        Args:
            specifications: List of STL specifications to monitor
            hw_arch: Optional hardware architecture (for signal extraction)
        """
        super().__init__(specifications)
        self.hw_arch = hw_arch
        self.extractor = SignalExtractor(hw_arch)

    def evaluate(self, stats_dict: Dict) -> List[Dict]:
        """
        Evaluate all STL specifications on completed simulation.

        Args:
            stats_dict: Statistics dictionary from accelerator.get_statistics()

        Returns:
            List of evaluation results, one per specification.
            Each result contains:
            - specification: Formula string
            - name: Specification name
            - robustness: Robustness degree (>= 0: satisfied, < 0: violated)
            - satisfied: Boolean satisfaction flag
            - signals_used: List of signal names used

        Raises:
            ValueError: If required signals are not available
        """
        logger = get_logger()
        debugger = get_debugger()

        logger.info(f"Starting offline STL monitoring ({len(self.specifications)} specifications)")

        # Extract all signals from statistics
        try:
            all_signals = self.extractor.extract_signals(stats_dict)
            logger.info(f"  Extracted {len(all_signals)} signals from statistics")
            logger.debug(f"  Available signals: {', '.join(list(all_signals.keys())[:10])}...")
        except Exception as e:
            error_msg = f"Failed to extract signals from statistics: {e}"
            logger.error(error_msg)
            debugger.add_error(error_msg, context={'num_specs': len(self.specifications)})
            raise RuntimeError(error_msg)

        # Evaluate each specification
        results = []
        for i, spec in enumerate(self.specifications):
            logger.debug(f"\nEvaluating specification {i+1}/{len(self.specifications)}: {spec.name}")

            try:
                # Get required signals for this specification
                required_signals = {
                    name: all_signals[name]
                    for name in spec.signal_names
                }

                # Evaluate specification
                robustness = spec.evaluate(required_signals)

                # Store result
                result = {
                    'specification': spec.formula,
                    'name': spec.name,
                    'robustness': robustness,
                    'satisfied': robustness >= 0,
                    'signals_used': spec.signal_names,
                    'spec_object': spec
                }

                results.append(result)

                if robustness >= 0:
                    logger.info(f"  ✓ {spec.name}: SATISFIED (ρ = {robustness:.6f})")
                else:
                    logger.warning(f"  ✗ {spec.name}: VIOLATED (ρ = {robustness:.6f})")

            except KeyError as e:
                error_msg = f"Required signal not available for specification '{spec.name}': {e}"
                logger.error(error_msg)
                logger.error(f"  Available signals: {list(all_signals.keys())}")
                debugger.add_error(error_msg, context={
                    'specification': spec.name,
                    'required_signals': spec.signal_names,
                    'available_signals': list(all_signals.keys())
                })
                raise ValueError(error_msg)
            except Exception as e:
                error_msg = f"Failed to evaluate specification '{spec.name}': {e}"
                logger.error(error_msg)

                # Run diagnostic
                from ..utils.diagnostics import diagnose_monitor_failure
                diagnosis = debugger.diagnose_evaluation_failure(spec, all_signals, e)

                debugger.add_error(error_msg, context={
                    'specification': spec.name,
                    'error_type': type(e).__name__
                })
                raise RuntimeError(error_msg)

        # Store results and mark as evaluated
        self.results = results
        self._evaluated = True

        # Summary
        satisfied_count = sum(1 for r in results if r['satisfied'])
        violated_count = len(results) - satisfied_count

        logger.info(f"\n=== Monitoring Summary ===")
        logger.info(f"  Total specifications: {len(results)}")
        logger.info(f"  Satisfied: {satisfied_count}")
        logger.info(f"  Violated: {violated_count}")

        return results

    def evaluate_and_report(self, stats_dict: Dict) -> Dict:
        """
        Evaluate specifications and return a comprehensive report.

        Args:
            stats_dict: Statistics dictionary from accelerator.get_statistics()

        Returns:
            Dictionary containing:
            - results: List of individual specification results
            - summary: Summary statistics
            - all_satisfied: Boolean flag
        """
        results = self.evaluate(stats_dict)
        summary = self.get_summary()

        return {
            'results': results,
            'summary': summary,
            'all_satisfied': summary['all_satisfied'],
            'violations': self.get_violations(),
            'satisfied': self.get_satisfied()
        }

    def get_available_signals(self, stats_dict: Dict) -> List[str]:
        """
        Get list of available signals from statistics.

        Useful for debugging and understanding what signals can be monitored.

        Args:
            stats_dict: Statistics dictionary

        Returns:
            List of available signal names
        """
        return self.extractor.get_available_signals(stats_dict)

    def validate_specifications(self, stats_dict: Dict) -> Dict[str, bool]:
        """
        Check if all required signals are available for each specification.

        Args:
            stats_dict: Statistics dictionary

        Returns:
            Dictionary mapping specification names to availability flags
        """
        available_signals = set(self.extractor.get_available_signals(stats_dict))

        validation = {}
        for spec in self.specifications:
            required_signals = set(spec.signal_names)
            is_valid = required_signals.issubset(available_signals)
            validation[spec.name] = {
                'valid': is_valid,
                'required_signals': list(required_signals),
                'missing_signals': list(required_signals - available_signals)
            }

        return validation
