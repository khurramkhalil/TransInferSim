"""
STL Specification class for defining and evaluating Signal Temporal Logic formulas.
"""

from typing import Dict, List, Tuple, Union, Optional
import warnings
from ..utils.logger import get_logger
from ..utils.debug import get_debugger


class STLSpecification:
    """
    Represents a Signal Temporal Logic (STL) temporal formula.

    STL allows expressing temporal properties over continuous and discrete signals,
    with quantitative robustness semantics that indicate "how much" a specification
    is satisfied or violated.

    Supported operators:
    - G (Globally/Always): Property holds at all times in interval
    - F (Finally/Eventually): Property holds at some time in interval
    - U (Until): First property holds until second becomes true
    - & (And): Both properties hold
    - | (Or): At least one property holds
    - -> (Implies): If first holds, then second must hold

    Examples:
        # Latency always below 10ms
        spec = STLSpecification(
            formula="always(latency < 0.01)",
            signal_names=['latency']
        )

        # Utilization eventually exceeds 70%
        spec = STLSpecification(
            formula="eventually(utilization > 0.7)",
            signal_names=['utilization']
        )
    """

    def __init__(
        self,
        formula: str,
        signal_names: List[str],
        time_bounds: Optional[Tuple[float, float]] = None,
        name: Optional[str] = None
    ):
        """
        Initialize an STL specification.

        Args:
            formula: STL formula as a string (e.g., "always(latency < 0.01)")
            signal_names: List of signal variable names used in the formula
            time_bounds: Optional (min_time, max_time) bounds for temporal operators
            name: Optional descriptive name for the specification
        """
        self.formula = formula
        self.signal_names = signal_names
        self.time_bounds = time_bounds if time_bounds else (0, float('inf'))
        self.name = name if name else formula

        # Parsed specification (populated when using rtamt)
        self._spec = None
        self._parsed = False

    def __str__(self):
        return f"STLSpec(name='{self.name}', formula='{self.formula}')"

    def __repr__(self):
        return self.__str__()

    def parse(self):
        """
        Parse the STL formula using rtamt library.

        Returns:
            bool: True if parsing succeeded, False otherwise
        """
        logger = get_logger()
        debugger = get_debugger()

        logger.debug(f"Parsing STL specification: {self.name}")
        logger.trace(f"  Formula: {self.formula}")
        logger.trace(f"  Signals: {self.signal_names}")

        try:
            import rtamt

            # Create discrete-time specification
            spec = rtamt.STLDiscreteTimeSpecification()

            # Declare all signal variables
            for signal_name in self.signal_names:
                spec.declare_var(signal_name, 'float')
                logger.trace(f"  Declared variable: {signal_name}")

            # Set the specification formula
            spec.spec = self.formula

            # Parse the specification
            spec.parse()

            logger.debug(f"  Successfully parsed: {self.name}")

            self._spec = spec
            self._parsed = True
            return True

        except ImportError as e:
            error_msg = "rtamt not installed. STL parsing unavailable. Install with: pip install rtamt"
            logger.error(error_msg)
            debugger.add_error(error_msg, context={'specification': self.name})
            warnings.warn(error_msg)
            return False
        except Exception as e:
            error_msg = f"Failed to parse STL formula '{self.formula}': {e}"
            logger.error(error_msg)
            debugger.add_error(error_msg, context={
                'specification': self.name,
                'formula': self.formula,
                'signals': self.signal_names
            })
            warnings.warn(error_msg)
            return False

    def evaluate(
        self,
        signals: Dict[str, List[Tuple[float, float]]]
    ) -> float:
        """
        Evaluate the STL specification against provided signals.

        Args:
            signals: Dictionary mapping signal names to time-series data
                    Each signal is a list of (time, value) tuples

        Returns:
            float: Robustness degree
                   > 0: Specification is satisfied (larger = more robust)
                   < 0: Specification is violated (more negative = worse violation)
                   = 0: Borderline case

        Raises:
            ValueError: If required signals are missing or rtamt not available
        """
        logger = get_logger()
        debugger = get_debugger()

        logger.debug(f"Evaluating STL specification: {self.name}")

        # Validate specification first
        spec_validation = debugger.validate_specification(self)
        if not spec_validation['valid']:
            error_msg = f"Invalid specification '{self.name}': {spec_validation['issues']}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate signals
        signal_validation = debugger.validate_signals(signals, self.signal_names)
        if not signal_validation['valid']:
            error_msg = f"Signal validation failed for '{self.name}'"
            logger.error(error_msg)
            logger.error(f"  Issues: {signal_validation['issues']}")

            # Try diagnosis
            diagnosis = debugger.diagnose_evaluation_failure(
                self, signals,
                ValueError(f"Missing signals: {signal_validation['missing_signals']}")
            )

            raise ValueError(
                f"Required signal not provided for specification '{self.name}'. "
                f"Missing: {signal_validation['missing_signals']}. "
                f"Available signals: {list(signals.keys())}"
            )

        # Parse if not already done
        if not self._parsed:
            logger.debug(f"  Specification not parsed, parsing now...")
            success = self.parse()
            if not success:
                error_msg = "Cannot evaluate STL specification: parsing failed. Ensure rtamt is installed."
                logger.error(error_msg)
                raise RuntimeError(error_msg)

        # Evaluate using rtamt
        try:
            # Build argument list: alternating signal names and data
            eval_args = []
            for signal_name in self.signal_names:
                eval_args.append(signal_name)
                eval_args.append(signals[signal_name])
                logger.trace(f"  Signal {signal_name}: {len(signals[signal_name])} points")

            robustness = self._spec.evaluate(*eval_args)

            logger.debug(f"  Evaluation result: robustness = {robustness:.6f}")
            logger.debug(f"  Satisfied: {robustness >= 0}")

            return robustness

        except Exception as e:
            error_msg = f"STL evaluation failed for '{self.name}': {e}"
            logger.error(error_msg)

            # Diagnose the failure
            diagnosis = debugger.diagnose_evaluation_failure(self, signals, e)

            raise RuntimeError(error_msg)

    def evaluate_simple(
        self,
        signals: Dict[str, List[Tuple[float, float]]]
    ) -> Dict[str, Union[float, bool]]:
        """
        Simplified evaluation that returns both robustness and satisfaction.

        Args:
            signals: Dictionary mapping signal names to time-series data

        Returns:
            dict: {
                'robustness': float,
                'satisfied': bool,
                'specification': str
            }
        """
        robustness = self.evaluate(signals)

        return {
            'robustness': robustness,
            'satisfied': robustness >= 0,
            'specification': self.formula,
            'name': self.name
        }

    @staticmethod
    def from_simple_predicate(
        signal_name: str,
        operator: str,
        threshold: float,
        temporal_op: str = 'always',
        time_interval: Optional[Tuple[int, int]] = None
    ):
        """
        Create a simple STL specification from basic components.

        Args:
            signal_name: Name of the signal to monitor
            operator: Comparison operator ('<', '<=', '>', '>=', '==', '!=')
            threshold: Threshold value for comparison
            temporal_op: Temporal operator ('always', 'eventually')
            time_interval: Optional (start, end) time interval

        Returns:
            STLSpecification instance

        Example:
            # Create: always(latency < 0.01)
            spec = STLSpecification.from_simple_predicate(
                'latency', '<', 0.01, 'always'
            )
        """
        # Build predicate
        predicate = f"{signal_name} {operator} {threshold}"

        # Add temporal operator
        if time_interval:
            start, end = time_interval
            formula = f"{temporal_op}[{start}:{end}]({predicate})"
        else:
            formula = f"{temporal_op}({predicate})"

        return STLSpecification(
            formula=formula,
            signal_names=[signal_name],
            time_bounds=time_interval,
            name=f"{temporal_op}_{signal_name}_{operator}_{threshold}"
        )
