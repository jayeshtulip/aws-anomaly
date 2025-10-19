"""Data quality validation for log data."""
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger


class DataValidator:
    """Validate data quality for ML pipeline."""
    
    def __init__(self):
        """Initialize data validator."""
        self.validation_results = {}
    
    def check_missing_values(
        self,
        df: pd.DataFrame,
        critical_columns: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, float]]:
        """Check for missing values.
        
        Args:
            df: DataFrame to validate
            critical_columns: Columns that cannot have missing values
            
        Returns:
            Tuple of (is_valid, missing_percentages)
        """
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        
        if critical_columns:
            critical_missing = {
                col: missing_pct.get(col, 0) 
                for col in critical_columns
            }
            
            has_critical_missing = any(
                pct > 0 for pct in critical_missing.values()
            )
            
            if has_critical_missing:
                logger.warning(
                    f"Critical columns have missing values: {critical_missing}"
                )
                return False, missing_pct
        
        logger.info(f"Missing value check passed")
        return True, missing_pct
    
    def check_data_types(
        self,
        df: pd.DataFrame,
        expected_types: Dict[str, str]
    ) -> Tuple[bool, List[str]]:
        """Check if columns have expected data types.
        
        Args:
            df: DataFrame to validate
            expected_types: Dict of column -> expected dtype
            
        Returns:
            Tuple of (is_valid, mismatched_columns)
        """
        mismatched = []
        
        for col, expected_type in expected_types.items():
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                mismatched.append(col)
                continue
            
            actual_type = str(df[col].dtype)
            
            if expected_type not in actual_type:
                logger.warning(
                    f"Column {col}: expected {expected_type}, got {actual_type}"
                )
                mismatched.append(col)
        
        is_valid = len(mismatched) == 0
        
        if is_valid:
            logger.info("Data type check passed")
        
        return is_valid, mismatched
    
    def check_value_ranges(
        self,
        df: pd.DataFrame,
        ranges: Dict[str, Tuple[float, float]]
    ) -> Tuple[bool, Dict[str, int]]:
        """Check if numeric values are within expected ranges.
        
        Args:
            df: DataFrame to validate
            ranges: Dict of column -> (min, max) tuple
            
        Returns:
            Tuple of (is_valid, out_of_range_counts)
        """
        out_of_range = {}
        
        for col, (min_val, max_val) in ranges.items():
            if col not in df.columns:
                continue
            
            col_data = pd.to_numeric(df[col], errors='coerce')
            
            out_of_range_count = (
                (col_data < min_val) | (col_data > max_val)
            ).sum()
            
            if out_of_range_count > 0:
                out_of_range[col] = out_of_range_count
                logger.warning(
                    f"Column {col}: {out_of_range_count} values out of range "
                    f"[{min_val}, {max_val}]"
                )
        
        is_valid = len(out_of_range) == 0
        
        if is_valid:
            logger.info("Value range check passed")
        
        return is_valid, out_of_range
    
    def check_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None
    ) -> Tuple[bool, int]:
        """Check for duplicate rows.
        
        Args:
            df: DataFrame to validate
            subset: Columns to consider for duplicates
            
        Returns:
            Tuple of (is_valid, duplicate_count)
        """
        duplicate_count = df.duplicated(subset=subset).sum()
        
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate rows")
            return False, duplicate_count
        
        logger.info("No duplicates found")
        return True, 0
    
    def check_timestamp_continuity(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        max_gap_minutes: int = 60
    ) -> Tuple[bool, List[Dict]]:
        """Check for gaps in time series data.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            max_gap_minutes: Maximum acceptable gap in minutes
            
        Returns:
            Tuple of (is_valid, list of gaps)
        """
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column {timestamp_col} not found")
            return False, []
        
        df_sorted = df.sort_values(timestamp_col)
        
        # Calculate time differences
        time_diffs = df_sorted[timestamp_col].diff()
        
        # Find large gaps
        large_gaps = time_diffs[
            time_diffs > pd.Timedelta(minutes=max_gap_minutes)
        ]
        
        gaps = []
        for idx in large_gaps.index:
            gaps.append({
                'index': idx,
                'gap_minutes': time_diffs[idx].total_seconds() / 60,
                'timestamp': df_sorted.loc[idx, timestamp_col]
            })
        
        if gaps:
            logger.warning(f"Found {len(gaps)} time gaps > {max_gap_minutes} minutes")
            return False, gaps
        
        logger.info("Timestamp continuity check passed")
        return True, []
    
    def check_categorical_values(
        self,
        df: pd.DataFrame,
        categorical_columns: Dict[str, List[str]]
    ) -> Tuple[bool, Dict[str, List]]:
        """Check if categorical columns have expected values.
        
        Args:
            df: DataFrame to validate
            categorical_columns: Dict of column -> list of valid values
            
        Returns:
            Tuple of (is_valid, unexpected_values)
        """
        unexpected = {}
        
        for col, valid_values in categorical_columns.items():
            if col not in df.columns:
                continue
            
            actual_values = set(df[col].dropna().unique())
            valid_set = set(valid_values)
            
            unexpected_values = actual_values - valid_set
            
            if unexpected_values:
                unexpected[col] = list(unexpected_values)
                logger.warning(
                    f"Column {col}: unexpected values {unexpected_values}"
                )
        
        is_valid = len(unexpected) == 0
        
        if is_valid:
            logger.info("Categorical values check passed")
        
        return is_valid, unexpected
    
    def validate_all(
        self,
        df: pd.DataFrame,
        critical_columns: Optional[List[str]] = None,
        expected_types: Optional[Dict[str, str]] = None,
        value_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        categorical_columns: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, any]:
        """Run all validation checks.
        
        Args:
            df: DataFrame to validate
            critical_columns: Columns that cannot have missing values
            expected_types: Expected data types
            value_ranges: Expected value ranges
            categorical_columns: Valid categorical values
            
        Returns:
            Dictionary with all validation results
        """
        logger.info(f"Starting data validation on DataFrame with shape {df.shape}")
        
        results = {
            'is_valid': True,
            'checks': {}
        }
        
        # Missing values check
        is_valid, missing = self.check_missing_values(df, critical_columns)
        results['checks']['missing_values'] = {
            'passed': is_valid,
            'details': missing
        }
        if not is_valid:
            results['is_valid'] = False
        
        # Data types check
        if expected_types:
            is_valid, mismatched = self.check_data_types(df, expected_types)
            results['checks']['data_types'] = {
                'passed': is_valid,
                'mismatched_columns': mismatched
            }
            if not is_valid:
                results['is_valid'] = False
        
        # Value ranges check
        if value_ranges:
            is_valid, out_of_range = self.check_value_ranges(df, value_ranges)
            results['checks']['value_ranges'] = {
                'passed': is_valid,
                'out_of_range': out_of_range
            }
            if not is_valid:
                results['is_valid'] = False
        
        # Duplicates check
        is_valid, dup_count = self.check_duplicates(df)
        results['checks']['duplicates'] = {
            'passed': is_valid,
            'count': dup_count
        }
        
        # Timestamp continuity check
        if 'timestamp' in df.columns:
            is_valid, gaps = self.check_timestamp_continuity(df)
            results['checks']['timestamp_continuity'] = {
                'passed': is_valid,
                'gaps': gaps
            }
        
        # Categorical values check
        if categorical_columns:
            is_valid, unexpected = self.check_categorical_values(
                df, categorical_columns
            )
            results['checks']['categorical_values'] = {
                'passed': is_valid,
                'unexpected': unexpected
            }
            if not is_valid:
                results['is_valid'] = False
        
        if results['is_valid']:
            logger.success("All validation checks passed!")
        else:
            logger.error("Some validation checks failed")
        
        self.validation_results = results
        return results


if __name__ == "__main__":
    # Example usage
    sample_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'log_level': ['INFO'] * 90 + ['ERROR'] * 10,
        'duration': range(100, 200),
        'service_name': ['api'] * 100
    })
    
    validator = DataValidator()
    
    results = validator.validate_all(
        sample_df,
        critical_columns=['timestamp', 'log_level'],
        expected_types={'duration': 'int'},
        value_ranges={'duration': (0, 10000)},
        categorical_columns={'log_level': ['INFO', 'WARN', 'ERROR', 'DEBUG']}
    )
    
    logger.info(f"Validation results: {results}")