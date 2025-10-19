"""Statistical analysis for A/B testing experiments."""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ABTestResult:
    """A/B test statistical result."""
    metric_name: str
    variant_a: str
    variant_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    sample_size_a: int
    sample_size_b: int
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    is_significant: bool
    power: float
    recommendation: str


class StatisticalAnalyzer:
    """Perform statistical analysis on A/B test results."""
    
    def __init__(self, alpha: float = 0.05, power: float = 0.8, min_effect_size: float = 0.02):
        """
        Initialize analyzer.
        
        Args:
            alpha: Significance level (default 0.05 for 95% confidence)
            power: Statistical power (default 0.8)
            min_effect_size: Minimum detectable effect size
        """
        self.alpha = alpha
        self.power = power
        self.min_effect_size = min_effect_size
        
        logger.info(f"Initialized analyzer with alpha={alpha}, power={power}, min_effect={min_effect_size}")
    
    def chi_square_test(
        self, 
        observed_a: np.ndarray, 
        observed_b: np.ndarray,
        variant_a: str = "A",
        variant_b: str = "B"
    ) -> Dict[str, any]:
        """
        Perform chi-square test for categorical data.
        
        Args:
            observed_a: Observed frequencies for variant A
            observed_b: Observed frequencies for variant B
            variant_a: Name of variant A
            variant_b: Name of variant B
            
        Returns:
            Dictionary with test results
        """
        # Create contingency table
        contingency_table = np.array([observed_a, observed_b])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramér's V (effect size for chi-square)
        n = np.sum(contingency_table)
        min_dim = min(contingency_table.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * min_dim))
        
        result = {
            'test_type': 'chi_square',
            'variant_a': variant_a,
            'variant_b': variant_b,
            'chi_square': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'effect_size': cramers_v,
            'is_significant': p_value < self.alpha,
            'contingency_table': contingency_table.tolist(),
            'expected_frequencies': expected.tolist()
        }
        
        logger.info(f"Chi-square test: χ²={chi2:.4f}, p={p_value:.4f}, significant={result['is_significant']}")
        return result
    
    def t_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        metric_name: str,
        variant_a: str = "A",
        variant_b: str = "B",
        equal_var: bool = True
    ) -> ABTestResult:
        """
        Perform independent samples t-test.
        
        Args:
            data_a: Data for variant A
            data_b: Data for variant B
            metric_name: Name of metric being tested
            variant_a: Name of variant A
            variant_b: Name of variant B
            equal_var: Assume equal variances (Welch's t-test if False)
            
        Returns:
            ABTestResult object
        """
        # Calculate statistics
        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
        std_a, std_b = np.std(data_a, ddof=1), np.std(data_b, ddof=1)
        n_a, n_b = len(data_a), len(data_b)
        
        # Perform t-test
        t_stat, p_value = ttest_ind(data_a, data_b, equal_var=equal_var)
        
        # Calculate confidence interval for difference in means
        if equal_var:
            pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            se = pooled_std * np.sqrt(1/n_a + 1/n_b)
            df = n_a + n_b - 2
        else:
            se = np.sqrt(std_a**2/n_a + std_b**2/n_b)
            df = ((std_a**2/n_a + std_b**2/n_b)**2) / ((std_a**2/n_a)**2/(n_a-1) + (std_b**2/n_b)**2/(n_b-1))
        
        critical_value = stats.t.ppf(1 - self.alpha/2, df)
        margin_of_error = critical_value * se
        mean_diff = mean_a - mean_b
        ci = (mean_diff - margin_of_error, mean_diff + margin_of_error)
        
        # Calculate Cohen's d (effect size)
        if equal_var:
            cohens_d = (mean_a - mean_b) / pooled_std
        else:
            cohens_d = (mean_a - mean_b) / np.sqrt((std_a**2 + std_b**2) / 2)
        
        # Calculate statistical power
        power = self._calculate_power(n_a, n_b, cohens_d, self.alpha)
        
        # Determine significance
        is_significant = p_value < self.alpha
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            is_significant, mean_diff, ci, cohens_d, power, n_a, n_b
        )
        
        result = ABTestResult(
            metric_name=metric_name,
            variant_a=variant_a,
            variant_b=variant_b,
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            sample_size_a=n_a,
            sample_size_b=n_b,
            test_statistic=t_stat,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=cohens_d,
            is_significant=is_significant,
            power=power,
            recommendation=recommendation
        )
        
        logger.info(f"T-test for {metric_name}: t={t_stat:.4f}, p={p_value:.4f}, d={cohens_d:.4f}")
        return result
    
    def mann_whitney_test(
        self,
        data_a: np.ndarray,
        data_b: np.ndarray,
        metric_name: str,
        variant_a: str = "A",
        variant_b: str = "B"
    ) -> Dict[str, any]:
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        
        Args:
            data_a: Data for variant A
            data_b: Data for variant B
            metric_name: Name of metric
            variant_a: Name of variant A
            variant_b: Name of variant B
            
        Returns:
            Dictionary with test results
        """
        u_stat, p_value = mannwhitneyu(data_a, data_b, alternative='two-sided')
        
        # Calculate effect size (r = Z / sqrt(N))
        n = len(data_a) + len(data_b)
        z_score = stats.norm.ppf(1 - p_value/2)
        effect_size = z_score / np.sqrt(n)
        
        result = {
            'test_type': 'mann_whitney',
            'metric_name': metric_name,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'u_statistic': u_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'median_a': np.median(data_a),
            'median_b': np.median(data_b),
            'is_significant': p_value < self.alpha
        }
        
        logger.info(f"Mann-Whitney test for {metric_name}: U={u_stat:.4f}, p={p_value:.4f}")
        return result
    
    def bayesian_ab_test(
        self,
        successes_a: int,
        trials_a: int,
        successes_b: int,
        trials_b: int,
        prior_alpha: float = 1,
        prior_beta: float = 1
    ) -> Dict[str, any]:
        """
        Perform Bayesian A/B test for binary outcomes.
        
        Args:
            successes_a: Number of successes in variant A
            trials_a: Total trials in variant A
            successes_b: Number of successes in variant B
            trials_b: Total trials in variant B
            prior_alpha: Prior alpha parameter for Beta distribution
            prior_beta: Prior beta parameter for Beta distribution
            
        Returns:
            Dictionary with Bayesian test results
        """
        # Posterior parameters
        alpha_a = prior_alpha + successes_a
        beta_a = prior_beta + trials_a - successes_a
        alpha_b = prior_alpha + successes_b
        beta_b = prior_beta + trials_b - successes_b
        
        # Sample from posterior distributions
        samples = 100000
        posterior_a = np.random.beta(alpha_a, beta_a, samples)
        posterior_b = np.random.beta(alpha_b, beta_b, samples)
        
        # Calculate probability that B > A
        prob_b_better = np.mean(posterior_b > posterior_a)
        
        # Calculate credible intervals
        ci_a = (np.percentile(posterior_a, 2.5), np.percentile(posterior_a, 97.5))
        ci_b = (np.percentile(posterior_b, 2.5), np.percentile(posterior_b, 97.5))
        
        # Expected loss
        expected_loss_a = np.mean(np.maximum(posterior_b - posterior_a, 0))
        expected_loss_b = np.mean(np.maximum(posterior_a - posterior_b, 0))
        
        result = {
            'test_type': 'bayesian',
            'prob_b_better_than_a': prob_b_better,
            'prob_a_better_than_b': 1 - prob_b_better,
            'expected_loss_a': expected_loss_a,
            'expected_loss_b': expected_loss_b,
            'credible_interval_a': ci_a,
            'credible_interval_b': ci_b,
            'posterior_mean_a': np.mean(posterior_a),
            'posterior_mean_b': np.mean(posterior_b),
            'recommendation': 'B' if prob_b_better > 0.95 else 'A' if prob_b_better < 0.05 else 'Inconclusive'
        }
        
        logger.info(f"Bayesian test: P(B>A)={prob_b_better:.4f}, recommendation={result['recommendation']}")
        return result
    
    def sequential_analysis(
        self,
        data_a: List[float],
        data_b: List[float],
        metric_name: str
    ) -> Dict[str, any]:
        """
        Perform sequential analysis to determine when to stop experiment.
        
        Args:
            data_a: Sequential data for variant A
            data_b: Sequential data for variant B
            metric_name: Name of metric
            
        Returns:
            Dictionary with sequential analysis results
        """
        results = []
        p_values = []
        
        # Minimum sample size
        min_n = 100
        
        for i in range(min_n, min(len(data_a), len(data_b))):
            t_stat, p_val = ttest_ind(data_a[:i], data_b[:i])
            p_values.append(p_val)
            
            # Check for early stopping
            if i >= min_n:
                if p_val < self.alpha / 2:  # Bonferroni correction
                    results.append({
                        'sample_size': i,
                        'p_value': p_val,
                        'stop': True,
                        'reason': 'Significant difference detected'
                    })
                    break
        
        if not results:
            results.append({
                'sample_size': min(len(data_a), len(data_b)),
                'p_value': p_values[-1] if p_values else None,
                'stop': False,
                'reason': 'Need more data'
            })
        
        return {
            'test_type': 'sequential',
            'metric_name': metric_name,
            'p_value_history': p_values,
            'final_result': results[-1]
        }
    
    def _calculate_power(
        self,
        n1: int,
        n2: int,
        effect_size: float,
        alpha: float
    ) -> float:
        """Calculate statistical power."""
        # Simplified power calculation
        noncentrality = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
        df = n1 + n2 - 2
        critical_value = stats.t.ppf(1 - alpha/2, df)
        
        # Power = P(reject H0 | H1 is true)
        power = 1 - stats.nct.cdf(critical_value, df, noncentrality)
        return power
    
    def _generate_recommendation(
        self,
        is_significant: bool,
        mean_diff: float,
        ci: Tuple[float, float],
        effect_size: float,
        power: float,
        n_a: int,
        n_b: int
    ) -> str:
        """Generate recommendation based on test results."""
        if not is_significant:
            if power < 0.8:
                return f"INCONCLUSIVE: Need more data (current power: {power:.2%}). Recommend collecting {int(n_a * 1.5)} more samples per variant."
            else:
                return "NO SIGNIFICANT DIFFERENCE: Variants perform similarly. Choose based on other factors (cost, complexity, etc.)"
        
        if abs(effect_size) < self.min_effect_size:
            return f"SIGNIFICANT BUT SMALL EFFECT: Difference is statistically significant but effect size ({effect_size:.4f}) is below minimum threshold ({self.min_effect_size}). Consider practical significance."
        
        better_variant = "A" if mean_diff > 0 else "B"
        return f"SIGNIFICANT IMPROVEMENT: Variant {better_variant} is significantly better (effect size: {abs(effect_size):.4f}). RECOMMEND deploying variant {better_variant}."
    
    def sample_size_calculation(
        self,
        baseline_rate: float,
        mde: float,  # Minimum detectable effect
        alpha: float = None,
        power: float = None
    ) -> int:
        """
        Calculate required sample size per variant.
        
        Args:
            baseline_rate: Current conversion/success rate
            mde: Minimum detectable effect (relative change)
            alpha: Significance level (uses self.alpha if None)
            power: Statistical power (uses self.power if None)
            
        Returns:
            Required sample size per variant
        """
        alpha = alpha or self.alpha
        power = power or self.power
        
        # Calculate effect size
        p1 = baseline_rate
        p2 = baseline_rate * (1 + mde)
        
        # Pooled proportion
        p_pooled = (p1 + p2) / 2
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta)**2 * 2 * p_pooled * (1 - p_pooled)) / ((p2 - p1)**2)
        
        return int(np.ceil(n))


def analyze_ab_experiment(
    results_file: str,
    output_dir: str = "ab_testing/analysis_results"
) -> Dict[str, any]:
    """
    Analyze A/B test experiment results from file.
    
    Args:
        results_file: Path to results CSV
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with all analysis results
    """
    logger.info(f"Analyzing experiment results from {results_file}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_csv(results_file)
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer()
    
    # Perform analyses
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'data_summary': {
            'total_samples': len(df),
            'variants': df['variant_id'].value_counts().to_dict(),
            'metrics': df.columns.tolist()
        },
        'tests': {}
    }
    
    # Group by variant
    variants = df['variant_id'].unique()
    
    # Perform pairwise comparisons
    for i, variant_a in enumerate(variants):
        for variant_b in variants[i+1:]:
            pair_key = f"{variant_a}_vs_{variant_b}"
            all_results['tests'][pair_key] = {}
            
            data_a = df[df['variant_id'] == variant_a]
            data_b = df[df['variant_id'] == variant_b]
            
            # Test each metric
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in data_a.columns and col in data_b.columns:
                    result = analyzer.t_test(
                        data_a[col].values,
                        data_b[col].values,
                        metric_name=col,
                        variant_a=variant_a,
                        variant_b=variant_b
                    )
                    all_results['tests'][pair_key][col] = result.__dict__
    
    # Save results
    with open(output_path / "analysis_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.success(f"✅ Analysis complete! Results saved to {output_path}")
    return all_results


if __name__ == "__main__":
    # Example usage
    analyzer = StatisticalAnalyzer()
    
    # Simulate data
    np.random.seed(42)
    data_a = np.random.normal(100, 15, 1000)
    data_b = np.random.normal(105, 15, 1000)
    
    # Perform t-test
    result = analyzer.t_test(data_a, data_b, "response_time", "Baseline", "Variant")
    
    print(f"\n{'='*60}")
    print(f"A/B Test Results: {result.metric_name}")
    print(f"{'='*60}")
    print(f"Variant A ({result.variant_a}): {result.mean_a:.2f} ± {result.std_a:.2f} (n={result.sample_size_a})")
    print(f"Variant B ({result.variant_b}): {result.mean_b:.2f} ± {result.std_b:.2f} (n={result.sample_size_b})")
    print(f"\nTest Statistic: {result.test_statistic:.4f}")
    print(f"P-value: {result.p_value:.6f}")
    print(f"95% CI: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")
    print(f"Effect Size (Cohen's d): {result.effect_size:.4f}")
    print(f"Statistical Power: {result.power:.2%}")
    print(f"Significant: {result.is_significant}")
    print(f"\nRecommendation: {result.recommendation}")
    print(f"{'='*60}\n")