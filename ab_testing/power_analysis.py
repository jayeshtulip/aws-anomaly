"""Power analysis and sample size calculations for A/B testing."""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger


def plot_power_curve(
    baseline_rate: float = 0.10,
    mde_range: tuple = (0.05, 0.30),
    alpha: float = 0.05,
    n_points: int = 50,
    output_dir: str = "ab_testing/power_analysis"
):
    """
    Plot power curve showing relationship between sample size and statistical power.
    
    Args:
        baseline_rate: Current conversion/success rate
        mde_range: Range of minimum detectable effects to test
        alpha: Significance level
        n_points: Number of points to plot
        output_dir: Output directory for plots
    """
    logger.info("Generating power curve...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mde_values = np.linspace(mde_range[0], mde_range[1], n_points)
    sample_sizes_80 = []
    sample_sizes_90 = []
    
    for mde in mde_values:
        # Calculate for 80% power
        n_80 = calculate_sample_size(baseline_rate, mde, alpha, 0.80)
        sample_sizes_80.append(n_80)
        
        # Calculate for 90% power
        n_90 = calculate_sample_size(baseline_rate, mde, alpha, 0.90)
        sample_sizes_90.append(n_90)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(mde_values * 100, sample_sizes_80, label='80% Power', linewidth=2)
    plt.plot(mde_values * 100, sample_sizes_90, label='90% Power', linewidth=2)
    plt.xlabel('Minimum Detectable Effect (%)', fontsize=12)
    plt.ylabel('Required Sample Size per Variant', fontsize=12)
    plt.title('Sample Size vs Effect Size', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Power vs sample size for fixed MDE
    plt.subplot(1, 2, 2)
    fixed_mde = 0.10
    sample_range = np.linspace(100, 5000, 100)
    powers = []
    
    for n in sample_range:
        power = calculate_power(n, baseline_rate, fixed_mde, alpha)
        powers.append(power)
    
    plt.plot(sample_range, powers, linewidth=2, color='green')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Power')
    plt.axhline(y=0.9, color='orange', linestyle='--', label='90% Power')
    plt.xlabel('Sample Size per Variant', fontsize=12)
    plt.ylabel('Statistical Power', fontsize=12)
    plt.title(f'Statistical Power (MDE={fixed_mde*100:.0f}%)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'power_curve.png', dpi=300, bbox_inches='tight')
    logger.success(f"✅ Power curve saved: {output_path / 'power_curve.png'}")
    plt.close()


def calculate_sample_size(
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> int:
    """Calculate required sample size."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    p_pooled = (p1 + p2) / 2
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = ((z_alpha + z_beta)**2 * 2 * p_pooled * (1 - p_pooled)) / ((p2 - p1)**2)
    return int(np.ceil(n))


def calculate_power(
    sample_size: int,
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05
) -> float:
    """Calculate statistical power for given sample size."""
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)
    p_pooled = (p1 + p2) / 2
    
    effect = abs(p2 - p1)
    se = np.sqrt(2 * p_pooled * (1 - p_pooled) / sample_size)
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z = effect / se
    
    power = 1 - stats.norm.cdf(z_alpha - z)
    return power


def generate_power_table(
    baseline_rate: float = 0.10,
    output_file: str = "ab_testing/power_analysis/sample_size_table.csv"
):
    """Generate sample size table for different scenarios."""
    logger.info("Generating sample size table...")
    
    mde_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    power_values = [0.80, 0.85, 0.90, 0.95]
    alpha = 0.05
    
    results = []
    
    for mde in mde_values:
        for power in power_values:
            n = calculate_sample_size(baseline_rate, mde, alpha, power)
            results.append({
                'MDE (%)': f"{mde*100:.0f}",
                'Power': f"{power*100:.0f}%",
                'Sample Size per Variant': n,
                'Total Sample Size': n * 2
            })
    
    import pandas as pd
    df = pd.DataFrame(results)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.success(f"✅ Sample size table saved: {output_path}")
    return df


if __name__ == "__main__":
    # Generate power analysis
    plot_power_curve(baseline_rate=0.10)
    
    # Generate sample size table
    df = generate_power_table(baseline_rate=0.10)
    print("\nSample Size Requirements:")
    print(df.to_string(index=False))