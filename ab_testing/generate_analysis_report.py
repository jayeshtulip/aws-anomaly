"""Generate comprehensive A/B test analysis report."""
import pandas as pd
import numpy as np
from pathlib import Path
from statistical_analysis import StatisticalAnalyzer, ABTestResult
from loguru import logger
import json
from datetime import datetime


def generate_comprehensive_report(data_file: str = "ab_testing/experiment_data.csv"):
    """Generate full A/B test analysis report."""
    logger.info("Generating comprehensive analysis report...")
    
    # Load data
    df = pd.read_csv(data_file)
    analyzer = StatisticalAnalyzer(alpha=0.05, power=0.8, min_effect_size=0.02)
    
    # Output directory
    output_dir = Path("ab_testing/analysis_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get variants
    variants = df['variant_id'].unique()
    logger.info(f"Analyzing {len(variants)} variants: {variants}")
    
    # Results storage
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(df),
            'variants': variants.tolist(),
            'alpha': analyzer.alpha,
            'power': analyzer.power,
            'min_effect_size': analyzer.min_effect_size
        },
        'variant_summary': {},
        'pairwise_tests': {},
        'recommendations': []
    }
    
    # Variant summaries
    for variant in variants:
        variant_data = df[df['variant_id'] == variant]
        all_results['variant_summary'][variant] = {
            'sample_size': len(variant_data),
            'metrics': {
                'f1_score': {
                    'mean': float(variant_data['f1_score'].mean()),
                    'std': float(variant_data['f1_score'].std()),
                    'median': float(variant_data['f1_score'].median())
                },
                'accuracy': {
                    'mean': float(variant_data['accuracy'].mean()),
                    'std': float(variant_data['accuracy'].std())
                },
                'latency_ms': {
                    'mean': float(variant_data['latency_ms'].mean()),
                    'std': float(variant_data['latency_ms'].std()),
                    'p50': float(variant_data['latency_ms'].quantile(0.5)),
                    'p95': float(variant_data['latency_ms'].quantile(0.95)),
                    'p99': float(variant_data['latency_ms'].quantile(0.99))
                }
            }
        }
    
    # Pairwise comparisons
    metrics_to_test = ['f1_score', 'accuracy', 'latency_ms']
    
    for i, variant_a in enumerate(variants):
        for variant_b in variants[i+1:]:
            pair_key = f"{variant_a}_vs_{variant_b}"
            logger.info(f"Comparing {pair_key}...")
            
            data_a = df[df['variant_id'] == variant_a]
            data_b = df[df['variant_id'] == variant_b]
            
            all_results['pairwise_tests'][pair_key] = {}
            
            for metric in metrics_to_test:
                result = analyzer.t_test(
                    data_a[metric].values,
                    data_b[metric].values,
                    metric_name=metric,
                    variant_a=variant_a,
                    variant_b=variant_b
                )
                
                all_results['pairwise_tests'][pair_key][metric] = {
                    'mean_a': float(result.mean_a),
                    'mean_b': float(result.mean_b),
                    'difference': float(result.mean_a - result.mean_b),
                    'p_value': float(result.p_value),
                    'confidence_interval': [float(result.confidence_interval[0]), float(result.confidence_interval[1])],
                    'effect_size': float(result.effect_size),
                    'is_significant': bool(result.is_significant),
                    'power': float(result.power),
                    'recommendation': str(result.recommendation)
                }
    
    # Overall recommendation
    best_variant = _determine_best_variant(all_results)
    all_results['recommendations'].append({
        'type': 'overall',
        'best_variant': best_variant,
        'rationale': _generate_rationale(all_results, best_variant)
    })
    
    # Save JSON report
    with open(output_dir / "full_analysis_report.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate human-readable report
    _generate_markdown_report(all_results, output_dir / "analysis_report.md")
    
    logger.success(f"Report generated: {output_dir}")
    return all_results


def _determine_best_variant(results: dict) -> str:
    """Determine best variant based on all metrics."""
    variant_scores = {}
    
    for variant in results['metadata']['variants']:
        score = 0
        summary = results['variant_summary'][variant]
        
        # Higher F1 is better
        score += summary['metrics']['f1_score']['mean'] * 100
        
        # Lower latency is better
        score -= summary['metrics']['latency_ms']['mean'] * 0.1
        
        variant_scores[variant] = score
    
    best_variant = max(variant_scores, key=variant_scores.get)
    return best_variant


def _generate_rationale(results: dict, best_variant: str) -> str:
    """Generate rationale for recommendation."""
    summary = results['variant_summary'][best_variant]
    f1 = summary['metrics']['f1_score']['mean']
    latency = summary['metrics']['latency_ms']['mean']
    
    return (
        f"{best_variant} is recommended based on the following:\n"
        f"  - Best F1 Score: {f1:.4f}\n"
        f"  - Latency: {latency:.2f}ms (p95: {summary['metrics']['latency_ms']['p95']:.2f}ms)\n"
        f"  - Statistically significant improvements in key metrics"
    )


def _generate_markdown_report(results: dict, output_file: Path):
    """Generate markdown report."""
    report = []
    report.append("# A/B Test Analysis Report\n\n")
    report.append(f"**Generated:** {results['metadata']['timestamp']}\n\n")
    report.append(f"**Total Samples:** {results['metadata']['total_samples']}\n\n")
    
    report.append("## Executive Summary\n\n")
    rec = results['recommendations'][0]
    report.append(f"**Recommended Variant:** {rec['best_variant']}\n\n")
    report.append(f"**Rationale:**\n\n{rec['rationale']}\n\n")
    
    report.append("## Variant Performance Summary\n\n")
    report.append("| Variant | F1 Score | Accuracy | Latency (p50) | Latency (p95) | Latency (p99) | Sample Size |\n")
    report.append("|---------|----------|----------|---------------|---------------|---------------|-------------|\n")
    
    for variant, summary in results['variant_summary'].items():
        metrics = summary['metrics']
        report.append(
            f"| {variant} | "
            f"{metrics['f1_score']['mean']:.4f} +/- {metrics['f1_score']['std']:.4f} | "
            f"{metrics['accuracy']['mean']:.4f} | "
            f"{metrics['latency_ms']['p50']:.2f}ms | "
            f"{metrics['latency_ms']['p95']:.2f}ms | "
            f"{metrics['latency_ms']['p99']:.2f}ms | "
            f"{summary['sample_size']} |\n"
        )
    
    report.append("\n## Statistical Tests\n\n")
    
    for pair, tests in results['pairwise_tests'].items():
        report.append(f"### {pair.replace('_', ' ')}\n\n")
        
        for metric, result in tests.items():
            report.append(f"#### {metric}\n\n")
            report.append(f"- **Mean A:** {result['mean_a']:.4f}\n")
            report.append(f"- **Mean B:** {result['mean_b']:.4f}\n")
            report.append(f"- **Difference:** {result['difference']:.4f}\n")
            report.append(f"- **95% CI:** [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]\n")
            report.append(f"- **P-value:** {result['p_value']:.6f}\n")
            report.append(f"- **Effect Size (Cohen's d):** {result['effect_size']:.4f}\n")
            report.append(f"- **Statistical Power:** {result['power']:.2%}\n")
            report.append(f"- **Significant:** {'Yes' if result['is_significant'] else 'No'}\n")
            report.append(f"- **Recommendation:** {result['recommendation']}\n\n")
    
    report.append("## Interpretation Guide\n\n")
    report.append("### Effect Size (Cohen's d)\n\n")
    report.append("- **Small:** 0.2 - 0.5\n")
    report.append("- **Medium:** 0.5 - 0.8\n")
    report.append("- **Large:** > 0.8\n\n")
    
    report.append("### P-value\n\n")
    report.append("- **< 0.05:** Statistically significant difference\n")
    report.append("- **>= 0.05:** No significant difference detected\n\n")
    
    report.append("### Statistical Power\n\n")
    report.append("- **< 80%:** Underpowered - may need more samples\n")
    report.append("- **>= 80%:** Adequately powered\n\n")
    
    # Write to file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    logger.info(f"Markdown report saved: {output_file}")


if __name__ == "__main__":
    # Generate synthetic data
    from generate_experiment_data import generate_experiment_data
    
    logger.info("Generating synthetic experiment data...")
    df = generate_experiment_data(n_samples_per_variant=1000)
    
    logger.info("Generating analysis report...")
    results = generate_comprehensive_report()
    
    logger.success("Complete! Check ab_testing/analysis_results/")