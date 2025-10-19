"""Run complete A/B test analysis pipeline."""
from loguru import logger
from generate_experiment_data import generate_experiment_data
from generate_analysis_report import generate_comprehensive_report
from power_analysis import plot_power_curve, generate_power_table
from pathlib import Path


def main():
    """Run complete analysis pipeline."""
    logger.info("="*60)
    logger.info("Starting Complete A/B Test Analysis Pipeline")
    logger.info("="*60)
    
    # Step 1: Generate synthetic experiment data
    logger.info("\n[1/4] Generating experiment data...")
    df = generate_experiment_data(n_samples_per_variant=1000)
    logger.info(f"✓ Generated {len(df)} samples")
    
    # Step 2: Generate comprehensive analysis report
    logger.info("\n[2/4] Running statistical analysis...")
    results = generate_comprehensive_report()
    logger.info("✓ Analysis complete")
    
    # Step 3: Power analysis
    logger.info("\n[3/4] Generating power analysis...")
    plot_power_curve(baseline_rate=0.10)
    df_power = generate_power_table(baseline_rate=0.10)
    logger.info("✓ Power analysis complete")
    
    # Step 4: Summary
    logger.info("\n[4/4] Generating summary...")
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print("\n📊 Variant Performance:")
    for variant, summary in results['variant_summary'].items():
        metrics = summary['metrics']
        print(f"\n{variant}:")
        print(f"  F1 Score:  {metrics['f1_score']['mean']:.4f} ± {metrics['f1_score']['std']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']['mean']:.4f}")
        print(f"  Latency:   {metrics['latency_ms']['mean']:.2f}ms (p95: {metrics['latency_ms']['p95']:.2f}ms)")
        print(f"  Samples:   {summary['sample_size']}")
    
    print("\n🏆 Recommendation:")
    rec = results['recommendations'][0]
    print(f"  Best Variant: {rec['best_variant']}")
    print(f"  {rec['rationale']}")
    
    print("\n📁 Output Files:")
    print("  • ab_testing/experiment_data.csv")
    print("  • ab_testing/analysis_results/full_analysis_report.json")
    print("  • ab_testing/analysis_results/analysis_report.md")
    print("  • ab_testing/power_analysis/power_curve.png")
    print("  • ab_testing/power_analysis/sample_size_table.csv")
    
    print("\n" + "="*60)
    logger.success("✅ COMPLETE! All analyses finished successfully.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()