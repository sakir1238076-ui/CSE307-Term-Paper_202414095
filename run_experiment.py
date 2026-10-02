from src.experiment import run_experiment_configs, generate_summaries
from src.plotting import generate_plots


def main():
    print("Starting CSE-307 Track 3 Experiment Pipeline...")
    configs = [
        {"name": "CONFIG_A", "processes": 4, "resources": 2},
        {"name": "CONFIG_B", "processes": 8, "resources": 4},
        {"name": "CONFIG_C", "processes": 12, "resources": 6},
    ]
    print("Executing configurations and calculating metrics...")
    results_df, metrics_df = run_experiment_configs(configs, n_train=1000, n_test=300)
    print("Generating summaries...")
    generate_summaries(metrics_df)
    print("Generating figures...")
    generate_plots(results_df, metrics_df)
    print("Experiment Pipeline Complete.")
    print("Results saved in 'results' directory.")


if __name__ == "__main__":
    main()
