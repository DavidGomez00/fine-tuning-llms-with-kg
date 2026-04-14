from typing import Dict

import pandas as pd

from configuration import CoTGenerationConfig, ExperimentConfig
from KG import (
    create_relation_mapping,
    load_id2relation_mapping,
    load_knowledge_graph,
    parse_kg,
)
from rules import (
    generate_CoTs,
    load_rules_from_path,
)
from model import load_model


def set_up(config: ExperimentConfig) -> None:
    """Set up method to load and generate data before training."""

    # Parse KG file
    kg_input = config.data.data_dir / config.data.kg_file
    kg_processed = config.data.data_dir / config.data.kg_file_processed
    parse_kg(kg_input, kg_processed)

    # Create relation file
    relations_file = config.data.data_dir / config.data.relation_file
    create_relation_mapping(kg_processed, relations_file)


def generate_data(config: ExperimentConfig) -> Dict[str, pd.DataFrame]:
    """Generates necessary data to perform an experiment."""

    # Load data from files
    kg_processed = config.data.data_dir / config.data.kg_file_processed
    graph, node_list = load_knowledge_graph(kg_processed)

    relations_file = config.data.data_dir / config.data.relation_file
    id2relation = load_id2relation_mapping(relations_file)

    rules_path = config.data.data_dir / config.data.rules_dir
    rules = load_rules_from_path(rules_path)

    # Define configurations for generating CoTs
    dataset_configs = {
        "train_data_without_rules": CoTGenerationConfig(samples=2000, use_rules=False),
        "train_data_with_rules": CoTGenerationConfig(samples=2000),
        "test_data_with_rules": CoTGenerationConfig(samples=1000),
    }

    # Generate datasets
    datasets: Dict[str, pd.DataFrame] = {}

    for split_name, data_config in dataset_configs.items():
        datasets[split_name] = generate_CoTs(
            graph=graph,
            node_list=node_list,
            id2relation=id2relation,
            rules=rules,
            config=data_config,
        )

    # Save generated data
    for split_name, dataset_df in datasets.items():
        output_path = config.data.output_dir / f"{split_name}.csv"
        dataset_df.to_csv(output_path, index=False)

    return datasets


def eval_base_model(config: ExperimentConfig) -> None:
    """Evaluates the base model without fine-tuning."""
    
    base_model, base_tokenizer = load_model(config)
    base_results = evaluate_model(
        base_model, base_tokenizer, test_df, max_samples=CONFIG["eval_samples"]
    )

    print(f"\nBASE MODEL RESULTS:")
    print(f"   Precision: {base_results.precision:.4f}")
    print(f"   Recall: {base_results.recall:.4f}")
    print(f"   F1 Score: {base_results.f1_score:.4f}")
    print(f"   Time: {base_results.eval_time_seconds:.2f}s")

    results["Base Model"] = base_results
    training_times["Base Model"] = 0.0

    del base_model
    torch.cuda.empty_cache()

def main() -> None:
    """Main method to execute an experiment."""

    config = ExperimentConfig()

    if config.run_settings.skip_base_eval:
        print("Skipping base model evaluation.")
    else:
        eval_base_model(config)
    


if __name__ == "__main__":
    main()






# Step 2: Fine-tune without rules
if not CONFIG["skip_baseline"]:
    print("\n" + "=" * 70)
    print("STEP 2: FINE-TUNING WITHOUT RULES (Baseline)")
    print("=" * 70)

    ft_model_baseline, ft_tokenizer_baseline = load_model(model_name, bnb_config)

    train_dataset_baseline = Dataset.from_pandas(train_df_without_rules)
    train_dataset_baseline = train_dataset_baseline.map(
        lambda batch: preprocess_batch(batch, ft_tokenizer_baseline), batched=True
    )

    output_dir_baseline = os.path.join(CONFIG["output_dir"], "finetuned_baseline")
    ft_model_baseline, train_time_baseline = fine_tune_model(
        ft_model_baseline,
        ft_tokenizer_baseline,
        train_dataset_baseline,
        output_dir_baseline,
        num_steps=CONFIG["num_training_steps"],
    )

    print("\nEVALUATING FINE-TUNED MODEL (WITHOUT RULES)")
    ft_results_baseline = evaluate_model(
        ft_model_baseline,
        ft_tokenizer_baseline,
        test_df,
        max_samples=CONFIG["eval_samples"],
    )

    print(f"\nBASELINE RESULTS:")
    print(f"   Precision: {ft_results_baseline.precision:.4f}")
    print(f"   Recall: {ft_results_baseline.recall:.4f}")
    print(f"   F1 Score: {ft_results_baseline.f1_score:.4f}")
    print(f"   Eval Time: {ft_results_baseline.eval_time_seconds:.2f}s")
    print(f"   Train Time: {train_time_baseline:.2f}s")

    results["Baseline"] = ft_results_baseline
    training_times["Baseline"] = train_time_baseline

    del ft_model_baseline
    torch.cuda.empty_cache()
else:
    print("Skipping baseline training (skip_baseline=True)")

# Step 3: Fine.tune with rules (MetaMine)
ft_model_rules, ft_tokenizer_rules = load_model(model_name, bnb_config)

train_dataset_rules = Dataset.from_pandas(train_df_with_rules)
train_dataset_rules = train_dataset_rules.map(
    lambda batch: preprocess_batch(batch, ft_tokenizer_rules), batched=True
)

output_dir_rules = os.path.join(CONFIG["output_dir"], "finetuned_metamine")
ft_model_rules, train_time_rules = fine_tune_model(
    ft_model_rules,
    ft_tokenizer_rules,
    train_dataset_rules,
    output_dir_rules,
    num_steps=CONFIG["num_training_steps"],
)


ft_results_rules = evaluate_model(
    ft_model_rules, ft_tokenizer_rules, test_df, max_samples=CONFIG["eval_samples"]
)

results["METAMINE"] = ft_results_rules
training_times["METAMINE"] = train_time_rules

# Print results table
print_results_table(results, training_times)

# Save all results
save_results(results, training_times)

# Generate comparison plot
if CONFIG["generate_plots"]:
    create_comparison_plot(results)


# Summary CSV
summary_path = os.path.join(CONFIG["output_dir"], CONFIG["summary_csv"])
print(f"\nResults Summary ({summary_path}):")
print("-" * 70)
summary_df = pd.read_csv(summary_path)
print(summary_df.to_string(index=False))
"""
