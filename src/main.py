# Preprocess the knowledge graph
kg_input = os.path.join(CONFIG['data_dir'], CONFIG['kg_file'])
kg_processed = os.path.join(CONFIG['data_dir'], 'train2id_processed.txt')
preprocess_kg_file(kg_input, kg_processed)
print("Knowledge graph preprocessed!")

# Create relation mapping file
relation_file = os.path.join(CONFIG['data_dir'], CONFIG['relation_file'])

relations = set()
with open(kg_processed, 'r') as f:
    n = int(f.readline())
    for line in f:
        parts = line.strip().split()
        if len(parts) == 3:
            relations.add(parts[2])

with open(relation_file, 'w') as f:
    f.write(f"{len(relations)}\n")
    for i, rel in enumerate(sorted(relations)):
        f.write(f"relation_{rel}\t{rel}\n")

print(f"Processed {n} triples with {len(relations)} unique relations")

print("LOADING KNOWLEDGE GRAPH AND SYMBOLIC RULES")

# Load knowledge graph
graph, node_list = load_knowledge_graph(kg_processed)
relation2id = load_relation_mapping(relation_file)
print(f"Graph loaded: {len(graph)} source nodes, {len(node_list)} total nodes")
print(f"Relations loaded: {len(relation2id)}")

# Load symbolic rules
rules_path = os.path.join(CONFIG['data_dir'], CONFIG['rules_dir'])
rules = []
if os.path.exists(rules_path):
    rules = load_all_rules(rules_path)
    print(f"\nTotal rules loaded: {len(rules)}")
else:
    print(f"\nWarning: Rules directory not found: {rules_path}")
    print("Training will proceed without symbolic rules")

print("\n" + "="*70)
print("GENERATING TRAINING DATA")
print("="*70)

# Generate training data with rules
print("\n--- Generating training data WITH rules ---")
train_df_with_rules = generate_training_data_with_rules(
    graph, node_list, relation2id, rules,
    total_samples=CONFIG['train_samples'],
    max_path_length=CONFIG['max_path_length'],
    include_reasoning=True,
    use_rules=True
)

# Generate training data without rules
print("\n--- Generating training data WITHOUT rules ---")
train_df_without_rules = generate_training_data_with_rules(
    graph, node_list, relation2id, rules,
    total_samples=CONFIG['train_samples'],
    max_path_length=CONFIG['max_path_length'],
    include_reasoning=True,
    use_rules=False
)

# Generate test data
print("\n--- Generating test data ---")
test_df = generate_training_data_with_rules(
    graph, node_list, relation2id, rules,
    total_samples=CONFIG['test_samples'],
    max_path_length=CONFIG['max_path_length'],
    include_reasoning=True,
    use_rules=True
)

# Save generated data
train_df_with_rules.to_csv(os.path.join(CONFIG['output_dir'], 'train_data_with_rules.csv'), index=False)
train_df_without_rules.to_csv(os.path.join(CONFIG['output_dir'], 'train_data_without_rules.csv'), index=False)
test_df.to_csv(os.path.join(CONFIG['output_dir'], 'test_data.csv'), index=False)

print(f"\n" + "="*70)
print(f"Training samples WITH rules: {len(train_df_with_rules)}")
print(f"Training samples WITHOUT rules: {len(train_df_without_rules)}")
print(f"Test samples: {len(test_df)}")
print("="*70)

# Step 1: Evaluate base model
if not CONFIG['skip_base_eval']:
    print("\n" + "="*70)
    print("STEP 1: Evaluating BASE MODEL")
    print("="*70)

    base_model, base_tokenizer = load_model(model_name, bnb_config)
    base_results = evaluate_model(base_model, base_tokenizer, test_df, max_samples=CONFIG['eval_samples'])

    print(f"\nBASE MODEL RESULTS:")
    print(f"   Precision: {base_results.precision:.4f}")
    print(f"   Recall: {base_results.recall:.4f}")
    print(f"   F1 Score: {base_results.f1_score:.4f}")
    print(f"   Time: {base_results.eval_time_seconds:.2f}s")

    results['Base Model'] = base_results
    training_times['Base Model'] = 0.0

    del base_model
    torch.cuda.empty_cache()
else:
    print("Skipping base model evaluation (skip_base_eval=True)")

# Step 2: Fine-tune without rules
if not CONFIG['skip_baseline']:
    print("\n" + "="*70)
    print("STEP 2: FINE-TUNING WITHOUT RULES (Baseline)")
    print("="*70)

    ft_model_baseline, ft_tokenizer_baseline = load_model(model_name, bnb_config)

    train_dataset_baseline = Dataset.from_pandas(train_df_without_rules)
    train_dataset_baseline = train_dataset_baseline.map(
        lambda batch: preprocess_batch(batch, ft_tokenizer_baseline),
        batched=True
    )

    output_dir_baseline = os.path.join(CONFIG['output_dir'], "finetuned_baseline")
    ft_model_baseline, train_time_baseline = fine_tune_model(
        ft_model_baseline, ft_tokenizer_baseline,
        train_dataset_baseline, output_dir_baseline,
        num_steps=CONFIG['num_training_steps']
    )

    print("\nEVALUATING FINE-TUNED MODEL (WITHOUT RULES)")
    ft_results_baseline = evaluate_model(
        ft_model_baseline, ft_tokenizer_baseline,
        test_df, max_samples=CONFIG['eval_samples']
    )

    print(f"\nBASELINE RESULTS:")
    print(f"   Precision: {ft_results_baseline.precision:.4f}")
    print(f"   Recall: {ft_results_baseline.recall:.4f}")
    print(f"   F1 Score: {ft_results_baseline.f1_score:.4f}")
    print(f"   Eval Time: {ft_results_baseline.eval_time_seconds:.2f}s")
    print(f"   Train Time: {train_time_baseline:.2f}s")

    results['Baseline'] = ft_results_baseline
    training_times['Baseline'] = train_time_baseline

    del ft_model_baseline
    torch.cuda.empty_cache()
else:
    print("Skipping baseline training (skip_baseline=True)")

# Step 3: Fine.tune with rules (MetaMine)
print("\n" + "="*70)
print("STEP 3: FINE-TUNING WITH RULES (MetaMine)")
print("="*70)

ft_model_rules, ft_tokenizer_rules = load_model(model_name, bnb_config)

train_dataset_rules = Dataset.from_pandas(train_df_with_rules)
train_dataset_rules = train_dataset_rules.map(
    lambda batch: preprocess_batch(batch, ft_tokenizer_rules),
    batched=True
)

output_dir_rules = os.path.join(CONFIG['output_dir'], "finetuned_metamine")
ft_model_rules, train_time_rules = fine_tune_model(
    ft_model_rules, ft_tokenizer_rules,
    train_dataset_rules, output_dir_rules,
    num_steps=CONFIG['num_training_steps']
)

print("\nEVALUATING FINE-TUNED MODEL (WITH RULES / MetaMine)")
ft_results_rules = evaluate_model(
    ft_model_rules, ft_tokenizer_rules,
    test_df, max_samples=CONFIG['eval_samples']
)
 8
print(f"\nMETAMINE RESULTS:")
print(f"   Precision: {ft_results_rules.precision:.4f}")
print(f"   Recall: {ft_results_rules.recall:.4f}")
print(f"   F1 Score: {ft_results_rules.f1_score:.4f}")
print(f"   Eval Time: {ft_results_rules.eval_time_seconds:.2f}s")
print(f"   Train Time: {train_time_rules:.2f}s")

results['METAMINE'] = ft_results_rules
training_times['METAMINE'] = train_time_rules

# Print results table
print_results_table(results, training_times)

# Save all results
save_results(results, training_times)

# Generate comparison plot
if CONFIG['generate_plots']:
    create_comparison_plot(results)

# Display the saved CSV results
print("\n" + "="*70)
print("SAVED RESULTS FILES")
print("="*70)

# Summary CSV
summary_path = os.path.join(CONFIG['output_dir'], CONFIG['summary_csv'])
print(f"\nResults Summary ({summary_path}):")
print("-"*70)
summary_df = pd.read_csv(summary_path)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print(f"All results saved to: {CONFIG['output_dir']}")
print("="*70)


