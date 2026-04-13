from typing import Optional, List

@dataclass
Class TrainingConfig:
    """ Configuration settings for fine-tuning LLMs with bitsandbytes and LoRA.
    
    Attributes:
        model_name (str, optional): Name of the model to be used (as huggingface path). Defaults
            to 'meta-llama/Llama-3.2-1B-Instruct'. 
        model_alias (str, optional): Alias for the model. Defaults to 'LLaMA-3.2-1B'.
        train_samples (int, optional): Number of samples for training. Defaults to 2000.
        eval_samples (int, optional): Number of samples for evaluation. Defaults to 2000.
        test_samples (int, optional): Number of samples for testing. Defaults to 1000.
        num_training_steps (int, optional): Number of training steps. Defaults to 500.
        batch_size (int, optional): Batch size. Defaults to 1.
        gradient_accumulation_steps (int, optional): Maximum number of steps with gradient
            accumulation. Defaults to 4.
        learning_rate (float, optional): Learning rate. Defaults to 2e-4.
        warmup_steps (int, optional): Number of steps for warming up before ??. Defaults to 10.
        max_seq_length (int, optional): TODO: ???. Defaults to 512.
        logging_steps (int, optional): TODO: ???. Defaults to 50.
        save_steps (int, optional): TODO: ???. Defaults to 100.
    """
    model_name: Optional[str] = "meta-llama/Llama-3.2-1B-Instruct",
    model_alias: Optional[str] = "LLaMA-3.2-1B",
    train_samples: Optional[int] = 2000,
    eval_samples: Optional[int] = 2000,
    test_samples: Optional[int] = 1000,
    num_training_steps: Optional[int] = 500,
    batch_size: Optional[int] = 1,
    gradient_accumulation_steps: Optional[int] = 4,
    learning_rate: Optional[float] = 2e-4,
    warmup_steps: Optional[int] = 10,
    max_seq_length: Optional[int] = 512,
    logging_steps: Optional[int] = 50,
    save_steps: Optional[int] = 100


def create_bnb_config():
    """ Create BitsAndBytes configuration for 4-bit quantization.
    
    Returns:
        BitsAndBytesConfig: Configuration for bits and bytes lib usage.
    """
    # TODO: Parse to stadardized code.
    compute_dtype = torch.bfloat16 if CONFIG['use_bf16'] else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )


def save_results(results, training_times):
    """
    Save comprehensive results to multiple formats.
    """
    output_dir = CONFIG['output_dir']

    # 1. Save JSON results
    json_path = os.path.join(output_dir, CONFIG['results_file'])
    results_dict = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'model': CONFIG['model_name'],
            'model_alias': CONFIG['model_alias'],
            'train_samples': CONFIG['train_samples'],
            'num_training_steps': CONFIG['num_training_steps'],
        },
        'results': {}
    }

    for model_name, metrics in results.items():
        metrics.training_time_seconds = training_times.get(model_name, 0)
        results_dict['results'][model_name] = metrics.to_dict()

    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"JSON results saved to {json_path}")

    # 2. Save summary CSV
    csv_path = os.path.join(output_dir, CONFIG['summary_csv'])
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            'Model': model_name,
            'Precision': round(metrics.precision, 4),
            'Recall': round(metrics.recall, 4),
            'F1': round(metrics.f1_score, 4),
            'Size': metrics.eval_size,
            'Time (s)': round(metrics.eval_time_seconds, 2),
            'Train Time (s)': round(training_times.get(model_name, 0), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")

def create_comparison_plot(results):
    """
    Create bar chart comparing model performances.
    """
    output_path = os.path.join(CONFIG['output_dir'], 'model_comparison.png')

    models = list(results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1'][:len(models)]

    metrics_data = [
        (axes[0, 0], [results[m].accuracy for m in models], 'Accuracy'),
        (axes[0, 1], [results[m].precision for m in models], 'Precision'),
        (axes[1, 0], [results[m].recall for m in models], 'Recall'),
        (axes[1, 1], [results[m].f1_score for m in models], 'F1 Score'),
    ]

    for ax, data, title in metrics_data:
        bars = ax.bar(models, data, color=colors, alpha=0.8)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved to {output_path}")







