from pathlib import Path
from config import KGConfig, RunConfig, DirConfig

from KG import load_knowledge_graph
from rules import convert_all_rules_to_natural_language
import logging
import pandas as pd
from rdflib import Graph

# This ensures the logger is identified by the module path
logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """Configures the root logger to output to the console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def _create_summary(
    kg_config: KGConfig, data_config: DirConfig, graph: Graph, rules_df: pd.DataFrame
) -> None:
    """TODO: work on docs"""
    summary_path = data_config.output_dir / "rules_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{kg_config.kg_name.upper()} CoT2 RULES — SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Rules        : {len(rules_df)}\n")
        f.write(f"Output Directory   : {data_config.output_dir.absolute()}\n")
        f.write(f"KG Triples         : {len(graph)}\n")
        f.write(f"PCA Threshold      : {kg_config.pca_threshold}\n")
        f.write(f"Namespace          : {kg_config.namespace}\n")
        f.write(f"Namespace Prefix   : {kg_config.namespace_prefix}\n\n")
        valid_pca = rules_df["PCA Confidence"].dropna()
        pos = (valid_pca >= kg_config.pca_threshold).sum()
        f.write(f"  Positive (>= {kg_config.pca_threshold}) : {pos}\n")
        neg = (valid_pca < kg_config.pca_threshold).sum()
        nan = rules_df["PCA Confidence"].isna().sum()
        f.write(f"  Negative (<  {kg_config.pca_threshold}) : {neg}\n")
        f.write(f"  Missing PCA                   : {nan}\n")

    logger.info(f"Summary saved to: {summary_path}")

def generate_cots_sparql(config_kg: KGConfig, data_config: DirConfig) -> None:
    """Generate natural language descriptions for each rule in the csv and each
    grounding of the rule in the KG."""

    kg_file_path = data_config.input_dir / config_kg.kg_file
    rules_csv_path = data_config.input_dir / config_kg.rules_csv

    if not kg_file_path.is_file(): # TODO: plan config checks
        raise FileNotFoundError(f"File {kg_file_path} does not exist.")
    if not rules_csv_path.is_file():
        raise FileNotFoundError(f"File {rules_csv_path} does not exist.")
    
    logger.info("NL-instances-CoT2  (SPARQL-based)\n")
    text = (
        f"\tKG file          : {config_kg.kg_file}\n"
        f"\tRules CSV        : {config_kg.rules_csv}\n"
        f"\tNamespace        : {config_kg.namespace}\n"
        f"\tNamespace prefix : {config_kg.namespace_prefix}\n"
        f"\tPCA threshold    : {config_kg.pca_threshold}\n"
    )
    logger.info(text)

    graph = load_knowledge_graph()
    rules_df = pd.read_csv(config_kg.rules_csv, encoding="utf-8")

    convert_all_rules_to_natural_language(
        config_kg=config_kg, 
        data_config=data_config, 
        graph=graph, 
        rules_df=rules_df
    )
    _create_summary(config, graph, rules_df)

if __name__ == "__main__":
    # Set up logger
    setup_logging()

    # Load configuration
    configuration_json = Path("configurations/gen_fr_cots.json")
    config = RunConfig.from_json(configuration_json)

    # Generate CoTs
    generate_cots_sparql(config=config)
