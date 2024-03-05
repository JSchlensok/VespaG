from pathlib import Path

import logging
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import rich.progress as progress
import scipy
import seaborn as sns
import typer
from dvc.api import params_show
from typing_extensions import Annotated

from src.vespag.style import create_paper_figure
from src.vespag.utils import setup_logger


def main(
        result_files: Annotated[list[Path], typer.Option("-f", "--file")],
        names: Annotated[list[str], typer.Option("-n", "--name")],
        comparison_methods: Annotated[list[str], typer.Option("--comparison")],
        output_dir: Annotated[Path, typer.Option("-o")],
):
    logger = setup_logger()
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(logging.INFO)

    logger.info("Reading in parameters")
    params = params_show()
    protein_reference_file = params["eval"]["proteingym"]["reference_files"][
        "per_protein"
    ]
    dms_reference_file = params["eval"]["proteingym"]["reference_files"]["per_dms"]
    dms_directory = params["eval"]["proteingym"]["dms_directory"]

    logger.info("Reading in DMS metadata")
    # Fetch nicely aggregated per-protein information from GEMME
    nefftab_df = pl.read_csv(protein_reference_file)

    # Fetch per-DMS information (version with modified UniProt_IDs we created ourselves earlier)
    proteingym_reference_df = pl.read_csv(dms_reference_file)

    # Manually map the two variants of P53_HUMAN to the same UniProt_ID
    proteingym_reference_df = proteingym_reference_df.with_columns(
        (
            pl.when(pl.col("UniProt_ID").str.starts_with("P53_HUMAN"))
            .then("P53_HUMAN")
            .otherwise(pl.col("UniProt_ID"))
        ).alias("UniProt_ID")
    )

    # Join with double index to avoid duplication of MSA_len column
    reference_df = proteingym_reference_df.join(
        nefftab_df, on=["MSA_filename", "MSA_len"]
    )

    # Fetch per-DMS performance of SOTA models
    performance_df = pl.read_csv(
        "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/Detailed_performance_files/Substitutions/Spearman/all_models_substitutions_Spearman_DMS_level.csv"
    ).rename({"": "DMS_id"})

    # Get rid of redundant columns that we already have in the reference DF
    performance_df = performance_df.drop(
        ["UniProt_ID", "number_mutants", "Neff_L_category", "Taxon"]
    )

    # Load per-DMS average Spearman from model-specific CSVs
    logger.info("Reading in result files")
    performance_df = performance_df.select(["DMS_id", *comparison_methods])
    for path, name in zip(result_files, names):
        model_results_df = pl.read_csv(path)
        model_results_df = model_results_df.select(["DMS_id", "spearman"])
        model_results_df.columns = ["DMS_id", name]
        performance_df = performance_df.join(model_results_df, on="DMS_id")

    ordered_method_columns = [*comparison_methods, *names]

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme()

    grouped_df = (
        performance_df.join(reference_df.select(["DMS_id", "UniProt_ID"]), on="DMS_id")
        .groupby("UniProt_ID")
        .mean()
        .drop("DMS_id")
    )

    pbar = progress.Progress(*progress.Progress.get_default_columns(), progress.TextColumn("Current DMS: {task.description}"))
    task = pbar.add_task("Creating summary statistics...", total=7)

    # 1 Non-bootstrapped (analogous to ProteinGym)
    # 1a) Average Spearman overall
    pbar.update(task, description="Average Spearman overall")
    df = (
        grouped_df.mean()
        .select([*comparison_methods, *names])
        .transpose(
            include_header=True, header_name="method", column_names=["average_spearman"]
        )
    )
    df.write_csv(output_dir / "spearman_overall.csv")

    fig, _ = create_paper_figure(colorblind=True, context="talk")
    ax = df.sns.barplot(x=pl.col("method"), y=pl.col("average_spearman"))
    ax.bar_label(ax.containers[0], fmt="%.3f")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg. Spearman")
    ax.set_title("ProteinGym overall performance")

    plt.savefig(output_dir / "spearman_overall.png", bbox_inches="tight")

    # 1b) Std. err of differential to best score
    pbar.update(task, advance=1, description="Std. err of differential to best score")
    df = pl.from_records(
        [
            {
                "method": method,
                "stderr": scipy.stats.bootstrap(
                    (grouped_df["TranceptEVE_L"] - grouped_df[method],),
                    np.mean,
                    n_resamples=10000,
                    random_state=42,
                ).standard_error,
            }
            for method in [*comparison_methods, *names]
        ]
    )

    df.write_csv(output_dir / "stderr_of_differential_to_best_score.csv")

    # 1c) Average Spearman by sequencing depth category
    pbar.update(task, advance=1, description="Average Spearman by sequencing depth")
    df = (
        performance_df.join(
            reference_df.select(["DMS_id", "UniProt_ID", "MSA_Neff_L_category"]),
            on="DMS_id",
        )
        .groupby(["UniProt_ID", "MSA_Neff_L_category"], maintain_order=True)
        .mean()
        .groupby("MSA_Neff_L_category", maintain_order=True)
        .mean()
        .select(ordered_method_columns)
        .transpose(
            include_header=True,
            header_name="method",
            column_names=["medium", "low", "high"],
        )
        .select(["method", "low", "medium", "high"])
    )

    df.write_csv(output_dir / "spearman_by_sequencing_depth_category.csv")

    # 1d) Average Spearman by taxon
    pbar.update(task, advance=1, description="Average Spearman by taxon")
    df = (
        performance_df.join(
            reference_df.select(["DMS_id", "UniProt_ID", "taxon"]), on="DMS_id"
        )
        .groupby(["UniProt_ID", "taxon"], maintain_order=True)
        .mean()
        .groupby("taxon", maintain_order=True)
        .mean()
        .select(ordered_method_columns)
        .transpose(
            include_header=True,
            header_name="method",
            column_names=["Virus", "Prokaryote", "Human", "Eukaryote"],
        )
    )
    df.write_csv(output_dir / "spearman_by_taxon.csv")

    # 2) Bootstrapped with 95% non-parametric confidence intervals
    # 2a) Average Spearman overall
    pbar.update(task, advance=1, description="Bootstrapped average Spearman overall")
    df = grouped_df.melt(
        id_vars="UniProt_ID", variable_name="method", value_name="spearman"
    )

    # 2a1) Boxplot with overlaid swarm plot
    fig, _ = create_paper_figure(colorblind=True, context="talk")
    ax = df.sns.swarmplot(
        x="method", y="spearman", hue="method", legend=False, alpha=0.8
    )
    df.sns.boxplot(x="method", y="spearman", showfliers=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg. Spearman")
    ax.set_title("ProteinGym overall performance")
    plt.savefig(
        output_dir / "spearman_overall_bootstrapped_boxplot.png", bbox_inches="tight"
    )

    # 2a2) Bar plot
    fig, _ = create_paper_figure(colorblind=True, context="talk")
    ax = df.sns.barplot(
        x=pl.col("method"),
        y=pl.col("spearman"),
        errorbar=("ci", 95),
        n_boot=1000,
        errwidth=1,
        capsize=0.2,
    )
    ax.bar_label(
        ax.containers[0],
        fmt="%.3f",
        label_type="center",
        fontsize="small",
        color="white",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg. Spearman")
    ax.set_title("ProteinGym overall performance")
    plt.savefig(
        output_dir / "spearman_overall_bootstrapped_barplot.png", bbox_inches="tight"
    )

    # 2a3) Extract bootstrap results from Seaborn
    records = []
    for method, bar, line in zip(ordered_method_columns, ax.patches, ax.lines[::3]):
        mean = bar.get_height()
        ydata = line.get_ydata()
        ci_lower_bound, ci_upper_bound = ydata.min(), ydata.max()
        records.append(
            {
                "method": method,
                "mean": mean,
                "lower_bound": ci_lower_bound,
                "upper_bound": ci_upper_bound,
            }
        )

    df = pl.from_records(records)
    df.write_csv(output_dir / "spearman_overall_bootstrapped.csv")

    # 2b) Average Spearman by sequencing depth category
    pbar.update(task, advance=1, description="Bootstrapped average Spearman by sequencing depth")
    df = (
        performance_df.join(
            reference_df.select(["DMS_id", "UniProt_ID", "MSA_Neff_L_category"]),
            on="DMS_id",
        )
        .drop("DMS_id")
        .groupby(["UniProt_ID", "MSA_Neff_L_category"], maintain_order=True)
        .mean()
        .melt(
            id_vars=["UniProt_ID", "MSA_Neff_L_category"],
            variable_name="method",
            value_name="spearman",
        )
    )

    category_order = ["low", "medium", "high"]

    fig, _ = create_paper_figure(colorblind=True, context="talk")
    ax = df.sns.barplot(
        x="method",
        y="spearman",
        hue="MSA_Neff_L_category",
        hue_order=category_order,
        errorbar=("ci", 95),
        n_boot=1000,
        errwidth=1,
        capsize=0.2,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg. Spearman")
    ax.set_title("ProteinGym performance by sequencing depth category")
    legend = plt.legend(
        title="ProteinGym\nsequencing\ndepth category",
        loc="lower left",
        fontsize="x-small",
    )
    legend.get_title().set_fontsize("x-small")
    plt.savefig(
        output_dir / "spearman_by_sequencing_depth_category_bootstrapped.png",
        bbox_inches="tight",
    )

    records = []
    for method, depth_category, bar, line in zip(
            ordered_method_columns * 3,
            category_order * len(ordered_method_columns),
            list(ax.patches) * 3,
            ax.lines[::3],
    ):
        mean = bar.get_height()
        ydata = line.get_ydata()
        ci_lower_bound, ci_upper_bound = ydata.min(), ydata.max()
        records.append(
            {
                "method": method,
                "depth_category": depth_category,
                "mean": mean,
                "lower_bound": ci_lower_bound,
                "upper_bound": ci_upper_bound,
            }
        )

    df = pl.from_records(records)
    df.write_csv(output_dir / "spearman_by_sequencing_depth_category_bootstrapped.csv")

    # 2c) Average Spearman by taxon
    pbar.update(task, advance=1, description="Bootstrapped average Spearman by taxon")
    df = (
        performance_df.join(
            reference_df.select(["DMS_id", "UniProt_ID", "taxon"]), on="DMS_id"
        )
        .drop("DMS_id")
        .groupby(["UniProt_ID", "taxon"], maintain_order=True)
        .mean()
        .melt(
            id_vars=["UniProt_ID", "taxon"],
            variable_name="method",
            value_name="spearman",
        )
    )

    taxon_order = ["Prokaryote", "Eukaryote", "Human", "Virus"]

    fig, _ = create_paper_figure(colorblind=True, context="talk")
    ax = df.sns.barplot(
        x="method",
        y="spearman",
        hue="taxon",
        hue_order=taxon_order,
        errorbar=("ci", 95),
        n_boot=1000,
        errwidth=1,
        capsize=0.2,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    legend.get_title().set_fontsize("x-small")
    ax.set_xlabel("Method")
    ax.set_ylabel("Avg. Spearman")
    ax.set_title("ProteinGym performance by taxon")
    legend = plt.legend(title="Taxon", loc="lower left", fontsize="x-small")
    plt.savefig(output_dir / "spearman_by_taxon_bootstrapped.png", bbox_inches="tight")

    for method, taxon, bar, line in zip(
            ordered_method_columns * 4,
            taxon_order * len(ordered_method_columns),
            list(ax.patches) * 4,
            ax.lines[::3],
    ):
        mean = bar.get_height()
        ydata = line.get_ydata()
        ci_lower_bound, ci_upper_bound = ydata.min(), ydata.max()
        records.append(
            {
                "method": method,
                "taxon": taxon,
                "mean": mean,
                "lower_bound": ci_lower_bound,
                "upper_bound": ci_upper_bound,
            }
        )

    df = pl.from_records(records)
    df.write_csv(output_dir / "spearman_by_taxon_bootstrapped.csv")


if __name__ == "__main__":
    typer.run(main)
