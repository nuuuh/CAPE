from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_simulation(
    csv_path: Path,
    title: Optional[str] = None,
    compartments: Optional[Sequence[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot a single simulation trajectory.
    
    Parameters
    ----------
    csv_path : Path
        Path to the simulation CSV file.
    title : str, optional
        Plot title. Defaults to filename.
    compartments : list of str, optional
        Subset of compartments to plot. If None, plots all.
    save_path : Path, optional
        If provided, saves the figure to this path.
    show : bool
        Whether to display the plot interactively.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    df = pd.read_csv(csv_path)
    time_col = "week" if "week" in df.columns else "day"
    compartment_cols = [c for c in df.columns if c != time_col]
    
    if compartments:
        compartment_cols = [c for c in compartment_cols if c in compartments]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for comp in compartment_cols:
        ax.plot(df[time_col], df[comp], label=comp, linewidth=2)
    
    time_label = "Time (weeks)" if time_col == "week" else "Time (days)"
    ax.set_xlabel(time_label, fontsize=12)
    ax.set_ylabel("Population", fontsize=12)
    ax.set_title(title or csv_path.stem, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_batch_simulations(
    metadata_path: Path,
    output_dir: Optional[Path] = None,
    max_plots: int = 10,
    show: bool = False,
) -> List[plt.Figure]:
    """Plot multiple simulations from a metadata.json ledger.
    
    Parameters
    ----------
    metadata_path : Path
        Path to metadata.json containing recipe ledger.
    output_dir : Path, optional
        Directory where plots will be saved. Defaults to metadata parent / "plots".
    max_plots : int
        Maximum number of simulations to plot.
    show : bool
        Whether to display plots interactively (blocks execution).
    
    Returns
    -------
    figs : list of Figure
        List of generated figures.
    """
    with metadata_path.open() as f:
        data = json.load(f)
    
    recipes = data.get("recipes", [])
    if not output_dir:
        output_dir = metadata_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figs = []
    for idx, recipe in enumerate(recipes[:max_plots]):
        csv_path = Path(recipe["csv_path"])
        if not csv_path.is_absolute():
            csv_path = metadata_path.parent / csv_path.name
        
        compartments_str = ", ".join(recipe["compartments"])
        title = f"{recipe['id']}: {compartments_str}"
        save_path = output_dir / f"{recipe['id']}.png"
        
        fig = plot_simulation(csv_path, title=title, save_path=save_path, show=show)
        figs.append(fig)
        print(f"[{idx + 1}/{min(len(recipes), max_plots)}] Plotted {recipe['id']}")
    
    print(f"Generated {len(figs)} plots in {output_dir}")
    return figs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize EpiRecipe simulations.")
    parser.add_argument("path", type=Path, help="CSV file or metadata.json to plot.")
    parser.add_argument("--output-dir", type=Path, help="Directory for saved plots.")
    parser.add_argument("--max-plots", type=int, default=10, help="Max plots for batch mode.")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.path.suffix == ".json":
        plot_batch_simulations(args.path, args.output_dir, args.max_plots, args.show)
    elif args.path.suffix == ".csv":
        output_path = None
        if args.output_dir:
            output_path = args.output_dir / f"{args.path.stem}.png"
        plot_simulation(args.path, save_path=output_path, show=args.show or not args.output_dir)
    else:
        raise ValueError(f"Unsupported file type: {args.path.suffix}")


if __name__ == "__main__":
    main()
