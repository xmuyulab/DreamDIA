"""
╔═════════════════════════════════════════════════════╗
║                     DreamDIA.py                     ║
╠═════════════════════════════════════════════════════╣
║         Description: DreamDIA user interfaces       ║
╠═════════════════════════════════════════════════════╣
║                Author: Mingxuan Gao                 ║
║           Contact: mingxuan.gao@utoronto.ca         ║
╚═════════════════════════════════════════════════════╝
"""

import os

import click
import tensorflow as tf

from dream_score import dream_score
from dream_prophet import dream_prophet
from art import logo

# Set TensorFlow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Configure TensorFlow logger to suppress warnings
tf.get_logger().setLevel('ERROR')

# Configure CLI context settings
CONTEXT_SETTINGS = dict(help_option_names = ['-h', '--help'], max_content_width = 120)

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo('DreamDIA Version 3.2.0')
    click.echo(logo)
    ctx.exit()

@click.group(context_settings = CONTEXT_SETTINGS)
@click.option('--version', is_flag = True, callback = print_version, expose_value = False, is_eager = True, help = "Print version and exit.")
def dreamdia():
    """DreamDIA: A DIA data analysis software suite. """
    pass

@dreamdia.command(context_settings = CONTEXT_SETTINGS)
@click.option("--file_dir", required = True, type = click.Path(exists = True), help = "Directory that contains only DIA data files. Centroided .mzXML, .mzML or .raw files from Thermo Fisher equipments are supported. (For Linux systems, `mono` tool has to be installed for the supporting of .raw files. https://www.mono-project.com/download/stable/#download-lin)")
@click.option("--lib", required = True, type = click.Path(exists = True), help = "File name of the spectral library. .tsv or .csv formats are supported.")
@click.option("--out", required = True, type = click.Path(exists = False), help = "Directory for output files.")
@click.option("--n_threads", default = 16, show_default = True, type = int, help = "Number of threads.")
@click.option("--seed", default = 123, show_default = True, type = int, help = "Random seed.")
@click.option("--mz_unit", default = "Da", show_default = True, type = click.Choice(['Da', 'ppm']), help = "m/z unit for m/z range and tolerance settings.")
@click.option("--mz_min", default = "99", show_default = True, type = int, help = "Minimum of m/z value.")
@click.option("--mz_max", default = "1801", show_default = True, type = int, help = "Maximum of m/z value.")
@click.option("--mz_tol_ms1", default = "0.01", show_default = True, type = float, help = "m/z tolerance for MS1.")
@click.option("--mz_tol_ms2", default = "0.03", show_default = True, type = float, help = "m/z tolerance for MS2.")
@click.option("--n_irts", default = "4000", show_default = True, type = int, help = "Number of endogenous precursors for RT normalization. Minimum: 150.")
@click.option("--irt_mode", default = "irt", show_default = True, type = click.Choice(['irt', 'rt']), help = "Whether the RT coordinates in the library are iRT values or RT values.")
@click.option("--irt_score_cutoff", default = "0.99", show_default = True, type = float, help = "Cut-off Dream score for RT normalization.")
@click.option("--irt_score_libcos_cutoff", default = "0.95", show_default = True, type = float, help = "Cut-off library cosine similarity score for RT normalization.")
@click.option("--rt_norm_model", default = "nonlinear", show_default = True, type = click.Choice(['linear', 'nonlinear', 'calib']), help = "Use linear, non-linear or Calib-RT model for RT normalizaiton.")
@click.option("--n_cycles", default = "300", show_default = True, type = int, help = "Number of RT cycles to search for RSMs for each precursor. Must be greater than 21.")
@click.option("--n_frags_each_precursor", default = "3", show_default = True, type = int, help = "Number of fragment ions at least for each precursor.")
@click.option("--decoy_method", default = "shuffle", show_default = True, type = click.Choice(["shuffle", "reverse", "pseudo_reverse", "shift", "mutate"]), help = "Decoy generation method.")
@click.option("--ipf_scoring", is_flag = True, help = "Whether to calculate IPF scores.")
def dreamScore(file_dir, lib, out, n_threads, seed, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, n_irts, irt_mode, irt_score_cutoff, irt_score_libcos_cutoff, rt_norm_model, n_cycles, n_frags_each_precursor, decoy_method, ipf_scoring):    
    """DreamDIA pipeline (Stage 1): scoring peaks of all runs."""
    dream_score(file_dir, lib, out, n_threads, seed, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, n_irts, irt_mode, irt_score_cutoff, irt_score_libcos_cutoff, rt_norm_model, n_cycles, n_frags_each_precursor, decoy_method, ipf_scoring)

@dreamdia.command(context_settings = CONTEXT_SETTINGS)
@click.option("--dream_dir", required = True, type = click.Path(exists = True), help = "Output directory of `dreamscore`.")
@click.option("--out", required = True, type = click.Path(exists = False), help = "Directory for output files.")
@click.option("--n_threads", default = 16, show_default = True, type = int, help = "Number of threads.")
@click.option("--seed", default = 12580, show_default = True, type = int, help = "Random seed.")
@click.option("--top_k", default = "10", show_default = True, type = int, help = "Number of RSMs for each precursor to extract.")
@click.option("--disc_model", default = "xgboost", show_default = True, type = click.Choice(["xgboost", "rf"]), help = "Type of the discriminative model.")
@click.option("--disc_sample_rate", default = None, show_default = True, type = float, help = "Sample rate for building discriminative model.")
@click.option("--fdr_precursor", default = "0.01", show_default = True, type = float, help = "FDR of precursor level.")
@click.option("--fdr_peptide", default = "0.01", show_default = True, type = float, help = "FDR of peptide level.")
@click.option("--fdr_protein", default = "0.01", show_default = True, type = float, help = "FDR of protein level.")
@click.option("--dreamdialignr", is_flag = True, help = "Whether to activate DreamDIAlignR cross-run analysis.")
@click.option("--r_home", default = None, show_default = True, type = click.Path(exists = True), help = "Directory of R home. (Required only when --dreamdialignr is specified.)")
@click.option("--mra_algorithm", default = "dialignr", show_default = True, type = click.Choice(['dialignr', 'global']), help = "Multi-run alignment (MRA) algorithm. (Valid only when --dreamdialignr is specified.)")
@click.option("--global_constraint_type", default = "lowess", show_default = True, type = click.Choice(['lowess', 'linear']), help = "Global alignment method. (Valid only when --dreamdialignr is specified.)")
@click.option("--span_value", default = "0.1", show_default = True, type = float, help = "Span value of lowess fit. (Valid only when --dreamdialignr is specified and --global_constraint_type is set to 'lowess'.)")
@click.option("--distance_metric", default = "nc", show_default = True, type = click.Choice(['nc', 'euclidean']), help = "Distance metric for global alignment of run pairs. (Valid only when --dreamdialignr is specified.)")
@click.option("--rt_tol", default = "3.3", show_default = True, type = float, help = "RT tolerance for multi-run alignment. Should be unit cycle time. (Valid only when --dreamdialignr is specified.)")
@click.option("--exp_decay", default = "50", show_default = True, type = float, help = "Exponential weight decay coeficient. (Valid only when --dreamdialignr is specified.)")
def dreamProphet(dream_dir, out, n_threads, seed, top_k, disc_model, disc_sample_rate, fdr_precursor, fdr_peptide, fdr_protein, dreamdialignr, r_home, mra_algorithm, global_constraint_type, span_value, distance_metric, rt_tol, exp_decay):    
    """DreamDIA pipeline (Stage 2): identifying and quantifying peptides and proteins."""
    dream_prophet(dream_dir, out, n_threads, seed, top_k, disc_model, disc_sample_rate, fdr_precursor, fdr_peptide, fdr_protein, dreamdialignr, r_home, mra_algorithm, global_constraint_type, span_value, distance_metric, rt_tol, exp_decay)

if __name__ == "__main__":
    dreamdia()