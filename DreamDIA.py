import os
import click
import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dream_score import dream_score

CONTEXT_SETTINGS = dict(help_option_names = ['-h', '--help'], max_content_width = 120)

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return
    click.echo('Dream-DIA Version 1.0.0')
    ctx.exit()

@click.command(context_settings = CONTEXT_SETTINGS)
@click.option("--file_dir", required = True, type = click.Path(exists = True), help = "Directory that contains only DIA data files. Centroided .mzXML, .mzML or .raw files from Thermo Fisher equipments are supported. (For Linux systems, `mono` tool has to be installed for the supporting of .raw files. https://www.mono-project.com/download/stable/#download-lin)")
@click.option("--lib", required = True, type = click.Path(exists = True), help = "File name of the spectral library. .tsv or .csv formats are supported.")
@click.option("--win", required = True, type = click.Path(exists = True), help = "Window settings of the acquisition with no overlaps. Each row has two numbers that describe the start and the end of a window, which are separated by a tab.")
@click.option("--out", required = True, type = click.Path(exists = False), help = "Directory for output files.")
@click.option("--n_threads", default = 32, show_default = True, type = int, help = "Number of threads.")
@click.option("--seed", default = 123, show_default = True, type = int, help = "Random seed for decoy generation.")
@click.option("--mz_unit", default = "Da", show_default = True, type = click.Choice(['Da', 'ppm']), help = "m/z unit for m/z range and tolerance settings.")
@click.option("--mz_min", default = "99", show_default = True, type = int, help = "Minimum of m/z value.")
@click.option("--mz_max", default = "1801", show_default = True, type = int, help = "Maximum of m/z value.")
@click.option("--mz_tol_ms1", default = "0.01", show_default = True, type = float, help = "m/z tolerance for MS1.")
@click.option("--mz_tol_ms2", default = "0.03", show_default = True, type = float, help = "m/z tolerance for MS2.")
@click.option("--fdr_precursor", default = "0.01", show_default = True, type = float, help = "FDR of precursor level.")
@click.option("--fdr_protein", default = "0.01", show_default = True, type = float, help = "FDR of protein level.")
@click.option("--n_irt", default = "500", show_default = True, type = int, help = "Number of endogenous precursors for RT normalization.")
@click.option("--top_k", default = "10", show_default = True, type = int, help = "Number of peak groups for each precursor to extract.")
@click.option("--n_cycles", default = "300", show_default = True, type = int, help = "Number of RT cycles to search for peak groups for each precursor.")
@click.option("--n_frags_each_precursor", default = "3", show_default = True, type = int, help = "Number of fragment ions at least for each precursor.")
@click.option("--do_not_output_library", is_flag = True, help = "Do not output the library with decoys generated by Dream-DIA. If this option is not activated, the library with decoys will be saved at the same directory of the input library.")
@click.option("--swath", is_flag = True, help = "Use optimized m/z tolerances and deep representation models for SWATH data. If --swath is used, options including --mz_unit, --mz_tol_ms1, --mz_tol_ms2 will be invalid.")
@click.option("--model_cycles", default = "12", show_default = True, type = int, help = "(# Do not modify this argument unless customed deep representation models are used.) Number of RT cycles of the XICs.")
@click.option("--n_lib_frags", default = "20", show_default = True, type = int, help = "(# Do not modify this argument unless customed deep representation models are used.) Number of XICs in 'lib' part of the input matrix of the deep representation models.")
@click.option("--n_self_frags", default = "50", show_default = True, type = int, help = "(# Do not modify this argument unless customed deep representation models are used.) Number of XICs in 'self' part of the input matrix of the deep representation models.")
@click.option("--n_qt3_frags", default = "50", show_default = True, type = int, help = "(# Do not modify this argument unless customed deep representation models are used.) Number of XICs in 'qt3' part of the input matrix of the deep representation models.")
@click.option("--n_ms1_frags", default = "10", show_default = True, type = int, help = "(# Do not modify this argument unless customed deep representation models are used.) Number of XICs in 'ms1' part of the input matrix of the deep representation models.")
@click.option("--prophet_mode", default = "local", show_default = True, type = click.Choice(["local", "global"]), help = "Train a disciminant model on the peak groups of each sample respectively (local) or train a discriminant model on all the peak groups from all the samples.")
@click.option("--disc_model", default = "xgboost", show_default = True, type = click.Choice(["xgboost", "rf"]), help = "Type of the discriminant model.")
@click.option("--dream_indicators", is_flag = True, help = "Activate Dream-Indicators to search better hyper-parameters for the discriminant model. If this option is not activated, the depths of the trees in the discriminant model will be heuristically set to 10.")
@click.option('--version', is_flag = True, callback = print_version, expose_value = False, is_eager = True, help = "Print version and exit.")
def dreamScore(file_dir, lib, win, out, n_threads, seed, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, fdr_precursor, fdr_protein, n_irt, top_k, n_cycles, n_frags_each_precursor, do_not_output_library, swath, model_cycles, n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, prophet_mode, disc_model, dream_indicators):
    
    dream_score(file_dir, lib, win, out, n_threads, seed, mz_unit, mz_min, mz_max, mz_tol_ms1, mz_tol_ms2, fdr_precursor, fdr_protein, n_irt, top_k, n_cycles, n_frags_each_precursor, do_not_output_library, swath, model_cycles, n_lib_frags, n_self_frags, n_qt3_frags, n_ms1_frags, prophet_mode, disc_model, dream_indicators)

if __name__ == "__main__":
    dreamScore()