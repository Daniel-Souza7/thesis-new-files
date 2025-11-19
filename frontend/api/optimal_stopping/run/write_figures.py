"""Writes all figures and PDF - SIMPLIFIED VERSION
This version just creates a minimal PDF wrapper around existing table files.
"""
import atexit
import os
import glob
import subprocess
from absl import app
from absl import flags
from optimal_stopping.utilities import configs_getter
from optimal_stopping.utilities import comparison_table
from optimal_stopping.utilities import plot_hurst
from optimal_stopping.utilities import plot_convergence_study

# TELEGRAM NOTIFICATION SETUP
SERVER = True
SEND = True


class SendBotMessage:
    def __init__(self):
        pass

    @staticmethod
    def send_notification(text, token=None, chat_id=None, files=None, *args, **kwargs):
        print(text)


try:
    from telegram_notifications import send_bot_message as SBM

    if SERVER:
        SEND = True
except Exception:
    SBM = SendBotMessage()
    SEND = False

# FLAGS
FLAGS = flags.FLAGS
flags.DEFINE_bool("send_telegram", True,
                  "Whether to send Telegram notification with PDF")
flags.DEFINE_string("telegram_token", "8239319342:AAGIIcoDaxJ1uauHbWfdByF4yzNYdQ5jpiA",
                    "Telegram bot token")
flags.DEFINE_string("telegram_chat_id", "798647521",
                    "Telegram chat ID")
flags.DEFINE_bool("check_latex", False,
                  "Check if LaTeX is installed before trying to compile")
flags.DEFINE_string("master_tex_name", "amc2",
                    "Name of the master LaTeX file (without .tex)")
flags.DEFINE_integer("latex_timeout", 300,
                     "Timeout in seconds for LaTeX compilation (default 300 = 5 minutes)")


def check_latex_installed():
    """Check if pdflatex is installed and accessible."""
    try:
        result = subprocess.run(['pdflatex', '--version'],
                                capture_output=True,
                                text=True,
                                timeout=5)
        if result.returncode == 0:
            print(f"✅ LaTeX found: {result.stdout.split(chr(10))[0]}")
            return True
        else:
            print("❌ LaTeX not working properly")
            return False
    except FileNotFoundError:
        print("❌ pdflatex not found in PATH")
        return False
    except Exception as e:
        print(f"❌ Error checking LaTeX: {e}")
        return False


def write_figures():
    """Generate LaTeX table files for each config."""
    generated_tables = []

    for config_name, config in configs_getter.get_configs():
        representations = list(config.representations)
        print(config_name, config, representations)
        for representation in representations:
            if len(representations) > 1:
                figure_name = f"{config_name}_{representation}"
            else:
                figure_name = config_name
            print(f"Writing {config_name}...")
            methods = {
                "TablePrice": comparison_table.write_table_price,
                "TableDuration": comparison_table.write_table_duration,
                "TablePriceDuration": comparison_table.write_table_price_duration,
                "ConvergenceStudy": plot_convergence_study.plot_convergence_study,
                "PlotHurst": plot_hurst.plot_hurst,
            }
            if representation not in methods:
                raise AssertionError(
                    f"Unknown representation type {config.representation}")
            try:
                if representation == "ConvergenceStudy":
                    methods[representation](config, x_axis="nb_paths")
                    methods[representation](config, x_axis="hidden_size")
                else:
                    methods[representation](figure_name, config)
                    # Track generated table files
                    if representation in ["TablePrice", "TableDuration", "TablePriceDuration"]:
                        generated_tables.append(figure_name)
            except BaseException as err:
                print("Error:", err)
                raise

    return generated_tables


def create_master_tex_file(latex_dir_path, table_names):
    """Create a minimal LaTeX document that directly includes table files."""
    master_file = os.path.join(latex_dir_path, f"{FLAGS.master_tex_name}.tex")

    # ULTRA-SIMPLE: Just the bare minimum to compile the tables
    latex_content = r"""\documentclass[10pt,a4paper]{article}

% Minimal packages needed
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[top=1cm, bottom=1.5cm, left=0.5cm, right=0.5cm]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{longtable}
\usepackage{array}
\usepackage{lmodern}
\usepackage[colorlinks=false]{hyperref}

\begin{document}

% Simple title
\begin{center}
  {\Huge\bfseries American Option Pricing Results}\\[0.3cm]
  {\large Daniel Souza}\\[0.2cm]
  {\large\today}
\end{center}

\tableofcontents
\clearpage

% Just input the table files directly - they already have all formatting
"""

    # Input each table file as-is
    for table_name in table_names:
        latex_content += f"\\input{{tables_draft/{table_name}.tex}}\n\n"

    latex_content += r"\end{document}"

    # Write the master file
    with open(master_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"✅ Created master LaTeX file: {master_file}")
    print(f"   - {len(table_names)} table(s) will be included directly")
    return master_file


def generate_pdf(table_names=None):
    """Generate PDF from LaTeX and return the path if successful."""

    # Get paths
    wd = os.getcwd()
    latex_dir_path = os.path.join(os.path.dirname(__file__), "../../../latex")
    latex_dir_path = os.path.abspath(latex_dir_path)

    # Create latex directory if it doesn't exist
    os.makedirs(latex_dir_path, exist_ok=True)
    os.makedirs(os.path.join(latex_dir_path, "tables_draft"), exist_ok=True)

    # If table names provided, create master tex file
    if table_names:
        print(f"\n📝 Creating master LaTeX document with {len(table_names)} table(s)...")
        tex_file = create_master_tex_file(latex_dir_path, table_names)
    else:
        # Check if master tex file exists
        tex_file = os.path.join(latex_dir_path, f"{FLAGS.master_tex_name}.tex")
        if not os.path.exists(tex_file):
            print(f"❌ LaTeX source file not found: {tex_file}")
            # Try to find table files and auto-generate
            tables_dir = os.path.join(latex_dir_path, "tables_draft")
            if os.path.exists(tables_dir):
                tex_files = glob.glob(os.path.join(tables_dir, "*.tex"))
                if tex_files:
                    print(f"\n💡 Found {len(tex_files)} table file(s), auto-generating master document...")
                    table_names = [os.path.splitext(os.path.basename(f))[0] for f in tex_files]
                    tex_file = create_master_tex_file(latex_dir_path, table_names)
                else:
                    print(f"   No table files found")
                    return None
            else:
                return None

    print(f"✅ Found LaTeX source: {tex_file}")

    # Change to latex directory
    os.chdir(latex_dir_path)
    atexit.register(os.chdir, wd)

    print("Generating PDF from LaTeX...")
    print("⏳ This may take a while for large tables...")

    # Try to compile with pdflatex
    try:
        # Run pdflatex twice for proper references
        for run in [1, 2]:
            print(f"📄 Compiling (pass {run}/2)... ", end="", flush=True)

            result = subprocess.run(
                ['pdflatex', '-synctex=1', '-interaction=nonstopmode',
                 f'{FLAGS.master_tex_name}.tex'],
                capture_output=True,
                text=True,
                timeout=FLAGS.latex_timeout
            )

            if result.returncode == 0:
                print("✅ Done")
            else:
                print(f"❌ Failed (return code {result.returncode})")

        if result.returncode == 0:
            pdf_path = os.path.abspath(os.path.join(latex_dir_path,
                                                    f"{FLAGS.master_tex_name}.pdf"))
            if os.path.exists(pdf_path):
                print(f"✅ {pdf_path} written successfully")
                return pdf_path
            else:
                print(f"❌ PDF not found at {pdf_path}")
                return None
        else:
            print(f"❌ pdflatex failed with return code {result.returncode}")
            print("\n--- LaTeX Error Output (last 30 lines) ---")
            error_lines = result.stdout.split('\n')
            for line in error_lines[-30:]:
                if line.strip() and ('!' in line or 'Error' in line or 'Warning' in line):
                    print(line)
            print("--- End of Error Output ---\n")

            log_file = os.path.join(latex_dir_path, f"{FLAGS.master_tex_name}.log")
            if os.path.exists(log_file):
                print(f"💾 Full log saved to: {log_file}")

            return None

    except FileNotFoundError:
        print("❌ ERROR: pdflatex not found!")
        print("\n🔧 Install MiKTeX (Windows) or TeX Live (Linux)")
        return None

    except subprocess.TimeoutExpired:
        print(f"❌ pdflatex timed out (>{FLAGS.latex_timeout} seconds)")
        print(f"💡 Increase timeout: --latex_timeout=600")
        return None

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def send_pdf_notification(pdf_path):
    """Send Telegram notification with PDF file."""
    if not SEND or not FLAGS.send_telegram:
        print("Telegram notifications disabled")
        return

    if pdf_path is None:
        print("No PDF to send (generation failed)")
        return

    try:
        print(f"Sending PDF via Telegram...")
        SBM.send_notification(
            token=FLAGS.telegram_token,
            text=f'📊 PDF generated!\n\nFile: {os.path.basename(pdf_path)}',
            files=[pdf_path],
            chat_id=FLAGS.telegram_chat_id
        )
        print("✅ PDF sent via Telegram successfully")
    except Exception as e:
        print(f"❌ Failed to send PDF via Telegram: {e}")


def main(argv):
    del argv

    # Check LaTeX installation if requested
    if FLAGS.check_latex:
        print("\n" + "=" * 60)
        print("CHECKING LATEX INSTALLATION")
        print("=" * 60)
        if not check_latex_installed():
            print("\n⚠️  LaTeX not installed or not in PATH")
            print("=" * 60 + "\n")
            return
        print("=" * 60 + "\n")

    try:
        # Send start notification
        if SEND and FLAGS.send_telegram:
            SBM.send_notification(
                token=FLAGS.telegram_token,
                text='📊 Starting LaTeX figure generation...',
                chat_id=FLAGS.telegram_chat_id
            )

        # Generate figures and collect table names
        print("\n" + "=" * 60)
        print("GENERATING FIGURES")
        print("=" * 60)
        table_names = write_figures()
        print("=" * 60 + "\n")

        # Generate PDF
        print("\n" + "=" * 60)
        print("GENERATING PDF")
        print("=" * 60)
        pdf_path = generate_pdf(table_names=table_names)
        print("=" * 60 + "\n")

        # Send PDF via Telegram
        if pdf_path:
            send_pdf_notification(pdf_path)
        else:
            print("\n⚠️  PDF generation failed - see errors above")
            if SEND and FLAGS.send_telegram:
                SBM.send_notification(
                    token=FLAGS.telegram_token,
                    text='⚠️ PDF generation failed',
                    chat_id=FLAGS.telegram_chat_id
                )

    except Exception as e:
        # Send error notification
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        if SEND and FLAGS.send_telegram:
            SBM.send_notification(
                token=FLAGS.telegram_token,
                text=f'❌ ERROR generating PDF:\n{e}',
                chat_id=FLAGS.telegram_chat_id
            )
        raise


if __name__ == "__main__":
    app.run(main)