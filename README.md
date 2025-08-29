# GliaPharm SA - Loïc Lengacher - 27.08.2025

--ELECTROPHYSIOLOGY SIGNAL ANALYSIS--

A python application with a graphical user interface (GUI) for visualizing and analyzing electrophysiology data stored in HDF5 (.h5) files.
It works with files generated from an application called SPOC linked with an electrophysiological recording system made by HEPIA, an article detailled this system (DOI 10.1109/TBCAS.2021.3097833).

This tool was developed for laboratory use in GliaPharm SA to facilitate signal inspection, electrode selection, spike and busrt activity detection on brain organoid.

FEATURES:

    File loading: open .h5 electrophysiology datasets.

    Visualization: plots raw signals from up to 32 electrodes.

    Electrode selection: choose which electrodes to include in analysis.

    Activity detection: automatic threshold-based detection of events.

    Results export: save processed results for further analysis.

    User-friendly GUI built with tkinter.


INSTALLATION

    Requirements

    Python 3.9+

    The following Python packages (see requirements.txt):

SETUP

    git clone https://github.com/LengaCodes/Electrophy_Analysis_Tool_GP.git
    cd Electrophy_Analysis_Tool_GP
    pip install -r requirements.txt

USAGE

    python main.py

WORKFLOW

    Welcome screen → select an .h5 file.

    Visualization screen → inspect signals and choose electrodes to exclude

    Analysis screen → compute activity and optionally export results.

CITATION

    Loïc Lengacher, Electrophysiology Signal Analysis (2025), GitHub repository: https://github.com/LengaCodes/Electrophy_Analysis_Tool_GP

