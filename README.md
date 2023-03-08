# copying-task-uu
This repository contains the code and outcome measures for the following manuscript: [TBA]

Main contributor: Alex J. Hoogerbrugge

Utrecht University, 2022, licensed under GNU GPLv3. 
We welcome all researchers to re-run, verify, and/or modify the code provided in this repository, granted that you cite
the original publication.

**All raw data and outcome measures are included in this repository. Individual raw data should be unzipped first and 
placed within `copying-task-uu/results/[ID-folder]` (NOTE: ~38GB unzipped!). `copying-task-uu/results` also contains 
aggregate outcome measures and any figures/animations.**

Please note that this has been a multi-year and multi-faceted projected, and thus there may be some redundancies in the 
codebase. Additionally, the experiment and the data processing are run with two different environments, such that the
analysis environment is more up to date.
See instructions below.

For help, contact a j hoogerbrugge@uu nl [replace spaces with dots]

# 1. The experiment

Below are instructions for those willing to collect data with this experimental paradigm. 
If you would like to implement a different manipulation, contact the author mentioned above.

## 1.1 Preparing the environment
Navigate to the `copying-task-uu` folder and run `conda env create --file environment.yml (optional: --name YOURNAME)` 
in an anaconda prompt. Then activate the environment.
A version of PyGaze is included in the repository because a small change was made to its source code in 
`copying-task-uu/PyGaze/pygaze/_eyetracker/libeyelink.py` [lines 607-609]. The PyGaze folder should be unzipped.

The experiment should be run with an Eyelink 1000 (probably, I haven't verified whether it works with other models).
Display parameters should be set in `copying-task-uu/constants.py` and `copying-task-uu/analysis/constants_analysis.py`.

## 1.2 Running the experiment
Everything should be ready to run now.
First, try running `copying-task-uu/run_experiment_training.py` from the anaconda prompt. This should ignore warnings 
about eye trackers and helps you verify whether the experiment runs at all.

Then, run `copying-task-uu/run_experiment.py` for the full experiment. Set the desired number of trials at the top 
of the file.

The experiment code is mostly located within `copying-task-uu/canvas.py`.

# 2. The analysis

Analyses can be re-run as verification, but these require several steps. 2.2 may be skipped in order to save time.
If you encounter any issues with these scripts, try setting `N_JOBS = 1` in 
`copying-task-uu/analysis/constants_analysis.py`.

All participants that are to be excluded in the analysis should be added to `copying-task-uu/results/participant_info.xlsx`.

Participants were excluded due to the following criteria:
* 1002: Stopped early
* 1003: Gaze data missing from last block
* 1004: Stopped early
* 1021: Insufficient calibration quality, stopped early (no data collected)
* 1022: Insufficient calibration quality, stopped early (no data collected)

## 2.1 Installing the environment
Navigate to the `copying-task-uu` folder and run 
`conda env create --file environment_analysis.yml (optional: --name YOURNAME)` 
in an anaconda prompt. Then activate the environment.

## 2.2 Converting the data
Raw gaze data (`.edf` files) should be converted to `.asc` first with SR Research's proprietary EDFConverter or EDF2ASC. 
This application can be downloaded from their [forum](https://www.sr-research.com/support/). 
See the docstring at the top of `copying-task-uu/analysis/asc_converter_samples.py` for the appropriate settings. 
Raw data should first be extracted from their zip-files.
Then run the converter script.

Then, run `copying-task-uu/analysis/load_and_merge_data.py`. This combines all individual raw data into one folder 
per participant, and adds trial, condition, etc. to the datafiles.

Lastly, run `copying-task-uu/analysis/fixation_detection.py`.

## 2.3 Processing outcome variables and making the figure
Run `copying-task-uu/analysis/process_and_plot.py` in order to recreate Figure 2.
This scripts computes many extra variables which are saved to `copying-task-uu/results`.

## 2.4 Analyses
Analyses were run in JASP. Included in `copying-task-uu/results` are appropriate tables and JASP savefiles. 
This is a bit circular with the above step, as Figure 2 relies on `copying-task-uu/results/tests.xlsx` being 
manually filled, which in turn relies on the plotting script to be run.

## 2.5 Animating a trial
`copying-task-uu/analysis/animate_trial.py` can be run in order to reconstruct a trial from the available data.
Note that some required data (only for animations, no effect on outcome measures) was not correctly stored for some 
participants, so the script may not generate an animation for all participants. 
The animation script may take a very long time to run, depending on trial length, framerate, and dpi.
