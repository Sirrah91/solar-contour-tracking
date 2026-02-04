from os import path

PROJECT_DIR = FULL PATH TO PROJECT
BACKUP_DIR =  FULL PATH TO BACKUP

PATH_CONTOURS = WHERE TO SAVE CONTOURS
PATH_CONTOURS_PHASES = WHERE TO SAVE PHASE-SEPARATED PRODUCTS

# Slopes defining the sunspot evolution. Good to precompute.
SLOPES_FILE = path.join(PATH_CONTOURS_PHASES, "all_slopes.parquet")

SUBDIRS = {
    "scr": "scr",
    "OpenPBS": "OpenPBS",
}

PATH_SCRIPTS = path.join(PROJECT_DIR, SUBDIRS["scr"])
PATH_PBS = path.join(PROJECT_DIR, SUBDIRS["OpenPBS"])

# base folder for all generated outputs
PATH_GRAPHIC_OUTPUT = path.join(PROJECT_DIR, "graphic_output")

# subfolders for different types
PATH_FIGURES = path.join(PATH_GRAPHIC_OUTPUT, "figures")
PATH_VIDEOS = path.join(PATH_GRAPHIC_OUTPUT, "videos")
PATH_INTERACTIVE = path.join(PATH_GRAPHIC_OUTPUT, "interactive")
