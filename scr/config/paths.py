from os import path

PROJECT_DIR = "/nfshome/david/Contours"
BACKUP_DIR = "/nfsscratch/david/backup/Contours/"

PATH_CONTOURS = "/nfsscratch/david/Contours/contours"
PATH_CONTOURS_PHASES = "/nfsscratch/david/Contours/contours_phases"

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

# Slopes defining the sunspot evolution. Good to precompute.
SLOPES_FILE = path.join(PATH_CONTOURS_PHASES, "all_slopes.parquet")
