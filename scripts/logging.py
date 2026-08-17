""""this is a script to create avery trivial log-file of the print outputs of this pipeline"""
import os
from scripts import config

log_messages = []

def log(message):
    log_messages.append(message)
    if os.path.exists(os.path.join(config.folder_project, 'md_analysis_pipeline_logfile.log')):
        append_write = 'a'
    else:
        append_write = 'w'
    with open(os.path.join(config.folder_project, 'md_analysis_pipeline_logfile.log'), append_write) as logfile:
        logfile.write(message + '\n')
