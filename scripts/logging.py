""""this is a script to create a very trivial log-file of the print outputs of this pipeline"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log_messages = []

def log(message):
    log_messages.append(message)
    if os.path.exists(os.path.join(config.PROJECT_ROOT, 'md_analysis_pipeline_logfile.log')):
        append_write = 'a'
    else:
        append_write = 'w'
    with open(os.path.join(config.PROJECT_ROOT, 'md_analysis_pipeline_logfile.log'), append_write) as logfile:
        logfile.write(message + '\n')
