"""Automatically summarize this long-running lifetime batch on completion."""
import subprocess
import sys
import time
from pathlib import Path
base=Path(__file__).resolve().parent
root=base/'sets_vs_qea_full_lifetime_budget10000_20261004'
log=Path('/tmp/sets_vs_qea_full_lifetime_budget10000_20261004.log')
while True:
    content=log.read_text() if log.exists() else ''
    if 'Traceback (most recent call last)' in content:
        raise RuntimeError('Experiment failed; inspect '+str(log))
    if ('result_dir '+str(root)) in content and len(list(root.glob('*/*/summary.csv')))==24:
        subprocess.run([sys.executable,str(base/'summarize_sets_vs_qea_lifetime.py')],check=True)
        break
    time.sleep(30)
