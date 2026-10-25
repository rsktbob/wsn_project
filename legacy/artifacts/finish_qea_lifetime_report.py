"""Wait for this specific experiment's saved results, then summarize once."""
import subprocess
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parent
log=Path('/tmp/qea_full_lifetime_budget10000_20261004.log')
result=root/'qea_full_lifetime_budget10000_20261004'
while True:
    content=log.read_text() if log.exists() else ''
    if 'Traceback (most recent call last)' in content:
        raise RuntimeError('Experiment failed; inspect '+str(log))
    if ('result_dir '+str(result)) in content and len(list(result.glob('*/*/summary.csv')))==24:
        subprocess.run([sys.executable,str(root/'summarize_qea_lifetime_10000.py')],check=True)
        break
    time.sleep(30)
