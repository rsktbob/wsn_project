import subprocess
import sys
import time
from pathlib import Path
base=Path(__file__).resolve().parent
root=base/'qea3_full_lifetime_budget10000_20261005'
log=Path('/tmp/qea3_full_lifetime_budget10000_20261005.log')
while True:
    content=log.read_text() if log.exists() else ''
    if 'Traceback (most recent call last)' in content:
        raise RuntimeError('Experiment failed; inspect '+str(log))
    if ('result_dir '+str(root)) in content and len(list(root.glob('qea3/*/summary.csv')))==6:
        subprocess.run([sys.executable,str(base/'summarize_qea3_full_lifetime.py')],check=True)
        break
    time.sleep(30)
