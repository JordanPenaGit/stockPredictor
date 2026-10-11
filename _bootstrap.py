"""Prefer isolated project dependencies without requiring shell activation."""
import os,sys
from pathlib import Path
def activate():
    root=Path(__file__).resolve().parent;env=root/'.venv';python=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if python.exists() and Path(sys.prefix).resolve()!=env.resolve():os.execv(str(python),[str(python),str(Path(sys.argv[0]).resolve()),*sys.argv[1:]])
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'2')
