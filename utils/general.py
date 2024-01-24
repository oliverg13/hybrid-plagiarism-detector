# Standard library imports
import time
from multiprocessing import cpu_count

N_PROCESSES = cpu_count() - 1 if cpu_count() > 1 else 1

def printt(text):
    print(f"[{time.strftime('%H:%M:%S')}]", text)

def print_sucess():
    print("\n" + "=" * 40)
    print("   SCRIPT FINISHED SUCCESSFULLY!   ")
    print("=" * 40 + "\n")