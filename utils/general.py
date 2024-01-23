# Standard library imports
import time
from multiprocessing import cpu_count

N_PROCESSES = 6 #cpu_count()

def printt(text):
    print(f"[{time.strftime('%H:%M:%S')}]", text)

def print_sucess():
    print("\n" + "=" * 40)
    print("   SCRIPT FINISHED SUCCESSFULLY!   ")
    print("=" * 40 + "\n")