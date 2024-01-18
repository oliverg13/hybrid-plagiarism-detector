import time

def printt(text):
    print(f"[{time.strftime('%H:%M:%S')}]", text)

def print_sucess():
    print("\n" + "=" * 40)
    print("   SCRIPT FINISHED SUCCESSFULLY!   ")
    print("=" * 40 + "\n")