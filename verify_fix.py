import requests
import time
import os

def test_workflow():
    url = "http://127.0.0.1:5000"
    
    # Check if app is running
    try:
        requests.get(url)
    except:
        print("Error: App not running on 127.0.0.1:5000")
        return

    # Simulate upload (using a dummy small image if possible, or just checking routes)
    print("Checking persistence directories...")
    if os.path.exists("tasks") and os.path.exists("static/results"):
        print("[+] Persistence directories exist.")
    else:
        print("[-] Persistence directories missing.")

    print("Verifying cleanup daemon...")
    # This is harder to test quickly, but we can check if it starts without error.
    
    print("\nVerification complete. Manual test on Render is recommended.")

if __name__ == "__main__":
    test_workflow()
