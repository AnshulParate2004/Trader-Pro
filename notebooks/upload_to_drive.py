"""
Google Drive Model Uploader for Trader-Pro
Uploads all trained models and metrics from notebooks/model to Google Drive.
"""
import os
import sys
import glob
import shutil

LOCAL_MODEL_DIR = r"D:\Projects_Main\Trade-Pro\Trader-Pro\notebooks\model"
COLAB_DRIVE_DIR = "/content/drive/MyDrive/crypto_models"

def main():
    print("=" * 60)
    print("        Trader-Pro Google Drive Model Sync")
    print("=" * 60)
    
    if not os.path.exists(LOCAL_MODEL_DIR):
        print(f"[!] Local model directory not found: {LOCAL_MODEL_DIR}")
        return

    # Check for files
    model_files = []
    for root, _, files in os.walk(LOCAL_MODEL_DIR):
        for f in files:
            model_files.append(os.path.join(root, f))

    if not model_files:
        print(f"[*] No model files found in {LOCAL_MODEL_DIR}. Run training notebooks first.")
        return

    print(f"[*] Found {len(model_files)} local model artifacts in: {LOCAL_MODEL_DIR}")
    for mf in model_files:
        rel = os.path.relpath(mf, LOCAL_MODEL_DIR)
        sz = os.path.getsize(mf) / 1024
        print(f"    - {rel:40s} ({sz:8.1f} KB)")

    # 1. Check if running in Colab or if Colab drive mounted
    if os.path.exists("/content/drive/MyDrive"):
        print(f"\n[+] Google Drive mounted at /content/drive/MyDrive. Syncing now...")
        for mf in model_files:
            rel = os.path.relpath(mf, LOCAL_MODEL_DIR)
            dest = os.path.join(COLAB_DRIVE_DIR, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(mf, dest)
            print(f"    Uploaded -> {dest}")
        print("\n[OK] Compulsory Google Drive upload complete!")
        return

    # 2. Check for local Google Drive desktop sync folder (e.g. G:\My Drive)
    possible_drive_paths = [
        r"G:\My Drive\crypto_models",
        os.path.expanduser(r"~\Google Drive\crypto_models"),
        os.path.expanduser(r"~\GoogleDrive\crypto_models"),
    ]
    detected_drive = next((p for p in possible_drive_paths if os.path.exists(os.path.dirname(p))), None)
    
    if detected_drive:
        print(f"\n[+] Detected Google Drive Desktop at: {detected_drive}")
        for mf in model_files:
            rel = os.path.relpath(mf, LOCAL_MODEL_DIR)
            dest = os.path.join(detected_drive, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(mf, dest)
            print(f"    Synced -> {dest}")
        print("\n[OK] Local-to-Drive sync complete!")
        return

    # 3. If neither Colab nor Desktop Drive detected
    print("\n" + "-" * 60)
    print("[INFO] Automatic local Google Drive desktop path not detected.")
    print("Your models are safely saved in:")
    print(f"  --> {LOCAL_MODEL_DIR}")
    print("\nTo satisfy the compulsory Google Drive requirement, you can:")
    print("1. If running in Google Colab: The notebooks automatically mount and sync to Drive.")
    print("2. If running locally: Drag and drop the 'notebooks/model' folder into Google Drive Web.")
    print("-" * 60)

if __name__ == "__main__":
    main()
