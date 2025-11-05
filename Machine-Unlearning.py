# ============================================
# Machine Unlearning (Interactive, single file)
# - Loads your Quesion_Answers.csv
# - Asks you at runtime which ROWS to delete
# - Retrains a tiny model (Country -> Company)
# - Saves updated CSV + proof JSON + ledger
# ============================================

import os, json, hashlib
from datetime import datetime, timezone

import pandas as pd                  # Reason: easy table loading/saving
import numpy as np                   # Reason: tiny numeric helpers
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# --------------- SETTINGS (edit these if needed) ---------------
CSV_PATH = r"/Users/itixa/Desktop/5-credit seminar/Quesion_Answers.csv"  # <- put your real path
OUT_DIR  = "unlearning_output"                                          # where proof/ledger go


# --------- SMALL UTILITIES (with reasons) ---------
def sha256_text(text: str) -> str:
    """Reason: we hash data to make the proof tamper-evident (change => new hash)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_people_table(path: str) -> pd.DataFrame:
    """
    Your file has 'Table 1' on the first line and columns separated by multiple spaces.
    Reason: we skip the first line and use a regex separator (2+ spaces or comma).
    If that fails, fall back to fixed-width reading.
    """
    try:
        df = pd.read_csv(path, skiprows=1, sep=r"\s{2,}|,", engine="python", header=0)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"File not found: {path}\nTip: pass a correct absolute path, use quotes if it has spaces."
        ) from e

    if df.shape[1] == 1:  # if everything got lumped into one column, try fixed-width format
        df = pd.read_fwf(path, skiprows=1)

    # Normalize column names (Reason: consistent names make the rest simple)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Map likely names to fixed ones (Reason: cope with different capitalization)
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"user_id", "userid", "id"}: rename[c] = "User_Id"
        elif lc == "name":                    rename[c] = "Name"
        elif lc == "country":                 rename[c] = "Country"
        elif lc == "company":                 rename[c] = "Company"
    df = df.rename(columns=rename)

    needed = {"Name", "Country", "Company"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Expected columns {needed}, found {list(df.columns)}")

    return df


def train_toy_model(df: pd.DataFrame):
    """
    Tiny demo model: predict Company from Country.
    Reason: gives us 'before/after' numbers to show unlearning changed the training set.
    """
    enc_country = LabelEncoder()
    enc_company = LabelEncoder()

    tmp = df.copy()
    tmp["Country_enc"] = enc_country.fit_transform(tmp["Country"].astype(str))
    tmp["Company_enc"] = enc_company.fit_transform(tmp["Company"].astype(str))

    X = tmp[["Country_enc"]].values
    y = tmp["Company_enc"].values

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))  # OK for a tiny classroom demo

    return model, acc


def write_proof(removed_df, kept_df, acc_before, acc_after, source_csv):
    """
    Reason: Create a verifiable 'certificate' of unlearning with:
    - who/what was removed (rows)
    - timestamp
    - hashes of removed/remaining data
    - before/after metrics
    Also append to a simple ledger (append-only text file).
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    proof = {
        "topic": "Machine Unlearning (people table demo)",
        "action": "unlearn_rows_by_index",
        "removed_names": removed_df["Name"].tolist(),
        "rows_removed": int(len(removed_df)),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hash_removed_rows": sha256_text(removed_df.to_json(orient="records")) if len(removed_df) else None,
        "hash_remaining_rows": sha256_text(kept_df.to_json(orient="records")),
        "metrics": {
            "accuracy_before_all": float(acc_before),
            "accuracy_after_remaining": float(acc_after),
        },
        "source_csv": os.path.abspath(source_csv),
    }

    proof_path = os.path.join(OUT_DIR, "proof_people_unlearning.json")
    with open(proof_path, "w", encoding="utf-8") as f:
        json.dump(proof, f, indent=2)

    ledger_path = os.path.join(OUT_DIR, "unlearning_ledger.jsonl")
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(proof) + "\n")

    return proof_path, ledger_path


# -------------------------- MAIN --------------------------
def main():
    # 1) Load table
    df = load_people_table(CSV_PATH)

    # Show the table with row numbers.
    # Reason: the user can clearly see which row index to remove.
    print("\n=== Loaded table (with row numbers) ===")
    print(df.reset_index().rename(columns={"index": "Row"}))  # adds a 'Row' column for clarity

    # 2) Train BEFORE unlearning
    _, acc_before = train_toy_model(df)
    print(f"\nAccuracy BEFORE unlearning (on all rows): {acc_before:.3f}")

    # 3) Ask the user which rows to delete
    # Reason: interactive, so you can demo various removals without editing code.
    raw = input("\nEnter row numbers to UNLEARN (comma-separated, e.g., 1,3): ").strip()
    if not raw:
        print("No rows provided. Exiting.")
        return

    try:
        rows = sorted({int(x.strip()) for x in raw.split(",") if x.strip() != ""})
    except ValueError:
        print("Invalid input. Please enter integers like: 0,2,5")
        return

    # Validate row numbers
    bad = [r for r in rows if r < 0 or r >= len(df)]
    if bad:
        print(f"Row index out of range: {bad}. Valid range is 0..{len(df)-1}")
        return

    # Split into kept/removed using index
    removed_df = df.iloc[rows].copy()
    kept_df    = df.drop(df.index[rows]).copy()

    print(f"\nRows to unlearn: {rows}")
    print("Removed records (preview):")
    print(removed_df)

    # 4) Train AFTER unlearning
    if kept_df.empty:
        print("\nAll rows removed — there is nothing left to train on.")
        acc_after = 0.0
    else:
        _, acc_after = train_toy_model(kept_df)

    print(f"\nAccuracy AFTER unlearning (on remaining rows): {acc_after:.3f}")

    # 5) Save updated CSV
    # We create a new filename alongside the original: *_updated.csv
    base, ext = os.path.splitext(CSV_PATH)
    updated_path = base + "_updated" + ext
    kept_df.to_csv(updated_path, index=False)
    print(f"\n✅ Updated CSV saved at:\n{updated_path}")

    # Optional: ask if you want to OVERWRITE the original (safer to keep a backup)
    ans = input("\nDo you also want to OVERWRITE the original CSV? (y/N): ").strip().lower()
    if ans == "y":
        # Make a quick backup first, then overwrite
        backup_path = base + "_backup" + ext
        if not os.path.exists(backup_path):
            df.to_csv(backup_path, index=False)
            print(f"Backup of original saved at: {backup_path}")
        kept_df.to_csv(CSV_PATH, index=False)
        print(f"Original CSV overwritten at: {CSV_PATH}")
    else:
        print("Keeping original CSV unchanged (you can use the *_updated.csv file).")

    # 6) Write proof JSON + ledger entry
    proof_path, ledger_path = write_proof(removed_df, kept_df, acc_before, acc_after, CSV_PATH)
    print(f"\n🔐 Proof saved at: {proof_path}")
    print(f"📜 Ledger appended at: {ledger_path}")

    # 7) Final message for your slides
    print("\n🎯 Done! Show in your presentation:")
    print(" - BEFORE vs AFTER accuracies")
    print(" - The updated CSV without the removed rows")
    print(" - The proof JSON (hashes + timestamp) and the ledger entry")


if __name__ == "__main__":
    main()
