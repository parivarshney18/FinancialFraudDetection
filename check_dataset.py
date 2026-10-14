import sys
from pathlib import Path
import pandas as pd

raw = Path("ai/data/raw")
if len(sys.argv) > 1:
    path = Path(sys.argv[1])
else:
    files = sorted(list(raw.glob("*.csv")) + list(raw.glob("*.xlsx")) + list(raw.glob("*.xls")))
    path = files[0] if files else None
if path is None or not path.exists():
    sys.exit("Dataset file nahi mili. Isse ai/data/raw/ mein rakho ya naam argument mein do.")

print("Reading:", path, "(Excel hai to thoda time lagega...)")
if path.suffix.lower() in (".xlsx", ".xls"):
    print("Sheets:", pd.ExcelFile(path).sheet_names)
    df = pd.read_excel(path, sheet_name=0)
else:
    df = pd.read_csv(path)

table3 = ["gender", "age", "state", "city", "bank_branch", "account_type",
          "transaction_date", "transaction_time", "transaction_amount", "merchant_id",
          "transaction_type", "merchant_category", "account_balance",
          "transaction_device", "transaction_location", "device_type",
          "transaction_currency", "transaction_description"]
cols = [c.lower().replace(" ", "_") for c in df.columns]

print("\nShape  :", df.shape)
print("Matched:", len([c for c in table3 if c in cols]), "of", len(table3))
print("Missing:", [c for c in table3 if c not in cols])
print("Extra  :", [c for c in cols if c not in table3])

label = next((c for c in df.columns if "fraud" in c.lower()), None)
if label:
    print("\nLabel column:", label)
    print(df[label].value_counts())
    print((df[label].value_counts(normalize=True) * 100).round(2))
else:
    print("\nFraud label column nahi mila. Columns:", list(df.columns))

out = Path("ai/data/processed") / (path.stem + ".csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print("\nCSV copy saved:", out)
