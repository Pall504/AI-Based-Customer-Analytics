
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FILES = {
    "telecom": "telecom_cleaned.csv",
    "wholesale": "wholesale_cleaned.csv",
    "retail": "online_retail_cleaned.csv",
    "customers": "online_retail_customer_features.csv",
}

def find_file(filename):
    candidates = [
        DATA_DIR / filename,
        BASE_DIR / filename,
        BASE_DIR.parent / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {filename}. Put it inside {DATA_DIR}."
    )

def load_telecom():
    return pd.read_csv(find_file(FILES["telecom"]))

def load_wholesale():
    return pd.read_csv(find_file(FILES["wholesale"]))

def load_online_retail():
    return pd.read_csv(find_file(FILES["retail"]))

def load_customer_features():
    return pd.read_csv(find_file(FILES["customers"]))

def dataset_info(df):
    return {
        "Rows": len(df),
        "Columns": df.shape[1],
        "Missing Values": int(df.isna().sum().sum()),
        "Duplicates": int(df.duplicated().sum()),
    }

def _numeric(df, cols):
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def segment_customers(df, k=4):
    required = ["Recency_Days", "Number_of_Invoices", "Total_Spending"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Customer feature file is missing: {missing}")

    data = _numeric(df, required).replace([np.inf, -np.inf], np.nan).dropna()
    data = data[
        (data["Recency_Days"] >= 0) &
        (data["Number_of_Invoices"] > 0) &
        (data["Total_Spending"] > 0)
    ].copy()

    if len(data) < k:
        raise ValueError(f"Only {len(data)} valid customers are available; k={k} is too large.")

    scaler = StandardScaler()
    X = scaler.fit_transform(data[required])

    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    data["Segment"] = model.fit_predict(X)

    summary = (
        data.groupby("Segment")
        .agg(
            Customers=("Segment", "size"),
            Avg_Recency=("Recency_Days", "mean"),
            Avg_Frequency=("Number_of_Invoices", "mean"),
            Avg_Spending=("Total_Spending", "mean"),
        )
        .reset_index()
    )

    # Give stable human-readable names based on segment behaviour.
    summary["Segment_Label"] = "Segment " + summary["Segment"].astype(str)
    med_spend = summary["Avg_Spending"].median()
    med_freq = summary["Avg_Frequency"].median()
    med_rec = summary["Avg_Recency"].median()

    def label(r):
        if r["Avg_Spending"] >= med_spend and r["Avg_Frequency"] >= med_freq and r["Avg_Recency"] <= med_rec:
            return "High-Value Loyal"
        if r["Avg_Recency"] > med_rec and r["Avg_Spending"] < med_spend:
            return "At-Risk / Low-Value"
        if r["Avg_Frequency"] >= med_freq:
            return "Frequent Buyers"
        return "Occasional Buyers"

    summary["Segment_Label"] = summary.apply(label, axis=1)
    data = data.merge(summary[["Segment", "Segment_Label"]], on="Segment", how="left")
    return data, summary

def _find_column(df, names):
    lower = {c.lower().strip(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None

def build_future_purchase_dataset(retail, horizon_days=30):
    """
    Leakage-safe target construction.

    For each customer:
      - history = transactions before the cutoff
      - target = 1 if the customer makes a purchase during the next horizon_days
      - features are calculated ONLY from history
    """
    df = retail.copy()

    customer_col = _find_column(df, ["CustomerID", "Customer Id", "Customer_ID"])
    date_col = _find_column(df, ["InvoiceDate", "Invoice_Date", "Date", "TransactionDate", "Transaction_Date"])
    amount_col = _find_column(df, ["Total_Amount", "TotalAmount", "Amount", "Revenue", "Sales"])

    if not customer_col or not date_col:
        raise ValueError(
            "Purchase prediction needs transaction-level data with CustomerID and InvoiceDate/date. "
            "The customer-feature file alone cannot create a valid future-purchase target."
        )

    df[customer_col] = df[customer_col].astype(str).str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[customer_col, date_col]).copy()

    if amount_col:
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
    else:
        amount_col = None

    df = df.sort_values(date_col)
    max_date = df[date_col].max()
    min_date = df[date_col].min()
    span_days = (max_date - min_date).days

    if span_days < horizon_days * 3:
        raise ValueError(
            f"Only {span_days} days of transaction history are available. "
            f"Use a smaller horizon than {horizon_days} days or provide a longer history."
        )

    # Use the last horizon_days as a future evaluation window.
    cutoff = max_date - pd.Timedelta(days=horizon_days)

    hist = df[df[date_col] < cutoff].copy()
    future = df[(df[date_col] >= cutoff) & (df[date_col] <= max_date)].copy()

    if hist.empty or future.empty:
        raise ValueError("History/future split produced no usable transactions.")

    ref_date = hist[date_col].max() + pd.Timedelta(days=1)

    g = hist.groupby(customer_col)
    features = g[date_col].agg(
        Last_Purchase_Date="max",
        First_Purchase_Date="min",
        Purchase_Frequency="count",
    ).reset_index()

    features["Recency_Days"] = (ref_date - features["Last_Purchase_Date"]).dt.days
    features["Customer_Lifetime_Days"] = (
        features["Last_Purchase_Date"] - features["First_Purchase_Date"]
    ).dt.days.clip(lower=0)

    # Number of unique invoices is preferable when InvoiceNo exists.
    invoice_col = _find_column(hist, ["InvoiceNo", "Invoice_No", "Invoice"])
    if invoice_col:
        freq = hist.groupby(customer_col)[invoice_col].nunique().rename("Number_of_Invoices")
    else:
        freq = g[date_col].nunique().rename("Number_of_Invoices")

    features = features.merge(freq.reset_index(), on=customer_col, how="left")

    if amount_col:
        spending = hist.groupby(customer_col)[amount_col].sum().rename("Total_Spending")
        features = features.merge(spending.reset_index(), on=customer_col, how="left")
    else:
        features["Total_Spending"] = 0.0

    features["Average_Order_Value"] = (
        features["Total_Spending"] / features["Number_of_Invoices"].replace(0, np.nan)
    )

    target_customers = set(future[customer_col].unique())
    features["Purchase_Target"] = features[customer_col].isin(target_customers).astype(int)

    features = features.drop(columns=["Last_Purchase_Date", "First_Purchase_Date"])
    features = features.replace([np.inf, -np.inf], np.nan).dropna()

    if features["Purchase_Target"].nunique() < 2:
        raise ValueError(
            "The future window contains only one target class. "
            "Change the horizon or provide more transaction history."
        )

    return features

def train_purchase_models(purchase_df):
    target = "Purchase_Target"
    exclude = {target}
    id_cols = [c for c in purchase_df.columns if c.lower() in {"customerid", "customer_id", "customer id"}]
    exclude.update(id_cols)

    candidate = [
        "Recency_Days",
        "Purchase_Frequency",
        "Number_of_Invoices",
        "Total_Spending",
        "Average_Order_Value",
        "Customer_Lifetime_Days",
    ]
    features = [c for c in candidate if c in purchase_df.columns and c not in exclude]

    if len(features) < 2:
        raise ValueError("Not enough leakage-safe numeric features for prediction.")

    X = purchase_df[features].apply(pd.to_numeric, errors="coerce")
    y = purchase_df[target].astype(int)

    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two customers.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1
            )),
        ]),
        "Gradient Boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]),
    }

    results = []
    fitted = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob),
        })
        fitted[name] = model

    metrics = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return fitted, features, metrics, X_test, y_test
