
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from backend import (
    load_telecom, load_wholesale, load_online_retail,
    load_customer_features, dataset_info,
    segment_customers, build_future_purchase_dataset,
    train_purchase_models
)

st.set_page_config(page_title="AI-Driven Customer Analytics", page_icon="📊", layout="wide")

st.title("AI-Driven Customer Analytics")
st.caption("Purchase Prediction • Customer Segmentation • Business Intelligence")

@st.cache_data
def load_all():
    return load_telecom(), load_wholesale(), load_online_retail(), load_customer_features()

try:
    telecom, wholesale, retail, customers = load_all()
except Exception as e:
    st.error("Dataset loading failed.")
    st.code(str(e))
    st.info("Create a data folder beside app.py and place the four CSV files inside it.")
    st.stop()

pages = [
    "Dashboard", "Dataset Explorer", "EDA", "Customer Intelligence",
    "Purchase Prediction", "Customer Segmentation", "Business Intelligence",
    "Download Results"
]
page = st.sidebar.radio("Navigation", pages)

if page == "Dashboard":
    st.header("Executive Dashboard")
    a, b, c, d = st.columns(4)
    a.metric("Telecom Customers", f"{len(telecom):,}")
    b.metric("Wholesale Customers", f"{len(wholesale):,}")
    d.metric("Retail Transactions", f"{len(retail):,}")
    if "CustomerID" in retail.columns:
        c.metric("Retail Customers", f"{retail['CustomerID'].nunique():,}")
    elif "Customer_ID" in retail.columns:
        c.metric("Retail Customers", f"{retail['Customer_ID'].nunique():,}")
    else:
        c.metric("Retail Customers", "N/A")

    st.divider()
    left, right = st.columns(2)

    with left:
        if "Month" in retail.columns and "Total_Amount" in retail.columns:
            monthly = retail.groupby("Month", as_index=False)["Total_Amount"].sum()
            st.plotly_chart(
                px.line(monthly, x="Month", y="Total_Amount", markers=True, title="Monthly Revenue"),
                use_container_width=True
            )
        else:
            st.info("Month/Total_Amount columns are not available for the revenue chart.")

    with right:
        if "Total_Spending" in customers.columns:
            st.plotly_chart(
                px.histogram(customers, x="Total_Spending", nbins=40, title="Customer Spending Distribution"),
                use_container_width=True
            )

    st.subheader("Dataset Health")
    health = pd.DataFrame([
        {"Dataset": "Telecom", **dataset_info(telecom)},
        {"Dataset": "Wholesale", **dataset_info(wholesale)},
        {"Dataset": "Online Retail", **dataset_info(retail)},
        {"Dataset": "Customer Features", **dataset_info(customers)},
    ])
    st.dataframe(health, use_container_width=True, hide_index=True)

elif page == "Dataset Explorer":
    st.header("Dataset Explorer")
    name = st.selectbox("Select Dataset", ["Telecom", "Wholesale", "Online Retail", "Customer Features"])
    df = {"Telecom": telecom, "Wholesale": wholesale, "Online Retail": retail, "Customer Features": customers}[name]
    info = dataset_info(df)
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{info['Rows']:,}")
    b.metric("Columns", info["Columns"])
    c.metric("Missing Values", info["Missing Values"])
    d.metric("Duplicates", info["Duplicates"])
    st.subheader("Data Preview")
    st.dataframe(df.head(100), use_container_width=True)
    st.subheader("Column Information")
    ci = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Unique": df.nunique().values
    })
    st.dataframe(ci, use_container_width=True, hide_index=True)

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    name = st.selectbox("Select Dataset", ["Telecom", "Wholesale", "Online Retail"])
    df = {"Telecom": telecom, "Wholesale": wholesale, "Online Retail": retail}[name]

    nums = df.select_dtypes(include=np.number).columns.tolist()
    if nums:
        col = st.selectbox("Numeric Variable", nums)
        st.plotly_chart(px.histogram(df, x=col, nbins=40, title=f"{col} Distribution"), use_container_width=True)

    if name == "Telecom" and "Churn" in df.columns:
        q = df["Churn"].value_counts().reset_index()
        q.columns = ["Churn", "Customers"]
        st.plotly_chart(px.pie(q, names="Churn", values="Customers", title="Churn Distribution"), use_container_width=True)

    if name == "Wholesale" and "Channel" in df.columns:
        q = df["Channel"].value_counts().reset_index()
        q.columns = ["Channel", "Customers"]
        st.plotly_chart(px.bar(q, x="Channel", y="Customers", title="Customers by Channel"), use_container_width=True)

    if name == "Online Retail" and {"Month", "Total_Amount"}.issubset(df.columns):
        q = df.groupby("Month", as_index=False)["Total_Amount"].sum()
        st.plotly_chart(px.line(q, x="Month", y="Total_Amount", markers=True, title="Monthly Revenue"), use_container_width=True)

elif page == "Customer Intelligence":
    st.header("Customer Intelligence")
    id_col = "CustomerID" if "CustomerID" in customers.columns else ("Customer_ID" if "Customer_ID" in customers.columns else None)
    if not id_col:
        st.warning("Customer ID column is not available.")
    else:
        cid = st.selectbox("Customer ID", customers[id_col].dropna().astype(str).tolist())
        row = customers[customers[id_col].astype(str) == cid].iloc[0]
        a, b, c, d = st.columns(4)
        for box, label, col, fmt in [
            (a, "Total Spending", "Total_Spending", ",.2f"),
            (b, "Orders", "Number_of_Invoices", ",.0f"),
            (c, "Avg Order Value", "Average_Order_Value", ",.2f"),
            (d, "Unique Products", "Unique_Products", ",.0f"),
        ]:
            if col in customers.columns:
                box.metric(label, format(row[col], fmt))
        st.dataframe(pd.DataFrame({"Metric": row.index, "Value": row.values}), use_container_width=True, hide_index=True)

elif page == "Purchase Prediction":
    st.header("Leakage-Safe Future Purchase Prediction")
    st.warning(
        "The old version defined the target as Number_of_Invoices >= 2 while also using "
        "historical invoice/spending variables as inputs. That is target leakage and can make "
        "the reported accuracy look artificially high. This version predicts purchases in a "
        "future time window using only earlier transactions."
    )

    horizon = st.slider("Future purchase horizon (days)", 7, 60, 30, 1)

    if st.button("Train and Evaluate Models", type="primary"):
        try:
            with st.spinner("Building historical features and training models..."):
                purchase_df = build_future_purchase_dataset(retail, horizon)
                fitted, features, metrics, X_test, y_test = train_purchase_models(purchase_df)

            st.session_state["purchase_df"] = purchase_df
            st.session_state["purchase_models"] = fitted
            st.session_state["purchase_features"] = features
            st.session_state["purchase_metrics"] = metrics
            st.session_state["purchase_test"] = (X_test, y_test)
        except Exception as e:
            st.error(str(e))

    if "purchase_metrics" in st.session_state:
        metrics = st.session_state["purchase_metrics"]
        st.subheader("Model Comparison")
        st.dataframe(
            metrics.style.format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}",
                "Recall": "{:.2%}", "F1": "{:.2%}", "ROC-AUC": "{:.3f}"
            }),
            use_container_width=True, hide_index=True
        )

        st.subheader("Target Distribution")
        p = st.session_state["purchase_df"]["Purchase_Target"].value_counts().rename(index={0: "No Future Purchase", 1: "Future Purchase"}).reset_index()
        p.columns = ["Class", "Customers"]
        st.plotly_chart(px.bar(p, x="Class", y="Customers", title="Future Purchase Target Distribution"), use_container_width=True)

        model_name = st.selectbox("Prediction Model", list(st.session_state["purchase_models"].keys()))
        model = st.session_state["purchase_models"][model_name]
        features = st.session_state["purchase_features"]

        st.subheader("Predict for a New Customer")
        values = {}
        cols = st.columns(2)
        for i, f in enumerate(features):
            med = float(st.session_state["purchase_df"][f].median())
            values[f] = cols[i % 2].number_input(f, value=med)

        if st.button("Predict Future Purchase"):
            inp = pd.DataFrame([values], columns=features)
            prob = model.predict_proba(inp)[0, 1]
            pred = int(prob >= 0.5)
            st.metric("Future Purchase Probability", f"{prob:.2%}")
            if pred:
                st.success("Predicted: Future Purchase")
            else:
                st.info("Predicted: No Future Purchase")

        st.download_button(
            "Download Model Metrics",
            metrics.to_csv(index=False),
            "purchase_model_metrics.csv",
            "text/csv"
        )

elif page == "Customer Segmentation":
    st.header("Customer Segmentation")
    k = st.slider("Number of Segments", 2, 8, 4)
    if st.button("Run Segmentation", type="primary"):
        try:
            segmented, summary = segment_customers(customers, k)
            st.session_state["segmented"] = segmented
            st.session_state["summary"] = summary
        except Exception as e:
            st.error(str(e))

    if "segmented" in st.session_state:
        st.subheader("Segment Summary")
        st.dataframe(st.session_state["summary"], use_container_width=True, hide_index=True)
        a, b = st.columns(2)
        with a:
            st.plotly_chart(
                px.scatter(
                    st.session_state["segmented"],
                    x="Recency_Days", y="Total_Spending",
                    color="Segment_Label", size="Number_of_Invoices",
                    title="Customer Segments"
                ),
                use_container_width=True
            )
        with b:
            st.plotly_chart(
                px.bar(
                    st.session_state["summary"],
                    x="Segment_Label", y="Customers",
                    title="Customers per Segment"
                ),
                use_container_width=True
            )
        st.download_button(
            "Download Segmented Customers",
            st.session_state["segmented"].to_csv(index=False),
            "customer_segments.csv",
            "text/csv"
        )

elif page == "Business Intelligence":
    st.header("Business Intelligence")
    if "Total_Spending" in customers.columns:
        q75 = customers["Total_Spending"].quantile(.75)
        high = customers[customers["Total_Spending"] >= q75].sort_values("Total_Spending", ascending=False)
        st.info(f"{len(high):,} customers are in the top 25% by spending.")
        st.dataframe(high.head(20), use_container_width=True, hide_index=True)
    if "Recency_Days" in customers.columns:
        q75 = customers["Recency_Days"].quantile(.75)
        risk = customers[customers["Recency_Days"] >= q75].sort_values("Recency_Days", ascending=False)
        st.info(f"{len(risk):,} customers are in the highest-recency quartile.")
        st.dataframe(risk.head(20), use_container_width=True, hide_index=True)

elif page == "Download Results":
    st.header("Download Results")
    for label, df, filename in [
        ("Telecom", telecom, "telecom_cleaned.csv"),
        ("Wholesale", wholesale, "wholesale_cleaned.csv"),
        ("Online Retail", retail, "online_retail_cleaned.csv"),
        ("Customer Features", customers, "online_retail_customer_features.csv"),
    ]:
        st.download_button(f"Download {label}", df.to_csv(index=False), filename, "text/csv")
