from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Smart Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- TITLE ---------------- #

st.title("📊 Smart Sales Analytics Dashboard")
st.markdown(
    """
    ### 🚀 Real-Time Business Intelligence Dashboard
    
    Analyze sales performance, forecast trends, and generate AI-driven business insights.
    """
)
# ---------------- LOAD DATA ---------------- #

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding="latin1")
else:
    df = pd.read_csv("data/superstore.csv", encoding="latin1")
    
df["Order Date"] = pd.to_datetime(df["Order Date"])
# ---------------- SIDEBAR ---------------- #

st.sidebar.header("🔍 Filters")
st.sidebar.markdown("### 📊 Dashboard Controls")

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

date_range = st.sidebar.date_input(
    "📅 Select Date Range",
    [
        df["Order Date"].min(),
        df["Order Date"].max()
    ]
)

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Order Date"] >= pd.to_datetime(date_range[0])) &
    (df["Order Date"] <= pd.to_datetime(date_range[1]))
]
csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name='filtered_sales_data.csv',
    mime='text/csv'
)
# ---------------- KPI SECTION ---------------- #

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()
avg_sales = filtered_df["Sales"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
col2.metric("📈 Total Profit", f"${total_profit:,.0f}")
col3.metric("🛒 Total Orders", total_orders)
col4.metric("📦 Average Sales", f"${avg_sales:,.0f}")

st.markdown("---")

# ---------------- CHARTS ---------------- #

col5, col6 = st.columns(2)

# Sales by Region
sales_by_region = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig1 = px.bar(
    sales_by_region,
    x="Region",
    y="Sales",
    color="Region",
    title="Sales by Region"
)

col5.plotly_chart(fig1, width='stretch')

# Profit by Category
profit_by_category = (
    filtered_df.groupby("Category")["Profit"]
    .sum()
    .reset_index()
)

fig2 = px.pie(
    profit_by_category,
    names="Category",
    values="Profit",
    title="Profit by Category"
)

col6.plotly_chart(fig2, width='stretch')

# ---------------- SALES TREND ---------------- #



sales_trend = (
    filtered_df.groupby("Order Date")["Sales"]
    .sum()
    .reset_index()
)

fig3 = px.line(
    sales_trend,
    x="Order Date",
    y="Sales",
    title="Sales Trend Over Time"
)

st.plotly_chart(fig3, width='stretch')
# ---------------- FORECASTING ---------------- #

st.subheader("📈 Sales Forecasting")

forecast_df = sales_trend.copy()

forecast_df["Days"] = np.arange(len(forecast_df))

X = forecast_df[["Days"]]
y = forecast_df["Sales"]

model = LinearRegression()
model.fit(X, y)

future_days = 30

future_x = np.arange(len(forecast_df), len(forecast_df) + future_days).reshape(-1, 1)

predictions = model.predict(future_x)

future_dates = pd.date_range(
    start=forecast_df["Order Date"].max(),
    periods=future_days + 1,
    freq="D"
)[1:]

forecast_future = pd.DataFrame({
    "Order Date": future_dates,
    "Predicted Sales": predictions
})

fig_forecast = px.line(
    forecast_future,
    x="Order Date",
    y="Predicted Sales",
    title="Next 30 Days Sales Forecast"
)

st.plotly_chart(fig_forecast, width='stretch')
# ---------------- AI INSIGHTS ---------------- #

st.subheader("🤖 AI Business Insights")

top_region = sales_by_region.sort_values(
    by="Sales",
    ascending=False
).iloc[0]

top_category = profit_by_category.sort_values(
    by="Profit",
    ascending=False
).iloc[0]

st.success(
    f"🏆 Highest Sales Region: {top_region['Region']} "
    f"with sales of ${top_region['Sales']:,.0f}"
)

st.info(
    f"📈 Most Profitable Category: {top_category['Category']} "
    f"with profit of ${top_category['Profit']:,.0f}"
)

sales_growth = sales_trend["Sales"].iloc[-1] - sales_trend["Sales"].iloc[0]

if sales_growth > 0:
    st.success("📊 Overall sales trend is increasing.")
else:
    st.error("📉 Overall sales trend is decreasing.")

# ---------------- TOP PRODUCTS ---------------- #

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig4 = px.bar(
    top_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    title="Top 10 Products"
)

st.plotly_chart(fig4, width='stretch')

# ---------------- DATA TABLE ---------------- #

st.subheader("📄 Sales Data")

st.dataframe(filtered_df)