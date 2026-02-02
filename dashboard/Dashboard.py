import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter
import os

sns.set(style="dark")


# HELPER FUNCTION
# Hitung revenue per state
def create_revenue_per_state_df(df):
    revenue_per_state = (
        df.groupby("customer_state")["price"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="revenue")
    )
    top_10_revenue_per_state = revenue_per_state.head(10)
    return top_10_revenue_per_state


# Hitung revenue per city
def create_revenue_per_city_df(df):
    revenue_per_city = (
        df.groupby("customer_city")["price"]
        .sum()
        .sort_values(ascending=False)
        .reset_index(name="revenue")
    )
    top_10_revenue_per_city = revenue_per_city.head(10)
    return top_10_revenue_per_city


# Hitung category performance
def create_category_performance_df(df):
    category_performance_df = (
        df.groupby("product_category_name_english")
        .agg({"price": "sum", "order_item_id": "count"})
        .reset_index()
    )

    # Ganti nama kolom untuk mempermudah
    category_performance_df.rename(
        columns={"price": "revenue", "order_item_id": "quantity"}, inplace=True
    )
    return category_performance_df


# Hitung RFM
def create_rfm_df(df):
    # Ambil tanggal terakhir sejak transaksi terkahir masuk
    max_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    # Hitung masing masing metrik RFM
    rfm_df = df.groupby("customer_unique_id").agg(
        {
            "order_purchase_timestamp": lambda x: (max_date - x.max()).days,
            "order_id": "nunique",
            "price": "sum",
        }
    )

    # Ganti nama kolom
    rfm_df.rename(
        columns={
            "order_purchase_timestamp": "recency",
            "order_id": "frequency",
            "price": "monetary",
        },
        inplace=True,
    )

    # Bagi customer untuk masing masing segment
    # Recency score
    rfm_df["r_score"] = pd.qcut(rfm_df["recency"], 5, labels=[5, 4, 3, 2, 1])
    # Frequency score
    rfm_df["f_score"] = pd.qcut(
        rfm_df["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    )
    # Gabung score menjadi satu string
    rfm_df["rf_score"] = rfm_df["r_score"].astype(str) + rfm_df["f_score"].astype(str)

    seg_map = {
        r"[4-5][4-5]": "Champions",  # 55, 54, 45, 44
        r"3[4-5]": "Loyal customers",  # 35, 34
        r"[3-5][1-3]": "Potential loyalist",  # 53-51, 43-41, 33-31
        r"[1-2][4-5]": "At risk",  # 25, 24, 15, 14
        r"2[2-3]": "Hibernating",  # 23, 22
        r"1[1-3]|21": "Lost",  # 13, 12, 11, 21
    }

    # Map score ke nama segmen menggunakan Regex
    rfm_df["customers_type"] = rfm_df["rf_score"].replace(seg_map, regex=True)

    # Hitung monetary
    revenue_per_type = rfm_df.groupby("customers_type")["monetary"].sum().reset_index()
    return rfm_df, revenue_per_type


# Load data
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "all_data.csv")
all_df = pd.read_csv(csv_path)

# Filtering tanggal
datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(inplace=True)
min_date = all_df["order_purchase_timestamp"].min()
max_date = all_df["order_purchase_timestamp"].max()

with st.sidebar:
    # Logo
    st.header("Filter Data")
    st.image("logo.png", width=95)
    # Input date
    try:
        start_date, end_date = st.date_input(
            label="Rentang Waktu",
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date],
        )
    except ValueError:
        st.error("Mohon pilih rentang tanggal yang valid (Start & End)")
        st.stop()

# Df utama yang akan digunakan
main_df = all_df[
    (all_df["order_purchase_timestamp"] >= str(start_date))
    & (all_df["order_purchase_timestamp"] <= str(end_date))
]

# Panggil fungsi helper
revenue_per_state = create_revenue_per_state_df(main_df)
revenue_per_city = create_revenue_per_city_df(main_df)
rfm_df, revenue_per_type = create_rfm_df(main_df)
category_performance = create_category_performance_df(main_df)

# VISUALISASI
st.header("E-commerce Product and Revenue Performance")
st.subheader("Product Performance by Location")


# Formatter ke Million (M)
def millions_formatter(x, pos):
    return f"{x/1_000_000:.1f}M"


col1, col2 = st.columns(2)

# Visualisasi revenue per state
with col1:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if i < 2 else "#d3d3d3" for i in range(10)]
    sns.barplot(
        x="customer_state",
        y="revenue",
        data=revenue_per_state,
        palette=colors,
        ax=ax,
    )
    ax.set_title("Top 10 States by Revenue", fontsize=15, weight="bold")
    ax.set_xlabel("State", fontsize=12)
    ax.set_ylabel("Revenue", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
    st.pyplot(fig)

# Visualisasi revenue per city
with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#1f77b4" if i < 2 else "#d3d3d3" for i in range(10)]
    sns.barplot(
        x="revenue", y="customer_city", data=revenue_per_city, palette=colors, ax=ax
    )
    ax.set_title("Top 10 Cities by Revenue", fontsize=15, weight="bold")
    ax.set_xlabel("Revenue", fontsize=12)
    ax.set_ylabel("City", fontsize=12)
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    st.pyplot(fig)

# Visualisasi revenue vs quantity
fig, ax = plt.subplots(figsize=(20, 9))
sns.set_style("whitegrid")
# Visualisasi dalam bentuk scatterplot
sns.scatterplot(
    data=category_performance,
    x="quantity",
    y="revenue",
    size="revenue",
    sizes=(50, 600),
    hue="revenue",
    color="#007bff",
    alpha=0.7,
    legend=False,
)

# Ambil top 8 sebagai label
top_points = category_performance.nlargest(8, "revenue")
for i in range(top_points.shape[0]):
    plt.text(
        x=top_points.quantity.iloc[i] + 100,
        y=top_points.revenue.iloc[i],
        s=top_points.product_category_name_english.iloc[i],
        fontdict=dict(color="black", size=10, weight="bold"),
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
    )

# format angka
plt.gca().yaxis.set_major_formatter(FuncFormatter(millions_formatter))

# Atur judul dan label
plt.title("Revenue vs Quantity")
plt.xlabel("Quantity Sold", fontsize=12)
plt.ylabel("Total Revenue (Millions)", fontsize=12)

# Garis rata rata sebagai benchmark
plt.axvline(
    category_performance["quantity"].mean(), color="red", linestyle="--", alpha=0.5
)
plt.axhline(
    category_performance["revenue"].mean(), color="red", linestyle="--", alpha=0.5
)
plt.text(
    category_performance["quantity"].max(),
    category_performance["revenue"].mean(),
    "Avg Revenue",
    color="red",
)
plt.tight_layout()
st.pyplot(fig)

col3, col4 = st.columns(2)
with col3:
    fig, ax = plt.subplots(figsize=(10, 6))
    count_data = rfm_df["customers_type"].value_counts()
    ax.pie(
        count_data,
        labels=count_data.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("viridis", len(count_data)),
    )
    ax.set_title("Customer Type Distribution", weight="bold")
    st.pyplot(fig)

with col4:
    fig, ax = plt.subplots(figsize=(10, 6))
    revenue_per_type = revenue_per_type.sort_values(by="monetary", ascending=False)
    sns.barplot(
        x="monetary",
        y="customers_type",
        color="#007bff",
        data=revenue_per_type,
        ax=ax,
    )
    ax.set_title("Customer Type by Revenue", weight="bold")
    ax.set_xlabel("Total Revenue")
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.bar_label(ax.containers[0], fmt=lambda x: f"{x/1e6:.1f}M", padding=3)
    st.pyplot(fig)
