import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from groq import Groq
import requests
from vnstock import Vnstock
import os

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="HRP-CVaR Portfolio Engine",
    layout="wide"
)

# Auto refresh every 60 seconds
st_autorefresh(interval=60000, key="data_refresh")

# ==============================
# API KEYS
# ==============================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOLD_TOKEN = os.getenv("GOLD_TOKEN")

client = Groq(api_key=GROQ_API_KEY)

GOLD_URL = "https://api.vnappmob.com/api/v2/gold/sjc"

# ==============================
# LOAD DATA
# ==============================

@st.cache_data(ttl=600)
def load_data():

    try:

        prices = pd.read_csv(
            "cleaned_prices_final.csv",
            index_col=0,
            parse_dates=True
        )

        returns = pd.read_csv(
            "cleaned_returns_final.csv",
            index_col=0,
            parse_dates=True
        )

        perf = pd.read_csv(
            "historical_performance.csv",
            index_col=0,
            parse_dates=True
        )

        return prices, returns, perf

    except:

        return None, None, None


df_prices, df_ret, df_perf = load_data()

# ==============================
# AI ADVICE
# ==============================

def get_ai_advice(prompt_text):

    try:

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content":
                    "Bạn là chuyên gia phân tích định lượng. "
                    "Hãy nhận xét danh mục HRP-CVaR về tail risk "
                    "và đa dạng hóa. Tối đa 3 câu."
                },
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.6
        )

        return completion.choices[0].message.content

    except:

        return "AI đang bận xử lý."

# ==============================
# GOLD PRICE
# ==============================

def get_sjc_realtime():

    headers = {"Authorization": f"Bearer {GOLD_TOKEN}"}

    try:

        res = requests.get(
            GOLD_URL,
            headers=headers,
            timeout=5
        ).json()

        return float(res["results"][0]["sell_1l"])

    except:

        return None

# ==============================
# STOCK PRICE
# ==============================

def get_stock_price(symbol):

    try:

        stock = Vnstock().stock(symbol=symbol, source="VCI")

        rt_price = stock.trading.price_board()["Lần cuối"].values[0] * 1000

        if rt_price > 0:
            return rt_price

    except:
        pass

    if symbol in df_prices.columns:
        return df_prices[symbol].iloc[-1] * 1000

    return None

# ==============================
# HRP-CVAR
# ==============================

def get_hrp_weights(returns):

    corr = returns.corr().fillna(0)

    dist = np.sqrt(2 * (1 - corr).clip(0, 2))

    dist_val = (dist.values + dist.values.T) / 2

    np.fill_diagonal(dist_val, 0)

    link = sch.linkage(squareform(dist_val), method="ward")

    sort_ix = returns.columns[sch.leaves_list(link)].tolist()

    def calc_cvar(r):

        q = r.quantile(0.05)

        tail = r[r <= q]

        return -tail.mean() if len(tail) > 0 else 0.0001

    w = pd.Series(1.0, index=sort_ix)

    items = [sort_ix]

    while len(items) > 0:

        items = [
            i[j:k]
            for i in items
            for j, k in ((0, len(i)//2), (len(i)//2, len(i)))
            if len(i) > 1
        ]

        for i in range(0, len(items), 2):

            c0 = items[i]
            c1 = items[i+1]

            cv0 = calc_cvar(returns[c0].mean(axis=1))
            cv1 = calc_cvar(returns[c1].mean(axis=1))

            alpha = 1 - cv0/(cv0+cv1+1e-9)

            alpha = np.clip(alpha, 0.2, 0.8)

            w[c0] *= alpha
            w[c1] *= 1-alpha

    return w

# ==============================
# MAIN APP
# ==============================

if df_prices is None:

    st.error("Chưa có dữ liệu. Hãy chạy file backtest trước.")

    st.stop()

st.sidebar.title("Hệ thống tư vấn")

mode = st.sidebar.radio(
    "Chọn mục tiêu đầu tư",
    ["Tối ưu danh mục hiện có", "Xây dựng danh mục mới"]
)

gold_price = get_sjc_realtime()

# ==============================
# PORTFOLIO OPTIMIZATION
# ==============================

if mode == "Tối ưu danh mục hiện có":

    st.title("Tối ưu danh mục hiện có")

    selected = st.sidebar.multiselect(
        "Chọn mã đang sở hữu",
        options=df_ret.columns.tolist()
    )

    holdings = {
        a: st.sidebar.number_input(
            f"Số lượng {a}",
            min_value=0.0,
            step=1.0
        )
        for a in selected
    }

    if len(selected) >= 2:

        prices = {
            a: (gold_price if a == "Gold" else get_stock_price(a))
            for a in selected
        }

        cur_vals = {a: holdings[a]*prices[a] for a in selected}

        total = sum(cur_vals.values())

        if total > 0:

            cur_w = {a: v/total for a,v in cur_vals.items()}

            opt_w = get_hrp_weights(df_ret[selected])

            c1,c2 = st.columns(2)

            with c1:

                fig = go.Figure(data=[go.Pie(
                    labels=list(cur_w.keys()),
                    values=list(cur_w.values()),
                    hole=0.4
                )])

                fig.update_layout(title="Tỷ trọng hiện tại")

                st.plotly_chart(fig,use_container_width=True)

            with c2:

                fig = go.Figure(data=[go.Pie(
                    labels=opt_w.index,
                    values=opt_w.values,
                    hole=0.4
                )])

                fig.update_layout(title="Tỷ trọng đề xuất")

                st.plotly_chart(fig,use_container_width=True)

# ==============================
# NEW PORTFOLIO
# ==============================

else:

    st.title("Xây dựng danh mục mới")

    capital = st.sidebar.number_input(
        "Vốn đầu tư",
        min_value=1000000,
        value=100000000
    )

    selected = st.sidebar.multiselect(
        "Chọn tài sản",
        options=df_ret.columns.tolist()
    )

    if len(selected) >= 2:

        prices = {
            a:(gold_price if a=="Gold" else get_stock_price(a))
            for a in selected
        }

        opt_w = get_hrp_weights(df_ret[selected])

        fig = go.Figure(data=[go.Pie(
            labels=opt_w.index,
            values=opt_w.values,
            hole=0.4
        )])

        fig.update_layout(title="Phân bổ vốn tối ưu")

        st.plotly_chart(fig,use_container_width=True)

# ==============================
# BACKTEST
# ==============================

st.markdown("---")

st.subheader("Hiệu suất tích lũy (Backtest 2018-2026)")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_perf.index,
    y=df_perf["HRP-CVaR"],
    name="HRP-CVaR"
))

fig.add_trace(go.Scatter(
    x=df_perf.index,
    y=df_perf["MVO-MaxSharpe"],
    name="MVO-MaxSharpe"
))

fig.update_layout(
    template="plotly_white",
    xaxis=dict(
        tickformat="%Y",
        dtick="M12",
        title="Year"
    ),
    yaxis=dict(title="Cumulative Growth"),
    height=400
)

st.plotly_chart(fig,use_container_width=True)
