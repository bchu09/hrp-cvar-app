import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from groq import Groq 
import requests
from vnstock import Vnstock
from datetime import datetime

# --- 1. CẤU HÌNH HỆ THỐNG & AI ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
GOLD_URL = "https://api.vnappmob.com/api/v2/gold/sjc"
GOLD_TOKEN = os.getenv("GOLD_TOKEN")

def get_ai_advice(prompt_text):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system", 
                    "content": (
                        "Bạn là chuyên gia phân tích định lượng tại một công ty đầu tư. "
                        "Hãy đánh giá danh mục dựa trên phân bổ HRP-CVaR bằng cách: "
                        "(1) nhận xét mức độ rủi ro đuôi (tail risk), "
                        "(2) phân tích cấu trúc phân cấp tài sản và mức độ đa dạng hóa, "
                        "(3) so sánh khả năng phòng vệ của CVaR so với cách phân bổ thông thường. "
                        "Giải thích ngắn gọn, chuyên môn nhưng dễ hiểu cho nhà đầu tư phổ thông. "
                        "Tối đa 3 câu, không dùng thuật ngữ quá kỹ thuật."
                    )
                },
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.6
        )
        return completion.choices[0].message.content
    except: return "Hệ thống AI đang bận xử lý dữ liệu định lượng."

def get_sjc_realtime():
    headers = {'Authorization': f'Bearer {GOLD_TOKEN}'}
    try:
        res = requests.get(GOLD_URL, headers=headers, timeout=5).json()
        price_tael = float(res['results'][0]['sell_1l'])
        return price_tael
    except: return None

def get_stock_price_hybrid(symbol, df_prices):
    try:
        stock = Vnstock().stock(symbol=symbol, source='VCI')
        rt_price = stock.trading.price_board()['Lần cuối'].values[0] * 1000
        if rt_price > 0: return rt_price
    except: pass
    return df_prices[symbol].iloc[-1] * 1000

@st.cache_data
def load_data():
    try:
        p = pd.read_csv("cleaned_prices_final.csv", index_col=0, parse_dates=True)
        r = pd.read_csv("cleaned_returns_final.csv", index_col=0, parse_dates=True)
        h = pd.read_csv("historical_performance.csv", index_col=0, parse_dates=True)
        return p, r, h
    except: return None, None, None

df_prices, df_ret, df_perf = load_data()

# --- 2. THUẬT TOÁN HRP-CVAR ---
def get_hrp_weights(returns):
    corr = returns.corr().fillna(0)
    dist = np.sqrt(2 * (1 - corr).clip(lower=0, upper=2))
    dist_val = (dist.values + dist.values.T) / 2
    np.fill_diagonal(dist_val, 0)
    link = sch.linkage(squareform(dist_val), method='ward')
    sort_ix = returns.columns[sch.leaves_list(link)].tolist()
    
    def calc_cvar(r):
        q = r.quantile(0.05)
        return -r[r <= q].mean() if len(r[r <= q]) > 0 else 0.0001

    w = pd.Series(1.0, index=sort_ix)
    items = [sort_ix]
    while len(items) > 0:
        items = [i[j:k] for i in items for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
        for i in range(0, len(items), 2):
            c0, c1 = items[i], items[i+1]
            cv0, cv1 = calc_cvar(returns[c0].mean(axis=1)), calc_cvar(returns[c1].mean(axis=1))
            alpha = 1 - cv0 / (cv0 + cv1 + 1e-9)
            alpha = np.clip(alpha, 0.2, 0.8)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w

# --- 3. GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="HRP-CVaR Portfolio Engine", layout="wide")

if df_prices is not None:
    st.sidebar.title("Hệ thống tư vấn")
    che_do = st.sidebar.radio("Chọn mục tiêu đầu tư:", ["Tối ưu danh mục hiện có", "Xây dựng danh mục mới"])
    
    gold_price_real = get_sjc_realtime()

    # --- TÌNH HUỐNG 1: TÁI CƠ CẤU ---
    if che_do == "Tối ưu danh mục hiện có":
        st.title("Tối ưu danh mục hiện có")
        selected = st.sidebar.multiselect("Chọn mã đang sở hữu:", options=df_ret.columns.tolist())
        
        holdings = {a: st.sidebar.number_input(f"Số lượng {a}:", min_value=0.0, step=1.0, key=f"h_{a}") for a in selected}
        
        if len(selected) >= 2:
            prices = {a: (gold_price_real if a=='Gold' else get_stock_price_hybrid(a, df_prices)) for a in selected}
            cur_vals = {a: holdings[a] * prices[a] for a in selected}
            total = sum(cur_vals.values())
            
            if total > 0:
                cur_w = {a: v/total for a, v in cur_vals.items()}
                opt_w = get_hrp_weights(df_ret[selected])
                
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(go.Figure(data=[go.Pie(labels=list(cur_w.keys()), values=list(cur_w.values()), hole=.4)]).update_layout(title="Tỷ trọng hiện tại"), use_container_width=True)
                with c2:
                    st.plotly_chart(go.Figure(data=[go.Pie(labels=opt_w.index, values=opt_w.values, hole=.4)]).update_layout(title="Tỷ trọng đề xuất (HRP-CVaR)"), use_container_width=True)
                
                st.subheader("Bảng khuyến nghị điều chỉnh chi tiết")
                advice_list = []
                for a in selected:
                    diff_money = (opt_w[a] - cur_w[a]) * total
                    action = "Mua thêm" if diff_money > 10000 else ("Bán bớt" if diff_money < -10000 else "Giữ nguyên")
                    advice_list.append({"Tài sản": a, "Giá thực tế (VND)": prices[a], "Hành động": action, "Số lượng cần thay đổi": abs(diff_money / prices[a])})
                st.table(pd.DataFrame(advice_list).style.format("{:,.0f}", subset=["Giá thực tế (VND)"]).format("{:.2f}", subset=["Số lượng cần thay đổi"]))
                
                if st.button("Hỏi ý kiến AI về danh mục này"):
                    with st.spinner("AI đang phân tích rủi ro đuôi..."):
                        prompt = f"Danh mục tập trung {cur_w} nhưng HRP-CVaR đề xuất {opt_w.to_dict()}. Hãy giải thích tại sao cần điều chỉnh để tối ưu biên an toàn CVaR."
                        st.info(get_ai_advice(prompt))

    # --- TÌNH HUỐNG 2: ĐẦU TƯ MỚI ---
    else:
        st.title("Xây dựng danh mục mới")
        von = st.sidebar.number_input("Vốn đầu tư (VND):", min_value=1000000, value=100000000)
        selected_new = st.sidebar.multiselect("Chọn mã muốn đầu tư:", options=df_ret.columns.tolist())
        
        if len(selected_new) >= 2:
            prices_new = {a: (gold_price_real if a=='Gold' else get_stock_price_hybrid(a, df_prices)) for a in selected_new}
            opt_w_new = get_hrp_weights(df_ret[selected_new])
            
            st.plotly_chart(go.Figure(data=[go.Pie(labels=opt_w_new.index, values=opt_w_new.values, hole=.4)]).update_layout(title="Phân bổ vốn tối ưu"), use_container_width=True)
            
            res_df = pd.DataFrame({
                "Tài sản": selected_new,
                "Giá thực tế (VND)": [prices_new[a] for a in selected_new],
                "Tỷ trọng (%)": [opt_w_new[a]*100 for a in selected_new],
                "Số tiền mua (VND)": [opt_w_new[a]*von for a in selected_new],
                "Số lượng mua dự kiến": [(opt_w_new[a]*von)/prices_new[a] for a in selected_new]
            })
            st.table(res_df.style.format("{:,.0f}", subset=["Giá thực tế (VND)", "Số tiền mua (VND)"]).format("{:.2f}", subset=["Tỷ trọng (%)", "Số lượng mua dự kiến"]))
            
            if st.button("Hỏi ý kiến AI về chiến lược này"):
                with st.spinner("AI đang lập chiến lược phòng vệ..."):
                    prompt = f"Người dùng đầu tư {von} VND vào {selected_new} theo HRP-CVaR {opt_w_new.to_dict()}. Hãy nhận xét ưu điểm cấu trúc phân cấp này."
                    st.success(get_ai_advice(prompt))

    # --- 4. BACKTEST (FIX TRỤC X) ---
    st.markdown("---")
    st.subheader("Hiệu suất tích lũy lịch sử (Backtest 2018 - 2026)")
    if df_perf is not None:
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=df_perf.index, y=df_perf['HRP-CVaR'], name='HRP-CVaR', line=dict(color='#0066cc')))
        fig_perf.add_trace(go.Scatter(x=df_perf.index, y=df_perf['MVO-MaxSharpe'], name='MVO-MaxSharpe', line=dict(color='#87ceeb')))
        fig_perf.update_layout(
            template="plotly_white",
            xaxis=dict(tickformat="%Y", dtick="M12", title="Năm"), # Chỉ hiện Năm
            yaxis=dict(title="Tăng trưởng tích lũy (Base = 1)"),
            height=400
        )
        st.plotly_chart(fig_perf, use_container_width=True)

else:
    st.error("Lỗi: Hãy chạy file test_2.py trước để tạo dữ liệu!")