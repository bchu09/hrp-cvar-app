import pandas as pd
import numpy as np
import yfinance as yf
from vnstock import Vnstock, register_user
from datetime import datetime
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
# Thêm các module từ PyPortfolioOpt
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- 1. CAU HINH HE THONG ---
import os
register_user(api_key=os.getenv("VNSTOCK_API_KEY"))
START_DATE = '2018-01-01'
TODAY = datetime.now().strftime('%Y-%m-%d')

# --- 2. CAC HAM THUAT TOAN HRP-CVAR ---
def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i, j = df0.index, df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]).sort_index()
    return sort_ix.tolist()

def get_rec_bisection_cvar(returns, sort_ix):
    w = pd.Series(1.0, index=sort_ix)
    items = [sort_ix]
    while len(items) > 0:
        items = [i[j:k] for i in items for j, k in ((0, len(i)//2), (len(i)//2, len(i))) if len(i) > 1]
        for i in range(0, len(items), 2):
            c0, c1 = items[i], items[i+1]
            def calc_cvar(r):
                port_r = r.mean(axis=1)
                q = port_r.quantile(0.05)
                tail = port_r[port_r <= q]
                return -tail.mean() if len(tail) > 0 else 0.0001
            
            cv0, cv1 = calc_cvar(returns[c0]), calc_cvar(returns[c1])
            alpha = 1 - cv0 / (cv0 + cv1 + 1e-9)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
    return w

# --- MOI: THUAT TOAN MVO MAX SHARPE (Dung PyPortfolioOpt) ---
def get_mvo_weights(train_returns):
    try:
        # Tính toán lợi nhuận kỳ vọng và ma trận hiệp phương sai
        mu = expected_returns.mean_historical_return(train_returns, returns_data=True)
        S = risk_models.sample_cov(train_returns, returns_data=True)
        
        # Tối ưu hóa Max Sharpe
        ef = EfficientFrontier(mu, S)
        # Ràng buộc long-only (trọng số từ 0 đến 1) đã được mặc định trong EF
        raw_weights = ef.max_sharpe(risk_free_rate=0.04)
        cleaned_weights = ef.clean_weights()
        
        return pd.Series(cleaned_weights)
    except Exception as e:
        # Fallback nếu tối ưu hóa không hội tụ
        n = train_returns.shape[1]
        return pd.Series(1/n, index=train_returns.columns)

# --- 3. QUY TRINH TAI VA XU LY DU LIEU (GIE NGUYEN) ---
def update_dataset():
    print("Bat dau tai du lieu VN30...")
    symbols = ['ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG', 
               'MBB', 'MSN', 'MWG', 'PLX', 'POW', 'REE', 'SAB', 'SHB', 'SSB', 'SSI', 
               'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VNM', 'VPB', 'VRE']
    
    all_stocks = {}
    for s in symbols:
        try:
            stock = Vnstock().stock(symbol=s, source='KBS')
            df = stock.quote.history(start=START_DATE, end=TODAY)
            if not df.empty:
                df['time'] = pd.to_datetime(df['time']).dt.normalize()
                df = df.drop_duplicates(subset='time', keep='last')
                all_stocks[s] = df.set_index('time')['close']
                print(f"Da tai: {s}")
        except: continue
    
    df_vn30 = pd.DataFrame(all_stocks)
    print("Dang quy doi gia Vang...")
    gold_raw = yf.download(["GC=F", "USDVND=X"], start=START_DATE, end=TODAY)['Close']
    g_val = gold_raw['GC=F'].iloc[:, 0] if isinstance(gold_raw['GC=F'], pd.DataFrame) else gold_raw['GC=F']
    f_val = gold_raw['USDVND=X'].iloc[:, 0] if isinstance(gold_raw['USDVND=X'], pd.DataFrame) else gold_raw['USDVND=X']
    gold_vnd = (g_val * f_val).rename("Gold")

    full_df = pd.concat([df_vn30, gold_vnd], axis=1).sort_index()
    full_df = full_df.ffill().bfill() 
    full_df.to_csv("cleaned_prices_final.csv")
    
    returns = np.log(full_df / full_df.shift(1)).dropna()
    returns.to_csv("cleaned_returns_final.csv")
    print("Da xuat xong file gia va file return.")
    return returns

# --- 4. CHAY BACKTEST SO SANH HRP-CVAR vs MVO ---
def run_backtest_for_app(returns):
    print("Dang chay Backtest (Rolling Window): HRP-CVaR vs MVO Max Sharpe...")
    window = 252
    step = 21
    hrp_returns = []
    mvo_returns = []
    
    for i in range(window, len(returns) - step, step):
        train = returns.iloc[i-window:i]
        test = returns.iloc[i:i+step]
        
        # 1. Tinh HRP-CVaR
        corr = train.corr().fillna(0)
        dist = np.sqrt((1 - corr) / 2).clip(lower=0, upper=1)
        dist_val = dist.values
        i_upper = np.triu_indices(dist_val.shape[0], k=1)
        dist_val[i_upper] = dist_val.T[i_upper]
        np.fill_diagonal(dist_val, 0)
        
        link = sch.linkage(squareform(dist_val), method='ward')
        sort_ix = train.columns[sch.leaves_list(link)].tolist()
        w_hrp = get_rec_bisection_cvar(train, sort_ix)
        
        # 2. Tinh MVO Max Sharpe
        w_mvo = get_mvo_weights(train)
        
        # Ket qua (đảm bảo dot product đúng thứ tự cột)
        hrp_returns.append(test.dot(w_hrp))
        mvo_returns.append(test.dot(w_mvo[test.columns]))
        
    res_hrp = np.exp(pd.concat(hrp_returns).cumsum())
    res_mvo = np.exp(pd.concat(mvo_returns).cumsum())
    
    performance = pd.DataFrame({'HRP-CVaR': res_hrp, 'MVO-MaxSharpe': res_mvo})
    performance.to_csv("historical_performance.csv")
    print("Da xuat xong file historical_performance.csv")

# --- THUC THI ---
if __name__ == "__main__":
    df_ret = update_dataset()
    run_backtest_for_app(df_ret)