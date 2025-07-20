import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from io import StringIO
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Portfolio Analytics Suite", layout="wide")

st.sidebar.title("Portfolio Analyzer")
st.sidebar.write("Input your stock portfolio:")

# --- Portfolio Input ---
portfolio_method = st.sidebar.radio("Portfolio entry method:", ["Manual Entry", "Upload CSV"])

if portfolio_method == "Manual Entry":
    tickers = st.sidebar.text_area("Enter stock tickers (comma-separated)", "AAPL,MSFT,GOOGL")
    quantities = st.sidebar.text_area("Enter quantities (comma-separated)", "10,5,3")
    try:
        symbols = [t.strip().upper() for t in tickers.split(",")]
        qtys = [float(q) for q in quantities.split(",")]
        portfolio_df = pd.DataFrame({'Ticker': symbols, 'Quantity': qtys})
    except Exception:
        st.sidebar.error("Check your tickers/quantities formatting!")
        st.stop()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (columns: Ticker,Quantity)", type=["csv"])
    if uploaded_file:
        try:
            portfolio_df = pd.read_csv(uploaded_file)
            portfolio_df['Ticker'] = portfolio_df['Ticker'].str.upper()
        except Exception as e:
            st.sidebar.error("CSV parsing error.")
            st.stop()
    else:
        st.stop()

start_date = st.sidebar.date_input("Start date", datetime.date.today()-datetime.timedelta(365))
end_date = st.sidebar.date_input("End date", datetime.date.today())

benchmark = st.sidebar.selectbox("Benchmark index", ["^GSPC", "^DJI", "^IXIC"])
investment_amount = st.sidebar.number_input("Amount for optimization ($)", value=10000, step=100)

# --- Data Download ---
@st.cache_data(show_spinner=False)
def get_data(tickers, start, end):
    data = {t: yf.download(t, start=start, end=end) for t in tickers}
    info = {t: yf.Ticker(t).info for t in tickers}
    return data, info

tickers_all = list(portfolio_df['Ticker']) + [benchmark]
data, info = get_data(tickers_all, start_date, end_date)

# --- Prepare Portfolio Prices ---
prices = pd.concat([data[t]['Adj Close'].rename(t) for t in portfolio_df['Ticker']], axis=1).dropna()
benchmark_price = data[benchmark]['Adj Close'].rename(benchmark).dropna()
aligned_idx = prices.index.intersection(benchmark_price.index)
prices, benchmark_price = prices.loc[aligned_idx], benchmark_price.loc[aligned_idx]

weights = portfolio_df['Quantity'] * prices.iloc[0].values
weights = weights / weights.sum()

returns = prices.pct_change().dropna()
benchmark_ret = benchmark_price.pct_change().dropna()

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Overview", "🔬 Risk Analysis", "💡 Optimization", "📋 Detailed Data"])

# 1. PERFORMANCE OVERVIEW
with tab1:
    col1, col2 = st.columns([2, 1])
    total_return = (prices.iloc[-1].dot(portfolio_df['Quantity']) - prices.iloc[0].dot(portfolio_df['Quantity'])) / prices.iloc[0].dot(portfolio_df['Quantity'])
    bench_return = (benchmark_price.iloc[-1] - benchmark_price.iloc[0]) / benchmark_price.iloc[0]
    annual_vol = returns.mul(weights, axis=1).sum(axis=1).std() * np.sqrt(252)

    col1.metric("Portfolio Return", f"{100*total_return:.2f}%")
    col1.metric("Benchmark Return", f"{100*bench_return:.2f}%")
    col1.metric("Annualized Volatility", f"{100*annual_vol:.2f}%")
    
    # Cumulative Chart
    cum_port = (1 + returns.mul(weights, axis=1).sum(axis=1)).cumprod()
    cum_bench = (1 + benchmark_ret).cumprod()
    fig, ax = plt.subplots()
    ax.plot(cum_port, label="Portfolio")
    ax.plot(cum_bench, label="Benchmark")
    ax.set_title("Cumulative Returns")
    ax.legend()
    col1.pyplot(fig)
    
    # Pie Charts
    allocation = portfolio_df['Quantity'] * prices.iloc[-1].values
    fig1, ax1 = plt.subplots()
    ax1.pie(allocation, labels=portfolio_df['Ticker'], autopct='%1.1f%%')
    ax1.set_title("Allocation by Asset")
    col2.pyplot(fig1)
    
    sectors = [info[t]['sector'] if 'sector' in info[t] else "Unknown" for t in portfolio_df['Ticker']]
    df_sector = pd.DataFrame({'Sector': sectors, 'Value': allocation}).groupby('Sector').sum()
    fig2, ax2 = plt.subplots()
    ax2.pie(df_sector['Value'], labels=df_sector.index, autopct='%1.1f%%')
    ax2.set_title("Allocation by Sector")
    col2.pyplot(fig2)

# 2. RISK ANALYSIS
with tab2:
    sharpe = (returns.mul(weights, axis=1).sum(axis=1).mean() - benchmark_ret.mean()) / returns.mul(weights, axis=1).sum(axis=1).std() * np.sqrt(252)
    var_95 = np.percentile(returns.mul(weights, axis=1).sum(axis=1), 5)
    colA, colB = st.columns(2)
    colA.metric("Sharpe Ratio", f"{sharpe:.2f}")
    colA.metric("Value at Risk (5%)", f"{100*var_95:.2f}%")
    
    beta_list = []
    for t in portfolio_df['Ticker']:
        cov = np.cov(returns[t], benchmark_ret)[0,1]
        beta = cov / benchmark_ret.var()
        beta_list.append(beta)
    beta_df = pd.DataFrame({'Stock': portfolio_df['Ticker'], 'Beta': beta_list})
    colB.dataframe(beta_df)
    
    # Correlation Heatmap
    fig_c, ax_c = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap='RdBu', ax=ax_c)
    ax_c.set_title("Correlation Matrix")
    st.pyplot(fig_c)
    
    # Drawdown Plot
    running_max = cum_port.cummax()
    drawdown = cum_port / running_max - 1
    fig_d, ax_d = plt.subplots()
    ax_d.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.3)
    ax_d.set_title("Portfolio Drawdown")
    st.pyplot(fig_d)

# 3. OPTIMIZATION & SUGGESTIONS
with tab3:
    # Modern Portfolio Theory Optimization (Max Sharpe)
    def sharpe_neg(w):
        r = (returns * w).sum(axis=1)
        return -(r.mean() - benchmark_ret.mean()) / r.std()
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1})
    bnds = tuple((0, 1) for _ in portfolio_df['Ticker'])
    res = minimize(sharpe_neg, np.ones(len(portfolio_df))/len(portfolio_df), bounds=bnds, constraints=cons)
    opt_w = res.x
    st.metric("Optimal Sharpe Ratio Portfolio", f"{(returns.mul(opt_w, axis=1).sum(axis=1).mean() - benchmark_ret.mean()) / returns.mul(opt_w, axis=1).sum(axis=1).std() * np.sqrt(252):.2f}")
    
    # Buy/Sell/Hold Suggestions
    suggestion = []
    for i, orig in enumerate(weights):
        if opt_w[i] > orig + 0.01: suggestion.append("Buy")
        elif opt_w[i] < orig - 0.01: suggestion.append("Sell")
        else: suggestion.append("Hold")
    rec_df = pd.DataFrame({"Ticker": portfolio_df['Ticker'], "Your Weight": weights, "Optimal Weight": opt_w, "Action": suggestion})
    st.dataframe(rec_df.style.format({'Your Weight':'{:.2%}','Optimal Weight':'{:.2%}'}))
    
    # Discrete Allocation
    latest_prices = prices.iloc[-1].values
    allocated_dollars = investment_amount * opt_w
    discrete = np.floor(allocated_dollars / latest_prices).astype(int)
    disc_alloc_df = pd.DataFrame({'Ticker':portfolio_df['Ticker'], 'Buy Shares': discrete, 'Allocated $': allocated_dollars})
    st.dataframe(disc_alloc_df)
    
    # Monte Carlo Simulation (year forecast)
    num_sim = 200
    forecast = []
    port_mu = returns.mul(opt_w, axis=1).sum(axis=1).mean()
    port_sigma = returns.mul(opt_w, axis=1).sum(axis=1).std()
    S0 = investment_amount
    days = 252
    for _ in range(num_sim):
        sim = [S0]
        for _ in range(days):
            sim.append(sim[-1] * np.exp(np.random.normal(port_mu - port_sigma**2/2, port_sigma)))
        forecast.append(sim)
    forecast = np.array(forecast)
    fig_mc, ax_mc = plt.subplots()
    ax_mc.plot(forecast.T, color="blue", alpha=0.05)
    ax_mc.set_title("Monte Carlo Portfolio Value Forecast")
    ax_mc.set_xlabel("Days")
    ax_mc.set_ylabel("Portfolio Value")
    st.pyplot(fig_mc)

# 4. DETAILED DATA
with tab4:
    cur_val = prices.iloc[-1].values * portfolio_df['Quantity']
    port_weight = cur_val / cur_val.sum()
    detailed_df = pd.DataFrame({'Ticker':portfolio_df['Ticker'], 'Quantity':portfolio_df['Quantity'], 'Latest Price':prices.iloc[-1].values, 'Market Value':cur_val, 'Portfolio Weight':port_weight})
    st.dataframe(detailed_df.style.format({'Latest Price':'${:.2f}','Market Value':'${:.2f}','Portfolio Weight':'{:.2%}'}))
    csv = detailed_df.to_csv(index=False)
    st.download_button("Download Holdings as CSV", csv, "holdings.csv", "text/csv")
    
    # Dividends History
    div_days = pd.date_range(start=start_date, end=end_date)
    div_rows = []
    for t in portfolio_df['Ticker']:
        div = yf.Ticker(t).dividends
        div = div[div.index.isin(div_days)]
        for date, val in div.items():
            div_rows.append({'Ticker': t, 'Date': date, 'Dividend': val})
    div_hist = pd.DataFrame(div_rows)
    if not div_hist.empty:
        st.write("Dividend History in Date Range")
        st.dataframe(div_hist)
    else:
        st.write("No dividends for the selected period.")
