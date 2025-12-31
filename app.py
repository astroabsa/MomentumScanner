import streamlit as st
import yfinance as yf
import pandas as pd
import time

# Set up the UI
st.set_page_config(page_title="Live F&O Screener", layout="wide")
st.title("🚀 Live F&O Intraday Momentum")

# List of top F&O stocks (Add more as needed)
FNO_SYMBOLS = [
    'ABFRL.NS', 'ADANIENSOL.NS', 'ADANIENT.NS', 'ADANIGREEN.NS', 'ADANIPORTS.NS', 'ALKEM.NS', 
    'AUROPHARMA.NS', 'AXISBANK.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS', 'BANKINDIA.NS', 'BDL.NS', 
    'BEL.NS', 'BEML.NS', 'BHARTIARTL.NS', 'BHEL.NS', 'BIOCON.NS', 'BPCL.NS', 'BRITANNIA.NS', 
    'BSE.NS', 'CAMS.NS', 'CANBK.NS', 'CDSL.NS', 'CGPOWER.NS', 'CHAMBLFERT.NS', 'CHOLAFIN.NS', 
    'CIPLA.NS', 'COALINDIA.NS', 'COFORGE.NS', 'COLPAL.NS', 'CONCOR.NS', 'COROMANDEL.NS', 
    'CROMPTON.NS', 'CUMMINSIND.NS', 'CYIENT.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DEEPAKNTR.NS', 
    'DELHIVERY.NS', 'DIVISLAB.NS', 'DIXON.NS', 'DMART.NS', 'DRREDDY.NS', 'FSL.NS', 'GAIL.NS', 
    'GLENMARK.NS', 'GMRINFRA.NS', 'GNFC.NS', 'GODREJCP.NS', 'GODREJPROP.NS', 'GRANULES.NS', 
    'GUJGASLTD.NS', 'HAL.NS', 'HAVELLS.NS', 'HCLTECH.NS', 'HDFCAMC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 
    'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 
    'ICICIGI.NS', 'IDFC.NS', 'IDFCFIRSTB.NS', 'IEX.NS', 'IGL.NS', 'INDHOTEL.NS', 'INDIACEM.NS', 'INDIAMART.NS', 
    'INDIGO.NS', 'INDUSINDBK.NS', 'INDUSTOWER.NS', 'INFY.NS', 'IOC.NS', 'IPCALAB.NS', 'IRCTC.NS', 'IRFC.NS', 
    'ITC.NS', 'JINDALSTEL.NS', 'JSWSTEEL.NS', 'JUBLFOOD.NS', 'KOTAKBANK.NS', 'LALPATHLAB.NS', 
    'LAURUSLABS.NS', 'LICHSGFIN.NS', 'LICI.NS', 'LT.NS', 'LTIM.NS', 'LTTS.NS', 
    'LUPIN.NS', 'M&M.NS', 'M&MFIN.NS', 'MANAPPURAM.NS', 'MARICO.NS', 'MARUTI.NS', 
    'MCDOWELL-N.NS', 'MCX.NS', 'METROPOLIS.NS', 'MGL.NS', 'MOTHERSON.NS', 'MPHASIS.NS', 
    'MRF.NS', 'MUTHOOTFIN.NS', 'NATIONALUM.NS', 'NAUKRI.NS', 'NAVINFLUOR.NS', 'NBCC.NS', 
    'NESTLEIND.NS', 'NHPC.NS', 'NMDC.NS', 'NTPC.NS', 'NYKAA.NS', 'OBEROIRLTY.NS', 
    'OFSS.NS', 'OIL.NS', 'ONGC.NS', 'PAGEIND.NS', 'PATANJALI.NS', 'PEL.NS', 
    'PERSISTENT.NS', 'PETRONET.NS', 'PFC.NS', 'PHOENIXLTD.NS', 'PIDILITIND.NS', 'PIIND.NS', 
    'PNB.NS', 'POLYCAP.NS', 'POWERTARID.NS', 'PRESTIGE.NS', 'PVRINOX.NS', 'RAMCOCEM.NS', 
    'RBLBANK.NS', 'RECLTD.NS', 'RELIANCE.NS', 'SAIL.NS', 'SBICARD.NS', 'SBILIFE.NS', 
    'SBIN.NS', 'SHREECEM.NS', 'SHRIRAMFIN.NS', 'SIEMENS.NS', 'SONACOMS.NS', 'SRF.NS', 
    'SUNPHARMA.NS', 'SUNTV.NS', 'SUPREMEIND.NS', 'SYNGENE.NS', 'TATACHEMICAL.NS', 
    'TATACOMM.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATAPOWER.NS', 'TATASTEEL.NS', 
    'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'TORNTPHARM.NS', 'TRENT.NS', 'TVSMOTOR.NS', 
    'UNIONBANK.NS', 'UNITDSPIRITS.NS', 'UPL.NS', 'VBL.NS', 'VEDL.NS', 
    'VOLTAS.NS', 'WIPRO.NS', 'YESBANK.NS', 'ZOMATO.NS', 'ZYDUSLIFE.NS'
]

def fetch_live_data(symbols):
    bullish, bearish = [], []
    
    # Show progress bar
    progress_bar = st.progress(0, text="Fetching Market Data...")
    
    for i, sym in enumerate(symbols):
        try:
            # Fetch 1-day intraday data
            ticker = yf.Ticker(sym)
            data = ticker.history(period='1d', interval='5m')
            
            if not data.empty:
                ltp = data['Close'].iloc[-1]
                prev_close = ticker.fast_info['previous_close']
                p_change = ((ltp - prev_close) / prev_close) * 100
                volume = data['Volume'].sum()
                
                stock_row = {
                    "Symbol": sym.replace(".NS", ""),
                    "LTP": round(ltp, 2),
                    "Change %": round(p_change, 2),
                    "Volume": f"{int(volume):,}"
                }

                # LOGIC: Bullish if price is up > 0.5%, Bearish if down < -0.5%
                if p_change > 0.5:
                    bullish.append(stock_row)
                elif p_change < -0.5:
                    bearish.append(stock_row)
                    
            progress_bar.progress((i + 1) / len(symbols))
        except:
            continue
            
    progress_bar.empty()
    return pd.DataFrame(bullish), pd.DataFrame(bearish)

# Main App Container
container = st.empty()

while True:
    with container.container():
        df_bull, df_bear = fetch_live_data(FNO_SYMBOLS)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("🟢 BULLISH MOMENTUM")
            if not df_bull.empty:
                st.dataframe(df_bull.sort_values(by="Change %", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.write("No bullish breakouts yet.")

        with col2:
            st.error("🔴 BEARISH MOMENTUM")
            if not df_bear.empty:
                st.dataframe(df_bear.sort_values(by="Change %"), use_container_width=True, hide_index=True)
            else:
                st.write("No bearish breakdowns yet.")
        
        st.caption(f"Last updated: {time.strftime('%H:%M:%S')} | Auto-refreshing in 2 mins...")

    time.sleep(300) # Update every 300 seconds
    st.rerun()