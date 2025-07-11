from binance.client import Client
import pandas as pd
import datetime
import time

# 初始化币安客户端（无需API密钥即可获取公开市场数据）
client = Client()

def get_historical_klines(symbol, interval, start_str, end_str=None):
    """分页获取历史K线数据"""
    data = []
    start_time = datetime.datetime.strptime(start_str, "%d %b %Y")
    end_time = datetime.datetime.now() if not end_str else datetime.datetime.strptime(end_str, "%d %b %Y")
    
    while start_time < end_time:
        temp_end = min(start_time + datetime.timedelta(days=30), end_time)
        print(f"Fetching data from {start_time} to {temp_end}")
        
        # 调用API
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=str(int(start_time.timestamp() * 1000)),
            end_str=str(int(temp_end.timestamp() * 1000)),
            limit=1000
        )
        
        # 合并数据
        data += klines
        
        # 更新起始时间（使用最后一条数据的时间戳+5分钟）
        if klines:
            last_time = klines[-1][0]
            start_time = datetime.datetime.fromtimestamp(last_time/1000) + datetime.timedelta(minutes=5)
        else:
            start_time = temp_end
        
        # 防止请求过于频繁
        time.sleep(0.2)
    
    return data

def save_to_csv(data, filename):
    """保存数据到CSV文件"""
    columns = [
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # 转换时间戳
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
    
    # 保存文件
    df.to_csv(filename, index=False)
    print(f"数据已保存到 {filename}")

if __name__ == "__main__":
    # 配置参数
    symbol = "BTCUSDT"  # 交易对
    interval = Client.KLINE_INTERVAL_5MINUTE  # 5分钟间隔
    start_date = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime("%d %b %Y")  # 两年前
    
    # 获取数据
    print("开始获取数据...")
    kline_data = get_historical_klines(symbol, interval, start_date)
    
    # 保存数据
    filename = f"{symbol}_{interval}_2years.csv"
    save_to_csv(kline_data, filename) 