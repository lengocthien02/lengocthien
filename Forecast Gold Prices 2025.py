import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from itertools import product
import warnings
from scipy import stats
import statsmodels.api as sm

warnings.filterwarnings('ignore')

# Chọn style hợp lệ
plt.style.use('ggplot')

# Đọc dữ liệu
df = pd.read_csv("/Users/thien/Downloads/XAU_1d_data.csv", sep=';')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Xử lý dữ liệu
df = df[-1500:]
numeric_df = df.select_dtypes(include=["number"])
df_month = numeric_df.resample('M').mean()

# Vẽ biểu đồ giá vàng theo tháng
plt.figure(figsize=(15, 7))
plt.plot(df_month['Close'], label='By Months', color='tomato', linewidth=2)
plt.title('Gold (Ounce) by Month', fontsize=18)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price (USD)', fontsize=14)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# Kiểm tra tính dừng của chuỗi
df_month['Close_box'], lmbda = stats.boxcox(df_month['Close'])
adf_result = sm.tsa.stattools.adfuller(df_month['Close_box'])[1]
print(f'The Dickey–Fuller test p-value is: {adf_result:.5f}')

# Khác biệt hóa để làm dừng chuỗi
df_month['Close_box_diff'] = df_month['Close_box'] - df_month['Close_box'].shift(12)
df_month['Close_box_diff2'] = df_month['Close_box_diff'] - df_month['Close_box_diff'].shift(1)

# Kiểm tra lại tính dừng sau khi khác biệt hóa
adf_result = sm.tsa.stattools.adfuller(df_month['Close_box_diff2'].dropna())[1]
print(f'After differencing: The Dickey–Fuller test p-value is: {adf_result:.5f}')

# Grid search tìm SARIMA tốt nhất
Qs = range(0, 2)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D, d = 1, 1

parameters_list = list(product(ps, qs, Ps, Qs))
best_aic = float("inf")
best_model = None
best_param = None

for param in parameters_list:
    try:
        model = sm.tsa.statespace.SARIMAX(df_month['Close_box'].dropna(),
                                          order=(param[0], d, param[1]),
                                          seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
        
        if model.aic < best_aic:
            best_aic = model.aic
            best_model = model
            best_param = param
    except:
        continue

print(f'Best SARIMA Model: {best_param} with AIC: {best_aic}')

# Hàm nghịch biến đổi Box-Cox
def invboxcox(y, lmbda):
    return np.exp(y) if lmbda == 0 else np.power((lmbda * y + 1), 1 / lmbda)

# Dự báo tương lai
future_dates = pd.date_range(start=df_month.index[-1] + timedelta(days=30), periods=24, freq='M')
future_df = pd.DataFrame(index=future_dates, columns=df_month.columns)
df_month2 = pd.concat([df_month, future_df])

df_month2['forecast'] = invboxcox(best_model.predict(start=len(df_month), end=len(df_month2)-1), lmbda)

# Vẽ biểu đồ dự báo
plt.figure(figsize=(15, 7))
df_month2['Close'].plot(label='Actual Price')
df_month2['forecast'].plot(color='r', linestyle='--', label='Predicted Price')
plt.legend()
plt.title('Gold Price Forecast by Months')
plt.ylabel('Mean USD')
plt.show()
