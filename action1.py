import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# 读入数据集
df = pd.read_csv('./DataSet/train.csv')

# 拟合模型
model = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.1)
model.fit(df)
# 构建待预测日期数据框，七个月后
future = model.make_future_dataframe(periods=213)

# 预测数据集
forecast = model.predict(future)

# 预测结果
model.plot(forecast)
plt.show()
