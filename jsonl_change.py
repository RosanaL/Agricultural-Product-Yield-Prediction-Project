# -*- coding: utf-8 -*-

import pandas as pd

# 读取 JSONL 文件
file_path = '/mnt/random_forest/task1/data1.jsonl'  # 替换为你的文件路径
data = pd.read_json(file_path, lines=True, encoding='utf-8')

# 替换键名
data.rename(columns={
    '平均低温/℃': 'avg_low_temp',
    '平均高温/℃': 'avg_high_temp',
    '极限低温/℃': 'min_temp',
    '极限高温/℃': 'max_temp',
    '共下雨/雪天数/d': 'days_of_rainsnow',
    '总降雨量/mm': 'rain_num',
    '平均空气质量': 'airquality',
    '平均每日最大风速/ km/h': 'avg_max_windspeed',
    '平均能见度/km': 'avg_see',
    '空气湿度/%': 'humidity',
    '低层大气中CO2浓度/ppm': 'CO2',
    '低层大气中氧气浓度/%': 'O2',
    '光照强度/W/m²': 'light',
    '野生羊肚菌干产量/Kg': 'product'
}, inplace=True)

# 将处理后的数据保存为新的 JSONL 文件
output_path = '/mnt/random_forest/task1/new_data1.jsonl'  # 替换为你想要保存的文件路径
data.to_json(output_path, orient='records', lines=True, force_ascii=False)

print("处理完成，已保存为", output_path)
