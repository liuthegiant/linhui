import pandas as pd
import folium

# 读取传感器位置
locs = pd.read_csv("graph_sensor_locations.csv")

# 建立地图（洛杉矶中心）
map_la = folium.Map(location=[34.05, -118.25], zoom_start=10, tiles="OpenStreetMap")

# 循环添加点
for _, row in locs.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=3,
        color="red",
        fill=True,
        fill_opacity=0.7,
        popup=f"Sensor {row['sensor_id']}"
    ).add_to(map_la)

# 保存地图
map_la.save("metrla_sensors.html")
print("地图已保存，打开 metrla_sensors.html 查看。")
