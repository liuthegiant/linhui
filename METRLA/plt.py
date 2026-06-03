import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pandas as pd

# 读传感器坐标
df = pd.read_csv("graph_sensor_locations.csv")
lats = df["latitude"].values
lons = df["longitude"].values

# 构造多边形边界 (传感器点的 convex hull)
gdf_points = gpd.GeoDataFrame(geometry=[Point(xy) for xy in zip(lons, lats)], crs="EPSG:4326")
hull = gdf_points.unary_union.convex_hull.buffer(0.01)  # 稍微扩展

# 从 OSM 下载这个区域的路网
G = ox.graph_from_polygon(hull, network_type="drive")

# 找到每个传感器对应的最近 OSM 节点
sensor_nodes = []
for lon, lat in zip(lons, lats):
    node = ox.distance.nearest_nodes(G, lon, lat)
    sensor_nodes.append(node)

# 提取只包含这些节点的子图
H = G.subgraph(sensor_nodes).copy()

# 绘制子图
fig, ax = ox.plot_graph(H, show=False, close=False, node_size=20, node_color="red", edge_color="blue")

plt.savefig("sensor_subgraph.png", dpi=300, bbox_inches="tight")
plt.close()
print("已保存 sensor_subgraph.png")
