# plant_recommender.py

import requests
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
import numpy as np

def get_location_info(latitude, longitude):
    """获取地理位置信息"""
    geolocator = Nominatim(user_agent="plant_app")
    location = geolocator.reverse(f"{latitude}, {longitude}")
    return {
        "address": location.address,
        "timezone": requests.get(
            f"https://api.timezonedb.com/v2.1/get-time-zone?key=YOUR_API_KEY&format=json&by=position&lat={latitude}&lon={longitude}"
        ).json()["zoneName"],
    }

def analyze_environment(latitude, longitude, light_condition):
    """分析环境条件"""
    location_info = get_location_info(latitude, longitude)
    return {
        "light": light_condition,
        "location": location_info["address"],
        "timezone": location_info["timezone"],
    }

def recommend_plants(base_plant, environment):
    """推荐植物"""
    # 基于简单的规则推荐
    recommendations = []
    if base_plant in PLANTS:
        recommendations.extend(PLANTS[base_plant]["companion_plants"])
    if environment["light"] in RECOMMENDATION_RULES:
        recommendations.extend(RECOMMENDATION_RULES[environment["light"]])
    
    # 去重并返回
    return list(set(recommendations))

def generate_design_plot(base_plant, recommended_plants):
    """生成设计概念图"""
    plt.figure(figsize=(10, 6))
    plt.title(f"Plant Design for {base_plant.capitalize()}")
    plt.xlabel("Position")
    plt.ylabel("Distance from Base Plant")

    # 简单的布局示例
    positions = np.arange(len(recommended_plants)) + 1
    plt.bar(positions, np.random.rand(len(recommended_plants)), align="center", alpha=0.5)
    plt.xticks(positions, recommended_plants, rotation=45)

    plt.show()
