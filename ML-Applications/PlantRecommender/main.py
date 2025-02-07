# main.py

import sys
from plant_recommender import *

def main():
    print("Plant Recommender App")
    print("--------------------")
    
    # 获取用户输入
    base_plant = input("Enter the name of your existing plant: ").lower()
    latitude = float(input("Enter your latitude: "))
    longitude = float(input("Enter your longitude: "))
    light_condition = input(
        "Enter the light condition (full_sun, partial_shade, full_shade): "
    ).lower()

    # 分析环境
    environment = analyze_environment(latitude, longitude, light_condition)
    print("\nEnvironment Analysis:")
    for key, value in environment.items():
        print(f"{key.capitalize()}: {value}")

    # 推荐植物
    recommendations = recommend_plants(base_plant, environment)
    print("\nRecommended Companion Plants:")
    for plant in recommendations:
        print(f"- {plant.capitalize()}")

    # 生成设计图
    generate_design_plot(base_plant, recommendations)

if __name__ == "__main__":
    main()
