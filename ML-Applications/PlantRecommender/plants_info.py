# plants_info.py

PLANTS = {
    "rose": {
        "name": "Rose",
        "light": "full_sun",
        "soil_type": "well_drained",
        "hardiness_zone": 6,
        "companion_plants": [" lavender", "garlic"],
    },
    "lily": {
        "name": "Lily",
        "light": "partial_shade",
        "soil_type": "moist",
        "hardiness_zone": 5,
        "companion_plants": ["iris", "daylily"],
    },
    # Add more plants as needed
}

# 简单的环境推荐规则
RECOMMENDATION_RULES = {
    "full_sun": ["sunflower", "cactus"],
    "partial_shade": ["hosta", "fern"],
    "full_shade": ["bamboo", "ivy"],
}
