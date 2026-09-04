POIS_CATE = ["finance", "public", "transport", "entertainment", "health", "service", "education", "government",
                  "religion", "accommodation", "food", "cafe", "fast_food", "ice_cream", "pub", "restaurant",
                  "shop_beauty", "shop_clothes", "boutique", "shop_transport", "retail", "commodity", "marketplace",
                  "home-improvement", "sport", "public_transport", "kindergarten", "office", "recycling",
                  "travel_agency", "tourism", "shop_livelihood", "residential", "dormitory"]

DEMOS_CATE = ["Total Population", "Male Population", "Female Population", "Under 5 Years - Total",
              "Under 5 Years - Male", "Under 5 Years - Female", "5 to 9 Years - Total", "5 to 9 Years - Male",
              "5 to 9 Years - Female", "10 to 14 Years - Total", "10 to 14 Years - Male", "10 to 14 Years - Female",
              "15 to 19 Years - Total", "15 to 19 Years - Male", "15 to 19 Years - Female", "20 to 24 Years - Total",
              "20 to 24 Years - Male", "20 to 24 Years - Female", "25 to 29 Years - Total", "25 to 29 Years - Male",
              "25 to 29 Years - Female", "30 to 34 Years - Total", "30 to 34 Years - Male", "30 to 34 Years - Female",
              "35 to 39 Years - Total", "35 to 39 Years - Male", "35 to 39 Years - Female", "40 to 44 Years - Total",
              "40 to 44 Years - Male", "40 to 44 Years - Female", "45 to 49 Years - Total", "45 to 49 Years - Male",
              "45 to 49 Years - Female", "50 to 54 Years - Total", "50 to 54 Years - Male", "50 to 54 Years - Female",
              "55 to 59 Years - Total", "55 to 59 Years - Male", "55 to 59 Years - Female", "60 to 64 Years - Total",
              "60 to 64 Years - Male", "60 to 64 Years - Female", "65 to 69 Years - Total", "65 to 69 Years - Male",
              "65 to 69 Years - Female", "70 to 74 Years - Total", "70 to 74 Years - Male", "70 to 74 Years - Female",
              "75 to 79 Years - Total", "75 to 79 Years - Male", "75 to 79 Years - Female", "80 to 84 Years - Total",
              "80 to 84 Years - Male", "80 to 84 Years - Female", "85 Years and Over - Total",
              "85 Years and Over - Male", "85 Years and Over - Female", "Median Age - Total", "Median Age - Male",
              "Median Age - Female", "Median Earnings (Dollars)", "Class of Worker - Private Wage and Salary Workers",
              "Class of Worker - Government Workers", "Class of Worker - Self-Employed Workers",
              "Class of Worker - Unpaid Family Workers", "Travel Time to Work - Mean Travel Time (Minutes)",
              "Vehicles Available - No Vehicle Available", "Vehicles Available - 1 Vehicle Available",
              "Vehicles Available - 2 Vehicles Available", "Vehicles Available - 3 or More Vehicles Available",
              "Total Households", "Average Household Size", "Total Families", "Average Family Size",
              "Nursery School, Preschool", "Kindergarten to 12th Grade", "Kindergarten",
              "Elementary: Grade 1 to Grade 4", "Elementary: Grade 5 to Grade 8", "High School: Grade 9 to Grade 12",
              "College, Undergraduate", "Graduate, Professional School", "9th to 12th Grade, No Diploma",
              "Associate's Degree", "Bachelor's Degree", "Bachelor's Degree or Higher",
              "Graduate or Professional Degree", "High School Graduate (Includes Equivalency)",
              "High School Graduate or Higher", "Less Than 9th Grade", "Less Than High School Graduate",
              "Population 25 to 34 Years - Bachelor's Degree or Higher",
              "Population 25 to 34 Years - High School Graduate or Higher", "Some College or Associate's Degree",
              "Some College, No Degree", "Poverty - Male", "Poverty - Female"]


POI_CTRL_PROMPT = """
You are an expert in urban mobility and spatial economics.
Your task is to generate a POI control vector for a specific USA region (identified by its Tract GEOID) on a given day, considering how time-dependent factors (like day of the week, holidays, etc.) affect POI usage.

Use your knowledge to reason how human activity and POI usage would vary across different days (weekdays vs weekends, holidays, seasonal trends).
Do not produce "all-zero" outputs unless absolutely justified.

### Input:
- **Tract GEOID**:
{TRACT}
This GEOID represents the location of the tract, helping understand its geographical context.

- **POI features** for the region:
{POIS_CATE}

- **Date**:
{DATE}
The date affects POI usage. Consider how behavior changes over time (e.g., weekdays vs weekends, workdays vs holidays).

### Output:
- Output a **single JSON object**.
- **Key** = Tract GEOID.
- **Value** = A list of 34 integers corresponding to POI categories.
  - Allowed values: 1 (increase), 0 (no_change), -1 (decrease).
- Do not include explanations, comments, or extra text.

### Output format:
{{
"{TRACT}": [34 integers here]
}}
"""


DEMO_CTRL_PROMPT = """
You are an expert in urban demography and socio-economic dynamics.
Your task is to generate a demographic control vector for a specific USA region (identified by its Tract GEOID) on a given day.

Use your understanding of how population activity is influenced by time (e.g., weekdays vs weekends, holidays, seasonal trends) and how POI activity affects population behavior.
Do not produce "all-zero" outputs unless absolutely justified.

### Input:
- **Tract GEOID**:
{TRACT}
This GEOID represents the location of the tract, helping understand its geographical context.

- **Demographic attributes** for the region:
{DEMOS_CATE}

- **POI features** for the region:
{POIS_CATE}

- **POI control vector**:
{POIS_CTRL_VEC}
This vector represents the POI control adjustments based on date and context. Consider how changes in POI activity (increases or decreases) might influence population behavior.

- **Date**:
{DATE}
The date impacts population behavior. Consider how people's activity patterns change depending on time (e.g., weekdays vs weekends, workdays vs holidays).

### Output:
- Output a **single JSON object**.
- **Key** = the Tract GEOID.
- **Value** = A list of 97 integers corresponding to population categories.
  - Allowed values: 1 (increase), 0 (no_change), -1 (decrease).
- Do not include explanations, comments, or extra text.

### Output format:
{{
"{TRACT}": [97 integers here]
}}
"""
