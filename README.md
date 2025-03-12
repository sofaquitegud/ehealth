# Staff Health Analysis System

A comprehensive solution for analyzing and reporting on employee health metrics across an organization. This system provides both snapshot (latest) and trend analysis capabilities for various health measures.

## Overview

The Staff Health Analysis System enables organizations to monitor, analyze, and visualize employee health data through:
- Comprehensive health metrics (overall health, hypertension risk, stress level, wellness level, and BMI)
- Flexible visualization categories (by age range, gender, BMI)
- Time-based trend analysis (weekly, monthly, quarterly, yearly)
- Optional AI-powered natural language summaries of the data

## Features

- **Dual Report Types**: Generate both current snapshot ("Latest") and time-based trend ("Trending") reports
- **Multiple Health Measures**: Analyze various health metrics including overall health, hypertension risk, stress levels, wellness levels, and BMI
- **Flexible Categorization**: View data by age range, gender type, or BMI category
- **Time Period Analysis**: Track trends over weekly, monthly, quarterly, or yearly periods
- **Data Visualization**: Built-in visualization capabilities with matplotlib
- **Intelligent Summaries**: Optional natural language summaries using OpenAI's GPT-4o (requires API key)
- **REST API**: Fully featured API for integration with dashboards or other applications

## Installation

### Prerequisites
- Python 3.8+
- Pandas
- Matplotlib
- NumPy
- FastAPI (for API functionality)
- Uvicorn (for running the API server)
- LangChain and OpenAI libraries (optional, for natural language summaries)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/staff-health-analysis.git
cd staff-health-analysis
```

2. Install dependencies:
```bash
pip install pandas matplotlib numpy fastapi uvicorn langchain-openai
```

3. Set up environment variables (optional):
```bash
export HEALTH_DATA_PATH="path/to/your/staff_health_data.csv"
export OPENAI_API_KEY="your-openai-api-key"  # Optional, for natural language summaries
```

## Usage

### Command Line Interface

Run the analyzer directly:

```python
from staff_health_analyzer import StaffHealthAnalyzer

# Initialize the analyzer
analyzer = StaffHealthAnalyzer(data_path='staff_health_data.csv')

# Generate a report
report = analyzer.run_analysis(
    report_type='Latest',  # 'Latest' or 'Trending'
    health_measure='Overall',  # 'Overall', 'Hypertension', 'Stress', 'Wellness', 'BMI'
    category='Age range',  # For Latest: 'Overall', 'Age range', 'Gender type', 'BMI'
                           # For Trending: 'Weekly', 'Monthly', 'Quarterly', 'Yearly'
    with_summary=False  # Set to True if you have an OpenAI API key
)

# Display the results
print(report['data'])
```

### REST API

Start the API server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Access the API documentation at http://localhost:8000/docs

### API Endpoints

- **GET /** - Root endpoint with API information
- **GET /health-measure** - Get all available health measures
- **GET /visualization-categories** - Get all available visualization categories for latest reports
- **GET /time-periods** - Get all available time periods for trending reports
- **GET /report_types** - Get all available report types
- **GET /report** - Generate a health analysis report
- **GET /all-latest-reports** - Generate all latest report combinations
- **GET /all-trending-reports** - Generate all trending report combinations

## Data Structure

The system expects a CSV file with the following columns:
- `staff_id`: Unique identifier for each employee
- `date`: Date of the health record
- `age`: Employee age
- `gender`: Employee gender
- `hypertensionRisk`: Hypertension risk level (low, medium, high)
- `stressLevel`: Stress level (low, normal, mild, high, very high)
- `wellnessLevel`: Wellness level (high, medium, low)
- `bmi`: Body Mass Index value or category

## Customization

### Modifying Value Mappings

You can adjust the value mappings in the `StaffHealthAnalyzer` class to customize how categorical health measures are converted to numerical values:

```python
# Example: Adjusting stress level mapping
self.stress_map = {
    "low": 1,
    "normal": 2,
    "mild": 3,
    "high": 4,
    "very high": 5
}
```

### Visualization

The system includes built-in visualization capabilities using matplotlib. The visualization is controlled by the `Enable_Display` parameter in the `StaffHealthAnalyzer` class.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License