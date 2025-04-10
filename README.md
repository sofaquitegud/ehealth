# StaffHealthAnalyzer

A comprehensive tool for analyzing and reporting staff health metrics with support for both standalone usage and API deployment.

## Features

- **Dual Mode Operation**: Supports both `kiosk` and `mobile` modes with appropriate data handling for each context
- **Flexible Data Sources**: Works with both CSV files and PostgreSQL database connections
- **Comprehensive Health Metrics**: Analyzes Overall Health, Hypertension Risk, Stress Level, Wellness Level, and BMI
- **Rich Reporting**: Generates both latest snapshot and trending reports across various demographics and time periods
- **Natural Language Summaries**: Optional AI-powered summaries of health reports (requires OpenAI API key)
- **Health Recommendations**: AI-generated actionable recommendations based on health data
- **Visualization Support**: Built-in data visualization capabilities
- **FastAPI Integration**: Complete REST API for integration with other systems

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL (if using database mode)
- OpenAI API key (optional, for natural language features)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/staff-health-analyzer.git
   cd staff-health-analyzer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration:

   ```
   # Database configuration
   host=your_db_host
   port=your_db_port
   dbname=your_db_name
   user=your_db_user
   pass=your_db_password

   # OpenAI configuration (optional)
   OPENAI_API_KEY=your_openai_api_key

   # CSV file paths (optional, if not using database)
   STAFF_HEALTH_DATA_MOBILE=path/to/mobile_data.csv
   STAFF_HEALTH_DATA_KIOSK=path/to/kiosk_data.csv
   ```

## Usage

### Using as a Library

```python
from staff_health_analyzer import StaffHealthAnalyzer, MODE_MOBILE, MODE_KIOSK

# Initialize with database connection (recommended)
analyzer = StaffHealthAnalyzer(
    data_path="",  # Empty string triggers database fallback
    mode=MODE_MOBILE,
    api_key="your_openai_api_key"  # Optional
)

# Or initialize with CSV file
analyzer = StaffHealthAnalyzer(
    data_path="path/to/data.csv",
    mode=MODE_MOBILE,
    api_key="your_openai_api_key"  # Optional
)

# Generate a specific report
report = analyzer.run_analysis(
    report_type="Latest",
    health_measure="Overall",
    category="Age_range",
    with_summary=True,
    enable_display=True,
    with_recommendations=True
)

# Generate all reports of a specific type
all_latest_reports = analyzer.generate_all_reports("Latest")
```

### Using the FastAPI Server

1. Start the server:

   ```bash
   uvicorn api:app --reload
   ```

2. Access the API documentation:

   - Open your browser and navigate to `http://localhost:8000/docs`

3. Example API requests:

   ```bash
   # Get a specific analysis
   curl -X POST "http://localhost:8000/analyze" \
        -H "Content-Type: application/json" \
        -d '{"report_type": "Latest", "health_measure": "Overall", "category": "Age_range", "with_summary": true, "with_recommendations": true}'

   # Get all reports of a specific type
   curl -X GET "http://localhost:8000/reports/Latest?mode=mobile"
   ```

## API Endpoints

| Endpoint                 | Method | Description                             |
| ------------------------ | ------ | --------------------------------------- |
| `/`                      | GET    | Welcome message                         |
| `/health`                | GET    | Health check endpoint                   |
| `/analyze`               | POST   | Generate a specific analysis report     |
| `/reports/{report_type}` | GET    | Generate all reports of a specific type |
| `/data`                  | GET    | Get raw data (admin only)               |
| `/config`                | GET    | Get non-sensitive configuration info    |

## Data Models

### Health Measures

- `Overall`: Combined health score
- `Hypertension`: Blood pressure risk level
- `Stress`: Mental stress level
- `Wellness`: General wellness score
- `BMI`: Body Mass Index category

### Report Types

- `Latest`: Current snapshot of health metrics
- `Trending`: Time-series analysis of health metrics

### Visualization Categories (for Latest reports)

- `Overall`: Entire staff population
- `Age_range`: Breakdown by age groups
- `Gender`: Breakdown by gender
- `BMI`: Breakdown by BMI category

### Time Periods (for Trending reports)

- `Weekly`: Analysis by week
- `Monthly`: Analysis by month
- `Quarterly`: Analysis by quarter
- `Yearly`: Analysis by year

## Mode-Specific Features

### Mobile Mode

- Full access to all features including BMI analysis
- Complete demographic analysis
- Works with both CSV and database sources

### Kiosk Mode

- Limited access to BMI data (uses placeholder values)
- Limited demographic analysis
- Works with both CSV and database sources

## Development

### Project Structure

```
staff-health-analyzer/
├── staff_health_analyzer.py  # Core analysis library
├── api.py                    # FastAPI server implementation
├── requirements.txt          # Project dependencies
├── .env                      # Environment configuration
└── README.md                 # This file
```

### Running Tests

```bash
pytest tests/
```

## License

[MIT License](LICENSE)
