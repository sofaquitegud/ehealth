# Import libraries
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import os
import difflib

# Import StaffHealthAnalyzer class
from staff_health_analyzer import (
    StaffHealthAnalyzer,
    HealthMeasure,
    VisualizationCategory,
    TimePeriod,
    ReportType,
)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Valid modes
VALID_MODES = ["mobile", "kiosk"]


# Function to get closest match for incorrect mode
def get_closest_match(input_mode):
    if not input_mode:
        return None
    # Find the closest match using difflib
    closest_match = difflib.get_close_matches(
        input_mode.lower(), VALID_MODES, n=1, cutoff=0.6
    )
    return closest_match[0] if closest_match else None


# Shared dependency to get the analyser instance
def get_analyzer(
    mode: str = Query("mobile", description="Application mode: 'mobile' or 'kiosk'")
):
    # Validate mode
    if mode not in VALID_MODES:
        closest_match = get_closest_match(mode)
        error_message = f"Invalid mode: {mode}. Allowed modes are 'mobile' or 'kiosk'"
        if closest_match:
            error_message += f". Did you mean '{closest_match}'?"

        return JSONResponse(content={"error": error_message}, status_code=400)

    # Choose data path based on mode
    if mode == "mobile":
        data_path = os.getenv("HEALTH_DATA_PATH", "staff_health_data.csv")
    else:  # mode == 'kiosk'
        data_path = os.getenv("HEALTH_DATA_KIOSK_PATH", "staff_health_data_kiosk.csv")

    api_key = os.getenv("OPENAI_API_KEY", None)

    # Get database config from environment variables
    db_config = None
    if all(os.getenv(key) for key in ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]):
        db_config = {
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }

    # Create and return analyser instance
    return StaffHealthAnalyzer(
        data_path=data_path, api_key=api_key, mode=mode, db_config=db_config
    )


# Response models
class ReportResponse(BaseModel):
    report_type: str
    health_measure: str
    category: str
    mode: str
    data: list
    summary: Optional[str] = None


# API routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Staff Health Analysis API is running",
        "version": "1.1.3",
        "docs": "/docs",
    }


@app.get("/health-measure", tags=["Metadata"])
async def get_health_measure(
    mode: str = Query("mobile", description="Application mode: 'mobile' or 'kiosk'")
):
    """Get all available health measures based on mode"""
    if mode not in VALID_MODES:
        closest_match = get_closest_match(mode)
        error_message = f"Invalid mode: {mode}. Allowed modes are 'mobile' or 'kiosk'"
        if closest_match:
            error_message += f". Did you mean '{closest_match}'?"

        return JSONResponse(content={"error": error_message}, status_code=400)

    measures = [measure.value for measure in HealthMeasure]
    # Remove BMI from available measures in kiosk mode
    if mode == "kiosk":
        measures = [m for m in measures if m != "BMI"]
    return measures


@app.get("/visualization-categories", tags=["Metadata"])
async def get_visualization_categories(
    mode: str = Query("mobile", description="Application mode: 'mobile' or 'kiosk'")
):
    """Get all available visualization categories for latest reports based on mode"""
    if mode not in VALID_MODES:
        closest_match = get_closest_match(mode)
        error_message = f"Invalid mode: {mode}. Allowed modes are 'mobile' or 'kiosk'"
        if closest_match:
            error_message += f". Did you mean '{closest_match}'?"

        return JSONResponse(content={"error": error_message}, status_code=400)

    categories = [category.value for category in VisualizationCategory]
    # Remove BMI from available categories in kiosk mode
    if mode == "kiosk":
        categories = [c for c in categories if c != "BMI"]
    return categories


@app.get("/time-periods", tags=["Metadata"])
async def get_time_periods():
    """Get all available time periods for trending reports"""
    return [period.value for period in TimePeriod]


@app.get("/report_types", tags=["Metadata"])
async def get_report_types():
    """Get all available report types"""
    return [report_type.value for report_type in ReportType]


@app.get("/report", response_model=ReportResponse, tags=["Reports"])
async def get_report(
    report_type: str = Query(
        "Trending", description="Type of report: Latest or Trending"
    ),
    health_measure: str = Query("BMI", description="Health measure to analyze"),
    category: str = Query(
        "Weekly", description="Category or time period for the report"
    ),
    with_summary: bool = Query(
        False, description="Whether to include a natural language summary"
    ),
    mode: str = Query("mobile", description="Application mode: mobile or kiosk"),
    analyzer: StaffHealthAnalyzer = Depends(get_analyzer),
):
    """
    Generate a health analysis report based on specified parameters

    - **report_type**: 'Latest' for current snapshot or 'Trending' for time-based analysis
    - **health_measure**: The health metric to analyze (Overall, Hypertension, BMI, Stress, Wellness)
    - **category**:
        - For Latest reports: Overall, Age_range, Gender, BMI
        - For Trending reports: Weekly, Monthly, Quarterly, Yearly
    - **with_summary**: Set to true to include AI-generated natural language summary
    - **mode**: Set application mode 'mobile' (default) or 'kiosk' (BMI not supported)
    """
    try:
        # Validate inputs
        if report_type not in [rt.value for rt in ReportType]:
            return JSONResponse(
                content={
                    "error": f"Invalid report type: {report_type}. Expected 'Latest' or 'Trending'."
                },
                status_code=400,
            )

        if health_measure not in [hm.value for hm in HealthMeasure]:
            return JSONResponse(
                content={
                    "error": f"Invalid health measure: {health_measure}. Expected 'Overall', 'Hypertension', 'BMI', 'Stress' or 'Wellness'."
                },
                status_code=400,
            )

        # Validate category based on report_type
        if report_type == "Latest":
            if category not in [vc.value for vc in VisualizationCategory]:
                return JSONResponse(
                    content={
                        "error": f"Invalid visualization category: {category}. Expected 'Overall', 'Age_range', 'Gender' or 'BMI'."
                    },
                    status_code=400,
                )
        elif report_type == "Trending":
            if category not in [tp.value for tp in TimePeriod]:
                return JSONResponse(
                    content={
                        "error": f"Invalid time period: {category}. Expected 'Weekly', 'Monthly', 'Quarterly' or 'Yearly'."
                    },
                    status_code=400,
                )

        # Validate mode
        if mode not in VALID_MODES:
            closest_match = get_closest_match(mode)
            error_message = (
                f"Invalid mode: {mode}. Allowed modes are 'mobile' or 'kiosk'"
            )
            if closest_match:
                error_message += f". Did you mean '{closest_match}'?"

            return JSONResponse(content={"error": error_message}, status_code=400)

        # Run analysis
        result = analyzer.run_analysis(
            report_type=report_type,
            health_measure=health_measure,
            category=category,
            with_summary=with_summary,
        )

        # Convert DataFrame to list for JSON serialization
        result["data"] = result["data"].to_dict(orient="records")

        return result

    except ValueError as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(
            content={"error": f"Unexpected error: {str(e)}"}, status_code=500
        )


@app.get("/all-latest-reports", tags=["Batch Reports"])
async def get_all_latest_reports(
    mode: str = Query("mobile", description="Application mode: mobile or kiosk"),
    analyzer: StaffHealthAnalyzer = Depends(get_analyzer),
):
    """Generate all latest report combinations based on mode"""
    try:
        reports = analyzer.generate_all_reports("Latest")

        # Filter out BMI reports in kiosk mode
        if mode == "kiosk":
            reports = {k: v for k, v in reports.items() if "bmi" not in k.lower()}

        return {
            key: report.to_dict(orient="records") for key, report in reports.items()
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/all-trending-reports", tags=["Batch Reports"])
async def get_all_trending_reports(
    mode: str = Query("mobile", description="Application mode: mobile or kiosk"),
    analyzer: StaffHealthAnalyzer = Depends(get_analyzer),
):
    """Generate all trending report combinations based on mode"""
    try:
        reports = analyzer.generate_all_reports("Trending")

        # Filter out BMI reports in kiosk mode
        if mode == "kiosk":
            reports = {k: v for k, v in reports.items() if "bmi" not in k.lower()}

        return {
            key: report.to_dict(orient="records") for key, report in reports.items()
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# Main entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
