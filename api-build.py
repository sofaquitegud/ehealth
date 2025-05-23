# Import required libraries
from fastapi import FastAPI, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import os
from enum import Enum
from dotenv import load_dotenv
import json
from datetime import date, datetime

# Import the StaffHealthAnalyzer class
from staff_health_analyzer import (
    StaffHealthAnalyzer,
    HealthMeasure,
    VisualizationCategory,
    TimePeriod,
    ReportType,
    MODE_KIOSK,
    MODE_MOBILE,
)

# Load environment variables
load_dotenv()

# Custom JSON encoder to handle date objects
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

# Create FastAPI app
app = FastAPI(
    title="Staff Health Analyzer API",
    description="API for analyzing and reporting staff health metrics",
    version="1.2.0",
)

# Define Pydantic models for request/response
class AnalysisRequest(BaseModel):
    report_type: str = Field("Latest", description="Latest or Trending")
    health_measure: str = Field(
        "Overall", description="Overall, Hypertension, Stress, Wellness, or BMI"
    )
    category: str = Field(
        "Age_range",
        description="For Latest: Overall, Age_range, Gender, BMI. For Trending: Weekly, Monthly, Quarterly, Yearly",
    )
    with_summary: bool = Field(
        False, description="Whether to include natural language summary"
    )
    with_recommendations: bool = Field(
        False, description="Whether to include health recommendations"
    )


class AnalysisResponse(BaseModel):
    report_type: str
    health_measure: str
    category: str
    mode: str
    data: List[Dict[str, Any]]
    summary: Optional[str] = None
    recommendations: Optional[Dict[str, Any]] = None


# Define API mode enum
class AnalysisMode(str, Enum):
    KIOSK = MODE_KIOSK
    MOBILE = MODE_MOBILE


# Function to convert DataFrame to JSON-serializable format
def dataframe_to_dict(df):
    """Convert DataFrame to JSON-serializable dict with proper date handling"""
    result = df.to_dict(orient="records")
    # Use the custom encoder to handle date objects
    return json.loads(json.dumps(result, cls=DateEncoder))

# Dependency for getting the appropriate analyzer
def get_analyzer(
    mode: AnalysisMode = Query(
        AnalysisMode.MOBILE, description="Analysis mode (kiosk or mobile)"
    )
):
    """Get the appropriate StaffHealthAnalyzer instance based on mode"""
    # Get API key from environment variables
    api_key = os.getenv("OPENAI_API_KEY")

    # Use empty string to trigger database fallback
    data_path = ""

    return StaffHealthAnalyzer(
        data_path=data_path,
        mode=mode,
        api_key=api_key
    )


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Staff Health Analyzer API"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "Available"}


# Analyze endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: AnalysisRequest, analyzer: StaffHealthAnalyzer = Depends(get_analyzer)
):
    """
    Generate a health analysis report based on specified parameters

    - **report_type**: 'Latest' for current snapshot or 'Trending' for time-based analysis
    - **health_measure**: The health metric to analyze (Overall, Hypertension, BMI, Stress, Wellness)
    - **category**:
        - For Latest reports: Overall, Age_range, Gender, BMI
        - For Trending reports: Weekly, Monthly, Quarterly, Yearly
    - **with_summary**: Set to true to include AI-generated natural language summary
    - **with_recommendations**: Set to true to include AI-generated health recommendations
    - **mode**: Set application mode 'mobile' (default) or 'kiosk' (BMI not supported)
    """
    try:
        # Validate enum values
        if request.report_type not in [rt.value for rt in ReportType]:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid report_type: {request.report_type}"},
            )

        if request.health_measure not in [hm.value for hm in HealthMeasure]:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid health_measure: {request.health_measure}"},
            )

        # Validate category based on report type
        if request.report_type == "Latest":
            valid_categories = [vc.value for vc in VisualizationCategory]
            if request.category not in valid_categories:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Invalid category for Latest report: {request.category}"
                    },
                )
        elif request.report_type == "Trending":
            valid_periods = [tp.value for tp in TimePeriod]
            if request.category not in valid_periods:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"Invalid time period for Trending report: {request.category}"
                    },
                )

        # Run analysis
        result = analyzer.run_analysis(
            report_type=request.report_type,
            health_measure=request.health_measure,
            category=request.category,
            with_summary=request.with_summary,
            with_recommendations=request.with_recommendations,
            enable_display=False,  # Disable visualization for API
        )

        # Convert DataFrame to list of dicts for JSON serialization with date handling
        result["data"] = dataframe_to_dict(result["data"])

        # Handle recommendations with date objects if they exist
        if result.get("recommendations") and isinstance(result["recommendations"], dict):
            result["recommendations"] = json.loads(json.dumps(result["recommendations"], cls=DateEncoder))

        return result

    except ValueError as e:
        return JSONResponse(
            status_code=400, content={"error": str(e)}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )


# Get all reports endpoint
@app.get("/reports/{report_type}")
async def get_all_reports(
    report_type: str, analyzer: StaffHealthAnalyzer = Depends(get_analyzer)
):
    """Generate all reports of a specific type"""
    try:
        if report_type not in [rt.value for rt in ReportType]:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid report_type: {report_type}"},
            )

        # Generate all reports
        reports = analyzer.generate_all_reports(report_type)

        # Convert DataFrames to lists of dicts for JSON serialization with date handling
        result = {}
        for key, df in reports.items():
            result[key] = dataframe_to_dict(df)

        return result

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )


# Get raw data endpoint (admin only)
@app.get("/data")
async def get_raw_data(analyzer: StaffHealthAnalyzer = Depends(get_analyzer)):
    """Get raw data (this could be protected by authentication in production)"""
    try:
        # Convert DataFrame to list of dicts for JSON serialization with date handling
        return {"data": dataframe_to_dict(analyzer.df)}

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )


# Configuration endpoint (without exposing sensitive info)
@app.get("/config")
async def get_config_info():
    """Get non-sensitive configuration information"""
    try:
        return {
            "db": {
                "host": os.getenv("host"),
                "port": os.getenv("port"),
                "dbname": os.getenv("dbname"),
                "user": os.getenv("user"),
                # Password is intentionally not included
            },
            "llm": {
                "api_key_configured": bool(os.getenv("OPENAI_API_KEY"))
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
