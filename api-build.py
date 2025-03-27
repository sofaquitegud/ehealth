# Import required libraries
from fastapi import FastAPI, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import os
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the StaffHealthAnalyzer class
from staff_health_analyzer import (
    StaffHealthAnalyzer,
    HealthMeasure,
    VisualizationCategory,
    TimePeriod,
    ReportType,
)

# Create FastAPI app
app = FastAPI(
    title="Staff Health Analyzer API",
    description="API for analyzing and reporting staff health metrics",
    version="1.0.0",
)


# Get database configuration from environment variables
def get_db_config():
    """Get database configuration from environment variables"""
    return {
        "host": os.getenv("host"),
        "port": os.getenv("port"),
        "dbname": os.getenv("dbname"),
        "user": os.getenv("user"),
        "password": os.getenv("pass"),
    }


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
    KIOSK = "kiosk"
    MOBILE = "mobile"


# Dependency for getting the appropriate analyzer
def get_analyzer(
    mode: AnalysisMode = Query(
        AnalysisMode.MOBILE, description="Analysis mode (kiosk or mobile)"
    )
):
    """Get the appropriate StaffHealthAnalyzer instance based on mode"""
    data_path = (
        os.getenv("STAFF_HEALTH_DATA_MOBILE")
        if mode == AnalysisMode.MOBILE
        else os.getenv("STAFF_HEALTH_DATA_KIOSK")
    )

    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    return StaffHealthAnalyzer(
        data_path=data_path, mode=mode, api_key=api_key, db_config=get_db_config()
    )


# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Staff Health Analyzer API"}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# Analyze endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: AnalysisRequest, analyzer: StaffHealthAnalyzer = Depends(get_analyzer)
):
    """Generate a health analysis report based on specified parameters"""
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
            enable_display=False,  # Disable visualization for API
            with_recommendations=request.with_recommendations,
        )

        # Convert DataFrame to list of dicts for JSON serialization
        result["data"] = result["data"].to_dict(orient="records")

        return result

    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

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

        # Convert DataFrames to lists of dicts for JSON serialization
        result = {}
        for key, df in reports.items():
            result[key] = df.to_dict(orient="records")

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
        # Convert DataFrame to list of dicts for JSON serialization
        return {"data": analyzer.df.to_dict(orient="records")}

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )


# Configuration endpoint (without exposing sensitive info)
@app.get("/config")
async def get_config_info():
    """Get non-sensitive configuration information"""
    try:
        db_config = get_db_config()
        return {
            "db": {
                "host": db_config["host"],
                "port": db_config["port"],
                "dbname": db_config["dbname"],
                "user": db_config["user"],
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
