# Import required libraries
from fastapi import FastAPI, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import configparser
import os
from enum import Enum

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


# Read configuration file
def get_config():
    """Get configuration from config.ini file"""
    config = configparser.ConfigParser()
    config.read("config.ini")
    return config


# Get database configuration from config.ini
def get_db_config(config=None):
    """Get database configuration from config.ini"""
    if config is None:
        config = get_config()

    return {
        "host": config["db"]["host"],
        "port": config["db"]["port"],
        "dbname": config["db"]["dbname"],
        "user": config["db"]["user"],
        "password": config["db"]["password"],
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


class AnalysisResponse(BaseModel):
    report_type: str
    health_measure: str
    category: str
    mode: str
    data: List[Dict[str, Any]]
    summary: Optional[str] = None


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
    config = get_config()

    data_path = (
        "staff_health_data.csv"
        if mode == AnalysisMode.MOBILE
        else "staff_health_data_kiosk.csv"
    )

    # Get API key from config
    api_key = (
        config["llm"]["api_key"]
        if "llm" in config and "api_key" in config["llm"]
        else None
    )

    return StaffHealthAnalyzer(
        data_path=data_path, mode=mode, api_key=api_key, db_config=get_db_config(config)
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
        config = get_config()
        return {
            "db": {
                "host": config["db"]["host"],
                "port": config["db"]["port"],
                "dbname": config["db"]["dbname"],
                "user": config["db"]["user"],
                # Password is intentionally not included
            },
            "llm": {
                "api_key_configured": "api_key" in config["llm"]
                and bool(config["llm"]["api_key"])
            },
        }
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"An error occurred: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
