# Import libraries
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel
import pandas as pd
import os

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

# Add CORS middlewareto allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Shared dependency to get the analyser instance
def get_analyzer(
    mode: str = Query("mobile", description="Application mode: 'mobile' or 'kiosk'")
):
    data_path = os.getenv("HEALTH_DATA_PATH", "staff_health_data.csv")
    api_key = os.getenv("OPENAI_API_KEY", None)

    # Validate mode
    if mode not in ["mobile", "kiosk"]:
        return JSONResponse(
            content={
                "error": f"Invalid mode: {mode}. Allowed modes are 'mobile' or 'kiosk'"
            },
            status_code=400,
        )

    # Create and return analyser instance
    return StaffHealthAnalyzer(data_path=data_path, api_key=api_key, mode=mode)


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
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health-measure", tags=["Metadata"])
async def get_health_measure(
    mode: str = Query("mobile", description="Application mode: 'mobile' or 'kiosk'")
):
    """Get all available health measures based on mode"""
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
        - For Latest reports: Overall, Age range, Gender type, BMI
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
                        "error": f"Invalid visualization category: {category}. Expected 'Overall', 'Age range', 'Gender type' or 'BMI'."
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
        reports = analyzer.generate_all_latest_reports()

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
        reports = analyzer.generate_all_trending_reports()

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
