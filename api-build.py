# Import libraries
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from typing import Dict, Optional, Any
from pydantic import BaseModel
import pandas as pd
import os

# Import StaffHealthAnalyzer class
from staff_health_analyzer import StaffHealthAnalyzer, HealthMeasure, VisualizationCategory, TimePeriod, ReportType

# Create FastAPI app
app = FastAPI()

# Add CORS middlewareto allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"] # Allow all headers
)

# Shared dependency to get the analyser instance
def get_analyzer():
    data_path = os.getenv('HEALTH_DATA_PATH', 'staff_health_data.csv')
    api_key = os.getenv('OPENAI_API_KEY')

    # Create and return analyser instance
    return StaffHealthAnalyzer(data_path=data_path, api_key=api_key)

# Response models
class ReportResponse(BaseModel):
    report_type: str
    health_measure: str
    category: str
    data: list
    summary: bool = False
    display: bool = False

# API routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Staff Health Analysis API is running",
        "version": "1.1.0",
        "docs": "/docs"
    }

@app.get("/health-measure", tags=["Metadata"])
async def get_health_measure():
    """Get all available health measures"""
    return [measure.value for measure in HealthMeasure]

@app.get("/visualization-categories", tags=["Metadata"])
async def get_visualization_categories():
    """Get all available visualization categories for latest reports"""
    return [category.value for category in VisualizationCategory]

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
    report_type: str = Query('Trending', description="Type of report: Latest or Trending"),
    health_measure: str = Query('Overall', description="Health measure to analyze"),
    category: str = Query('Quarterly', description="Category or time period for the report"),
    with_summary: bool = Query(False, description="Whether to include a natural language summary"),
    enable_display: bool = Query(False, description='Whether to include visualisation'),
    analyzer: StaffHealthAnalyzer = Depends(get_analyzer)
):
    """
    Generate a health analysis report based on specified parameters
    
    - **report_type**: 'Latest' for current snapshot or 'Trending' for time-based analysis
    - **health_measure**: The health metric to analyze (Overall, Hypertension, BMI, Stress, Wellness)
    - **category**:
        - For Latest reports: Overall, Age range, Gender type, BMI
        - For Trending reports: Weekly, Monthly, Quarterly, Yearly
    - **with_summary**: Set to true to include AI-generated natural language summary
    """
    try:
        # Validate inputs
        if report_type not in [rt.value for rt in ReportType]:
            raise HTTPException(status_code=400, detail=f"Invalid report type: {report_type}")
        
        if health_measure not in [hm.value for hm in HealthMeasure]:
            raise HTTPException(status_code=400, detail=f"Invalid health measure: {health_measure}")
        
        if report_type == 'Latest' and category not in [vc.value for vc in VisualizationCategory]:
            raise HTTPException(status_code=400, detail=f"Invalid visualization category: {category}")
        
        if report_type =='Trending' and category not in [tp.value for tp in TimePeriod]:
            raise HTTPException(status_code=400, detail=f"Invalid time period {category}")
        
        # Run analysis
        result = analyzer.run_analysis(
            report_type=report_type,
            health_measure=health_measure,
            category=category,
            with_summary=with_summary,
        )

        # Convert DataFrame to list for JSON serialization
        result['data'] = result['data'].to_dict(orient='records')

        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/all-latest-reports", tags=["Batch Reports"])
async def get_all_latest_reports(analyzer: StaffHealthAnalyzer = Depends(get_analyzer)):
    """Generate all latest report combinations"""
    try:
        reports = analyzer.generate_all_latest_reports()

        return {key: report.to_dict(orient='records') for key, report in reports.items()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/all-trending-reports", tags=["Batch Reports"])
async def get_all_trending_reports(analyzer: StaffHealthAnalyzer = Depends(get_analyzer)):
    """Generate all trending report combinations"""
    try:
        reports = analyzer.generate_all_trending_reports()

        return {key: report.to_dict(orient='records') for key, report in reports.items()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
