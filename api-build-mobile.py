from fastapi.responses import JSONResponse
from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
from typing import Dict, Optional, Any
from matplotlib import category
from openai import api_key
from pydantic import BaseModel
import pandas as pd
import os
import uvicorn

from sqlalchemy import JSON, desc
from torch import Value
import uvicorn

# Import StaffHealthAnalyzer class
from staff_health_analyzer import StaffHealthAnalyzer, HealthMeasure, VisualizationCategory, TimePeriod, ReportType
import staff_health_analyzer

# Create FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Shared dependency to get the analyzer instance
def get_analyzer():
    data_path = os.getenv('HEALTH_DATA_PATH', 'staff_health_data.csv')
    api_key = os.getenv('OPENAI_API_KEY', None)

    return StaffHealthAnalyzer(data_path=data_path, api_key=api_key)

# Response models
class ReportResponse(BaseModel):
    report_type: str
    health_measure: str
    category: str
    data: list
    summary: Optional[str] = None

# API routes
@app.get('/', tags=['Root'])
async def root():
    """Root endpoint"""
    return {
        'message': 'Staff Health Analysis API is running properly',
        'version': '1.0.0',
        'docs': '/docs'
    }

@app.get('/health-measure', tags=['Metadata'])
async def get_health_measure():
    """Get all available health measures"""
    measures = [measure.value for measure in HealthMeasure]
    
    return measures

@app.get('/visualization-categories', tags=['Metadata'])
async def get_visualization_categories():
    """Get all available visualization categories for latest reports"""
    categories = [category.value for category in VisualizationCategory]

    return categories

@app.get('/time-periods', tags=['Metadata'])
async def get_time_periods():
    """Get all available time periods for trending reports"""
    periods = [period.value for period in TimePeriod]

    return periods

@app.get('/report-types', tags=['Metadata'])
async def get_report_types():
    """Get all available report types"""
    report_types = [report_type.value for report_type in ReportType]

    return report_types

@app.get('/report', response_model=ReportResponse, tags=['Reports'])
async def get_report(
    report_type: str = Query(..., description='Type of report: Latest or Trending'),
    health_measure: str = Query(..., description='Health measure to analyze'),
    category: str = Query(..., description='Category or time period for the report'),
    with_summary: bool = Query(False, description='Whether to include a natural language summary'),
    analyzer: StaffHealthAnalyzer = Depends(get_analyzer)
):
    """
    Generate a health analysis report based on specified parameters
    
    - **report_type**: 'Latest' for current snapshot or 'Trending' for time-based analysis
    - **health_measure**: The health metric to analyze (Overall, BMI, Hypertension, Wellness)
    - **category**:
        - For Latest report: Overall, Age_range, Gender, BMI
        - For Trending report: Weekly, Monthly, Quarterly, Yearly
    - **with_summary**: Set to true to include AI-generated natural language summary
    """
    try:
        # Validate inputs
        if report_type not in [rt.value for rt in ReportType]:
            return JSONResponse(
                status_code=400,
                content={'error': f"Invalid report type: {report_type}. Expected 'Latest' or 'Trending'"}
            )
        
        if health_measure not in [hm.value for hm in HealthMeasure]:
            return JSONResponse(
                status_code=400,
                content={'error': f"Invalid health measure: {health_measure}. Expected 'Overall', 'BMI', 'Hypertension' or 'Wellness'"}
            )
        
        # Validate category based on report type
        if report_type == 'Latest':
            if category not in [vc.value for vc in VisualizationCategory]:
                return JSONResponse(
                    status_code=400,
                    content={'error': f"Invalid category: {category}. Expected 'Overall', 'Age_range', 'Gender', 'BMI'"}
                )
        elif report_type == 'Trending':
            if category not in [tp.value for tp in TimePeriod]:
                return JSONResponse(
                    status_code=400,
                    content={'error': f"Invalid periods: {category}. Expected 'Weekly', 'Monthly', 'Quarerly', 'Yearly'"}
                )
        
        # Run analysis
        result = analyzer.run_analysis(
            report_type=report_type,
            health_measure=health_measure,
            category=category,
            with_summary=with_summary
        )

        # Convert DataFrame to list for JSON serialization
        result['data'] = result['data'].to_dict(orient='records')
        
        return result
    
    except ValueError as e:
        
        return JSONResponse(
            status_code=400,
            content={'error': str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': f"Unexpected error: {str(e)}"}
        )
    
@app.get('/all-latest-reports', tags=['Batch Reports'])
async def get_all_latest_reports(
    analyzer: StaffHealthAnalyzer = Depends(get_analyzer)
):
    """Generate all latest report combinations"""
    try:
        reports = analyzer.generate_all_reports('Latest')

        return {
            key: report.to_dict(orient='records') for key, report in reports.items()
        }
    except Exception as e:
        
        return JSONResponse(
            status_code=400,
            content={'error': str(e)}
        )
    
@app.get('/all-trending-reports', tags=['Batch Reports'])
async def get_all_trending_reports(
    analyzer: StaffHealthAnalyzer = Depends(get_analyzer)
):
    """Generate all trending report combinations"""
    try:
        reports = analyzer.generate_all_reports('Trending')

        return {
            key: report.to_dict(orient='records') for key, report in reports.items()
        }
    except Exception as e:

        return JSONResponse(
            status_code=400,
            content={'error': str(e)}
        )
    
# Main entry point
if __name__ == '__main__':
    uvicorn.run('api-build-mobile:app', host='0.0.0.0', port=8000, reload=True)