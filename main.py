# main.py

import os
from analytics import StaffHealthAnalyzer
from preprocessing import MODE_MOBILE, MODE_KIOSK

# Main execution section
if __name__ == "__main__":
    # For mobile data
    mobile_analyzer = StaffHealthAnalyzer(
        data_path="staff_health_data.csv",
        mode=MODE_MOBILE,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # For kiosk data
    kiosk_analyzer = StaffHealthAnalyzer(
        data_path="staff_health_data_kiosk.csv",
        mode=MODE_KIOSK,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Generate report
    generated_report = mobile_analyzer.run_analysis(
        report_type="Trending",  # Latest | Trending
        health_measure="BMI",  # Overall | BMI | Hypertension | Stress | Wellness
        category="Monthly",  # Age_range, Gender, BMI, Overall | Weekly, Monthly, Quarterly, Yearly
        with_summary=False,  # Set to True if have an API key
        enable_display=True,  # Set to True to display visualization
    )

    print("Generated Report:")
    print(generated_report["data"])

    # Print summary if available
    if "summary" in generated_report:
        print("\nSummary:")
        print(generated_report["summary"])
