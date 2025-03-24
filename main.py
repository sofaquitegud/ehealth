# main.py

import os
import configparser
from analytics import StaffHealthAnalyzer
from preprocessing import MODE_MOBILE, MODE_KIOSK

# Main execution section
if __name__ == "__main__":
    # Load configuration
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    # Set up database configuration
    db_config = {
        "host": config["db"]["host"],
        "port": config["db"]["port"],
        "dbname": config["db"]["dbname"],
        "user": config["db"]["user"],
        "password": config["db"]["password"],
    }

    # For mobile data
    mobile_analyzer = StaffHealthAnalyzer(
        data_path="staff_health_data.csv",
        mode=MODE_MOBILE,
        api_key=os.environ.get("OPENAI_API_KEY", config["llm"]["api_key"]),
        db_config=db_config,
    )

    # For kiosk data
    kiosk_analyzer = StaffHealthAnalyzer(
        data_path="staff_health_data_kiosk.csv",
        mode=MODE_KIOSK,
        api_key=os.environ.get("OPENAI_API_KEY", config["llm"]["api_key"]),
        db_config=db_config,
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

    # Example of generating all reports
    """
    # Generate and save all latest reports
    print('Generating and saving latest reports...')
    all_latest_reports = mobile_analyzer.generate_all_reports("Latest")
    for key, report in all_latest_reports.items():
        os.makedirs("reports/latest", exist_ok=True)
        filename = f"reports/latest/{key}.csv"
        report.to_csv(filename, index=False)
        print(f"Saved {filename}")

    # Generate and save all trending reports
    print('Generating and saving trending reports...')
    all_trending_reports = mobile_analyzer.generate_all_reports("Trending")
    for key, report in all_trending_reports.items():
        os.makedirs("reports/trending", exist_ok=True)
        filename = f"reports/trending/{key}.csv"
        report.to_csv(filename, index=False)
        print(f"Saved {filename}")
    """
