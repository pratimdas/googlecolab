Below is a consolidated, detailed specification based on our discussion. Please review and let me know if any adjustments are needed before handing off to the developer.
1. Overview

Objective:
Develop a Marketing Funnel Health Performance Dashboard for executive-level use. The dashboard will condense extensive marketing performance data into 2–3 high-impact slides that answer the key question: “Is our marketing funnel healthy?”

Primary Audience:
VP-level executives who need a concise, high-level view of key marketing performance metrics.
2. Key Metrics (KPIs)

The dashboard must focus on these top 5 KPIs:

    Marketing-Influenced Revenue (5/5 Importance)
        Direct measure of marketing's contribution to overall revenue.
        Bridges marketing activities with tangible business growth.

    Customer Acquisition Cost (CAC) (4/5 Importance)
        Shows the financial efficiency of marketing campaigns.
        Essential for evaluating profitability and resource allocation.

    Return on Marketing Investment (ROMI) (4/5 Importance)
        Reflects net profit generated by marketing efforts.
        Helps justify budgets and investments.

    Customer Lifetime Value (CLV) (4/5 Importance)
        Indicates long-term value and quality of acquired customers.
        Key for strategic planning and customer retention.

    Marketing Qualified Leads (MQL) Conversion Rate (4/5 Importance)
        Demonstrates the effectiveness of lead generation.
        Measures the alignment and effectiveness between marketing and sales.

3. Presentation and Visualization

    Slide Layout:
        2–3 slides containing big-number KPIs, trend charts, and comparisons against targets.
        Visual Components:
            Trend Charts: Visualize KPI trends over time (default intervals: monthly and quarterly, with weekly data where available).
            Comparisons vs. Targets: Highlight deviations from expected performance using static thresholds (for MVP anomaly detection).

    Design & Styling:
        Minimalist design using a Google/GCP color scheme.
        Clean layout ensuring that “content is king.”
        Executive-level default filters applied on load.

4. Interactivity and Filtering

    Default View:
        The dashboard should load with executive-level filters (e.g., aggregated view by region, product, campaign) already applied.

    Interactive Capabilities:
        Custom Date Range/Interval: Allow users to toggle between weekly, monthly, and quarterly views.
        Drill-Down: Enable interactive filtering to break down data by region, product, or campaign.
        Future Enhancement: Ability to save custom filters as “custom reports” (this feature can be prioritized for later iterations).

5. Anomaly Detection and Forecasting

    MVP Approach:
        Anomaly Detection: Implement a simple static threshold-based alert system with visual indicators when KPIs deviate from set benchmarks.
        Forecasting: Use basic forecasting methods (e.g., moving averages) to project future trends.

    Future Enhancements:
        Transition to dynamic, data-driven models for anomaly detection and refined forecasting.

6. Data Sources and Integration

    Data Sources:
        CRM (Salesforce)
        Marketing automation tools
        Web analytics
        Campaign performance data

    Data Integration:
        Data will be ingested into a SQL data warehouse (e.g., BigQuery) through existing automated and manual pipelines managed by the backend team.
        The dashboard will have full read access and will generate its own views and aggregates as needed.
        Note: There is no control over data quality or pipeline performance from the dashboard’s perspective.

7. Technical Environment

    MVP Implementation:
        Develop within an iPython notebook environment (e.g., Google Colab).
        Build charts and interactive controls using Python libraries (e.g., matplotlib) within notebook cells.
        This environment will allow rapid iteration and cell-based toggling between various views.

    Production Environment:
        Upon approval, the solution will be migrated to a production dashboard using Looker.

8. Download and Export Features

    Download Options:
        CSV: Export raw data.
        PDF: Export tables and charts as they appear in the report.
    Export Characteristics:
        For PDF exports, the design will be simple and plain (no additional layout customizations such as headers/footers or logos are required for the MVP).

9. Timeline

    Delivery:
        The MVP should be delivered ASAP to meet urgent VP presentation requirements.

10. Additional Considerations

    Future Enhancements:
        No additional requirements or enhancements are prioritized at this stage.
        Focus on building a robust MVP with potential future upgrades (custom filter saving, advanced analytics, etc.).
