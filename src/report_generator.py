import os
import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_pdf_report(df_results, summary_metrics):
    """
    Generates downloadable PDF executive report summarizing dataset metrics,
    revenue at risk, priority customer list, and retention strategies.
    Returns bytes buffer.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=6,
        alignment=1
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.HexColor('#2563EB'),
        spaceAfter=15,
        alignment=1
    )
    
    heading2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    elements = []

    # Title & Subtitle Header
    elements.append(Paragraph("SMART CUSTOMER CHURN INTELLIGENCE REPORT", title_style))
    elements.append(Paragraph("EXECUTIVE SUMMARY & RETENTION OPTIMIZATION PLAN", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563EB'), spaceAfter=15))

    # Executive Summary Paragraph
    summary_text = (
        f"This executive intelligence report synthesizes predictive machine learning analytics across "
        f"<b>{summary_metrics.get('total_customers', len(df_results)):,} customer accounts</b>. "
        f"The platform identified <b>{summary_metrics.get('high_risk_count', 0):,} high-risk churners</b> "
        f"({summary_metrics.get('churn_rate_pct', 0.0):.1f}% churn rate) putting an estimated "
        f"<b>${summary_metrics.get('annual_revenue_at_risk', 0.0):,.2f} in annual recurring revenue at risk</b>."
    )
    elements.append(Paragraph(summary_text, body_style))
    elements.append(Spacer(1, 10))

    # KPI Table
    elements.append(Paragraph("1. Key Performance Indicators", heading2_style))
    
    kpi_data = [
        ["Metric Indicator", "Value / Volume"],
        ["Total Customers Analyzed", f"{summary_metrics.get('total_customers', len(df_results)):,}"],
        ["High-Risk Churn Accounts", f"{summary_metrics.get('high_risk_count', 0):,} ({summary_metrics.get('churn_rate_pct', 0.0):.1f}%)"],
        ["Monthly Revenue at Risk", f"${summary_metrics.get('monthly_revenue_at_risk', 0.0):,.2f}"],
        ["Annual Revenue at Risk", f"${summary_metrics.get('annual_revenue_at_risk', 0.0):,.2f}"],
        ["High-Value Annual Risk", f"${summary_metrics.get('high_value_annual_at_risk', 0.0):,.2f}"],
        ["Estimated Retention ROI", f"{summary_metrics.get('overall_roi_pct', 380.0):.1f}%"]
    ]

    t_kpi = Table(kpi_data, colWidths=[280, 240])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8FAFC')),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(t_kpi)
    elements.append(Spacer(1, 12))

    # Top Priority Customers Table
    elements.append(Paragraph("2. Critical High-Priority Customers for Immediate Action", heading2_style))
    
    if "Priority_Score" in df_results.columns:
        top_df = df_results.sort_values(by="Priority_Score", ascending=False).head(8)
    else:
        top_df = df_results.head(8)

    prio_data = [["Customer ID", "Contract", "Monthly Bill", "Churn Prob", "Priority Score", "Risk Level"]]
    for _, r in top_df.iterrows():
        prio_data.append([
            str(r.get("customerID", "N/A")),
            str(r.get("Contract", "N/A")),
            f"${float(r.get('MonthlyCharges', 0)):.2f}",
            f"{float(r.get('Churn_Probability', 0))*100:.1f}%",
            f"{float(r.get('Priority_Score', 0)):.1f}",
            str(r.get("Risk_Level", "High Risk"))
        ])

    t_prio = Table(prio_data, colWidths=[80, 90, 80, 80, 90, 100])
    t_prio.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (2, 1), (4, -1), 'CENTER')
    ]))
    elements.append(t_prio)
    elements.append(Spacer(1, 12))

    # Strategic Recommendations Section
    elements.append(Paragraph("3. Strategic Action Plan & ROI Optimization", heading2_style))
    strat_text = (
        "<b>Actionable Retention Playbook:</b><br/>"
        "1. <b>Convert Month-to-Month Contracts:</b> Deploy 10-15% discount incentives for 1-year contract lock-ins.<br/>"
        "2. <b>VIP Tech Support Bundles:</b> Offer free 90-day tech support for customers with no support add-ons.<br/>"
        "3. <b>Automated Billing:</b> Transition paper check payers to automated credit/bank billing to eliminate friction.<br/>"
        "4. <b>High-Value Outreach:</b> Deploy personal success calls for churners with CLV exceeding $1,500."
    )
    elements.append(Paragraph(strat_text, body_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
