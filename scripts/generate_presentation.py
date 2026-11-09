"""Generate professional academic PowerPoint presentation for ETAFlow."""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_path: str = "ETAFlow_Presentation.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette: Clean Warm Light Theme with Orange Accent
    BG_WARM = RGBColor(250, 249, 246)       # Light Warm Off-White
    NAVY = RGBColor(15, 23, 42)             # Slate-900 / Dark Navy
    SLATE = RGBColor(51, 65, 85)            # Slate-700
    MUTED = RGBColor(100, 116, 139)         # Slate-500
    ORANGE = RGBColor(234, 88, 12)          # Vibrant Orange (#EA580C)
    ORANGE_LIGHT = RGBColor(255, 237, 213)  # Soft Orange Background (#FFEDD5)
    ORANGE_BORDER = RGBColor(253, 186, 116) # Orange Border
    CARD_BG = RGBColor(255, 255, 255)       # Pure White Card
    CARD_BORDER = RGBColor(226, 232, 240)   # Light Grey Border (#E2E8F0)
    ALT_ROW_BG = RGBColor(248, 250, 252)    # Subtle Row Background
    GREEN = RGBColor(5, 150, 105)           # Mint Green Accent
    GREEN_LIGHT = RGBColor(209, 250, 229)

    FONT_HEADING = "Segoe UI"
    FONT_BODY = "Segoe UI"

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_WARM
        bg.line.fill.background()
        return bg

    def add_header(slide, title_text, subtitle_text, category_badge="MLOPS PLATFORM • B.TECH CAPSTONE"):
        # Category Badge
        badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.4), Inches(3.2), Inches(0.32))
        badge.fill.solid()
        badge.fill.fore_color.rgb = ORANGE_LIGHT
        badge.line.color.rgb = ORANGE_BORDER
        badge.line.width = Pt(1)
        tf = badge.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.text = category_badge
        p.font.name = FONT_BODY
        p.font.size = Pt(9)
        p.font.bold = True
        p.font.color.rgb = ORANGE
        p.alignment = PP_ALIGN.CENTER

        # Title
        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.72), Inches(11.5), Inches(0.55))
        tf_title = tb_title.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.name = FONT_HEADING
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = NAVY

        # Subtitle
        tb_sub = slide.shapes.add_textbox(Inches(0.8), Inches(1.28), Inches(11.5), Inches(0.35))
        tf_sub = tb_sub.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle_text
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = MUTED

        # Subtle divider line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.68), Inches(11.733), Inches(0.02))
        line.fill.solid()
        line.fill.fore_color.rgb = CARD_BORDER
        line.line.fill.background()

    def format_cell(cell, text, bold=False, color=NAVY, size=10, align=PP_ALIGN.LEFT, bg_color=None):
        if bg_color:
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_color
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.12)
        tf.margin_right = Inches(0.12)
        tf.margin_top = Inches(0.08)
        tf.margin_bottom = Inches(0.08)
        tf.clear()
        p = tf.paragraphs[0]
        p.text = str(text)
        p.font.name = FONT_BODY
        p.font.bold = bold
        p.font.color.rgb = color
        p.font.size = Pt(size)
        p.alignment = align

    # ==========================================================
    # SLIDE 1: PROJECT INTRODUCTION
    # ==========================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)

    # Top Badge
    b1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.8), Inches(4.5), Inches(0.38))
    b1.fill.solid()
    b1.fill.fore_color.rgb = ORANGE_LIGHT
    b1.line.color.rgb = ORANGE_BORDER
    tf = b1.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = "B.TECH CAPSTONE PROJECT  •  ACADEMIC YEAR 2026"
    p.font.bold = True
    p.font.size = Pt(10)
    p.font.color.rgb = ORANGE
    p.alignment = PP_ALIGN.CENTER

    # Main Project Title
    tb = s1.shapes.add_textbox(Inches(0.8), Inches(1.35), Inches(11.5), Inches(1.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "ETAFlow"
    p.font.name = FONT_HEADING
    p.font.size = Pt(46)
    p.font.bold = True
    p.font.color.rgb = NAVY

    p2 = tf.add_paragraph()
    p2.text = "MLOps Platform for Food Delivery ETA Prediction"
    p2.font.name = FONT_HEADING
    p2.font.size = Pt(22)
    p2.font.bold = True
    p2.font.color.rgb = ORANGE

    p3 = tf.add_paragraph()
    p3.text = "An applied machine learning and MLOps system predicting end-to-end food delivery duration at checkout, preventing data leakage, tracking experiments, and monitoring statistical drift in production."
    p3.font.name = FONT_BODY
    p3.font.size = Pt(12)
    p3.font.color.rgb = SLATE

    # Left Card: Problem Statement & Objectives
    c_prob = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(3.6), Inches(6.0), Inches(3.2))
    c_prob.fill.solid()
    c_prob.fill.fore_color.rgb = CARD_BG
    c_prob.line.color.rgb = CARD_BORDER
    c_prob.line.width = Pt(1)
    tf = c_prob.text_frame
    tf.margin_left = tf.margin_right = Inches(0.3)
    tf.margin_top = Inches(0.25)
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "PROBLEM STATEMENT & OBJECTIVES"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ORANGE

    points_prob = [
        ("The Challenge: ", "In food delivery platforms, inaccurate delivery time predictions (ETAs) lead to customer frustration, bad reviews, and inefficient kitchen dispatching."),
        ("The Technical Risk: ", "Most baseline models suffer from data leakage by accidentally using post-booking signals (like actual pickup time or driver speed) unavailable at checkout."),
        ("The ETAFlow Solution: ", "We built a leakage-safe ML pipeline that predicts delivery duration strictly at booking time using 11 clean spatial, temporal, and order attributes."),
        ("Production MLOps: ", "Integrated MLflow experiment tracking, model registry governance, statistical drift detection (PSI, KS-test), and a real-time web dashboard.")
    ]
    for bold_prefix, text in points_prob:
        p = tf.add_paragraph()
        p.space_before = Pt(8)
        run_b = p.add_run()
        run_b.text = bold_prefix
        run_b.font.bold = True
        run_b.font.size = Pt(10)
        run_b.font.color.rgb = NAVY
        run_t = p.add_run()
        run_t.text = text
        run_t.font.size = Pt(10)
        run_t.font.color.rgb = SLATE

    # Right Card: Academic Institution & Team Members
    c_team = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.1), Inches(3.6), Inches(5.433), Inches(3.2))
    c_team.fill.solid()
    c_team.fill.fore_color.rgb = CARD_BG
    c_team.line.color.rgb = CARD_BORDER
    c_team.line.width = Pt(1)
    tf = c_team.text_frame
    tf.margin_left = tf.margin_right = Inches(0.3)
    tf.margin_top = Inches(0.25)
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "COLLEGE & PROJECT TEAM"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = ORANGE

    p = tf.add_paragraph()
    p.space_before = Pt(6)
    p.text = "Vishwakarma Institute of Technology, Pune"
    p.font.bold = True
    p.font.size = Pt(12)
    p.font.color.rgb = NAVY

    p = tf.add_paragraph()
    p.text = "Department of Artificial Intelligence and Data Science"
    p.font.size = Pt(10)
    p.font.color.rgb = MUTED

    members = [
        ("1. Achal Chidrawar", "Data Engineering, Validation & Leakage Prevention Lead"),
        ("2. Chinmayee Choudhari", "Machine Learning, Benchmarking & Experimentation Lead"),
        ("3. Vidal Deo", "MLOps, API Development, Drift Observability & Testing Lead")
    ]
    for name, role in members:
        p = tf.add_paragraph()
        p.space_before = Pt(12)
        run_n = p.add_run()
        run_n.text = name + "\n"
        run_n.font.bold = True
        run_n.font.size = Pt(11)
        run_n.font.color.rgb = NAVY
        run_r = p.add_run()
        run_r.text = role
        run_r.font.size = Pt(9.5)
        run_r.font.color.rgb = SLATE

    # ==========================================================
    # SLIDE 2: ETAFLOW AT A GLANCE
    # ==========================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2)
    add_header(s2, "ETAFlow at a Glance", "End-to-End System Architecture, Data Pipeline, and Verified Benchmark Results")

    # 6-Step Horizontal Architecture Workflow
    workflow_steps = [
        ("1. Public Data", "38,964 records\n22 raw attributes\nSHA-256 verified"),
        ("2. Leakage Guard", "11 booking features\nPost-booking excluded\nHTTP 400 rejection"),
        ("3. 80/20 Split", "31,171 train rows\n7,793 test rows\nMedian/One-hot prep"),
        ("4. ML Models", "Dummy floor (7.43m)\nRidge linear (6.62m)\nRandom Forest (6.14m)"),
        ("5. MLOps Store", "MLflow SQLite log\nmodels/registry.json\nActive version v1.0.0"),
        ("6. Serving & UI", "Standard library API\n<4ms p50 latency\nDrift Guard (PSI: 0.01)")
    ]

    x_start = Inches(0.8)
    box_width = Inches(1.8)
    box_gap = Inches(0.186)
    y_box = Inches(1.85)

    for i, (title, desc) in enumerate(workflow_steps):
        cur_x = x_start + i * (box_width + box_gap)
        box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cur_x, y_box, box_width, Inches(1.4))
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = ORANGE if i == 3 else CARD_BORDER
        box.line.width = Pt(1.5 if i == 3 else 1)
        tf = box.text_frame
        tf.margin_left = tf.margin_right = Inches(0.1)
        tf.margin_top = Inches(0.12)
        p = tf.paragraphs[0]
        p.text = title
        p.font.bold = True
        p.font.size = Pt(10)
        p.font.color.rgb = ORANGE if i == 3 else NAVY
        p.alignment = PP_ALIGN.CENTER
        p2 = tf.add_paragraph()
        p2.space_before = Pt(4)
        p2.text = desc
        p2.font.size = Pt(8.5)
        p2.font.color.rgb = SLATE
        p2.alignment = PP_ALIGN.CENTER

    # 3 Large Metric Callout Cards (Left side, matching PredictOps Slide 2)
    callout_metrics = [
        ("6.14 min", "MAE of Random Forest champion", "down from 7.43 min for the Dummy baseline (-17.4% error reduction; R²: 0.3036)"),
        ("11 / 22", "Booking-time features retained", "Target leakage (delivery_speed) and 10 post-booking timing columns rejected"),
        ("0.0101", "Prediction drift PSI on test split", "Verified stable across 7,793 test rows with automated Kolmogorov-Smirnov alerts")
    ]

    for i, (metric, sub1, sub2) in enumerate(callout_metrics):
        cx = Inches(0.8) + i * Inches(2.3)
        card = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, Inches(3.45), Inches(2.15), Inches(3.5))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = ORANGE if i == 0 else CARD_BORDER
        card.line.width = Pt(1.5 if i == 0 else 1)
        tf = card.text_frame
        tf.margin_left = tf.margin_right = Inches(0.15)
        tf.margin_top = Inches(0.2)
        tf.word_wrap = True

        p_m = tf.paragraphs[0]
        p_m.text = metric
        p_m.font.name = FONT_HEADING
        p_m.font.size = Pt(28)
        p_m.font.bold = True
        p_m.font.color.rgb = ORANGE if i == 0 else NAVY
        p_m.alignment = PP_ALIGN.CENTER

        p_s1 = tf.add_paragraph()
        p_s1.space_before = Pt(8)
        p_s1.text = sub1
        p_s1.font.bold = True
        p_s1.font.size = Pt(10)
        p_s1.font.color.rgb = NAVY
        p_s1.alignment = PP_ALIGN.CENTER

        p_s2 = tf.add_paragraph()
        p_s2.space_before = Pt(6)
        p_s2.text = sub2
        p_s2.font.size = Pt(8.5)
        p_s2.font.color.rgb = MUTED
        p_s2.alignment = PP_ALIGN.CENTER

    # Actual Model Benchmarking Table (Right side)
    tb_table_title = s2.shapes.add_textbox(Inches(7.75), Inches(3.45), Inches(4.783), Inches(0.35))
    tf = tb_table_title.text_frame
    p = tf.paragraphs[0]
    p.text = "VERIFIED MODEL PERFORMANCE (7,793 TEST ROWS)"
    p.font.bold = True
    p.font.size = Pt(10.5)
    p.font.color.rgb = ORANGE

    rows, cols = 5, 5
    tbl_shape = s2.shapes.add_table(rows, cols, Inches(7.75), Inches(3.85), Inches(4.783), Inches(2.2))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(1.3)
    tbl.columns[1].width = Inches(0.85)
    tbl.columns[2].width = Inches(0.85)
    tbl.columns[3].width = Inches(0.8)
    tbl.columns[4].width = Inches(0.983)

    headers = ["Model Architecture", "MAE", "RMSE", "R²", "MAPE"]
    for j, h in enumerate(headers):
        format_cell(tbl.cell(0, j), h, bold=True, color=NAVY, size=9, align=PP_ALIGN.CENTER, bg_color=ORANGE_LIGHT)

    table_data = [
        ("DummyRegressor (Mean)", "7.4305 m", "9.2257 m", "-0.0001", "33.33%"),
        ("Ridge Regression", "6.6154 m", "8.1635 m", "0.2169", "29.54%"),
        ("Random Forest (Champion)", "6.1406 m", "7.6987 m", "0.3036", "27.41%"),
        ("Improvement vs Baseline", "-1.2900 m", "-1.5270 m", "+0.3037", "-5.92%")
    ]
    for i, row in enumerate(table_data):
        is_champion = (i == 2)
        is_delta = (i == 3)
        row_bg = ORANGE_LIGHT if is_champion else (ALT_ROW_BG if i % 2 == 1 else CARD_BG)
        txt_color = ORANGE if (is_champion or is_delta) else NAVY
        for j, val in enumerate(row):
            format_cell(tbl.cell(i+1, j), val, bold=(is_champion or is_delta), color=txt_color, size=8.5,
                        align=PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER, bg_color=row_bg)

    # Footer callout in Slide 2
    tb_foot = s2.shapes.add_textbox(Inches(7.75), Inches(6.15), Inches(4.783), Inches(0.8))
    tf = tb_foot.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Methodological Guarantee: Models were trained strictly on 31,171 training rows using the preprocessor fitted on the training split. Zero test-set snooping or target leakage occurred."
    p.font.size = Pt(8.5)
    p.font.color.rgb = MUTED

    # ==========================================================
    # SLIDE 3: SKILLS NEEDED TO START THIS PROJECT
    # ==========================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3)
    add_header(s3, "Skills Needed to Start This Project", "Foundational Technical Capabilities and Engineering Prerequisities")

    rows, cols = 11, 3
    tbl_shape = s3.shapes.add_table(rows, cols, Inches(0.8), Inches(1.85), Inches(11.733), Inches(4.6))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(2.5)
    tbl.columns[1].width = Inches(7.433)
    tbl.columns[2].width = Inches(1.8)

    format_cell(tbl.cell(0, 0), "Technical Skill", bold=True, color=NAVY, size=10, align=PP_ALIGN.LEFT, bg_color=ORANGE_LIGHT)
    format_cell(tbl.cell(0, 1), "Why ETAFlow Needs It", bold=True, color=NAVY, size=10, align=PP_ALIGN.LEFT, bg_color=ORANGE_LIGHT)
    format_cell(tbl.cell(0, 2), "Level to Start", bold=True, color=NAVY, size=10, align=PP_ALIGN.CENTER, bg_color=ORANGE_LIGHT)

    skills_data = [
        ("Python", "Core programming language for data loading, pipeline execution, and standard library API server implementation.", "Basic"),
        ("Pandas", "Ingesting, inspecting, validating schemas, and manipulating the 38,964 historical food delivery records.", "Basic"),
        ("Data Preprocessing", "Cleaning missing values, median imputation, standard scaling, and one-hot encoding categorical variables.", "Basic"),
        ("Machine Learning Fundamentals", "Understanding supervised learning, bias-variance tradeoff, train/test partitioning, and tree ensembles.", "Basic"),
        ("Regression Models", "Formulating baseline floor models (mean forecast), regularized linear surfaces, and non-linear random forests.", "Basic"),
        ("Model Evaluation", "Quantifying predictive error using domain metrics: MAE (minutes), RMSE (outlier penalty), R² score, and MAPE.", "Basic"),
        ("API Development", "Implementing HTTP request/response lifecycles, JSON payload parsing, status codes, and input contract validation.", "Beginner"),
        ("MLflow", "Digital experiment tracking lab: recording model parameters, metrics, run tags, and serialized artifacts in SQLite.", "Beginner"),
        ("HTML, CSS & JavaScript", "Building a modern, responsive browser interface with Leaflet maps, live telemetry metrics, and drift toggles.", "Beginner"),
        ("Software Testing", "Writing unit and integration tests using pytest to ensure data schema integrity, zero leakage, and API stability.", "Beginner")
    ]

    for i, (sk, why, lvl) in enumerate(skills_data):
        bg = ALT_ROW_BG if i % 2 == 1 else CARD_BG
        format_cell(tbl.cell(i+1, 0), sk, bold=True, color=NAVY, size=9, align=PP_ALIGN.LEFT, bg_color=bg)
        format_cell(tbl.cell(i+1, 1), why, bold=False, color=SLATE, size=9, align=PP_ALIGN.LEFT, bg_color=bg)
        format_cell(tbl.cell(i+1, 2), lvl, bold=True, color=ORANGE if lvl == "Basic" else MUTED, size=9, align=PP_ALIGN.CENTER, bg_color=bg)

    # Slide 3 Bottom Note
    tb_note = s3.shapes.add_textbox(Inches(0.8), Inches(6.6), Inches(11.733), Inches(0.45))
    tf = tb_note.text_frame
    p = tf.paragraphs[0]
    p.text = "Practical engineering insight: We did not need advanced cloud certifications to begin. Solid fundamentals in Python, Pandas, and basic regression allowed us to build the entire production MLOps system step-by-step."
    p.font.size = Pt(9.5)
    p.font.color.rgb = MUTED

    # ==========================================================
    # SLIDE 4: ACHAL CHIDRAWAR - DATA ENGINEERING & VALIDATION
    # ==========================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4)
    add_header(s4, "Person 1: Achal Chidrawar", "Role: Data Engineering & Validation Lead  •  Before the Project → After the Project", "TEAM MEMBER 1")

    # Contribution Summary Banner
    b_sum = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.85), Inches(11.733), Inches(0.65))
    b_sum.fill.solid()
    b_sum.fill.fore_color.rgb = CARD_BG
    b_sum.line.color.rgb = ORANGE
    b_sum.line.width = Pt(1)
    tf = b_sum.text_frame
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.12)
    p = tf.paragraphs[0]
    p.text = "CONTRIBUTION SUMMARY: Responsible for dataset preparation, data validation, feature engineering and leakage prevention."
    p.font.bold = True
    p.font.size = Pt(10.5)
    p.font.color.rgb = ORANGE
    p2 = tf.add_paragraph()
    p2.text = "Key Deliverables: data/raw audit, src/etaflow/data/loader.py, validation.py, reports/validation/data_validation_report.json, and SHA-256 fingerprinting."
    p2.font.size = Pt(9)
    p2.font.color.rgb = MUTED

    # Exact Before vs After Table for Achal
    rows, cols = 6, 3
    tbl_shape = s4.shapes.add_table(rows, cols, Inches(0.8), Inches(2.65), Inches(11.733), Inches(4.0))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(2.4)
    tbl.columns[1].width = Inches(4.4)
    tbl.columns[2].width = Inches(4.933)

    format_cell(tbl.cell(0, 0), "Skills", bold=True, color=NAVY, size=10.5, align=PP_ALIGN.LEFT, bg_color=ORANGE_LIGHT)
    format_cell(tbl.cell(0, 1), "Before the Project", bold=True, color=NAVY, size=10.5, align=PP_ALIGN.LEFT, bg_color=CARD_BORDER)
    format_cell(tbl.cell(0, 2), "After the Project", bold=True, color=ORANGE, size=10.5, align=PP_ALIGN.LEFT, bg_color=ORANGE_LIGHT)

    p1_data = [
        ("Python & Pandas", "Basic knowledge of Python and data handling", "Used Pandas to load, clean and process the food delivery dataset"),
        ("Data Validation", "Beginner-level understanding", "Learned to check missing values, data types and dataset quality"),
        ("Feature Engineering", "Basic understanding of features", "Created useful booking-time features, including delivery distance"),
        ("Data Leakage", "Beginner-level understanding", "Learned to prevent the use of information unavailable at booking time"),
        ("Data Processing", "Basic knowledge", "Prepared training and testing datasets for machine learning")
    ]

    for i, (sk, bfr, aft) in enumerate(p1_data):
        bg_left = ALT_ROW_BG if i % 2 == 1 else CARD_BG
        bg_right = RGBColor(255, 247, 237) if i % 2 == 1 else CARD_BG
        format_cell(tbl.cell(i+1, 0), sk, bold=True, color=NAVY, size=9.5, align=PP_ALIGN.LEFT, bg_color=bg_left)
        format_cell(tbl.cell(i+1, 1), bfr, bold=False, color=SLATE, size=9.5, align=PP_ALIGN.LEFT, bg_color=bg_left)
        format_cell(tbl.cell(i+1, 2), aft, bold=True, color=NAVY, size=9.5, align=PP_ALIGN.LEFT, bg_color=bg_right)

    tb_foot = s4.shapes.add_textbox(Inches(0.8), Inches(6.8), Inches(11.733), Inches(0.35))
    tf = tb_foot.text_frame
    p = tf.paragraphs[0]
    p.text = "Practical Outcome: Zero target leakage. Banned delivery_speed and lookahead pickup timestamps; verified 38,964 rows with cryptographic SHA-256 hash."
    p.font.size = Pt(8.5)
    p.font.color.rgb = MUTED

    # ==========================================================
    # SLIDE 5: CHINMAYEE CHOUDHARI - MACHINE LEARNING & EXPERIMENTATION
    # ==========================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "Person 2: Chinmayee Choudhari", "Role: Machine Learning & Experimentation Lead  •  Before the Project → After the Project", "TEAM MEMBER 2")

    b_sum = s5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.85), Inches(11.733), Inches(0.65))
    b_sum.fill.solid()
    b_sum.fill.fore_color.rgb = CARD_BG
    b_sum.line.color.rgb = ORANGE
    b_sum.line.width = Pt(1)
    tf = b_sum.text_frame
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.12)
    p = tf.paragraphs[0]
    p.text = "CONTRIBUTION SUMMARY: Responsible for machine learning development, model evaluation, comparison and MLflow experiment tracking."
    p.font.bold = True
    p.font.size = Pt(10.5)
    p.font.color.rgb = ORANGE
    p2 = tf.add_paragraph()
    p2.text = "Key Deliverables: src/etaflow/models/baseline.py, regressors.py, pipelines/train_mlflow.py, reports/experiments/experiment_summary.json (MAE 6.14m)."
    p2.font.size = Pt(9)
    p2.font.color.rgb = MUTED

    rows, cols = 6, 3
    tbl_shape = s5.shapes.add_table(rows, cols, Inches(0.8), Inches(2.65), Inches(11.733), Inches(4.0))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(2.4)
    tbl.columns[1].width = Inches(4.4)
    tbl.columns[2].width = Inches(4.933)

    format_cell(tbl.cell(0, 0), "Skills", bold=True, color=NAVY, size=10.5, align=PP_ALIGN.LEFT, bg_color=ORANGE_LIGHT)
    format_cell(tbl.cell(0, 1), "Before the Project", bold=True, color=NAVY, size=10.5, align=PP_ALIGN.LEFT, bg_color=CARD_BORDER)
    format_cell(tbl.cell(0, 2), "After the Project", bold=True, color=ORANGE, size=10.5, align=PP_ALIGN.LEFT, bg_color=ORANGE_LIGHT)

    p2_data = [
        ("Machine Learning", "Basic theoretical knowledge", "Applied machine learning to predict food delivery duration"),
        ("Model Training", "Beginner-level understanding", "Trained Dummy, Ridge and Random Forest models"),
        ("Model Evaluation", "Basic understanding of evaluation", "Compared models using MAE, RMSE, R² and MAPE"),
        ("Hyperparameter Tuning", "Beginner-level understanding", "Learned how model settings affect prediction performance"),
        ("Experiment Tracking", "No practical experience", "Used MLflow to record model experiments, parameters and results")
    ]

    for i, (sk, bfr, aft) in enumerate(p2_data):
        bg_left = ALT_ROW_BG if i % 2 == 1 else CARD_BG
        bg_right = RGBColor(255, 247, 237) if i % 2 == 1 else CARD_BG
        format_cell(tbl.cell(i+1, 0), sk, bold=True, color=NAVY, size=9.5, align=PP_ALIGN.LEFT, bg_color=bg_left)
        format_cell(tbl.cell(i+1, 1), bfr, bold=False, color=SLATE, size=9.5, align=PP_ALIGN.LEFT, bg_color=bg_left)
        format_cell(tbl.cell(i+1, 2), aft, bold=True, color=NAVY, size=9.5, align=PP_ALIGN.LEFT, bg_color=bg_right)

    tb_foot = s5.shapes.add_textbox(Inches(0.8), Inches(6.8), Inches(11.733), Inches(0.35))
    tf = tb_foot.text_frame
    p = tf.paragraphs[0]
    p.text = "Practical Outcome: Random Forest Champion reduced MAE from 7.43m to 6.14m (-17.4%); logged 3 reproducible runs in SQLite MLflow experiment store."
    p.font.size = Pt(8.5)
    p.font.color.rgb = MUTED

    # ==========================================================
    # SLIDE 6: VIDAL DEO - MLOPS, API & OBSERVABILITY
    # ==========================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "Person 3: Vidal Deo", "Role: MLOps, API & Observability Lead  •  Before the Project → After the Project", "TEAM MEMBER 3")

    b_sum = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.85), Inches(11.733), Inches(0.65))
    b_sum.fill.solid()
    b_sum.fill.fore_color.rgb = CARD_BG
    b_sum.line.color.rgb = ORANGE
    b_sum.line.width = Pt(1)
    tf = b_sum.text_frame
    tf.margin_left = Inches(0.2)
    tf.margin_top = Inches(0.12)
    p = tf.paragraphs[0]
    p.text = "CONTRIBUTION SUMMARY: Responsible for API integration, model registry, drift monitoring, dashboard development and testing."
    p.font.bold = True
    p.font.size = Pt(10.5)
    p.font.color.rgb = ORANGE
    p2 = tf.add_paragraph()
    p2.text = "Key Deliverables: src/etaflow/api/app.py, schema.py, telemetry.py, src/etaflow/mlops/drift.py, registry.json, app/ dashboard, 79 pytest cases."
    p2.font.size = Pt(9)
    p2.font.color.rgb = MUTED

    rows, cols = 7, 3
    tbl_shape = s6.shapes.add_table(rows, cols, Inches(0.8), Inches(2.65), Inches(11.733), Inches(4.0))
    tbl = tbl_shape.table
    tbl.columns[0].width = Inches(2.4)
    tbl.columns[1].width = Inches(4.4)
    tbl.columns[2].width = Inches(4.933)

    format_cell(tbl.cell(0, 0), "Skills", bold=True, color=NAVY, size=10.5, align=PP_ALIGN.LEFT, bg_color=ORANGE_LIGHT)
    format_cell(tbl.cell(0, 1), "Before the Project", bold=True, color=NAVY, size=10.5, align=PP_ALIGN.LEFT, bg_color=CARD_BORDER)
    format_cell(tbl.cell(0, 2), "After the Project", bold=True, color=ORANGE, size=10.5, align=PP_ALIGN.LEFT, bg_color=ORANGE_LIGHT)

    p3_data = [
        ("API Development", "Beginner-level understanding", "Developed REST API endpoints for ETA prediction"),
        ("Model Deployment", "Basic theoretical understanding", "Learned to load and serve a saved machine learning model"),
        ("Model Registry", "Beginner-level understanding", "Implemented model version tracking and artifact verification"),
        ("Drift Monitoring", "Basic theoretical understanding", "Applied PSI, KS-test and Chi-Square techniques"),
        ("Dashboard Development", "Basic HTML, CSS and JavaScript knowledge", "Developed an interactive MLOps dashboard"),
        ("Software Testing", "Beginner-level understanding", "Worked with automated testing using pytest")
    ]

    for i, (sk, bfr, aft) in enumerate(p3_data):
        bg_left = ALT_ROW_BG if i % 2 == 1 else CARD_BG
        bg_right = RGBColor(255, 247, 237) if i % 2 == 1 else CARD_BG
        format_cell(tbl.cell(i+1, 0), sk, bold=True, color=NAVY, size=9, align=PP_ALIGN.LEFT, bg_color=bg_left)
        format_cell(tbl.cell(i+1, 1), bfr, bold=False, color=SLATE, size=9, align=PP_ALIGN.LEFT, bg_color=bg_left)
        format_cell(tbl.cell(i+1, 2), aft, bold=True, color=NAVY, size=9, align=PP_ALIGN.LEFT, bg_color=bg_right)

    tb_foot = s6.shapes.add_textbox(Inches(0.8), Inches(6.8), Inches(11.733), Inches(0.35))
    tf = tb_foot.text_frame
    p = tf.paragraphs[0]
    p.text = "Practical Outcome: Sub-4ms p50 latency on POST /predict; mathematical drift detection alerting on distribution shifts; 79 automated pytest tests passing."
    p.font.size = Pt(8.5)
    p.font.color.rgb = MUTED

    # ==========================================================
    # SLIDE 7: TEAM ACHIEVEMENTS
    # ==========================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "What We Achieved Together", "Collective Engineering Outcomes Across the Complete MLOps Lifecycle")

    achieve_pillars = [
        ("1. Data Engineering & Governance",
         "Sourced and audited 38,964 records",
         [
             "Prepared and validated the food delivery dataset with zero duplicate rows.",
             "Created 11 leakage-safe booking-time features (Haversine distance, cyclical order hours, day of week).",
             "Prevented data leakage by discovering and excluding delivery_speed (direct target leakage) and 10 post-booking timing fields.",
             "Generated SHA-256 cryptographic provenance hash: a76ed049cd698b189797b180ffce60dcdad7b45b99729cd171b5562ec585eb0c."
         ]),
        ("2. Machine Learning & Benchmarking",
         "Trained and compared 3 regression models",
         [
             "Established DummyRegressor floor (7.43m MAE) and Ridge linear surface (6.62m MAE).",
             "Trained Random Forest Regressor (100 trees, depth 16) achieving champion score of 6.1406 min MAE.",
             "Achieved 17.4% error reduction over baseline and explained 30.4% of delivery duration variance (R² = 0.3036).",
             "Extracted 17 Gini impurity feature importances led by distance_km (30.6%) and order_hour (26.1%)."
         ]),
        ("3. Production MLOps & Observability",
         "Delivered tracking, registry, API and dashboard",
         [
             "Implemented MLflow experiment tracking recording parameters, metrics and artifacts in SQLite.",
             "Created model registry manifest (models/registry.json) with SHA-256 weights verification.",
             "Developed lightweight HTTP inference API with input validation (<4ms p50 latency).",
             "Built drift monitoring (PSI, KS-test, Chi-Square) and interactive web dashboard.",
             "Verified all 79 automated unit and integration tests passing with pytest."
         ])
    ]

    for i, (title, subtitle, bullets) in enumerate(achieve_pillars):
        cx = Inches(0.8) + i * Inches(4.0)
        card = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, Inches(1.85), Inches(3.733), Inches(4.5))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = ORANGE if i == 2 else CARD_BORDER
        card.line.width = Pt(1.5 if i == 2 else 1)
        tf = card.text_frame
        tf.margin_left = tf.margin_right = Inches(0.2)
        tf.margin_top = Inches(0.25)
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEADING
        p.font.bold = True
        p.font.size = Pt(12)
        p.font.color.rgb = ORANGE

        p_sub = tf.add_paragraph()
        p_sub.space_before = Pt(3)
        p_sub.text = subtitle
        p_sub.font.bold = True
        p_sub.font.size = Pt(9.5)
        p_sub.font.color.rgb = NAVY

        for b in bullets:
            p_b = tf.add_paragraph()
            p_b.space_before = Pt(8)
            p_b.text = "• " + b
            p_b.font.size = Pt(8.5)
            p_b.font.color.rgb = SLATE

    # Slide 7 Bottom Banner
    b_banner = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(6.5), Inches(11.733), Inches(0.65))
    b_banner.fill.solid()
    b_banner.fill.fore_color.rgb = NAVY
    b_banner.line.fill.background()
    tf = b_banner.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = "From raw dataset exploration to a versioned, tracked, tested and deployable ML platform: practical MLOps skills none of us had before."
    p.font.bold = True
    p.font.size = Pt(10.5)
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.alignment = PP_ALIGN.CENTER

    # ==========================================================
    # HELPER FOR CERTIFICATE SLIDES (8, 9, 10)
    # ==========================================================
    def add_certificate_slide(slide, member_name, cert1_name, cert2_name, slide_num_label):
        set_slide_background(slide)
        add_header(slide, f"Individual Course Certificates – {member_name}",
                   f"Academic Credential Portfolio  •  B.Tech Artificial Intelligence & Data Science, VIT Pune",
                   slide_num_label)

        certs = [cert1_name, cert2_name]
        for i, cert_name in enumerate(certs):
            cx = Inches(0.8) + i * Inches(6.0)
            card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx, Inches(1.85), Inches(5.733), Inches(5.2))
            card.fill.solid()
            card.fill.fore_color.rgb = CARD_BG
            card.line.color.rgb = CARD_BORDER
            card.line.width = Pt(1)

            # Header Badge inside card
            b_badge = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx + Inches(0.2), Inches(2.05), Inches(2.6), Inches(0.28))
            b_badge.fill.solid()
            b_badge.fill.fore_color.rgb = ORANGE_LIGHT
            b_badge.line.color.rgb = ORANGE_BORDER
            tf = b_badge.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.text = f"COURSE CERTIFICATE {i+1}"
            p.font.bold = True
            p.font.size = Pt(8.5)
            p.font.color.rgb = ORANGE
            p.alignment = PP_ALIGN.CENTER

            # Certificate Course Title
            tb_title = slide.shapes.add_textbox(cx + Inches(0.2), Inches(2.38), Inches(5.333), Inches(0.6))
            tf = tb_title.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = cert_name
            p.font.name = FONT_HEADING
            p.font.bold = True
            p.font.size = Pt(13)
            p.font.color.rgb = NAVY

            p_sub = tf.add_paragraph()
            p_sub.text = f"Learner: {member_name}"
            p_sub.font.size = Pt(9.5)
            p_sub.font.color.rgb = MUTED

            # Large Rectangular Placeholder for Certificate Image
            box_placeholder = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx + Inches(0.2), Inches(3.05), Inches(5.333), Inches(3.2))
            box_placeholder.fill.solid()
            box_placeholder.fill.fore_color.rgb = ALT_ROW_BG
            box_placeholder.line.color.rgb = ORANGE_BORDER
            box_placeholder.line.width = Pt(1.5)
            box_placeholder.line.dash_style = 2  # Dashed line
            tf = box_placeholder.text_frame
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.text = f"[ Insert Certificate Image Here ]\n\n{cert_name}\n({member_name})"
            p.font.bold = True
            p.font.size = Pt(11)
            p.font.color.rgb = MUTED
            p.alignment = PP_ALIGN.CENTER

            # Footer space for details
            tb_foot = slide.shapes.add_textbox(cx + Inches(0.2), Inches(6.35), Inches(5.333), Inches(0.6))
            tf = tb_foot.text_frame
            tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
            p = tf.paragraphs[0]
            p.text = "Issuing Organization: [ Enter Issuing Platform / University ]"
            p.font.size = Pt(8.5)
            p.font.color.rgb = SLATE
            p2 = tf.add_paragraph()
            p2.text = "Verification URL / Certificate ID: [ Enter Verification Link ]"
            p2.font.size = Pt(8.5)
            p2.font.color.rgb = MUTED

    # SLIDE 8: ACHAL CHIDRAWAR
    s8 = prs.slides.add_slide(blank_layout)
    add_certificate_slide(s8, "Achal Chidrawar", "FinTech Course Certificate", "AWS Cloud Course Certificate", "SLIDE 8: CERTIFICATES")

    # SLIDE 9: CHINMAYEE CHOUDHARI
    s9 = prs.slides.add_slide(blank_layout)
    add_certificate_slide(s9, "Chinmayee Choudhari", "Full Stack Development Course Certificate", "FinTech Course Certificate", "SLIDE 9: CERTIFICATES")

    # SLIDE 10: VIDAL DEO
    s10 = prs.slides.add_slide(blank_layout)
    add_certificate_slide(s10, "Vidal Deo", "DevOps Course Certificate", "FinTech Course Certificate", "SLIDE 10: CERTIFICATES")

    # Save presentation
    prs.save(output_path)
    print(f"Successfully generated PowerPoint presentation at: {output_path}")

if __name__ == "__main__":
    create_presentation()

