# ==========================================================
# Employee Attrition Prediction System
# Streamlit Web Application
# ==========================================================
# This app loads the model, scaler, encoders and feature list
# that were already saved by the project notebooks, and uses
# them to predict whether an employee is likely to leave.
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ==========================================================
# Page setup
# ==========================================================

st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Some custom styling to make the app look professional
# ==========================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

h1 {
    color: #0068C9;
}

.result-card {
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 10px;
}

.stay-card {
    background-color: #E8F8EF;
    border: 2px solid #28A745;
}

.leave-card {
    background-color: #FDECEC;
    border: 2px solid #DC3545;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    padding-top: 30px;
    border-top: 1px solid #E0E0E0;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# File locations
# ==========================================================
# The saved files can either sit next to app.py, or inside a
# Model / Dataset folder. We check both so the app does not
# break because of folder structure.

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))


def find_file(file_name):
    # first check the same folder as app.py
    direct_path = os.path.join(BASE_FOLDER, file_name)
    if os.path.exists(direct_path):
        return direct_path

    # then check inside a Model folder
    model_path = os.path.join(BASE_FOLDER, "Model", file_name)
    if os.path.exists(model_path):
        return model_path

    # then check inside a Dataset folder
    dataset_path = os.path.join(BASE_FOLDER, "Dataset", file_name)
    if os.path.exists(dataset_path):
        return dataset_path

    return None


# ==========================================================
# Load model, scaler, encoders and feature list
# ==========================================================
# IMPORTANT: these files were saved using joblib.dump(), not
# plain pickle. They must be loaded with joblib.load(), or
# loading will fail with a pickle error.

@st.cache_resource
def load_saved_files():

    required_files = ["best_model.pkl", "scaler.pkl", "encoders.pkl", "features.pkl"]
    missing_files = []
    file_paths = {}

    for file_name in required_files:
        path = find_file(file_name)
        if path is None:
            missing_files.append(file_name)
        else:
            file_paths[file_name] = path

    if missing_files:
        return None, None, None, None, missing_files

    model = joblib.load(file_paths["best_model.pkl"])
    scaler = joblib.load(file_paths["scaler.pkl"])
    encoders = joblib.load(file_paths["encoders.pkl"])
    feature_order = joblib.load(file_paths["features.pkl"])

    return model, scaler, encoders, feature_order, []


model, scaler, encoders, feature_order, missing_files = load_saved_files()

# Stop the app early with a clear message if something is missing
if missing_files:
    st.error(
        "The app could not find the following required file(s): "
        + ", ".join(missing_files)
        + ". Please make sure best_model.pkl, scaler.pkl, encoders.pkl and "
          "features.pkl are placed in the same folder as app.py."
    )
    st.stop()

# ==========================================================
# Load the raw dataset (only used for the dataset preview tab)
# ==========================================================

dataset_path = find_file("WA_Fn-UseC_-HR-Employee-Attrition.csv")

if dataset_path is not None:
    raw_data = pd.read_csv(dataset_path)
else:
    raw_data = None

# ==========================================================
# Fixed values for columns that carry no real information
# ==========================================================
# The saved scaler / model expect 34 columns because of how the
# preprocessing notebook was written, but 4 of these columns
# never actually change across employees:
#
#   EmployeeCount   -> always 1 in the original dataset
#   StandardHours   -> always 80 in the original dataset
#   Over18          -> always "Y" in the original dataset
#   EmployeeNumber  -> just a row ID, not a real HR attribute
#
# Rather than asking the user to type meaningless values for
# these, the app fills them in automatically using the same
# constant values seen in the training data.

FIXED_EMPLOYEE_COUNT = 1
FIXED_STANDARD_HOURS = 80
FIXED_OVER18 = "Y"
FIXED_EMPLOYEE_NUMBER = 9999

# Columns that were label encoded during preprocessing
CATEGORICAL_COLUMNS = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "Over18", "OverTime"
]

# Labels shown to the user for ordinal / rating style columns.
# The dataset documentation defines these rating scales.
EDUCATION_LABELS = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}
SATISFACTION_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
WORK_LIFE_LABELS = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}
PERFORMANCE_LABELS = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}
INVOLVEMENT_LABELS = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}

# ==========================================================
# Sidebar
# ==========================================================

with st.sidebar:

    st.title("📊 HR Analytics")
    st.markdown("---")

    st.subheader("Developer Information")
    st.write("Project: Employee Attrition Prediction System")
    st.write("Internship: AIML Summer Internship 2026")
    st.write("Institute: IIHMF, MNNIT Allahabad")

    st.markdown("---")

    st.subheader("Model Information")
    st.write("Algorithm:", type(model).__name__)
    st.write("Total Features Used:", len(feature_order))
    st.write("Scaling Method: StandardScaler")
    st.write("Encoding Method: Label Encoding")

    st.markdown("---")

    if raw_data is not None:
        st.subheader("Dataset Information")
        st.write("Total Records:", raw_data.shape[0])
        st.write("Total Columns:", raw_data.shape[1])
        leave_count = (raw_data["Attrition"] == "Yes").sum()
        stay_count = (raw_data["Attrition"] == "No").sum()
        st.write("Employees who left:", leave_count)
        st.write("Employees who stayed:", stay_count)

    st.markdown("---")
    st.caption("Built with Streamlit")

# ==========================================================
# Main title
# ==========================================================

st.title("Employee Attrition Prediction System")
st.write(
    "This tool predicts whether an employee is likely to leave the "
    "company, using a machine learning model trained on the IBM HR "
    "Analytics Employee Attrition dataset."
)

st.markdown("---")

# ==========================================================
# Tabs for a cleaner layout
# ==========================================================

tab1, tab2 = st.tabs(["🔍 Predict Attrition", "📁 Dataset Preview"])

# ==========================================================
# TAB 1 - Prediction form
# ==========================================================

with tab1:

    st.subheader("Enter Employee Details")

    with st.expander("Personal Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=60, value=30)
            gender = st.selectbox("Gender", encoders["Gender"].classes_)

        with col2:
            marital_status = st.selectbox("Marital Status", encoders["MaritalStatus"].classes_)
            distance_from_home = st.number_input("Distance From Home (km)", min_value=1, max_value=29, value=5)

        with col3:
            education_level = st.selectbox(
                "Education Level",
                options=list(EDUCATION_LABELS.keys()),
                format_func=lambda x: EDUCATION_LABELS[x],
                index=2
            )
            education_field = st.selectbox("Education Field", encoders["EducationField"].classes_)

    with st.expander("Job Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            department = st.selectbox("Department", encoders["Department"].classes_)
            job_role = st.selectbox("Job Role", encoders["JobRole"].classes_)

        with col2:
            job_level = st.number_input("Job Level", min_value=1, max_value=5, value=2)
            business_travel = st.selectbox("Business Travel", encoders["BusinessTravel"].classes_)

        with col3:
            overtime = st.selectbox("OverTime", encoders["OverTime"].classes_)
            num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=9, value=2)

    with st.expander("Compensation Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            monthly_income = st.number_input("Monthly Income", min_value=1009, max_value=19999, value=5000)
            daily_rate = st.number_input("Daily Rate", min_value=102, max_value=1499, value=800)

        with col2:
            monthly_rate = st.number_input("Monthly Rate", min_value=2094, max_value=26999, value=14000)
            hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=65)

        with col3:
            percent_salary_hike = st.number_input("Percent Salary Hike", min_value=11, max_value=25, value=15)
            stock_option_level = st.number_input("Stock Option Level", min_value=0, max_value=3, value=1)

    with st.expander("Satisfaction and Work-Life Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            environment_satisfaction = st.selectbox(
                "Environment Satisfaction",
                options=list(SATISFACTION_LABELS.keys()),
                format_func=lambda x: SATISFACTION_LABELS[x],
                index=2
            )
            job_satisfaction = st.selectbox(
                "Job Satisfaction",
                options=list(SATISFACTION_LABELS.keys()),
                format_func=lambda x: SATISFACTION_LABELS[x],
                index=2
            )

        with col2:
            relationship_satisfaction = st.selectbox(
                "Relationship Satisfaction",
                options=list(SATISFACTION_LABELS.keys()),
                format_func=lambda x: SATISFACTION_LABELS[x],
                index=2
            )
            work_life_balance = st.selectbox(
                "Work Life Balance",
                options=list(WORK_LIFE_LABELS.keys()),
                format_func=lambda x: WORK_LIFE_LABELS[x],
                index=2
            )

        with col3:
            job_involvement = st.selectbox(
                "Job Involvement",
                options=list(INVOLVEMENT_LABELS.keys()),
                format_func=lambda x: INVOLVEMENT_LABELS[x],
                index=2
            )
            performance_rating = st.selectbox(
                "Performance Rating",
                options=list(PERFORMANCE_LABELS.keys()),
                format_func=lambda x: PERFORMANCE_LABELS[x],
                index=0
            )

    with st.expander("Experience and Tenure Details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=8)
            years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)

        with col2:
            years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=18, value=3)
            years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)

        with col3:
            years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=17, value=3)
            training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=2)

    st.markdown("---")

    predict_button = st.button("Predict Attrition", type="primary", use_container_width=True)

    # ==========================================================
    # Run prediction when the button is clicked
    # ==========================================================

    if predict_button:

        try:
            # Create employee dataframe
            # Every key here matches a column the model was trained on.
            # EmployeeCount, EmployeeNumber, Over18 and StandardHours are
            # filled in automatically since they never change per employee.

            employee_data = {
                "Age": age,
                "BusinessTravel": business_travel,
                "DailyRate": daily_rate,
                "Department": department,
                "DistanceFromHome": distance_from_home,
                "Education": education_level,
                "EducationField": education_field,
                "EmployeeCount": FIXED_EMPLOYEE_COUNT,
                "EmployeeNumber": FIXED_EMPLOYEE_NUMBER,
                "EnvironmentSatisfaction": environment_satisfaction,
                "Gender": gender,
                "HourlyRate": hourly_rate,
                "JobInvolvement": job_involvement,
                "JobLevel": job_level,
                "JobRole": job_role,
                "JobSatisfaction": job_satisfaction,
                "MaritalStatus": marital_status,
                "MonthlyIncome": monthly_income,
                "MonthlyRate": monthly_rate,
                "NumCompaniesWorked": num_companies_worked,
                "Over18": FIXED_OVER18,
                "OverTime": overtime,
                "PercentSalaryHike": percent_salary_hike,
                "PerformanceRating": performance_rating,
                "RelationshipSatisfaction": relationship_satisfaction,
                "StandardHours": FIXED_STANDARD_HOURS,
                "StockOptionLevel": stock_option_level,
                "TotalWorkingYears": total_working_years,
                "TrainingTimesLastYear": training_times_last_year,
                "WorkLifeBalance": work_life_balance,
                "YearsAtCompany": years_at_company,
                "YearsInCurrentRole": years_in_current_role,
                "YearsSinceLastPromotion": years_since_last_promotion,
                "YearsWithCurrManager": years_with_curr_manager,
            }

            employee_df = pd.DataFrame([employee_data])

            # Encode categorical columns using the saved encoders
            for column in CATEGORICAL_COLUMNS:
                encoder = encoders[column]
                value = employee_df.loc[0, column]

                if value not in encoder.classes_:
                    st.error(
                        f"The value '{value}' for '{column}' was not seen during "
                        f"training, so it cannot be encoded. Allowed values are: "
                        f"{list(encoder.classes_)}"
                    )
                    st.stop()

                employee_df[column] = encoder.transform(employee_df[column])

            # Arrange columns in the exact order the model expects
            employee_df = employee_df[feature_order]

            # Scale the features
            scaled_employee = scaler.transform(employee_df)

            # Predict
            prediction = model.predict(scaled_employee)[0]
            probability = model.predict_proba(scaled_employee)[0]

            stay_probability = probability[0] * 100
            leave_probability = probability[1] * 100

            st.markdown("---")
            st.subheader("Prediction Result")

            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:
                if prediction == 1:
                    st.markdown(
                        f"""
                        <div class="result-card leave-card">
                        <h2>⚠️ Likely to Leave</h2>
                        <p>This employee shows a higher risk of attrition.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="result-card stay-card">
                        <h2>✅ Likely to Stay</h2>
                        <p>This employee shows a lower risk of attrition.</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with result_col2:
                st.metric("Stay Probability", f"{stay_probability:.2f}%")
                st.metric("Leave Probability", f"{leave_probability:.2f}%")

                confidence_score = max(stay_probability, leave_probability)
                st.write("Model Confidence:")
                st.progress(int(confidence_score))
                st.write(f"{confidence_score:.2f}%")

            st.markdown("### Risk Meter")
            st.progress(int(leave_probability))

            if leave_probability >= 70:
                risk_level = "High Risk"
                recommendation = "Immediate attention recommended. Consider a one-on-one discussion, review compensation and career growth path."
            elif leave_probability >= 40:
                risk_level = "Moderate Risk"
                recommendation = "Keep monitoring engagement levels and check in periodically with this employee."
            else:
                risk_level = "Low Risk"
                recommendation = "Employee shows healthy retention indicators. Continue standard engagement practices."

            st.write("Risk Level:", risk_level)
            st.info(recommendation)

            # Employee summary
            st.markdown("### Employee Summary")
            summary_df = pd.DataFrame({
                "Attribute": list(employee_data.keys()),
                "Value": list(employee_data.values())
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            # Downloadable report
            report_lines = []
            report_lines.append("EMPLOYEE ATTRITION PREDICTION REPORT")
            report_lines.append("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            report_lines.append("")
            report_lines.append(f"Prediction: {'Likely to Leave' if prediction == 1 else 'Likely to Stay'}")
            report_lines.append(f"Stay Probability: {stay_probability:.2f}%")
            report_lines.append(f"Leave Probability: {leave_probability:.2f}%")
            report_lines.append(f"Risk Level: {risk_level}")
            report_lines.append(f"Recommendation: {recommendation}")
            report_lines.append("")
            report_lines.append("Employee Details:")
            for key, value in employee_data.items():
                report_lines.append(f"  {key}: {value}")

            report_text = "\n".join(report_lines)

            st.download_button(
                label="Download Prediction Report",
                data=report_text,
                file_name="attrition_prediction_report.txt",
                mime="text/plain"
            )

        except KeyError as error:
            st.error(f"A required column was missing while preparing the data: {error}")
        except ValueError as error:
            st.error(f"There was a problem with the values entered: {error}")
        except Exception as error:
            st.error(f"Something went wrong while making the prediction: {error}")

# ==========================================================
# TAB 2 - Dataset preview
# ==========================================================

with tab2:

    st.subheader("Dataset Preview")

    if raw_data is not None:
        st.write("Showing the first 20 rows of the original training dataset.")
        st.dataframe(raw_data.head(20), use_container_width=True)

        st.markdown("### Attrition Distribution")
        attrition_counts = raw_data["Attrition"].value_counts()
        st.bar_chart(attrition_counts)
    else:
        st.warning(
            "The original dataset file (WA_Fn-UseC_-HR-Employee-Attrition.csv) "
            "was not found next to app.py, so the preview cannot be shown."
        )

# ==========================================================
# Footer
# ==========================================================

st.markdown(
    """
    <div class="footer">
    Employee Attrition Prediction System | AIML Summer Internship 2026 | IIHMF, MNNIT Allahabad
    </div>
    """,
    unsafe_allow_html=True
)
