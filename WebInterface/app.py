import streamlit as st
import requests
import time
import torch
from DoubleLayer import DoubleLayer
from DoubleRegression import DoubleLayerRegression
from streamlit_lottie import st_lottie

# Function to load Lottie animations
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
lottie_approved = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json")
lottie_rejected = load_lottie_url("https://assets2.lottiefiles.com/private_files/lf30_m6j5igxb.json")
lottie_header = load_lottie_url("https://assets10.lottiefiles.com/packages/lf20_hzgq1iov.json")

# Page config
st.set_page_config(
    page_title="Loan Wizard ✨",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for ✨slay✨ styling
st.markdown("""
    <style>
        body {
            background-color: #f2f6ff;
        }
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #2e77d0;
        }
        .subtitle {
            font-size: 1.5em;
            color: #4b4b4b;
        }
        .stButton>button {
            border-radius: 10px;
            background-color: #2e77d0;
            color: white;
            font-weight: bold;
            padding: 10px 24px;
        }
    </style>
""", unsafe_allow_html=True)

# Header Layout
header_col1, header_col2 = st.columns([1, 3])
with header_col1:
    st.image("bank_logo.png", width=100)
with header_col2:
    st.markdown("<div class='title'>T.T. Bank</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Loan Prediction — but make it chic.</div>", unsafe_allow_html=True)

# Animated banner
st_lottie(lottie_header, height=200, key="header_anim")

# Tabs for prediction type
st.markdown("### 🔮 What would you like to predict today?")
option = st.radio("Select Prediction Type", ["Loan Approval", "Risk Score"], horizontal=True)

# Stylish input section
st.markdown("#### ✍️ Enter your details below:")

with st.expander("📋 Fill Out Your Loan Info", expanded=True):
    input_data = {
        'Age': st.number_input('🎂 Age', min_value=18, max_value=100, value=30),
        'AnnualIncome': st.number_input('💼 Annual Income', value=50000),
        'CreditScore': st.number_input('💳 Credit Score', value=700),
        'EmploymentStatus': st.selectbox('👩‍💻 Employment Status', [1, 0], format_func=lambda x: "Employed" if x == 1 else "Unemployed"),
        'EducationLevel': st.slider('🎓 Education Level (0 = None, 5 = PhD)', 0, 5, 2),
        'Experience': st.number_input('🧠 Work Experience (Years)', value=5),
        'LoanAmount': st.number_input('💰 Loan Amount', value=10000),
        'LoanDuration': st.number_input('⏳ Loan Duration (Months)', value=12),
        'MaritalStatus': st.selectbox('💍 Marital Status', [1, 0], format_func=lambda x: "Married" if x == 1 else "Single"),
        'NumberOfDependents': st.number_input('👨‍👩‍👧‍👦 Number of Dependents', value=2),
        'HomeOwnershipStatus': st.selectbox('🏡 Home Ownership', [1, 0], format_func=lambda x: "Owned" if x == 1 else "Rented"),
        'MonthlyDebtPayments': st.number_input('📉 Monthly Debt Payments', value=500),
        'CreditCardUtilizationRate': st.slider('💳 Credit Card Utilization Rate', 0.0, 1.0, 0.3),
        'NumberOfOpenCreditLines': st.number_input('🔓 Open Credit Lines', value=5),
        'NumberOfCreditInquiries': st.number_input('🔎 Credit Inquiries', value=2),
        'DebtToIncomeRatio': st.slider('💸 Debt-to-Income Ratio', 0.0, 1.0, 0.4),
        'BankruptcyHistory': st.selectbox('💥 Bankruptcy History', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
        'LoanPurpose': st.selectbox('🎯 Loan Purpose', [1, 2, 3], format_func=lambda x: ["Home", "Car", "Education"][x-1]),
        'PreviousLoanDefaults': st.selectbox('❌ Previous Defaults', [0, 1], format_func=lambda x: "No" if x == 0 else "Yes"),
        'PaymentHistory': st.slider('📈 Payment History', 0.0, 1.0, 0.8),
        'LengthOfCreditHistory': st.number_input('📆 Credit History (Years)', value=5),
        'SavingsAccountBalance': st.number_input('🏦 Savings Balance', value=5000),
        'CheckingAccountBalance': st.number_input('💳 Checking Balance', value=2000),
        'TotalAssets': st.number_input('📊 Total Assets', value=50000),
        'TotalLiabilities': st.number_input('📉 Total Liabilities', value=20000),
        'MonthlyIncome': st.number_input('📥 Monthly Income', value=3000),
        'UtilityBillsPaymentHistory': st.slider('🔌 Utility Bills Payment History', 0.0, 1.0, 0.9),
        'JobTenure': st.number_input('🕒 Job Tenure (Years)', value=3),
        'NetWorth': st.number_input('💎 Net Worth', value=80000),
        'BaseInterestRate': st.number_input('📈 Base Interest Rate', value=0.05),
        'InterestRate': st.number_input('📉 Interest Rate on Loan', value=0.04),
        'MonthlyLoanPayment': st.number_input('💸 Monthly Loan Payment', value=200),
        'TotalDebtToIncomeRatio': st.slider('📊 Total DTI Ratio', 0.0, 1.0, 0.3)
    }

input_tensor = torch.tensor([list(input_data.values())], dtype=torch.float32)

# Load models
loan_model = DoubleLayer(input_size=33, hidden1=64, hidden2=64)
loan_model.load_state_dict(torch.load("best_model.pt"))
loan_model.eval()

risk_model = DoubleLayerRegression(hidden1=64, hidden2=64)
risk_model.load_state_dict(torch.load("best_model_regression.pt"))
risk_model.eval()

# Loan Approval Prediction
# Loan Approval Prediction
if option == "Loan Approval":
    st.markdown("## 🚀 Loan Approval Result")
    if st.button("🔮 Predict Loan Approval"):
        with torch.no_grad():
            prediction = torch.sigmoid(loan_model(input_tensor)).item()
            if prediction >= 0.5:
                animation_spot = st.empty()
                with animation_spot:
                    st_lottie(lottie_approved, height=250)
                    time.sleep(1.5)
                    animation_spot.empty()
                st.success("✨ Congratulations! Your loan is approved.")
            else:
                st_lottie(lottie_rejected, height=250)
                st.error("😢 Sorry, your loan was not approved.")


# Risk Score Prediction
if option == "Risk Score":
    st.markdown("## 📉 Risk Score Result")
    if st.button("📊 Predict Risk Score"):
        with torch.no_grad():
            risk_score = torch.sigmoid(risk_model(input_tensor)).item() * 100
            st.info(f"💡 Your predicted **Risk Score** is: **{risk_score:.2f}**")

            if risk_score < 40:
                st.success("🟢 Low Risk — You're in the safe zone.")
            elif risk_score < 70:
                st.warning("🟠 Moderate Risk — Approval depends on other factors.")
            else:
                st.error("🔴 High Risk — Chances of approval are low.")
                st_lottie(lottie_rejected, height=250)
