import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import google.generativeai as genai
from gtts import gTTS
from io import BytesIO

# 1) PAGE CONFIGURATION & CUSTOM CSS

st.set_page_config(
    page_title="CreditIQ | AI Risk Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #0d1b2a; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; }
    div[data-testid="stMetricValue"] { font-size: 28px; color: #415a77; }
    [data-testid="stSidebar"] { background-color: #0d1b2a; color: white; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: white; }
    .chat-container { border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; background: white; }
</style>
""", unsafe_allow_html=True)

# 2) Api Key Config

gemini_key = "Please Insert Your API Key Here"

# 3) LOAD RESOURCES

@st.cache_resource
def load_data_and_models():
    # Load Models
    clf_model = joblib.load('best_class_model.pkl')
    reg_model = joblib.load('best_reg_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    class_metrics = joblib.load('class_metrics.pkl')
    reg_metrics = joblib.load('reg_metrics.pkl')
    return clf_model, reg_model, scaler, model_columns, class_metrics, reg_metrics

try:
    clf, reg, scaler, model_cols, c_metrics, r_metrics = load_data_and_models()
except FileNotFoundError:
    st.error("⚠️ Critical Error: Model files not found. Please run your Jupyter Notebook (Step 1) to generate the .pkl files first.")
    st.stop()

# 4) SIDEBAR :- INPUTS & SETTINGS

st.sidebar.image("https://img.icons8.com/color/96/000000/safe-ok.png", width=80)
st.sidebar.title("CreditIQ")
st.sidebar.markdown("AI-Powered Underwriting")

tab_input, tab_settings = st.sidebar.tabs(["📝 Application", "⚙️ Settings"])

with tab_input:
    loan_amnt = st.number_input("Loan Amount ($)", 500, 50000, 15000, step=500)
    term_val = st.radio("Term Duration", [36, 60], horizontal=True)
    dti = st.slider("DTI Ratio (%)", 0.0, 100.0, 15.0)
    installment = st.number_input("Monthly Installment ($)", 50.0, 2000.0, 400.0)
    grade = st.select_slider("Credit Grade", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    
    purpose = st.selectbox("Loan Purpose", [
        'credit_card', 'debt_consolidation', 'home_improvement', 'major_purchase', 
        'small_business', 'car', 'medical', 'moving', 'vacation', 'wedding', 'other'
    ])
    
    funded_amnt = loan_amnt 
    funded_amnt_inv = loan_amnt * 0.95 

with tab_settings:
    st.header("⚙️ Configuration")
    if gemini_key != "PASTE_YOUR_GEMINI_API_KEY_HERE":
        st.success("✅ Real AI Connected")
    else:
        st.warning("⚠️ API Key Missing")
        
    narration_mode = st.toggle("Enable Narration Mode")
    show_shap = st.checkbox("Show Explainer Charts", value=True)

# 5) Processing Logic

def preprocess_input(l_amt, term, dti_val, install, gr, purp, f_amt, f_inv):
    grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
    
    data = {
        'loan_amnt': l_amt, 'funded_amnt': f_amt, 'funded_amnt_inv': f_inv,
        'term': term, 'installment': install, 'dti': dti_val,
        'grade_mapped': grade_map[gr]
    }
    
    df_in = pd.DataFrame([data])
    
    for col in model_cols:
        if col.startswith('purpose_'):
            df_in[col] = 0
            
    chosen_purpose = f"purpose_{purp}"
    if chosen_purpose in df_in.columns:
        df_in[chosen_purpose] = 1
        
    df_in = df_in[model_cols] 
    return df_in, scaler.transform(df_in)

input_df, input_scaled = preprocess_input(loan_amnt, term_val, dti, installment, grade, purpose, funded_amnt, funded_amnt_inv)

# Predictions
prob_default = clf.predict_proba(input_scaled)[0][1]
pred_default = 1 if prob_default > 0.5 else 0
pred_int_rate = reg.predict(input_scaled)[0]


# 6) MAIN DASHBOARD UI

st.title("📊 Credit Risk Assessment")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Risk Analysis")
    if pred_default == 0:
        st.markdown(f"""
        <div class="chat-container" style="border-left: 5px solid #28a745;">
            <h2 style="color: #28a745;">✅ Approved / Low Risk</h2>
            <p>Probability of Default: <b>{prob_default:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-container" style="border-left: 5px solid #dc3545;">
            <h2 style="color: #dc3545;">⚠️ High Risk / Default</h2>
            <p>Probability of Default: <b>{prob_default:.1%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Pricing Engine")
    st.metric("Recommended Interest Rate", f"{pred_int_rate:.2f}%")
    if pred_int_rate < 10: st.caption("🌟 Prime Rate Eligible")
    elif pred_int_rate < 15: st.caption("🔹 Standard Market Rate")
    else: st.caption("🔥 Subprime Rate Warning")

# 7) EXPLAINABILITY & NARRATION (WITH AUDIO)

with col2:
    st.subheader("🔍 Decision Intelligence")
    
    narrative_text = f"The borrower requests ${loan_amnt:,.0f} for {purpose.replace('_', ' ')}. " \
                     f"Based on a credit grade of {grade} and DTI of {dti}%, " \
                     f"the model predicts a {prob_default:.1%} chance of default. " \
                     f"{'The loan is approved.' if pred_default == 0 else 'The loan is flagged as high risk.'}"
    
    # Narration Logic with Audio
    if narration_mode:
        st.info("🎙️ **Generating Audio Analysis...**")
        try:
            tts = gTTS(text=narrative_text, lang='en', slow=False)
            sound_file = BytesIO()
            tts.write_to_fp(sound_file)
            st.audio(sound_file)
        except Exception as e:
            st.error(f"Audio Error: {e}")
    
    with st.expander("Why this decision?", expanded=False):
        st.write(narrative_text)
        
        if show_shap:
            st.write("**Feature Impact:**")
            coeffs = clf.coef_[0]
            local_impact = coeffs * input_scaled[0]
            impact_df = pd.DataFrame({'Feature': model_cols, 'Impact': local_impact})
            impact_df = impact_df.reindex(impact_df.Impact.abs().sort_values(ascending=False).index).head(8)
            
            fig, ax = plt.subplots(figsize=(5, 3))
            colors = ['red' if x > 0 else 'green' for x in impact_df['Impact']]
            ax.barh(impact_df['Feature'], impact_df['Impact'], color=colors)
            ax.set_title("Risk Drivers (Red=Riskier)")
            st.pyplot(fig)


st.markdown("---")
if pred_default == 1:
    st.subheader("🔧 AI Optimizer")
    optimized_amt = loan_amnt
    is_safe = False
    steps = 0
    while not is_safe and steps < 20:
        optimized_amt *= 0.90 
        _, opt_scaled = preprocess_input(optimized_amt, term_val, dti, installment, grade, purpose, optimized_amt, optimized_amt*0.95)
        opt_prob = clf.predict_proba(opt_scaled)[0][1]
        if opt_prob < 0.45: is_safe = True
        steps += 1
            
    if is_safe:
        st.success(f"Suggestion: Reduce loan to **${optimized_amt:,.0f}** to lower risk to **{opt_prob:.1%}**.")
    else:
        st.warning("Risk remains high even with lower amounts.")


# 9) Ai ChatBot

st.markdown("---")
st.subheader("🤖 CreditIQ Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about this loan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response_text = "Thinking..."
    
    # AI CONNECTION
    if gemini_key and gemini_key != "PASTE_YOUR_GEMINI_API_KEY_HERE":
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            context = f"""
            Act as a Senior Fintech Risk Advisor.
            Loan Amount: ${loan_amnt} | Grade: {grade}
            Default Prob: {prob_default:.1%} | Rate: {pred_int_rate:.2f}%
            User Question: {prompt}
            Keep answer short and professional.
            """
            
            response = model.generate_content(context)
            response_text = response.text
        except Exception as e:
            response_text = f"AI Error: {e}. Check your API Key."
    else:
        response_text = "⚠️ Please paste your API Key in app.py Line 30."

    with st.chat_message("assistant"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})


# 10) Key Metrices

with st.expander("📉 Model Performance"):
    t1, t2 = st.tabs(["Classification", "Regression"])
    with t1:
        st.metric("Accuracy", f"{c_metrics['accuracy']:.2%}")
        st.metric("AUC", f"{c_metrics['auc']:.2f}")
    with t2:
        st.metric("MAE", f"{r_metrics['mae']:.4f}")

        st.metric("R2", f"{r_metrics['r2']:.4f}")

