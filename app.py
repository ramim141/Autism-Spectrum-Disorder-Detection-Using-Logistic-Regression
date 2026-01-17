import gradio as gr
import pandas as pd
import joblib
import numpy as np

# ---------------------------------------------------------
# 1. মডেল লোড এবং কলাম ডিটেকশন
# ---------------------------------------------------------
try:
    model = joblib.load('asd_model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# মডেল থেকে সঠিক ফিচার কলামগুলো বের করা
try:
    if hasattr(model, 'feature_names_in_'):
        model_columns = list(model.feature_names_in_)
    elif hasattr(model, 'best_estimator_'):
        model_columns = list(model.best_estimator_.steps[-1][1].feature_names_in_)
    elif hasattr(model, 'steps'):
         model_columns = list(model.steps[-1][1].feature_names_in_)
    else:
        # Fallback (তবে এটি অটোমেটিক ডিটেক্ট হওয়াই ভালো)
        print("⚠️ Warning: Using fallback columns list. This might cause casing issues.")
        model_columns = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 
            'Sex', 'Jaundice', 'Family_mem_with_ASD', 
            'Ethnicity_Latino', 'Ethnicity_Native Indian', 'Ethnicity_Others', 
            'Ethnicity_Pacifica', 'Ethnicity_White European', 'Ethnicity_asian', 'Ethnicity_black', 
            'Ethnicity_middle eastern', 'Ethnicity_mixed', 'Ethnicity_south asian',
            'Who completed the test_Others', 'Who completed the test_Self', 
            'Who completed the test_family member', 'Who completed the test_Health care professional'
        ]
        
    print(f"✅ Features detected: {len(model_columns)}")
    # print(model_columns) # ডিবাগিংয়ের জন্য

except Exception as e:
    print(f"Error detecting features: {e}")
    model_columns = []

# ---------------------------------------------------------
# 2. কনভার্সন ফাংশন
# ---------------------------------------------------------
def convert_to_score(response, q_type="standard"):
    if response is None: return None
    if q_type == "standard":
        return 0 if any(x in response for x in ["Always", "Usually", "Easy", "Typical"]) else 1
    else: # A10 Reverse
        return 0 if any(x in response for x in ["Never", "Rarely"]) else 1

# ---------------------------------------------------------
# 3. প্রেডিকশন ফাংশন (Smart Matching)
# ---------------------------------------------------------
def predict_asd(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, age, sex, jaundice, family_mem, ethnicity, who_completed):
    
    # 1. Validation
    inputs = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, age, sex, jaundice, family_mem, ethnicity, who_completed]
    if any(x is None for x in inputs):
        return "<h3 style='color:red'>⚠️ Please answer ALL questions.</h3>"

    # 2. Base DataFrame with 0s (শুধুমাত্র মডেল যা চিনে সেই কলামগুলোই থাকবে)
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # 3. Numerical & Binary Value Mapping
    input_df['A1'] = convert_to_score(a1)
    input_df['A2'] = convert_to_score(a2)
    input_df['A3'] = convert_to_score(a3)
    input_df['A4'] = convert_to_score(a4)
    input_df['A5'] = convert_to_score(a5)
    input_df['A6'] = convert_to_score(a6)
    input_df['A7'] = convert_to_score(a7)
    input_df['A8'] = convert_to_score(a8)
    input_df['A9'] = convert_to_score(a9)
    input_df['A10'] = convert_to_score(a10, "reverse")
    
    # কলামের নাম যদি মডেলে থাকে তবেই ভ্যালু বসাবে
    if 'Age_Mons' in model_columns: input_df['Age_Mons'] = float(age)
    if 'Sex' in model_columns: input_df['Sex'] = 1 if sex == 'Male' else 0
    if 'Jaundice' in model_columns: input_df['Jaundice'] = 1 if jaundice == 'Yes' else 0
    if 'Family_mem_with_ASD' in model_columns: input_df['Family_mem_with_ASD'] = 1 if family_mem == 'Yes' else 0
    
    # 4. One-Hot Encoding (FUZZY MATCHING)
    # আমরা মডেলের কলামগুলোর উপর লুপ চালাব এবং দেখব ইউজারের ইনপুট সেখানে আছে কিনা
    
    # Ethnicity Matching
    user_ethnicity = ethnicity.lower().replace(" ", "") # e.g. "South Asian" -> "southasian"
    for col in model_columns:
        if "ethnicity" in col.lower():
            # মডেলের কলাম: Ethnicity_south asian -> southasian
            col_part = col.split('_')[-1].lower().replace(" ", "")
            if col_part == user_ethnicity:
                input_df[col] = 1 # Match found!
                
    # Who Completed Matching
    user_who = who_completed.lower().replace(" ", "") # e.g. "Health Care Professional" -> "healthcareprofessional"
    for col in model_columns:
        if "who completed" in col.lower():
            # মডেলের কলাম: ..._Health care professional -> healthcareprofessional
            col_part = col.split('_')[-1].lower().replace(" ", "")
            if col_part == user_who:
                input_df[col] = 1 # Match found!

    # 5. Prediction
    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        
        # Calculate Score
        scores = [input_df[f'A{i}'][0] for i in range(1, 11)]
        qchat_score = sum(scores)
        
        # UI Styling
        if prediction == 1:
            color = "#DC2626"
            bg_color = "#FEF2F2"
            status = "POTENTIAL ASD TRAITS DETECTED"
            icon = "⚠️"
            advice = "Please consult a healthcare professional."
        else:
            color = "#059669"
            bg_color = "#ECFDF5"
            status = "NO ASD TRAITS DETECTED"
            icon = "✅"
            advice = "Development appears typical."

        return f"""
        <div style='background-color: {bg_color}; border: 2px solid {color}; border-radius: 10px; padding: 20px; text-align: center;'>
            <h2 style='color: {color}; margin:0;'>{icon} {status}</h2>
            <hr style='border-color: {color}40; margin: 15px 0;'>
            <div style='display:flex; justify-content:space-around;'>
                <div><b>Confidence:</b> {prob*100:.2f}%</div>
                <div><b>Q-Chat Score:</b> {qchat_score}/10</div>
            </div>
            <p style='margin-top:10px; font-style:italic;'>"{advice}"</p>
        </div>
        """
        
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# ---------------------------------------------------------
# 4. UI Layout
# ---------------------------------------------------------
opts_std = ["Always / Usually", "Sometimes / Rarely / Never"]
opts_easy = ["Very Easy / Quite Easy", "Quite Difficult / Very Difficult"]
opts_typical = ["Typical", "Non-typical / Delayed"]
opts_a10 = ["Never / Rarely", "Sometimes / Usually / Always"]

ethnicity_opts = ['Middle Eastern', 'White European', 'Hispanic', 'Black', 'Asian', 'South Asian', 'Native Indian', 'Others', 'Latino', 'Mixed', 'Pacifica']
who_opts = ['Family member', 'Health Care Professional', 'Self', 'Others']

custom_css = """
body { font-family: 'Inter', sans-serif; background-color: #f9fafb; }
.header { background: #2563EB; padding: 20px; border-radius: 10px; color: white; text-align: center; margin-bottom: 20px; }
"""

theme = gr.themes.Soft(primary_hue="blue")

with gr.Blocks(theme=theme, css=custom_css, title="ASD Screening") as demo:
    
    gr.HTML("""<div class="header"><h1>Toddler Autism Screening System</h1></div>""")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Q-Chat-10 Questions")
            a1 = gr.Radio(opts_std, label="A1. Look when called?", info="Does your child look at you when you call his/her name?")
            a2 = gr.Radio(opts_easy, label="A2. Eye Contact?", info="How easy is it for you to get eye contact with your child?")
            a3 = gr.Radio(opts_std, label="A3. Pointing?", info="Does your child point to indicate that s/he wants something?")
            a4 = gr.Radio(opts_std, label="A4. Sharing interest?", info="Does your child point to share interest with you?")
            a5 = gr.Radio(opts_std, label="A5. Pretend play?", info="Does your child pretend? (e.g. care for dolls)")
            a6 = gr.Radio(opts_std, label="A6. Follow look?", info="Does your child follow where you’re looking?")
            a7 = gr.Radio(opts_std, label="A7. Comfort others?", info="If you are visibly upset, does your child show signs of wanting to comfort them?")
            a8 = gr.Radio(opts_typical, label="A8. First words?", info="Would you describe your child’s first words as:")
            a9 = gr.Radio(opts_std, label="A9. Simple gestures?", info="Does your child use simple gestures?")
            a10 = gr.Radio(opts_a10, label="A10. Staring?", info="Does your child stare at nothing with no apparent purpose?")

        with gr.Column(scale=1):
            gr.Markdown("### 👤 Details")
            age = gr.Slider(12, 36, label="Age (Months)", step=1, value=24)
            sex = gr.Radio(['Male', 'Female'], label="Sex", value='Male')
            ethnicity = gr.Dropdown(choices=ethnicity_opts, label="Ethnicity", value='Asian')
            jaundice = gr.Radio(['Yes', 'No'], label="Born with Jaundice?", value='No')
            family_mem = gr.Radio(['Yes', 'No'], label="Family Member with ASD?", value='No')
            who_completed = gr.Dropdown(choices=who_opts, label="Who is completing this test?", value='Family member')
            
            gr.HTML("<br>")
            btn_predict = gr.Button("Analyze Risk", variant="primary")
            output_html = gr.HTML(label="Result")

    btn_predict.click(
        fn=predict_asd,
        inputs=[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, age, sex, jaundice, family_mem, ethnicity, who_completed],
        outputs=output_html
    )

if __name__ == "__main__":
    demo.launch()