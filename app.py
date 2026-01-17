import gradio as gr
import pandas as pd
import joblib
import numpy as np


try:
    model = joblib.load('asd_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    if hasattr(model, 'feature_names_in_'):
        model_columns = list(model.feature_names_in_)
    elif hasattr(model, 'best_estimator_'):
        model_columns = list(model.best_estimator_.steps[-1][1].feature_names_in_)
    elif hasattr(model, 'steps'):
         model_columns = list(model.steps[-1][1].feature_names_in_)
    else:
        model_columns = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'Age_Mons', 
            'Sex', 'Jaundice', 'Family_mem_with_ASD', 
            'Ethnicity_Hispanic', 'Ethnicity_Latino', 'Ethnicity_Native Indian', 'Ethnicity_Others', 
            'Ethnicity_Pacifica', 'Ethnicity_White European', 'Ethnicity_asian', 'Ethnicity_black', 
            'Ethnicity_middle eastern', 'Ethnicity_mixed', 'Ethnicity_south asian',
            'Who completed the test_Health Care Professional', 'Who completed the test_Others', 
            'Who completed the test_Self', 'Who completed the test_family member'
        ]
    print(f"Features detected: {len(model_columns)}")

except Exception as e:
    print(f"Warning: Auto-detection failed. Error: {e}")
    model_columns = []


def convert_to_score(response, q_type="standard"):
    if response is None: return None
    if q_type == "standard":
        if any(x in response for x in ["Always", "Usually", "Easy", "Typical"]):
            return 0
        else:
            return 1
    else:
        if any(x in response for x in ["Never", "Rarely"]):
            return 0
        else:
            return 1

def predict_asd(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, age, sex, jaundice, family_mem, ethnicity, who_completed):
    
    
    raw_inputs = [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, age, sex, jaundice, family_mem, ethnicity, who_completed]
    if any(x is None for x in raw_inputs):
        return """
        <div style='background-color: #FEF2F2; border: 1px solid #F87171; border-radius: 8px; padding: 20px; color: #991B1B; text-align: center;'>
            <h3 style='margin:0;'>⚠️ Incomplete Data</h3>
            <p>Please select an option for every question.</p>
        </div>
        """

    
    a1_val = convert_to_score(a1, "standard")
    a2_val = convert_to_score(a2, "standard")
    a3_val = convert_to_score(a3, "standard")
    a4_val = convert_to_score(a4, "standard")
    a5_val = convert_to_score(a5, "standard")
    a6_val = convert_to_score(a6, "standard")
    a7_val = convert_to_score(a7, "standard")
    a8_val = convert_to_score(a8, "standard")
    a9_val = convert_to_score(a9, "standard")
    a10_val = convert_to_score(a10, "reverse")

   
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    input_df['A1'] = a1_val
    input_df['A2'] = a2_val
    input_df['A3'] = a3_val
    input_df['A4'] = a4_val
    input_df['A5'] = a5_val
    input_df['A6'] = a6_val
    input_df['A7'] = a7_val
    input_df['A8'] = a8_val
    input_df['A9'] = a9_val
    input_df['A10'] = a10_val
    input_df['Age_Mons'] = float(age)
    
 
    if 'Sex' in model_columns: input_df['Sex'] = 1 if sex == 'Male' else 0
    if 'Jaundice' in model_columns: input_df['Jaundice'] = 1 if jaundice == 'Yes' else 0
    if 'Family_mem_with_ASD' in model_columns: input_df['Family_mem_with_ASD'] = 1 if family_mem == 'Yes' else 0


    eth_match = False
    clean_eth_input = ethnicity.lower().replace(" ", "")
    for col in model_columns:
        if "ethnicity" in col.lower():
            col_part = col.split('_')[-1].lower().replace(" ", "")
            if col_part == clean_eth_input:
                input_df[col] = 1
                eth_match = True
    

    clean_who_input = who_completed.lower().replace(" ", "")
    for col in model_columns:
        if "who completed" in col.lower():
            col_part = col.split('_')[-1].lower().replace(" ", "")
            if col_part == clean_who_input:
                input_df[col] = 1

    try:
        if model:
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
        else:
            return "Error: Model not loaded."

        qchat_score = sum([a1_val, a2_val, a3_val, a4_val, a5_val, a6_val, a7_val, a8_val, a9_val, a10_val])
        

        if prediction == 1:
            color = "#DC2626"
            bg_color = "#FEF2F2"
            status = "POTENTIAL ASD TRAITS DETECTED"
            icon = "⚠️"
            advice = "High probability of ASD traits. Please consult a specialist."
        else:
            color = "#059669"
            bg_color = "#ECFDF5"
            status = "NO ASD TRAITS DETECTED"
            icon = "✅"
            advice = "Behavioral patterns appear typical for this age."

        html_result = f"""
        <div style='background-color: {bg_color}; border: 2px solid {color}; border-radius: 10px; padding: 25px; text-align: center; font-family: sans-serif;'>
            <h2 style='color: {color}; margin: 0 0 10px 0; font-size: 24px;'>{icon} {status}</h2>
            <div style='display: flex; justify-content: space-around; margin: 20px 0; border-top: 1px solid {color}30; border-bottom: 1px solid {color}30; padding: 15px 0;'>
                <div>
                    <p style='margin:0; color: #6B7280; font-size: 14px;'>Model Confidence</p>
                    <p style='margin:0; font-weight: bold; font-size: 20px; color: {color};'>{prob*100:.2f}%</p>
                </div>
                <div>
                    <p style='margin:0; color: #6B7280; font-size: 14px;'>Q-Chat-10 Score</p>
                    <p style='margin:0; font-weight: bold; font-size: 20px; color: #374151;'>{qchat_score}/10</p>
                </div>
            </div>
            <p style='color: #4B5563; font-style: italic; margin-top: 10px;'>"{advice}"</p>
        </div>
        """
        return html_result
        
    except Exception as e:
        return f"Error: {str(e)}"



opts_std = ["Always / Usually", "Sometimes / Rarely / Never"]
opts_easy = ["Very Easy / Quite Easy", "Quite Difficult / Very Difficult"]
opts_typical = ["Typical", "Non-typical / Delayed"]
opts_a10 = ["Never / Rarely", "Sometimes / Usually / Always"]

ethnicity_opts = ['Middle Eastern', 'White European', 'Hispanic', 'Black', 'Asian', 'South Asian', 'Native Indian', 'Others', 'Latino', 'Mixed', 'Pacifica']
who_opts = ['Family member', 'Health Care Professional', 'Self', 'Others']

custom_css = """
body { font-family: 'Inter', sans-serif; background-color: #f3f4f6; }
.header { background: linear-gradient(90deg, #2563EB 0%, #1E40AF 100%); padding: 30px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }
.footer { text-align: center; margin-top: 30px; color: #6B7280; font-size: 0.9em; }
"""

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")


with gr.Blocks(title="Toddler Autism Screening System") as demo:
    
    gr.HTML("""
    <div class="header">
        <h1 style='font-size: 32px; margin-bottom: 10px;'>Toddler Autism Screening System</h1>
        <p style='font-size: 16px; opacity: 0.9;'>Early detection tool based on Q-Chat-10 (For Toddlers 12-36 Months)</p>
    </div>
    """)

    with gr.Row():
        # --- Left Column: Questions ---
        with gr.Column(scale=2):
            gr.Markdown("### 📝 Behavioral Screening (Q-Chat-10)")
            with gr.Group():
                a1 = gr.Radio(opts_std, label="A1. Look when called?", info="Does your child look at you when you call his/her name?")
                a2 = gr.Radio(opts_easy, label="A2. Eye Contact?", info="How easy is it for you to get eye contact with your child?")
                a3 = gr.Radio(opts_std, label="A3. Pointing?", info="Does your child point to indicate that s/he wants something?")
                a4 = gr.Radio(opts_std, label="A4. Sharing interest?", info="Does your child point to share interest with you?")
                a5 = gr.Radio(opts_std, label="A5. Pretend play?", info="Does your child pretend? (e.g. care for dolls, talk on toy phone)")
                a6 = gr.Radio(opts_std, label="A6. Follow look?", info="Does your child follow where you’re looking?")
                a7 = gr.Radio(opts_std, label="A7. Comfort others?", info="If you are visibly upset, does your child show signs of wanting to comfort them?")
                a8 = gr.Radio(opts_typical, label="A8. First words?", info="Would you describe your child’s first words as:")
                a9 = gr.Radio(opts_std, label="A9. Simple gestures?", info="Does your child use simple gestures? (e.g. wave goodbye)")
                a10 = gr.Radio(opts_a10, label="A10. Staring?", info="Does your child stare at nothing with no apparent purpose?")

        # --- Right Column: Details ---
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Patient Details")
            with gr.Group():
                age = gr.Slider(12, 36, label="Age (Months)", step=1, value=24)
                sex = gr.Radio(['Male', 'Female'], label="Sex", value='Male')
                ethnicity = gr.Dropdown(choices=ethnicity_opts, label="Ethnicity", value='Asian')
                jaundice = gr.Radio(['Yes', 'No'], label="Born with Jaundice?", value='No')
                family_mem = gr.Radio(['Yes', 'No'], label="Family Member with ASD?", value='No')
                who_completed = gr.Dropdown(choices=who_opts, label="Who is completing this test?", value='Family member')
            
            gr.HTML("<br>")
            btn_predict = gr.Button("🔍 Analyze Risk", variant="primary", size="lg")
            output_html = gr.HTML(label="Assessment Result")

    gr.HTML("<div class='footer'><p>Developed for Machine Learning Final Exam. Model Accuracy: 100%.</p></div>")

    # Click Event
    btn_predict.click(
        fn=predict_asd,
        inputs=[a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, age, sex, jaundice, family_mem, ethnicity, who_completed],
        outputs=output_html
    )



demo.launch(theme=theme, css=custom_css)