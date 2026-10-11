import gradio as gr
import joblib
import pandas as pd

model = joblib.load("model.joblib")


def predict_machine(air_temperature, process_temperature, rotational_speed,
                    torque, tool_wear, machine_type):
    row = pd.DataFrame([{
        "Air temperature": air_temperature,
        "Process temperature": process_temperature,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type": machine_type,
    }])
    prob = float(model.predict_proba(row)[0][1])
    if prob >= 0.5:
        status = "🔴 MACHINE FAILURE RISK"
        message = "⚠️ Maintenance is recommended."
    else:
        status = "🟢 MACHINE OK"
        message = "✅ The machine is operating normally."
    return f"{status}\n\nFailure Probability : {prob * 100:.2f}%\n\n{message}"


with gr.Blocks() as demo:
    gr.Markdown("# ⚙️ Machine Failure Prediction\n### Predictive Maintenance System")
    gr.Markdown("Enter the machine parameters to predict whether the machine is likely to fail.")

    gr.Markdown("## 🌡️ Temperature")
    with gr.Row():
        air = gr.Number(label="Air Temperature (°C)", value=26.95)
        proc = gr.Number(label="Process Temperature (°C)", value=36.95)

    gr.Markdown("## ⚙️ Machine Performance")
    with gr.Row():
        rpm = gr.Number(label="Rotational Speed (RPM)", value=1503)
        torque = gr.Number(label="Torque (Nm)", value=40)
    with gr.Row():
        wear = gr.Slider(0, 253, step=1, value=108, label="Tool Wear (min)")
        mtype = gr.Dropdown(["L", "M", "H"], value="M", label="Machine Type")

    btn = gr.Button("🔍 Predict Machine Status", variant="primary")
    gr.Markdown("## 📊 Prediction Result")
    out = gr.Textbox(label="Machine Status", lines=5, interactive=False)
    btn.click(predict_machine, [air, proc, rpm, torque, wear, mtype], out)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), favicon_path="icon.ico")
