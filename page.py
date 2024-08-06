import streamlit as st
import pandas as pd
import editdistance
import difflib
import re
import os

def load_data(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    dfs = [pd.read_csv(os.path.join(folder, file)) for file in files]
    combined_df = dfs[0][['audio_path', 'ground_truth']].copy()
    
    for i, df in enumerate(dfs):
        model_name = os.path.splitext(files[i])[0]
        df = df.rename(columns={'transcription': f'transcription_{model_name}', 'inference_time': f'inference_time_{model_name}'})
        combined_df = combined_df.merge(df[['audio_path', f'transcription_{model_name}', f'inference_time_{model_name}']], on='audio_path', how='left')
    
    return combined_df, [os.path.splitext(file)[0] for file in files]

def preprocess_text(text):
    text = text.replace('-', ' ').lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def calculate_wer(reference, hypothesis):
    reference = re.sub(r'[^\w\s]', '', reference.lower().strip())
    hypothesis = re.sub(r'[^\w\s]', '', hypothesis.lower().strip())
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    distance = editdistance.eval(ref_words, hyp_words)
    return distance / len(ref_words) if len(ref_words) > 0 else float('inf')

def style_text(original, transcription, wer, min_wer):
    original_words = preprocess_text(original)
    transcription_words = preprocess_text(transcription)
    styled_text = []
    s = difflib.SequenceMatcher(None, original_words, transcription_words)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'equal':
            styled_text.append(f"<span style='color: #32CD32;'>{' '.join(transcription_words[j1:j2])}</span>")
        else:
            styled_text.append(f"<span style='color: red;'>{' '.join(transcription_words[j1:j2])}</span>")
    wer_color = '#32CD32' if wer == min_wer else '#FF0000'
    return ' '.join(styled_text) + f"<br><b style='color: {wer_color};'>(WER: {wer:.2f})</b>"

def style_inference_time(time, min_time):
    color = '#32CD32' if time == min_time else '#FF0000'
    return f"<span style='color: {color};'>{time:.4f}</span>"

def main():
    st.title("Transcription Comparison Tool")

    data, model_names = load_data("Transcription Results")

    st.sidebar.title("Options")
    show_models = st.sidebar.multiselect("Select Models to Show", model_names, default=model_names)
    show_summary = st.sidebar.checkbox("Show Average Inference Time and WER")

    if not show_models:
        st.error("Please select at least one model to show.")
        return

    st.write(f"Total Transcriptions: {len(data)}")

    transcription_column_width = 60 / (len(show_models) * 2) if show_models else 60
    inference_time_column_width = transcription_column_width

    table_html = f"""
    <style>
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-left: -320px;
        }}
        th, td {{
            border: 1px solid black;
            padding: 10px;
            text-align: center;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
    </style>
    <table>
    <tr>
        <th style='width: 10%;'>Audio</th>
        <th style='width: 20%;'>Ground Truth</th>""" + "".join([f"<th style='width: {transcription_column_width}%;'>{model} Transcription</th><th style='width: {inference_time_column_width}%;'>{model} Inference Time</th>" for model in show_models]) + "</tr>"

    for i, row in data.iterrows():
        audio_path = os.path.join("Data Collection", row["audio_path"])
        table_html += "<tr>"
        table_html += f"<td><audio controls><source src='{audio_path}' type='audio/mpeg'></audio></td>"
        table_html += f"<td>{row['ground_truth']}</td>"

        wer_values = {model: calculate_wer(row['ground_truth'], row[f'transcription_{model}']) for model in show_models}
        min_wer = min(wer_values.values())

        inference_times = {model: row[f'inference_time_{model}'] for model in show_models}
        min_inference_time = min(inference_times.values())

        for model in show_models:
            styled_transcription = style_text(row['ground_truth'], row[f'transcription_{model}'], wer_values[model], min_wer)
            styled_inference_time = style_inference_time(row[f'inference_time_{model}'], min_inference_time)
            table_html += f"<td>{styled_transcription}</td><td>{styled_inference_time}</td>"
        table_html += "</tr>"
    table_html += "</table>"

    st.markdown(table_html, unsafe_allow_html=True)

    if show_summary:
        summary_data = []
        for model in show_models:
            avg_wer = data.apply(lambda row: calculate_wer(row['ground_truth'], row[f'transcription_{model}']), axis=1).mean()
            avg_inference_time = data[f'inference_time_{model}'].mean()
            summary_data.append((model, avg_wer, avg_inference_time))

        min_avg_wer = min(summary_data, key=lambda x: x[1])[1]
        min_avg_inference_time = min(summary_data, key=lambda x: x[2])[2]

        summary_html = """
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid black;
                padding: 10px;
                text-align: center;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
        </style>
        <table>
        <tr>
            <th>Model</th>
            <th>Average WER</th>
            <th>Average Inference Time</th>
        </tr>"""

        for model, avg_wer, avg_inference_time in summary_data:
            wer_color = '#32CD32' if avg_wer == min_avg_wer else '#FF0000'
            inference_time_color = '#32CD32' if avg_inference_time == min_avg_inference_time else '#FF0000'
            summary_html += f"""
            <tr>
                <td>{model}</td>
                <td style='color: {wer_color};'>{avg_wer:.2f}</td>
                <td style='color: {inference_time_color};'>{avg_inference_time:.4f}</td>
            </tr>"""
        summary_html += "</table>"

        st.markdown(summary_html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
