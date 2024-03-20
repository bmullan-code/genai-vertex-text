# app v2
""" Generation of text with gemini
    https://github.com/GoogleCloudPlatform/generative-ai/blob/main/gemini/getting-started/intro_gemini_python.ipynb
"""

from google.cloud import aiplatform
import google.cloud.logging
import vertexai
import gradio as gr
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part

PROJECT_ID = "sandbox2-363416"
LOCATION = "us-central1"

client = google.cloud.logging.Client(project=PROJECT_ID)
client.setup_logging()

log_name = "genai-vertex-text-log"
logger = client.logger(log_name)

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.0-pro")

templates = {
    "Prompt#1" : """
        Format the user input below as a set of bullet points.
        <UserInput>
        {user_input}
        </UserInput>
    """,
    "Prompt#2" : """
        Format the user input below as a markdown table.
        <UserInput>
        {user_input}
        </UserInput>
    """
}

def predict(template,prompt, max_output_tokens, temperature, top_p, top_k):
    logger.log_text(prompt)

    final_prompt = templates[template].format(user_input=prompt)
    print(final_prompt)
    config = GenerationConfig(max_output_tokens=int(max_output_tokens), temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))

    # responses = model.generate_content(prompt, stream=True)
    # responses = model.generate_content(prompt, generation_config=config, stream=True)
    responses = model.generate_content(final_prompt, generation_config=config, stream=True)

    final_response = []
    for response in responses:
        print(str(response))
        try:
            final_response.append(response.text)
        except IndexError:
            final_response.append("")
            continue
        except AttributeError:
            final_response.append("")
            continue

    return " ".join(final_response)


# examples = [
#     ["Best receipt for banana bread:"],
#     ["You are an equities analyst researching information for your report with relevant facts and figures. Tell me about the mortgage market in US."],
#     ["Brainstorm some ideas combining VR and fitness:"],
# 

with gr.Blocks() as demo:
    with gr.Accordion("Settings", open=False):
        max_o = gr.Textbox(label="Max Output Tokens (0-8096)", value="1024")
        temp = gr.Textbox(label="Temperature (0-1)", value="0.5")
        top_p = gr.Textbox(label="Top P (0-1)", value="0.8")
        top_k = gr.Textbox(label="Top K (0-40)", value="38")

    # max_o = gr.Slider(0,1024,step=32,label="Max Output Tokens")
    # max_o.value=32
    # slider = gr.Slider(0, 100, step=0.1,label="slider")
    # temp = gr.Slider(0,1,step=0.1,label = "temperature")
    # top_p = gr.Slider(0, 1, value=0.8, step = 0.1, label = "top_p")
    # top_k = gr.Slider(0, 40, value=38, step = 1, label = "top_k")

    dropdown = gr.Dropdown(templates.keys(),label="Prompt Template")
    with gr.Row():
        text_input = gr.Textbox(label="Enter prompt:", value="Best receipt for banana bread:",lines=20)
        with gr.Column():
            prompt_output = gr.Markdown( label="prompt")
            text_output = gr.Markdown( label="response")
    text_button = gr.Button("Submit")

    dropdown.change(fn=lambda x: templates[x], inputs=dropdown, outputs=prompt_output)
    text_button.click(fn=predict, inputs=[dropdown,text_input,max_o,temp,top_p,top_k], outputs=text_output)

# # demo = gr.Interface(
#     # fn = predict, 
#     inputs = [ 
#       gr.Dropdown(templates.keys() ),
#       gr.Textbox(label="Enter prompt:", value="Best receipt for banana bread:",lines=20),
#       gr.Slider(32, 1024, value=512, step = 32, label = "max_output_tokens"),
#       gr.Slider(0, 1, value=0.2, step = 0.1, label = "temperature"),
#       gr.Slider(0, 1, value=0.8, step = 0.1, label = "top_p"),
#       gr.Slider(1, 40, value=38, step = 1, label = "top_k"),
#     ],
#     outputs= "markdown"
#     # examples=examples
#     )

demo.launch(server_name="0.0.0.0", server_port=7860,share=True)
# demo.launch(server_name="0.0.0.0", server_port=7860)