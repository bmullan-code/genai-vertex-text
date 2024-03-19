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


def predict(prompt, max_output_tokens, temperature, top_p, top_k):
    logger.log_text(prompt)

    config = GenerationConfig(max_output_tokens=max_output_tokens, temperature=temperature, top_p=top_p, top_k=top_k)

    # responses = model.generate_content(prompt, stream=True)
    responses = model.generate_content(prompt, generation_config=config, stream=True)

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


examples = [
    ["Best receipt for banana bread:"],
    ["You are an equities analyst researching information for your report with relevant facts and figures. Tell me about the mortgage market in US."],
    ["Brainstorm some ideas combining VR and fitness:"],
]

demo = gr.Interface(
    predict, 
    [ gr.Textbox(label="Enter prompt:", value="Best receipt for banana bread:",lines=20),
      gr.Slider(32, 1024, value=512, step = 32, label = "max_output_tokens"),
      gr.Slider(0, 1, value=0.2, step = 0.1, label = "temperature"),
      gr.Slider(0, 1, value=0.8, step = 0.1, label = "top_p"),
      gr.Slider(1, 40, value=38, step = 1, label = "top_k"),
    ],
    "text",
    examples=examples
    )

# demo.launch(server_name="0.0.0.0", server_port=7860,share=True)
demo.launch(server_name="0.0.0.0", server_port=7860)