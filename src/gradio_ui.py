from utils.process_uploaded import process_uploaded_files
from utils.gradio_settings import toggle_sidebar, feedback
from utils.load_config import LoadConfig
from utils.chatbot import respond
import gradio as gr

CONFIG = LoadConfig()


with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("RAG-Mistral7b"):
            ##################
            # First ROW:
            ##################
            with gr.Row() as row_one:
                with gr.Column(visible=False) as reference_bar:
                    ref_output = gr.Markdown()

                with gr.Column() as chatbot_output:
                    chatbot = gr.Chatbot(
                        [],
                        elem_id="chatbot",
                        bubble_full_width=False,
                        height=500)
                    chatbot.like(feedback, None, None)
            ##################
            # SECOND ROW:
            ##################
            with gr.Row():
                input_txt = gr.Textbox(
                    lines=4,
                    scale=8,
                    placeholder="Enter text and press enter, or upload PDF files",
                    container=False)

            ##################
            # Third ROW:
            ##################
            with gr.Row() as row_two:
                text_submit_btn = gr.Button(value="Generate answer")
                sidebar_state = gr.State(False)
                btn_toggle_sidebar = gr.Button(value="References")
                btn_toggle_sidebar.click(toggle_sidebar, [sidebar_state], [
                    reference_bar, sidebar_state])
                upload_btn = gr.UploadButton(
                    "Upload PDF file", file_types=['.pdf'],
                    file_count="multiple")
                rag_with_dropdown = gr.Dropdown(
                    label="Choose action",
                    choices=["Preprocessed doc",
                             "Upload doc: Process for RAG",
                             "Upload doc: Summary"],
                    value="Preprocessed doc")
                choose_model = gr.Dropdown(
                    label="Choose model",
                    choices=["Open-Source model Mistral7b",
                             "gpt-4o-mini"],
                    value="Open-Source model Mistral7b")

                clear_button = gr.ClearButton([input_txt, chatbot])

            ##################
            # Process:
            ##################
            file_msg = upload_btn.upload(fn=process_uploaded_files,
                                         inputs=[upload_btn, chatbot, rag_with_dropdown, choose_model],
                                         outputs=[input_txt, chatbot], queue=False)

            txt_msg = input_txt.submit(fn=respond,
                                       inputs=[chatbot, input_txt, choose_model, rag_with_dropdown],
                                       outputs=[input_txt, chatbot, ref_output],
                                       queue=False).then(lambda: gr.Textbox(interactive=True),
                                                         None, [input_txt], queue=False)
            submit_btn = text_submit_btn.click(fn=respond,
                                               inputs=[chatbot, input_txt, choose_model, rag_with_dropdown],
                                               outputs=[input_txt, chatbot, ref_output],
                                               queue=False).then(lambda: gr.Textbox(interactive=True),
                                                                 None, [input_txt], queue=False)


if "__main__" == __name__:
    demo.launch()
