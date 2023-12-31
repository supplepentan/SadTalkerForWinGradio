import gradio as gr
from utils.gradio_demo import SadTalker

checkpoint_path = "checkpoints"
config_path = "config"
sad_talker = SadTalker(checkpoint_path, config_path, lazy_load=True)

app = gr.Blocks(analytics_enabled=False)
with app:
    with gr.Row():
        with gr.Column(variant="panel"):
            with gr.Tabs(elem_id="sadtalker_source_image"):
                with gr.TabItem("Upload image"):
                    with gr.Row():
                        source_image = gr.Image(
                            label="Source image",
                            type="filepath",
                            elem_id="img2img_image",
                            width=512,
                        )
            with gr.Tabs(elem_id="sadtalker_driven_audio"):
                with gr.TabItem("Upload OR TTS"):
                    with gr.Column(variant="panel"):
                        driven_audio = gr.Audio(label="Input audio", type="filepath")
        with gr.Column(variant="panel"):
            with gr.Tabs(elem_id="sadtalker_genearted"):
                gen_video = gr.Video(label="Generated video", format="mp4")
            with gr.Tabs(elem_id="sadtalker_checkbox"):
                with gr.TabItem("Settings"):
                    gr.Markdown(
                        "need help? please visit our [best practice page](https://github.com/OpenTalker/SadTalker/blob/main/docs/best_practice.md) for more detials"
                    )
                    with gr.Column(variant="panel"):
                        # width = gr.Slider(minimum=64, elem_id="img2img_width", maximum=2048, step=8, label="Manually Crop Width", value=512) # img2img_width
                        # height = gr.Slider(minimum=64, elem_id="img2img_height", maximum=2048, step=8, label="Manually Crop Height", value=512) # img2img_width
                        pose_style = gr.Slider(
                            minimum=0,
                            maximum=46,
                            step=1,
                            label="Pose style",
                            value=0,
                        )  #
                        size_of_image = gr.Radio(
                            [256, 512],
                            value=256,
                            label="face model resolution",
                            info="use 256/512 model?",
                        )  #
                        preprocess_type = gr.Radio(
                            ["crop", "resize", "full", "extcrop", "extfull"],
                            value="crop",
                            label="preprocess",
                            info="How to handle input image?",
                        )
                        is_still_mode = gr.Checkbox(
                            label="Still Mode (fewer head motion, works with preprocess `full`)"
                        )
                        batch_size = gr.Slider(
                            label="batch size in generation",
                            step=1,
                            maximum=10,
                            value=2,
                        )
                        enhancer = gr.Checkbox(label="GFPGAN as Face enhancer")
                        submit = gr.Button(
                            "Generate",
                            elem_id="sadtalker_generate",
                            variant="primary",
                        )
                        submit.click(
                            fn=sad_talker.test,
                            inputs=[
                                source_image,
                                driven_audio,
                                preprocess_type,
                                is_still_mode,
                                enhancer,
                                batch_size,
                                size_of_image,
                                pose_style,
                            ],
                            outputs=[gen_video],
                        )


if __name__ == "__main__":
    app.queue()
    app.launch()
