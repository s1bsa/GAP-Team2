import gradio as gr
from PIL import Image
import matplotlib.pyplot as plt
import io
import model.inference as inference
from gradio.flagging import CSVLogger 

# Flagging setup 
flagging_dir = "flagged_data" 
csv_logger = CSVLogger()

# Model and image gallery setup

models = inference.grab_model_names()
model_descriptions = {"CNN_11-General": "Our most general CNN model trained on the whole dataset", 
                      "ResNet3-General": "A ResNet based model trained on the whole dataset",
                      "CNN_11-Tomatoes": "A model we trained specifically to classify tomato diseases"}

images = [
    Image.open("UI/gallery_imgs/gallery_img1.png"),
    Image.open("UI/gallery_imgs/gallery_img2.png"),
    Image.open("UI/gallery_imgs/gallery_img3.png"),
    Image.open("UI/gallery_imgs/gallery_img4.png"),
    Image.open("UI/gallery_imgs/gallery_img5.png")
]

# Color definitions

LIGHT_COLORS = {
    "back": "#f5f5f4",
    "front": "#fbbf24",
    "graph": "#00210c",
    "button_bg": "#fbbf24",
    "button_text": "#00210c",
    "markdown_text": "#00210c",
    "hover_color": "#d29a1e"
}

DARK_COLORS = {
    "back": "#292524",
    "front": "#f59e0b",
    "graph": "#e7e6e4",
    "button_bg": "#f59e0b",
    "button_text": "#292524",
    "markdown_text": "#e7e6e4",
    "hover_color": "#ca8509"
}

# Functions

# adds images to the gallery of plant images. (also adds images that the users submitted)
def add_to_images(new_images_paths):
    for path in new_images_paths:
        try:
            if path is not None:
                img = Image.open(path).convert("RGB")
                images.append(img)
        except Exception as e:
            print(f"Could not open image for gallery: {path}, Error: {e}")
    return images

# gets the top5 predictions by the selected model and makes a motplotlib bar graph for each image submitted
def show_predictions(model, image_paths, mode):
    
    # handle the case where Gradio passes a single file path for gr.Image
    if isinstance(image_paths, str):
        image_paths = [image_paths] 

    if not image_paths or image_paths[0] is None:
        return None, images, "Please upload image(s).", [] 

    colors = LIGHT_COLORS if mode == "Light" else DARK_COLORS
    back_color = colors["back"]
    front_color = colors["front"]
    graph_color = colors["graph"]

    all_graphs = []
    first_data_string = "No Predictions."
    processed_input_paths = []

    # iterate through multiple images and produce graphs for each of them
    for i, path in enumerate(image_paths):
        if path is None:
            continue
            
        #call main method of inference script to run models as it is all handled there
        results_list = inference.main(model_name=model, image_path=path) 
    
        if not results_list:
            print(f"Inference failed for image: {path}")
            continue 
            
        data_dict = results_list[0]
        
        top_labels = data_dict.get("top5_labels", [])
        top_probs = data_dict.get("top5_probs", [])
        
        data = dict(zip(top_labels, top_probs))
    
        classes = list(data.keys())
        confidence_levels = list(data.values())

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(back_color)
        ax.set_facecolor(back_color)
        ax.barh(range(len(data)), confidence_levels, tick_label=classes, color=front_color)
        ax.xaxis.tick_top()
        ax.set_xlabel("Confidence Level", fontsize=12, color=graph_color, fontweight="bold")
        ax.xaxis.set_label_position("top")
        ax.set_ylabel("Plant Diseases", fontsize=12, color=graph_color, fontweight="bold")
        ax.invert_yaxis()
        ax.set_title(f"Prediction for Image {i+1}", fontsize=14, color=graph_color, fontweight="bold")
        ax.tick_params(axis="x", colors=graph_color, labelsize=10, rotation=15)
        ax.tick_params(axis="y", colors=graph_color, labelsize=10)
        ax.set_xlim(0, 1.1)
        for spine in ["bottom", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["top"].set_color(graph_color)
        ax.spines["left"].set_color(graph_color)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        
        all_graphs.append(Image.open(buf))
        processed_input_paths.append(path) 
        
        if i == 0:
            first_data_string = str(data)
            

    if not all_graphs:
        return None, images, "No images processed successfully.", []
    
    # updates the gallery with the user submitted images
    updated_gallery = add_to_images(processed_input_paths)
        
    return all_graphs[0], updated_gallery, first_data_string, all_graphs

# redraws the graph whenever the user switches between light and dark mode for coherent theming
def redraw_graph_if_exists(model, single_image, batch_images, mode):
    image_paths = single_image if single_image is not None else batch_images
    
    if image_paths is None:
        return None, images, None, [] 
    else:
        first_graph, updated_gallery, data_string, all_graphs = show_predictions(model, image_paths, mode)
        return first_graph, updated_gallery, data_string, all_graphs

# changes the UI depending on if the user wants to process a single image or multiple images
def change_processing_type(processing_type):
    if processing_type == "Single Image":
        return {
            single_image_input: gr.update(visible=True),
            batch_file_input: gr.update(visible=False),
            result: gr.update(visible = True),
            batch_result: gr.update(visible = False),
            flag_comment : gr.update(visible = True),
            flag_btn : gr.update(visible = True)

        }
    else: # Mutiple Images
        return {
            single_image_input: gr.update(visible=False),
            batch_file_input: gr.update(visible=True),
            result: gr.update(visible = False),
            batch_result: gr.update(visible = True),
            flag_comment : gr.update(visible = False),
            flag_btn : gr.update(visible = False)
        }

# processes the submission for show_predictions to run smoothly 
def handle_submission(model, single_img, batch_imgs, mode, processing_type):
    if processing_type == "Single Image":
        image_paths = single_img
    elif processing_type == "Mutiple Images":
        image_paths = batch_imgs
    else:
        return None, images, "Invalid processing type selected.", []

    return show_predictions(model, image_paths, mode)

# displays a short description of the model
def get_model_description(model_name):
    return model_descriptions.get(
        model_name,
        f"No detailed description available for model: **{model_name}**."
    )

# method that handles flagging and the display messages for flagging
def handle_flagging(image, pred_data, comment, proc_type, callback):
    
    if image is None:
        return "**Error!** Please input an image and make a prediction", comment
    
    # if no errors proceed with flagging
    image_data_for_flagging = {"path": image}
    flag_data = [image_data_for_flagging, pred_data, comment]
    callback.flag(flag_data)
    
    #returns success message
    return "This data point has been flagged and saved for review.", ""


# custom theme for the gradio app, enables clean dark and light mode switching
class CitrusTheme(gr.themes.Base):
    def __init__(self):
        super().__init__(
            primary_hue=gr.themes.colors.amber, 
            neutral_hue=gr.themes.colors.stone, 
            secondary_hue=gr.themes.colors.amber,
        )
        
        self.set(
            body_background_fill=LIGHT_COLORS["back"],
            block_background_fill=LIGHT_COLORS["back"],
            
            body_text_color_subdued=LIGHT_COLORS["markdown_text"],
            body_text_color=LIGHT_COLORS["markdown_text"],
            
            button_primary_background_fill=LIGHT_COLORS["button_bg"],
            button_primary_text_color=LIGHT_COLORS["button_text"],
            button_primary_background_fill_hover=LIGHT_COLORS["hover_color"],
            button_secondary_background_fill=LIGHT_COLORS["button_bg"],
            button_secondary_text_color=LIGHT_COLORS["button_text"],
            button_secondary_background_fill_hover=LIGHT_COLORS["hover_color"],

            body_background_fill_dark=DARK_COLORS["back"],
            block_background_fill_dark=DARK_COLORS["back"],
            
            body_text_color_subdued_dark=DARK_COLORS["markdown_text"],
            body_text_color_dark=DARK_COLORS["markdown_text"],

            button_primary_background_fill_dark=DARK_COLORS["button_bg"],
            button_primary_text_color_dark=DARK_COLORS["button_text"],
            button_primary_background_fill_hover_dark=DARK_COLORS["hover_color"],
            button_secondary_background_fill_dark=DARK_COLORS["button_bg"],
            button_secondary_text_color_dark=DARK_COLORS["button_text"],
            button_secondary_background_fill_hover_dark=DARK_COLORS["hover_color"],
            
            border_color_accent=LIGHT_COLORS["front"],
            input_background_fill=LIGHT_COLORS["back"],
            input_border_color=LIGHT_COLORS["graph"],
            border_color_accent_dark=DARK_COLORS["front"],
            input_background_fill_dark=DARK_COLORS["back"],
            input_border_color_dark=DARK_COLORS["graph"],
        )

citrus = CitrusTheme()

# java script which changes the theme of the app
js_toggle_mode = """
(mode) => {
    const darkBack = '#292524';
    const lightBack = '#f5f5f4';
    
    if (mode === "Dark") {
        document.body.classList.add('dark');
        document.querySelector('gradio-app').style.backgroundColor = darkBack; 
    } else {
        document.body.classList.remove('dark');
        document.querySelector('gradio-app').style.backgroundColor = lightBack;
    }
}
"""

# structure of the ui
with gr.Blocks() as demo:
    
    # initializing callback for the flag logger (csv_logger)
    demo.flagging_callback = csv_logger
    
    with gr.Row():
        md_title = gr.Markdown("# Plant Disease Classifier\nUpload a leaf image or a folder of images for analysis.")

        mode_selector = gr.Radio(
            choices=["Light", "Dark"],
            value="Dark",
            label="Toggle Light/Dark mode",
            interactive=True,
        )

    with gr.Row():
        with gr.Column():
            #define multiple image inputs to switch between depending on what the user wants
            single_image_input = gr.Image(type="filepath", label="Upload a Leaf Image (Single)", visible=True)
            batch_file_input = gr.File(type="filepath", label="Upload a Folder of Leaf Images (Batch)", file_count="directory", visible=False)
            
            # hidden element used to store the prediction data for flagging
            prediction_data = gr.Textbox(visible=False, label="Prediction Data") 
            
            with gr.Row():
                with gr.Column():
                    submit_btn = gr.Button("Submit", variant="primary") 
                    clear_btn = gr.Button("Clear", variant="secondary")
                processing_type = gr.Radio(
                    choices = ["Single Image", "Mutiple Images"], 
                    value = "Single Image", 
                    label = "Choose if you want to process one image or a folder of images",
                    interactive = True)
                    
        with gr.Column():
            result = gr.Image(label="Prediction Bar Chart (First Image)")
            batch_result = gr.Gallery(label="Batch Prediction Gallery", visible=False, columns=1, rows=1, object_fit="contain", height="auto", preview = True)

            flag_btn = gr.Button("Flag", variant="primary")

            flag_message = gr.Markdown("Click 'Flag' if the prediction is incorrect.") 
            flag_comment = gr.Textbox(visible = True, interactive= True, label ="Extra information for flagging", placeholder="Enter extra information on the issue here e.g 'The disease should be x'")
            
    with gr.Row():
        model_selector = gr.Dropdown(
            label="Select a CNN Model",
            choices=models,
            value = "CNN_11-General",
            interactive=True
        )
        model_info = gr.Textbox(
            value=get_model_description("CNN_11-General"), 
            label="Model Details",
            interactive = False
        )

    # gallery of leaf images 
    md_gallery = gr.Markdown("### Gallery of Plant Images")
    gallery = gr.Gallery(value=images, columns=5, rows=1, object_fit="contain", height="300", label="Gallery", allow_preview= False)

    # clears all relevant inputs and outputs
    def clear(processing_type):
        if processing_type == "Single Image":
            return None, None, None, [], None, "Click 'Flag' if the prediction is incorrect.", ""
        else:
            return None, None, None, [], None, "Cannot flag mutiple images at once. Please select the image you believe is incorrectly classified in single image processing and then flag it.", ""
    
    # sets up the logger and defines its inputs (each element processed in setup is a column)
    csv_logger.setup([single_image_input, prediction_data, flag_comment], flagging_dir=flagging_dir,) 
    demo.flagging_inputs = [single_image_input, prediction_data, flag_comment] 

    # Event handlers

    # changes the UI from single image to batch image processing 
    processing_type.change(
        fn = change_processing_type,
        inputs = [processing_type], 
        outputs = [single_image_input, batch_file_input, result, batch_result, flag_comment, flag_btn],
        queue = False,
    ).then(
        fn=clear,
        inputs= [processing_type],
        outputs=[single_image_input, batch_file_input, result, batch_result, prediction_data, flag_message, flag_comment],
        queue=False,
    )
    
    # passes the information needed to the handle_submissionn which then passes to show_predictions
    submit_btn.click(
        fn=handle_submission,
        inputs=[model_selector, single_image_input, batch_file_input, mode_selector, processing_type],
        outputs=[result, gallery, prediction_data, batch_result]
    )

    # changes hat model is being used
    model_selector.change(
        fn=get_model_description,
        inputs=[model_selector],
        outputs=[model_info],
        queue=False
    )

    # redraws the graph if light and dark mode is switched
    mode_selector.change(
        fn=redraw_graph_if_exists,
        inputs=[model_selector, single_image_input, batch_file_input, mode_selector],
        outputs=[result, gallery, prediction_data, batch_result]
    )
    
    mode_selector.change(
        fn=None, 
        inputs=[mode_selector], 
        outputs=None,
        js=js_toggle_mode,
        queue=False
    )

    # handles flagging and passes the inputs into the logger 
    flag_btn.click(
        fn=lambda img, data, cmt, p_type: handle_flagging(img, data, cmt, p_type, demo.flagging_callback),
        inputs=[
            single_image_input, 
            prediction_data, 
            flag_comment, 
            processing_type
        ],
        outputs=[flag_message, flag_comment],
        queue=False
    )

    # clears all the inputs and outputs
    clear_btn.click(
        fn=clear, 
        inputs = [processing_type],
        outputs=[single_image_input, batch_file_input, result, batch_result, prediction_data, flag_message, flag_comment]
    )

def launch_ui():
    demo.launch(theme=citrus)
