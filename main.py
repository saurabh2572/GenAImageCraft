import streamlit as st
import os
from pathlib import Path
from streamlit_chat import message
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import make_image_grid, load_image
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2
from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)
from PIL import Image
import numpy as np

from huggingface_hub import notebook_login
notebook_login()

class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

image_gen_model = StableDiffusionPipeline.from_pretrained(
  CFG.image_gen_model_id, torch_dtype=torch.float16,
  revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def stable_diffusion(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image
def sdxl_turbo(prompt):
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    image = pipe(prompt=prompt, num_inference_steps=1).images[0]
    return image


def image_to_image(init_image,input_prompt):
    image = np.array(init_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
    pipe = pipe.to("cuda")
    prompt = input_prompt
    generator = torch.Generator(device="cpu")
    output = pipe(
    prompt,
    canny_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=20,
    generator=generator,
    )

    return output.images

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def potrait_image(actor,init_image):
    image = np.array(init_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
    pipe = pipe.to("cuda")
    prompt = " ,dramatic Lighting, Divine, Realistic, highly detailed, most beautiful image I have seen ever"
    prompt = actor + prompt
    generator = torch.Generator(device="cpu")
    output = pipe(
    prompt,
    canny_image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=20,
    generator=generator,
    )

    return output.images



if __name__=="__main__":
    st.set_page_config(layout="wide")
    st.subheader("GenAImageCraft")

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    with st.chat_message("ai"):
        st.write("Hello 👋, I'm here to assist you in creating Images🙂. JUST GIVE AN IDEA")

    model=st.selectbox(label="Select LLM Model",options=['Text to Image',"Image to Image","Image to Sketch","Replace Potrait"], index=None, placeholder='select Model')

    if model:
        if model=="Stable Diffusion":
            input_prompt = st.text_input("Give an Idea")

            if input_prompt is not None:
                st.session_state.messages.append(
            HumanMessage(content=input_prompt))
                if st.button("Generate Image"):
                    image_output = stable_diffusion(input_prompt, image_gen_model)
                    st.info("Generating image.....")
                    st.success("Image Generated Successfully")
                    st.image(image_output, caption="Generated by Stable Diffusion")

        if model=="Text to Image":
            input_prompt = st.text_input("Give an Idea")
            st.session_state.messages.append(
        HumanMessage(content=input_prompt))
            if input_prompt is not None:
                st.session_state.messages.append(
            HumanMessage(content=input_prompt))
                if st.button("Generate Image"):
                    image_output = stable_diffusion(input_prompt, image_gen_model)
                    st.info("Generating image.....")
                    st.success("Image Generated Successfully")
                    st.image(image_output, caption="Generated by SDXL Turbo")
        if model=="Image to Image":
            uploaded_file=st.file_uploader("Upload a file:",type=["png","jpg","jpeg"])

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Your Image")
                bytes_data = uploaded_file.read()
                image_name = os.path.join('./', uploaded_file.name)

                with open(image_name, 'wb') as f:
                    f.write(bytes_data)
                init_image = load_image(image_name)

                input_prompt = st.text_input("Give an Idea")


                if input_prompt is not None:
                    if st.button("Generate Image"):
                        image_output = image_to_image(init_image,input_prompt)
                        st.info("Generating image.....")
                        st.success("Image Generated Successfully")
                        st.image(image_output, caption="Generated by stable-diffusion-v1-5")

        if model=="Image to Sketch":
            uploaded_file=st.file_uploader("Upload a file:",type=["png","jpg","jpeg"])

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Your Image")
                bytes_data = uploaded_file.read()
                image_name = os.path.join('./', uploaded_file.name)

                with open(image_name, 'wb') as f:
                    f.write(bytes_data)
                init_image = load_image(image_name)

                image = np.array(init_image)

                low_threshold = 100
                high_threshold = 200

                image = cv2.Canny(image, low_threshold, high_threshold)
                image = image[:, :, None]
                image = np.concatenate([image, image, image], axis=2)
                canny_image = Image.fromarray(image)
                st.image(canny_image, caption="Sketch")

        if model=="Replace Potrait":
            uploaded_file=st.file_uploader("Upload a file:",type=["png","jpg","jpeg"])

            if uploaded_file is not None:
                st.image(uploaded_file, caption="Your Image")
                bytes_data = uploaded_file.read()
                image_name = os.path.join('./', uploaded_file.name)

                with open(image_name, 'wb') as f:
                    f.write(bytes_data)
                init_image = load_image(image_name)
                actor=st.text_input("Enter your Favorite Actor")
                if st.button("Generate Image"):

                  image_grid=potrait_image(actor,init_image)
                  st.image(image_grid, caption="sd-controlnet-canny")

