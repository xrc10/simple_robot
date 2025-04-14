from openai import OpenAI
from PIL import Image
import io
import base64
import yaml
import cv2
import numpy as np


class VLM(object):

    def __init__(self, cfg_path, model_id):
        with open(cfg_path, "r") as f:
            llm_config = yaml.safe_load(f)

        self.client = OpenAI(api_key=llm_config["SiliconCloud"]["api_key"], 
                            base_url=llm_config["SiliconCloud"]["base_url"])
        self.prompt_history = []  # Store history of prompts and responses
        self.model_id = model_id

    def convert_image_to_webp_base64(self, image_data):
        # check if the image is a path or CV2 image, if it is a path open it with PIL , if it is a CV2 image convert it to base64
        try:
            if isinstance(image_data, np.ndarray):  # Check if it's a CV2 image (numpy array)
                # Convert CV2 image to PIL Image
                img = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
                byte_arr = io.BytesIO()
                img.save(byte_arr, format='webp')
                byte_arr = byte_arr.getvalue()
                base64_str = base64.b64encode(byte_arr).decode('utf-8')
            else:
                with Image.open(image_data) as img:
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='webp')
                    byte_arr = byte_arr.getvalue()
                    base64_str = base64.b64encode(byte_arr).decode('utf-8')
        except IOError:
            print(f"Error: Unable to open or convert the image {image_data}")
            return None
        
        return base64_str
    
    def get_response(self, image_data, prompt, stream=True, temperature=1.0, top_p=0.9, max_tokens=None):
        return self.run(image_data, self.model_id, prompt, stream, temperature, top_p, max_tokens)

    def run(self, image_data, model_id, prompt, stream=True, temperature=1.0, top_p=0.9, max_tokens=None):
        if type(image_data) is list:
            base64_images = [self.convert_image_to_webp_base64(image) for image in image_data]
            image_contents = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail":"low"}} for base64_image in base64_images]
            messages = [{"role": "user", "content": image_contents + [{"type": "text", "text": prompt}]}]

        else:
            base64_image = self.convert_image_to_webp_base64(image_data)
            messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail":"low"}},
                                                    {"type": "text", "text": prompt}]}]
          
        response = self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=stream,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        # collect the response string from the response
        response_string = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                response_string += chunk.choices[0].delta.content

        # Store the prompt and response in history
        self.prompt_history.append({
            "prompt": prompt,
            "response": response_string,
            "images": image_data if isinstance(image_data, list) else [image_data],
            "model_id": model_id
        })

        return response_string

    def get_prompt_history(self):
        return self.prompt_history


if __name__ == "__main__":
    vlm = VLM("llm.yaml")
    print(vlm.run("views/view_0_RotateLeft.png", "Qwen/Qwen2.5-VL-32B-Instruct", "Describe the image in 10 words"))
    print(vlm.run(["views/view_0_RotateLeft.png", "views/view_0_RotateLeft.png"], "Qwen/Qwen2.5-VL-32B-Instruct", "Describe the difference between the two images in 10 words"))

