from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
import os
import tempfile

app = FastAPI()

# Configure CORS to allow Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://incandescent-beijinho-b4545d.netlify.app"],  # Replace with your Netlify domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gradio Client
client = Client("lixin4ever/VideoLLaMA3", hf_token=os.getenv("HF_TOKEN"))

@app.post("/process-media")
async def process_media(file: UploadFile = File(...), query: str = Form(...)):
    try:
        # Validate file format
        if not file.filename.endswith((".mp4", ".png", ".jpg", ".jpeg")):
            return {"error": "Unsupported file format. Use .mp4, .png, .jpg, or .jpeg"}

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        # Step 1: Upload media (video or image)
        messages = []
        endpoint = "/_on_video_upload" if file.filename.endswith(".mp4") else "/_on_image_upload"
        media_param = (
            {"video": handle_file(tmp_file_path)} if endpoint == "/_on_video_upload"
            else {"path": handle_file(tmp_file_path)}
        )
        result = client.predict(
            messages=messages,
            **{endpoint.split("_")[-2]: media_param},  # video or image
            api_name=endpoint
        )
        messages = result[0]

        # Step 2: Add text query
        result = client.predict(
            messages=messages,
            text=query,
            api_name="/_on_text_submit"
        )
        messages = result[0]

        # Step 3: Generate response
        result = client.predict(
            messages=messages,
            input_text="",
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            max_new_tokens=2048,
            fps=1.0,
            max_frames=180,
            api_name="/_predict"
        )
        response = result[-1]["content"] if result else "No response generated"

        # Clean up temporary file
        os.unlink(tmp_file_path)
        return {"response": response}
    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return {"error": str(e)}