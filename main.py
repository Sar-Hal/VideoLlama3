from gradio_client import Client, handle_file
from huggingface_hub import HfApi
import time

SPACE_ID = "lixin4ever/VideoLLaMA3"

def wait_for_gpu():
    print("🔍 Checking GPU status...")
    api = HfApi()
    while True:
        info = api.get_space_runtime(SPACE_ID)
        stage = info.stage
        print(f"ℹ️ Space status: {stage}")
        if stage == "RUNNING":
            print("✅ GPU is available. Proceeding...\n")
            break
        elif stage in {"BUILDING", "WAITING", "SLEEPING", "PAUSED"}:
            print("⏳ Waiting for GPU to become available...")
            time.sleep(5)
        else:
            raise RuntimeError(f"Unexpected space state: {stage}")

# Step 1: Wait for GPU
wait_for_gpu()

# Step 2: Init client
client = Client(SPACE_ID)
print(f"Loaded as API: {client.src} ✔")

# Step 3: Upload video
video_url = "https://github.com/gradio-app/gradio/raw/main/demo/video_component/files/world.mp4"
print("▶️ Uploading video...")
video_path = handle_file(video_url)
upload_result = client.predict(
    messages=[],
    video={"video": video_path},
    api_name="/_on_video_upload"
)
print(f"✅ Uploaded: {upload_result}")

# Step 4: Prepare message
video_message = {
    "role": "user",
    "content": [
        {
            "type": "video",
            "video": {
                "video_path": video_path,
                "fps": 1,
                "max_frames": 180
            }
        },
        "Summarize the video"
    ]
}

# Step 5: Submit inference job
print("▶️ Asking question...")
response = client.predict(
    messages=[video_message],
    input_text="Summarize the video",
    do_sample=True,
    temperature=0.2,
    top_p=0.9,
    max_new_tokens=512,
    fps=1,
    max_frames=180,
    api_name="/_predict"
)

print("\n=== ✅ Final Response ===")
for msg in response:
    print(f"{msg['role'].upper()}: {msg['content']}")


print("\n=== 🔄 Streaming Response ===")
start = time.time()
chunk_count = 0
try:
    for response in job:
        chunk_count += 1
        print(f"🟢 Chunk #{chunk_count}")
        print("➡️ Raw:", response)

        if isinstance(response, list):
            for msg in response:
                print(f"🔹 {msg['role'].upper()}: {msg['content']}")
        else:
            print("🔸", response)
except Exception as e:
    print("❌ Error occurred during streaming:", e)
finally:
    print(f"⏱️ Total time waited: {round(time.time() - start, 2)} seconds")
