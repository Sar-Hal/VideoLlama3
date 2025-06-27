from gradio_client import Client, handle_file

client = Client("lixin4ever/VideoLLaMA3")
result = client.predict(
		messages=[],
		video={"video":handle_file('https://github.com/gradio-app/gradio/raw/main/demo/video_component/files/world.mp4')},
		api_name="/_on_video_upload"
)
print(result)