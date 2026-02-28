import uvicorn
import webbrowser
import threading

def open_browser():
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # Open browser automatically
    threading.Timer(1.5, open_browser).start()

    uvicorn.run(
        "backend.app:app",
        host="127.0.0.1",
        port=8000
    )