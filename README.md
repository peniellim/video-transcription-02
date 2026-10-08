# Video Transcription Tool

Upload a video (up to 60 min) on the website and get an Excel file with dialogue, FPS timecodes, on-screen text and English translation. Processing runs on a Google Colab GPU that the website talks to directly.

## Layout
- `docs/index.html` - the website (GitHub Pages)
- `notebooks/video_transcriber_colab.ipynb` - Colab backend (starts the server and prints a connection code)
- `tool/video_transcriber.py` - Whisper + EasyOCR pipeline
- `tool/server.py` - upload/progress/download API used by the website

## Publish the website
1. Push this folder to a GitHub repository (`main` branch).
2. **Settings -> Pages -> Deploy from a branch -> `main` / `/docs` -> Save.**
3. Your site: `https://<username>.github.io/<repo>/`

## Use
1. On the site, open the **backend notebook** link. In Colab set Runtime to **T4 GPU** and run all cells.
2. The last cell prints a connection code (`https://....trycloudflare.com#key`). Paste it on the site and click **Connect**.
3. Choose a video and click **Transcribe**. The video is uploaded in 40 MB chunks, progress is shown live, and the Excel file downloads automatically when finished (SRT and an in-page viewer are also provided).

Notes: the code changes every time the Colab cell is restarted; the site remembers the last one in your browser. Colab sessions stop after inactivity or the free-tier time limit.
