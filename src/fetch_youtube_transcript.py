from yt_dlp import YoutubeDL

url = "https://www.youtube.com/watch?v=UV81LAb3x2g"
ydl_opts = {
    "skip_download": True,
    "writesubtitles": True,            # human captions
    "writeautomaticsub": True,         # auto captions
    "subtitleslangs": ["en"],          # try English first
    "subtitlesformat": "srt",
    "outtmpl": "%(id)s.%(ext)s",
}
with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])   # creates VIDEO_ID.en.srt if available
