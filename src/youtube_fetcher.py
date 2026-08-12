import re
import logging
import http.cookiejar
import requests
from pathlib import Path
from typing import List, Tuple, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import yt_dlp



from src.models import VideoMetadata, TranscriptSegment

logger = logging.getLogger(__name__)


def extract_video_id(url_or_id: str) -> str:
    """
    Extract YouTube video ID from various URL formats or plain video ID string.
    
    Supported formats:
    - https://www.youtube.com/watch?v=dQw4w9WgXcQ
    - https://youtu.be/dQw4w9WgXcQ
    - https://www.youtube.com/embed/dQw4w9WgXcQ
    - dQw4w9WgXcQ
    """
    if len(url_or_id) == 11 and re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
        return url_or_id

    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/|v\/|vi\/|youtu\.be\/|\/v\/|\/e\/|watch\?v=|&v=)([0-9A-Za-z_-]{11})'
    ]

    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract YouTube video ID from input: {url_or_id}")


def fetch_video_metadata(
    url_or_id: str,
    cookie_file: Optional[str] = "cookies.txt",
    browser_for_cookies: Optional[str] = None
) -> VideoMetadata:
    """Fetch video metadata using yt-dlp, supporting authentication cookies."""
    video_id = extract_video_id(url_or_id)
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
        'format': 'worst'
    }


    # Apply cookie file if exists
    if cookie_file and Path(cookie_file).exists():
        ydl_opts['cookiefile'] = str(Path(cookie_file).resolve())
        logger.info(f"Using cookie file: {cookie_file}")
    elif browser_for_cookies:
        ydl_opts['cookiesfrombrowser'] = (browser_for_cookies,)
        logger.info(f"Using browser cookies from: {browser_for_cookies}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return VideoMetadata(
                video_id=video_id,
                title=info.get('title', f"YouTube Video ({video_id})"),
                url=url,
                channel=info.get('uploader', info.get('channel', info.get('uploader_id', 'Unknown Channel'))),
                upload_date=info.get('upload_date'),
                duration=info.get('duration', 0),
                description=info.get('description', ''),
                language=info.get('language', 'en')
            )

    except Exception as e:
        logger.warning(f"yt-dlp metadata fetch notice for {video_id}: {e}. Using extracted fallback metadata.")
        return VideoMetadata(
            video_id=video_id,
            title=f"Total War Video ({video_id})",
            url=url,
            channel="YouTube",
            language="en"
        )



def fetch_transcript(
    video_id: str,
    languages: List[str] = None,
    cookie_file: Optional[str] = "cookies.txt"
) -> List[TranscriptSegment]:
    """
    Fetch subtitles/captions with timestamp details using youtube-transcript-api.
    
    Supports authentication cookies from cookie_file (e.g. cookies.txt).
    """
    if languages is None:
        languages = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']

    segments = []
    try:
        session = requests.Session()
        if cookie_file and Path(cookie_file).exists():
            try:
                cookie_jar = http.cookiejar.MozillaCookieJar(cookie_file)
                cookie_jar.load(ignore_discard=True, ignore_expires=True)
                session.cookies = cookie_jar
                logger.info(f"Loaded authentication cookies into transcript fetcher from: {cookie_file}")
            except Exception as c_err:
                logger.warning(f"Could not load cookie file '{cookie_file}': {c_err}")

        api = YouTubeTranscriptApi(http_client=session)
        # Primary method: api.fetch
        if hasattr(api, 'fetch'):
            try:
                fetched = api.fetch(video_id, languages=languages)
                snippets = getattr(fetched, 'snippets', fetched)
                for item in snippets:
                    text = getattr(item, 'text', '') if hasattr(item, 'text') else item.get('text', '')
                    start = getattr(item, 'start', 0.0) if hasattr(item, 'start') else item.get('start', 0.0)
                    duration = getattr(item, 'duration', 0.0) if hasattr(item, 'duration') else item.get('duration', 0.0)
                    text = text.replace('\n', ' ').replace('&quot;', '"').replace('&#39;', "'").strip()
                    if text:
                        segments.append(TranscriptSegment(text=text, start=float(start), duration=float(duration)))
                if segments:
                    return segments
            except Exception as e:
                logger.info(f"api.fetch failed for {video_id}, trying api.list: {e}")


        # Secondary method: api.list
        if hasattr(api, 'list'):
            transcript_list = api.list(video_id)
            try:
                t = transcript_list.find_manually_created_transcript(languages)
            except Exception:
                t = transcript_list.find_generated_transcript(languages)
            fetched = t.fetch()
            snippets = getattr(fetched, 'snippets', fetched)
            for item in snippets:
                text = getattr(item, 'text', '') if hasattr(item, 'text') else item.get('text', '')
                start = getattr(item, 'start', 0.0) if hasattr(item, 'start') else item.get('start', 0.0)
                duration = getattr(item, 'duration', 0.0) if hasattr(item, 'duration') else item.get('duration', 0.0)
                text = text.replace('\n', ' ').replace('&quot;', '"').replace('&#39;', "'").strip()
                if text:
                    segments.append(TranscriptSegment(text=text, start=float(start), duration=float(duration)))
            return segments

        # Static class method fallback (older versions)
        if hasattr(YouTubeTranscriptApi, 'get_transcript'):
            raw_data = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            for item in raw_data:
                text = item.get('text', '').replace('\n', ' ').replace('&quot;', '"').replace('&#39;', "'").strip()
                if text:
                    segments.append(TranscriptSegment(text=text, start=float(item['start']), duration=float(item['duration'])))
            return segments

    except Exception as e:
        raise RuntimeError(f"Could not fetch transcript for YouTube video '{video_id}': {e}")

    if not segments:
        raise ValueError(f"No caption segments retrieved for video '{video_id}'.")

    return segments



def extract_playlist_video_urls(
    playlist_url: str,
    cookie_file: Optional[str] = "cookies.txt",
    browser_for_cookies: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Extract all (video_id, url) pairs from a YouTube playlist URL using yt-dlp.
    """
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'skip_download': True
    }
    if cookie_file and Path(cookie_file).exists():
        ydl_opts['cookiefile'] = str(Path(cookie_file).resolve())
    elif browser_for_cookies:
        ydl_opts['cookiesfrombrowser'] = (browser_for_cookies,)

    videos = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)
        if 'entries' in result:
            for entry in result['entries']:
                if entry and 'id' in entry:
                    vid = entry['id']
                    videos.append((vid, f"https://www.youtube.com/watch?v={vid}"))
    return videos

