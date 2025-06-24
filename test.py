import os
import requests
from dotenv import load_dotenv

load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def search_youtube_videos(query, max_results=2):
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()
    
    if 'items' not in data:
        print(f"❌ Error: {data}")
        return []

    videos = []
    for item in data['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        thumbnail = item['snippet']['thumbnails']['high']['url']
        videos.append({
            "title": title,
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "thumbnail": thumbnail
        })

    return videos

# 🔍 Test it with weak topics
weak_topics = ["PCA", "AI agents", "Neural Networks"]

for topic in weak_topics:
    print(f"\n--- {topic} ---")
    videos = search_youtube_videos(topic)
    for v in videos:
        print(f"🎬 {v['title']}")
        print(f"🔗 {v['url']}")
        print(f"🖼️ {v['thumbnail']}")
