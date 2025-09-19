# app.py — minimal Streamlit app showing only prediction (no table view)
import os
import sys
import uuid
import json
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone
import praw
import joblib

# make sure your text_processing.py is importable
sys.path.append(os.path.join(os.getcwd(), "src", "topic_modeling"))
from text_processing import preprocess_series

# load env
load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
MODEL_PATH = os.getenv("MODEL_PATH", "models/topic_classifier.pkl")

# nice category mapping
CATEGORY_MAP = {
    "alt.atheism": "Atheism / Religion (Debates)",
    "comp.graphics": "Computer Graphics",
    "comp.os.ms-windows.misc": "Windows OS",
    "comp.sys.ibm.pc.hardware": "PC Hardware",
    "comp.sys.mac.hardware": "Mac Hardware",
    "comp.windows.x": "Computer Windows",
    "misc.forsale": "Marketplace / For Sale",
    "rec.autos": "Automobiles",
    "rec.motorcycles": "Motorcycles",
    "rec.sport.baseball": "Baseball",
    "rec.sport.hockey": "Hockey",
    "sci.crypt": "Cryptography / Security",
    "sci.electronics": "Electronics",
    "sci.med": "Medicine / Health",
    "sci.space": "Space / Astronomy",
    "soc.religion.christian": "Christian Religion",
    "talk.politics.guns": "Politics — Guns",
    "talk.politics.mideast": "Politics — Middle East",
    "talk.politics.misc": "Politics — Miscellaneous",
    "talk.religion.misc": "Religion — Miscellaneous"
}

# initialize reddit if creds available
reddit = None
if REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT:
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )
    except Exception:
        reddit = None

def fetch_reddit_post(url):
    if reddit is None:
        raise RuntimeError("Reddit client not configured. Set REDDIT_CLIENT_ID/SECRET/USER_AGENT in .env")
    submission = reddit.submission(url=url)
    return {
        "id": str(uuid.uuid4()),
        "source": "reddit",
        "author": submission.author.name if submission.author else "unknown",
        "timestamp": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
        "text": (submission.title or "") + "\n" + (submission.selftext or ""),
        "metadata": {"language": "en", "likes": submission.score, "url": url}
    }

def fetch_news(query):
    if not NEWS_API_KEY:
        raise RuntimeError("NEWS_API_KEY not configured. Set NEWS_API_KEY in .env")
    url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&apiKey={NEWS_API_KEY}&pageSize=1"
    resp = requests.get(url)
    resp.raise_for_status()
    payload = resp.json()
    if "articles" not in payload or len(payload["articles"]) == 0:
        return None
    a = payload["articles"][0]
    return {
        "id": str(uuid.uuid4()),
        "source": "news",
        "author": a.get("author") or "unknown",
        "timestamp": a.get("publishedAt"),
        "text": (a.get("title") or "") + "\n" + (a.get("description") or ""),
        "metadata": {"language": a.get("language", "en"), "likes": None, "url": a.get("url")}
    }

def save_data(records, format_choice):
    df = pd.json_normalize(records)
    if format_choice == "CSV":
        df.to_csv("output_data.csv", index=False, encoding="utf-8")
        return "Data saved to output_data.csv"
    else:
        with open("output_data.json", "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)
        return "Data saved to output_data.json"

# Streamlit UI
st.set_page_config(page_title="NarrativeNexus — Predict Only", layout="centered")
st.title("NarrativeNexus — Fetch & Predict")
st.write("Paste a Reddit post link, News query, or raw text. The app will fetch, predict the topic, and save the record.")

option = st.radio("Choose Source:", ["Reddit Post", "News Article", "Raw Text"])
user_input = st.text_input("Enter Reddit link, News query, or raw text:")
save_format = st.selectbox("Save as:", ["CSV", "JSON"])

# preload model
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.warning(f"Could not load model at {MODEL_PATH}: {e}")
        model = None

if st.button("Fetch, Predict & Save"):
    records = []
    try:
        if option == "Reddit Post":
            if not user_input.startswith("http"):
                st.error("Please paste a valid Reddit URL starting with http(s)://")
            else:
                records.append(fetch_reddit_post(user_input))

        elif option == "News Article":
            if not user_input.strip():
                st.error("Please enter a search query for NewsAPI")
            else:
                rec = fetch_news(user_input)
                if rec:
                    records.append(rec)
                else:
                    st.error("No news found for this query.")

        else:  # Raw Text
            if not user_input.strip():
                st.error("Please enter some text to predict")
            else:
                records.append({
                    "id": str(uuid.uuid4()),
                    "source": "raw",
                    "author": "user",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "text": user_input,
                    "metadata": {"language": "en", "likes": None, "url": None}
                })

        if not records:
            # nothing to do
            pass
        else:
            raw_text = records[0].get("text", "")
            # handle empty or deleted content
            if not raw_text or raw_text.strip().lower() in ("[deleted]", "[removed]"):
                nice_label = "No content (deleted/removed)"
            elif model is None:
                nice_label = "Model not available"
            else:
                processed = preprocess_series([raw_text])
                try:
                    raw_pred = model.predict(processed)[0]
                    nice_label = CATEGORY_MAP.get(raw_pred, raw_pred)
                except Exception as e:
                    nice_label = f"Prediction error: {e}"

            records[0]["predicted_category"] = nice_label

            # Save and show only JSON preview + predicted label
            message = save_data(records, save_format)
            st.success(message)
            st.subheader("Preview")
            st.json(records[0])

            st.subheader("Predicted Category")
            st.write(nice_label)

    except Exception as e:
        st.error(f"Error: {e}")

# status
st.markdown("---")
st.write("Model path:", MODEL_PATH)
if model is None:
    st.warning("Trained model not loaded — set MODEL_PATH or place the model at models/topic_classifier.pkl")
else:
    st.success("Trained model loaded — ready to predict.")
