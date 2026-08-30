"""
Lunaite Architecture — Internet & Live Information Tools
========================================================
Provides real-time information retrieval tools:
- DuckDuckGo live web search
- Clean webpage/article scraper
- Wikipedia knowledge lookup
- Open-Meteo high-accuracy global weather

Author: Swasthik Shetty <swasthik.mk3@gmail.com>
License: MIT
"""

import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional


def web_search(query: str, max_results: int = 5) -> str:
    """Perform live web search and return ranked factual snippets with URL citations."""
    clean_query = query.strip()
    results = []

    # 1. DuckDuckGo Instant Answer API (fast, clean structured facts)
    try:
        ia_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_query)}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(ia_url, headers={"User-Agent": "Lunaite/3.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            abstract = data.get("AbstractText", "")
            heading = data.get("Heading", "")
            source_url = data.get("AbstractURL", "")
            if abstract:
                results.append(f"[Instant Answer: {heading}]\nURL: {source_url}\n{abstract}")
    except Exception:
        pass

    # 2. DuckDuckGo Live HTML Search (real-time news and full web results)
    try:
        encoded = urllib.parse.quote(clean_query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        snippets = re.findall(r'<a class="result__snippet[^>]*>(.*?)</a>', html, re.DOTALL)
        titles = re.findall(r'<h2 class="result__title">.*?<a[^>]*>(.*?)</a>', html, re.DOTALL)
        urls = re.findall(r'<a class="result__url"[^>]*href="([^"]*)"', html, re.DOTALL)

        for i in range(min(len(snippets), max_results)):
            clean_snippet = re.sub(r'<.*?>', '', snippets[i]).strip()
            clean_title = re.sub(r'<.*?>', '', titles[i]).strip() if i < len(titles) else f"Result {i+1}"
            raw_url = urls[i].strip() if i < len(urls) else ""

            if "uddg=" in raw_url:
                m = re.search(r'uddg=([^&]+)', raw_url)
                if m:
                    raw_url = urllib.parse.unquote(m.group(1))

            if clean_snippet and clean_snippet not in [r for r in results]:
                results.append(f"[{len(results)+1}] {clean_title}\n    URL: {raw_url}\n    Snippet: {clean_snippet}")

    except Exception as e:
        if not results:
            results.append(f"[Live Web Search Notice]: Search query executed for '{clean_query}' ({e})")

    # 3. Wikipedia Fallback Lookup if search results are sparse
    if len(results) < 2:
        try:
            wiki_res = wiki_lookup(clean_query)
            if wiki_res and "No Wikipedia extract" not in wiki_res:
                results.append(f"\n[Wikipedia Background]:\n{wiki_res}")
        except Exception:
            pass

    if results:
        return "\n\n".join(results)

    return f"Searched the web for '{clean_query}'. No direct results found."


def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch live web page, extract readable article text, and strip HTML noise."""
    try:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        html = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<header.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)

        text = re.sub(r'<.*?>', ' ', html)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [Truncated {len(text)-max_chars} additional characters from {url}]"

        return f"**Live Webpage Content ({url}):**\n\n{text}"
    except Exception as e:
        return f"[URL Fetch Error for {url}]: {e}"


def wiki_lookup(topic: str) -> str:
    """Fetch structured summary from Wikipedia REST API."""
    try:
        clean_topic = re.sub(r'^(?:who\s+was|who\s+is|what\s+is|tell\s+me\s+about)\s+', '', topic, flags=re.IGNORECASE).strip("?.! ")
        encoded = urllib.parse.quote(clean_topic)
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"
        headers = {"User-Agent": "LunaiteAI-Architecture/3.0 (Swasthik Shetty)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            title = data.get("title", topic)
            extract = data.get("extract", "")
            url_link = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            return f"**Wikipedia: {title}**\nURL: {url_link}\n\n{extract}" if extract else f"No Wikipedia extract for '{topic}'."
    except Exception as e:
        return f"[Wikipedia Error]: {e}"


def fetch_weather(location: str = "London") -> str:
    """Fetch live weather forecast using Open-Meteo Geocoding & Weather API."""
    loc = location.strip() or "London"
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(loc)}&count=1&language=en&format=json"
        req = urllib.request.Request(geo_url, headers={"User-Agent": "LunaiteAI-Architecture/3.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            geo_data = json.loads(resp.read().decode("utf-8"))

        if "results" in geo_data and len(geo_data["results"]) > 0:
            top = geo_data["results"][0]
            lat = top.get("latitude")
            lon = top.get("longitude")
            place_name = top.get("name", loc)
            country = top.get("country", "")
            admin = top.get("admin1", "")

            w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            w_req = urllib.request.Request(w_url, headers={"User-Agent": "LunaiteAI-Architecture/3.0"})
            with urllib.request.urlopen(w_req, timeout=6) as w_resp:
                w_data = json.loads(w_resp.read().decode("utf-8"))

            cw = w_data.get("current_weather", {})
            temp_c = cw.get("temperature", "N/A")
            temp_f = round(temp_c * 9/5 + 32, 1) if isinstance(temp_c, (int, float)) else "N/A"
            wind_km = cw.get("windspeed", "N/A")
            wcode = cw.get("weathercode", 0)

            wmo_map = {
                0: "Clear Sky", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
                45: "Foggy", 48: "Depositing Rime Fog",
                51: "Light Drizzle", 53: "Moderate Drizzle", 55: "Dense Drizzle",
                61: "Slight Rain", 63: "Moderate Rain", 65: "Heavy Rain",
                71: "Slight Snow", 73: "Moderate Snow", 75: "Heavy Snow",
                80: "Slight Rain Showers", 81: "Moderate Rain Showers", 82: "Violent Rain Showers",
                95: "Thunderstorm", 96: "Thunderstorm with Slight Hail", 99: "Thunderstorm with Heavy Hail"
            }
            condition = wmo_map.get(wcode, "Clear")
            location_str = f"{place_name}, {admin + ', ' if admin else ''}{country}".strip(", ")

            return (
                f"**Live Weather for {location_str}:**\n"
                f"• Condition: {condition}\n"
                f"• Temperature: {temp_c}°C ({temp_f}°F)\n"
                f"• Wind Speed: {wind_km} km/h"
            )
    except Exception:
        pass

    return f"Unable to retrieve live weather for '{location}' currently."
