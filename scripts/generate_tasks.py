"""Generate benchmark tasks targeting underserved MCP server categories.

Uses GPT-4o-mini to generate realistic one-shot tasks that exercise servers
in categories with insufficient existing tasks (code, visualization,
entertainment, discovery, geolocation, travel, finance, music, documents).
"""

import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Category → (target count, description, example tools for the prompt)
CATEGORIES = {
    "code": {
        "count": 25,
        "description": "Software development, package management, library documentation, code generation",
        "example_tools": [
            "resolve-library-id / query-docs (Context7 — library documentation lookup)",
            "get_github_trending_repositories / get_github_trending_developers (GitHub Trending)",
            "validateMermaid (Mermaid diagram validation)",
            "get-component-docs / list-components (Ant Design component docs)",
            "getUIComponents / getAnimations (Magic UI design components)",
            "npmVersions / npmLatest / npmDeps / npmSize (NPM package info)",
            "get_latest_release / list_maven_versions (Maven dependency lookup)",
            "get-problem / search-problems / get-user-profile (LeetCode problems)",
        ],
    },
    "visualization": {
        "count": 20,
        "description": "Charts, diagrams, presentations, icons, image processing",
        "example_tools": [
            "generate_bar_chart / generate_area_chart / generate_boxplot_chart (Chart generation)",
            "convert_markdown_to_mindmap (Mindmap creation)",
            "list_icons / search_icons (Hugeicons icon search)",
            "create_presentation / get_presentation_info (PowerPoint creation)",
            "drawing_generateCanvas / drawing_fillRectangle (Drawing/painting)",
            "extract_image_from_url / extract_image_from_file (Image extraction)",
        ],
    },
    "entertainment": {
        "count": 15,
        "description": "Games, fortune telling, randomization, leisure activities",
        "example_tools": [
            "getBaziDetail / getChineseCalendar (Chinese fortune telling / Bazi)",
            "get_character_info / get_artifact_info (Wuthering Waves game data)",
            "lol_search_champion_meta / lol_get_champion_analysis / lol_get_lane_matchup_guide (League of Legends stats)",
            "random_int / random_choices / random_shuffle (Random number generation)",
        ],
    },
    "discovery": {
        "count": 40,
        "description": "Information retrieval, news, recipes, academic papers, museums, weather, domain lookups",
        "example_tools": [
            "mcp_howtocook_getAllRecipes / mcp_howtocook_recommendMeals (Chinese recipe lookup)",
            "get_current_time / convert_time (Time zone queries)",
            "get-36kr-trending / get-bbc-news / get-bilibili-rank (Trending news aggregation)",
            "get_transcript (YouTube video transcript extraction)",
            "search / article_searcher / article_getter (Biomedical paper search — BioMCP)",
            "search_wikipedia / get_article (Wikipedia article lookup)",
            "package_search / package_show (Data.gov dataset search)",
            "search / getStory / getStoryWithComments (Hacker News stories)",
            "whois_domain / whois_ip (WHOIS domain/IP lookups)",
            "search_cards / get_card_by_id (Yu-Gi-Oh card database)",
            "search_papers / get_paper_data (ArXiv paper search)",
            "search-museum-objects / get-museum-object / list-departments (Met Museum collection)",
            "get_weather / get_weather_by_city (Weather forecasts)",
            "deepwiki_fetch (Deepwiki repository documentation)",
            "fetch_html / fetch_markdown (Web content fetching)",
        ],
    },
    "geolocation": {
        "count": 35,
        "description": "Maps, geocoding, routing, nearby places, neighborhood analysis, GIS",
        "example_tools": [
            "geocode_address / reverse_geocode (Address ↔ coordinates conversion)",
            "find_nearby_places / search_category (POI search near location)",
            "get_route_directions (Turn-by-turn navigation between locations)",
            "suggest_meeting_point (Optimal meeting point for multiple people)",
            "analyze_neighborhood / explore_area (Area livability analysis)",
            "find_schools_nearby / find_ev_charging_stations / find_parking_facilities (Nearby amenity search)",
            "analyze_commute (Commute analysis between home and work)",
            "wkt_to_geojson / geojson_to_wkt / csv_to_geojson (GIS data format conversion)",
            "mcp_geo_calculate_distance / mcp_geo_calculate_area (Geospatial calculations)",
            "get_bounds / search_overpass (Bounding box and OpenStreetMap queries)",
        ],
    },
    "travel": {
        "count": 25,
        "description": "Transportation, accommodation, tickets, bus schedules, theme parks",
        "example_tools": [
            "query-tickets / query-ticket-price / query-transfer (China train ticket search — 12306)",
            "airbnb_search / airbnb_listing_details (Airbnb accommodation search)",
            "getOneDayTicketPrice / getTwoDayTicketPrice (Shanghai Disney ticket prices)",
            "get_timetable / get_approach_for_station (Nagoya bus schedules)",
            "get_forecast (Japanese weather forecast for travel planning)",
        ],
    },
    "finance_extra": {
        "count": 20,
        "description": "Stock market, crypto, exchange rates, market analysis, financial data",
        "example_tools": [
            "get_current_stock_price / get_stock_price_date_range / get_dividends (Yahoo Finance stock data)",
            "load_all_tickers / get_stock_ohlcv / get_stock_fundamental (Korean KOSPI/KOSDAQ market data)",
            "get_asset_price / list_assets (Asset price lookup)",
            "yfinance_get_ticker_info / yfinance_get_ticker_news / yfinance_get_top (Yahoo Finance ticker info)",
            "get_market_movers / get_cnn_fear_greed_index / get_crypto_fear_greed_index (Market sentiment)",
            "exchange_rate (Currency exchange rate conversion)",
            "get-crypto-price / get-market-analysis / get-historical-analysis (Crypto price and analysis)",
            "get_hist_data / get_realtime_data / get_balance_sheet (China A-share stock data)",
        ],
    },
    "music_misc": {
        "count": 10,
        "description": "Music analysis, calculations, barcodes, refund checking, document processing",
        "example_tools": [
            "load / get_duration / tempo / chroma_cqt / mfcc (Audio/music analysis)",
            "calculate (Mathematical calculations)",
            "decode_image / generate_qr / generate_barcode (Barcode/QR code operations)",
            "refund_eligibility (Consumer refund eligibility check)",
            "create_document / get_document_text (Word document operations)",
            "read_pdf / pdf_merger / pdf_splitter (PDF operations)",
        ],
    },
}

SYSTEM_PROMPT = """You are a benchmark task generator for an MCP (Model Context Protocol) tool recommendation system.

Generate realistic, diverse, one-shot questions that a user might ask an AI assistant that would require using external tools to answer. Each question should be answerable with 1-3 tool calls.

Rules:
- Questions should be natural language, like a real user would ask
- Questions should be specific enough to have a concrete answer
- Questions should NOT reference any specific tool name or API
- Questions should be varied — different phrasings, different sub-topics within the category
- Each question should be independent (not building on previous questions)
- Mix difficulty: some simple lookups, some requiring combining information
- Include questions that require real-time or current data where applicable
- Do NOT include questions that need local file access or user-specific data

Output format: JSON array of objects with fields:
- "query": the question text
- "call_type": "single" if answerable with 1 tool call, "multi" if 2-3 needed
- "subcategory": a short label for the sub-topic (e.g., "stock_price", "recipe_search")
"""


def generate_for_category(client: OpenAI, category: str, info: dict) -> list[dict]:
    """Generate tasks for one category."""
    user_prompt = f"""Generate exactly {info['count']} benchmark questions for the category: {category}

Category description: {info['description']}

These are the kinds of tools available (for context only — do NOT mention tool names in questions):
{chr(10).join(f"  - {t}" for t in info['example_tools'])}

Generate {info['count']} diverse questions. Return a JSON array."""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        max_tokens=4000,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    parsed = json.loads(content)

    # Handle various JSON formats the model might return
    if isinstance(parsed, list):
        tasks = parsed
    elif isinstance(parsed, dict):
        # Try common keys
        for key in ("tasks", "questions", "data", "results", "items", "benchmarks"):
            if key in parsed and isinstance(parsed[key], list):
                tasks = parsed[key]
                break
        else:
            # Use first list value found
            tasks = []
            for v in parsed.values():
                if isinstance(v, list):
                    tasks = v
                    break
    else:
        tasks = []

    # Normalize and add metadata
    result = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        # Find the query text — try multiple field names
        query = t.get("query") or t.get("question") or t.get("text") or t.get("prompt") or ""
        if not query:
            continue
        call_type = t.get("call_type", "single")
        if call_type not in ("single", "multi"):
            call_type = "single"
        result.append({
            "uuid": str(uuid.uuid4()),
            "category": category if category != "finance_extra" else "finance",
            "call_type": call_type,
            "source": "generated",
            "query": query,
            "subcategory": t.get("subcategory", ""),
        })

    print(f"  {category}: {len(result)} tasks generated (requested {info['count']})")
    return result


def main():
    client = OpenAI()

    all_tasks = []
    for category, info in CATEGORIES.items():
        print(f"Generating {info['count']} tasks for {category}...")
        tasks = generate_for_category(client, category, info)
        all_tasks.extend(tasks)

    print(f"\nTotal generated: {len(all_tasks)} tasks")

    # Category breakdown
    from collections import Counter
    cats = Counter(t["category"] for t in all_tasks)
    print(f"By category: {dict(cats)}")

    # Save
    out = ROOT / "data" / "tasks_generated.json"
    with open(out, "w") as f:
        json.dump(all_tasks, f, indent=2)
    print(f"Saved to {out}")

    # Also rebuild the combined task file
    print("\nRebuilding combined task file...")

    # Load existing MCPToolBench++ tasks
    existing_tasks = []
    task_files = {
        "search": ROOT / "data" / "search" / "search_0725_single_v2.json",
        "browser": ROOT / "data" / "browser" / "browser_0724_single_v3.json",
        "finance": ROOT / "data" / "finance" / "finance_0724_single_v3.json",
    }
    for domain, path in task_files.items():
        if path.exists():
            with open(path) as f:
                tasks = json.load(f)
            for t in tasks:
                t["source"] = "mcptoolbench"
                if "query" not in t:
                    t["query"] = t.get("question", t.get("instruction", ""))
            existing_tasks.extend(tasks)
            print(f"  {domain}: {len(tasks)} existing tasks")

    # Load LiveMCPBench tasks
    lmcb_path = ROOT / "data" / "tasks_livemcpbench.json"
    if lmcb_path.exists():
        with open(lmcb_path) as f:
            lmcb_tasks = json.load(f)
        existing_tasks.extend(lmcb_tasks)
        print(f"  livemcpbench: {len(lmcb_tasks)} tasks")

    combined = existing_tasks + all_tasks
    cats_combined = Counter(t.get("category", "unknown") for t in combined)
    sources = Counter(t.get("source", "unknown") for t in combined)
    print(f"\nCombined: {len(combined)} tasks")
    print(f"By source: {dict(sources)}")
    print(f"By category: {dict(cats_combined)}")

    out_combined = ROOT / "data" / "tasks_combined.json"
    with open(out_combined, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"Saved to {out_combined}")


if __name__ == "__main__":
    main()
