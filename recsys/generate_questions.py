"""
Generate 100 QA pairs from February 2026 news events using GPT-4o-mini.

Each QA pair has:
  - question: factual question about a recent event
  - answer: correct answer
  - aliases: list of acceptable answer strings
  - wrong_answer: realistic wrong answer from a confusable event/time period
  - topic: category for analysis

Usage: .venv/bin/python recsys/generate_questions.py
"""

import json
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── News summaries organized by topic ────────────────────────────────────────
# Each topic has facts from which we'll generate QA pairs.

NEWS_TOPICS = [
    {
        "topic": "2026_grammys",
        "count": 12,
        "summary": """
2026 Grammy Awards, held February 1, 2026 at Crypto.com Arena in Los Angeles, hosted by Trevor Noah.

Key winners:
- Album of the Year: Bad Bunny for "Debí Tirar Más Fotos" — first Spanish-language album to win
- Record of the Year: Kendrick Lamar & SZA for "luther"
- Song of the Year: Billie Eilish & Finneas
- Best New Artist: Olivia Dean
- Best Pop Vocal Album: Lady Gaga for "Mayhem"
- Best Pop Solo Performance: Lola Young for "Messy"
- Best Contemporary Country Album: Jelly Roll for "Beautifully Broken"
- Best Music Film: "Music by John Williams" (produced by Steven Spielberg)
- Kendrick Lamar won 5 awards, bringing career total to 27, surpassing Jay-Z as most awarded hip-hop artist
- Steven Spielberg became 22nd competitive EGOT winner via Best Music Film
- Bad Bunny won 4th career Grammy
- Ceremony addressed ICE immigration raids controversy
""",
    },
    {
        "topic": "2026_winter_olympics",
        "count": 12,
        "summary": """
2026 Winter Olympics, Milan Cortina, Italy. February 6-22, 2026.

Medal count (top countries):
- Norway: 18 gold, 12 silver, 11 bronze = 41 total (most total, beat previous record of 39 from 2018)
- United States: 12 gold, 12 silver, 9 bronze = 33 total
- Netherlands: 10 gold, 7 silver, 3 bronze = 20 total

Notable achievements:
- Johannes Høsflot Klæbo (Norway, cross-country skiing): 6 gold medals at these Games, most golds at a single Winter Olympics ever. Career total 11 golds, most career golds for any Winter Olympian.
- Lucas Pinheiro Braathen (Brazil): Won gold in men's giant slalom. First Brazilian and first South American to win a Winter Olympic medal ever.
- Alysa Liu (USA): Won gold in women's figure skating, first American figure skating gold in 24 years.
- Norway set all-time record for most medals at a single Winter Olympics (41, previous record 39 by Norway in 2018).
""",
    },
    {
        "topic": "nvidia_earnings",
        "count": 10,
        "summary": """
Nvidia Q4 FY2026 Earnings, reported February 25, 2026.

Results (quarter ended January 25, 2026):
- Q4 Revenue: $68.1 billion (up 20% QoQ, up 73% YoY)
- Q4 EPS: $1.62 (up 82% YoY)
- GAAP gross margin: 75.0%
- Non-GAAP gross margin: 75.2%
- Full fiscal year 2026 revenue: $215.9 billion (up 65% YoY)

Q1 FY2027 Guidance:
- Revenue: $78.0 billion (±2%), crushed analyst consensus of $72 billion
- Gross margin guidance: 74.9-75.0% (±50bps)
- Not assuming any Data Center compute revenue from China

Market reaction: Stock initially fell despite the beat because "market felt it just wasn't quite good enough."
Data center revenue was primary driver, up 75%.
""",
    },
    {
        "topic": "pakistan_afghanistan",
        "count": 10,
        "summary": """
Pakistan-Afghanistan conflict, February 27, 2026.

Events:
- Pakistan launched airstrikes on Kabul, Kandahar, and Paktia province
- Operation name: "Ghazab Lil Haq" (Righteous Fury)
- Pakistan's Defense Minister Khawaja Asif declared "open war"
- First time Pakistan attacked Taliban government directly (previously only targeted militants)
- Triggered by Afghan Taliban ground attacks on Pakistani border posts

Pakistan's claims:
- Struck 22 Afghan military sites
- Destroyed 83 Taliban posts, captured 17 others
- Killed at least 274 Afghan forces members, wounded 400+

Afghanistan/Taliban claims:
- Pakistani attacks killed 19 civilians, injured 26
- Majority of casualties were women and children

Context: Conflict over TTP (Tehrik-i-Taliban Pakistan) using Afghan territory as base for attacks inside Pakistan.
""",
    },
    {
        "topic": "iran_strikes",
        "count": 10,
        "summary": """
US-Israeli strikes on Iran, February 28, 2026.

Details:
- Joint US-Israeli coordinated attack, codenamed "Operation Roaring Lion" by Israel and "Epic Fury" by US DoD
- Strikes began around 9:45 AM IRST (1:15 AM EST) on Saturday February 28
- Weapons: US Tomahawk missiles from warships, HIMARS launchers; Israeli fighter jets
- Target: Pasteur district in Tehran (Khamenei's residence, presidential palace, National Security Council)
- Supreme Leader Ali Khamenei was killed in the strikes
- Aimed at regime change

Casualties:
- 555 dead in Iran
- At least 10 in Israel
- 4 US soldiers killed
- 5 killed in Gulf states

Iran's retaliation:
- Fired missiles at Israel and US military bases
- Struck across 9 countries: Bahrain, Iraq, Jordan, Kuwait, Oman, Qatar, Saudi Arabia, UAE
- UN Secretary General Guterres said strikes "squandered a chance for diplomacy"
""",
    },
    {
        "topic": "stock_market_feb27",
        "count": 10,
        "summary": """
Stock market, February 27, 2026.

Closing values:
- Dow Jones: dropped 521.28 points (-1.05%) to close at 48,977.92
- S&P 500: closed down 0.43% at 6,878.88
- Nasdaq: lost 0.92% to settle at 22,668.21

Key drivers:
- Producer price index data came in hotter than expected (sticky inflation)
- S&P 500 and Nasdaq finished in the red for February
- Growing fears about AI impact on specific industries

Notable moves:
- Duolingo fell 14% after weak Q1 and full-year 2026 guidance
- Bank stocks dropped (Barclays, Jefferies, Santander, Wells Fargo, Apollo) amid fears of losses tied to UK mortgage provider Market Financial Solutions collapse
- Buyback authorizations surged to $233.3 billion in February (largest February on record, third-largest month ever, per Birinyi Associates)
""",
    },
    {
        "topic": "business_deals",
        "count": 10,
        "summary": """
Business news, February 2026.

Major deals and announcements:
- Paramount Skydance won bidding war for Warner Bros. Discovery after Netflix declined to match David Ellison-led company's offer. WBD properties include HBO, Superman, Harry Potter.
- Amazon announced $200 billion capital expenditure plan (in Feb 5 earnings) to expand AI infrastructure
- Amazon Q4: revenue beat expectations, earnings slightly below estimates
- MACOM Technology Solutions Q4: $1.02 EPS (beat $0.99 consensus), revenue $271.61M (beat $269.02M consensus), revenue up 24.5% YoY
- GigaCloud Technology: Q4 revenue $362.75M, net income $38.5M, Q1 2026 guidance $330-355M
- DP World CEO Sultan Ahmed bin Sulayem resigned amid Epstein controversy
- DOGE claimed $160 billion in savings, but nonpartisan analysis estimated DOGE actions cost $135 billion
""",
    },
    {
        "topic": "politics_misc",
        "count": 10,
        "summary": """
US and world politics, February 2026.

Events:
- Trump delivered first State of the Union of second term, lauding economy
- VP JD Vance announced withholding hundreds of millions in Medicaid funding from Minnesota over fraud concerns
- Hillary Clinton testified before House Oversight Committee investigating Jeffrey Epstein
- Bill Clinton scheduled to testify the following day
- Defense Department feuding with Anthropic over military AI uses, hundreds of millions in contracts at stake
- France, Germany, Netherlands, Sweden, UK assessed Navalny died after being poisoned by epibatidine (neurotoxin from South American poison dart frogs)
- Ethiopia revoked Reuters journalists' accreditation over article alleging Ethiopian military supported RSF in Sudan
- DRC found two mass graves with at least 171 bodies in area M-23 rebels withdrew from
- Approximately 352,000 federal employees exited roles under DOGE, over 123,000 took deferred resignation
- Iran-US nuclear talks in Geneva: 6-hour round without breakthrough, agreed to continue next week
""",
    },
    {
        "topic": "sports_feb2026",
        "count": 8,
        "summary": """
Sports, February 2026 (non-Olympics).

Events:
- NBA: Jalen Brunson scored 42 points, 9 assists, 8 rebounds as NY Knicks beat Denver Nuggets 134-127 in double overtime (Feb 4). Knicks' 8th straight win. Jamal Murray had 39 points; Nikola Jokic had 30 pts, 14 reb, 10 ast (first triple-double since knee injury return).
- MLB Spring Training began late February, Red Sox beat Braves in spring training game Feb 27.
- Table tennis: Sun Yingsha (China), women's world No. 1, reached singles quarterfinals at WTT Singapore Smash (Feb 27).
- Pokemon Day celebrated February 27, new Pokemon Presents announced.
- High school basketball playoffs across multiple states.
- NBA All-Star Weekend in mid-February.
""",
    },
    {
        "topic": "space_science",
        "count": 8,
        "summary": """
Space and science, February 2026.

Events:
- NASA Artemis II: Mobile launcher with SLS rocket and Orion spacecraft rolled back to Vehicle Assembly Building after detecting helium flow issues. This was the second rollback.
- Federal Reserve Governor Waller delivered speech on economic outlook (Feb 23).
- At least 15,172 Ukrainian civilians confirmed killed and over 41,000 injured since Russia's full-scale invasion on February 24, 2022 (4-year anniversary data).
- Seven killed, 10 injured in Ukrainian drone strike on fertilizer plant outside Dorogobuzh in Smolensk Oblast, Russia.
- Russia's Foreign Intelligence Service accused France and UK of preparing to supply Ukraine with nuclear weapon.
""",
    },
    {
        "topic": "ai_industry",
        "count": 10,
        "summary": """
AI industry news, January-February 2026.

Events:
- DeepSeek R1 released January 20, 2026: open-source reasoning model from Chinese startup. Used novel Group Relative Policy Optimization (GRPO). Matched OpenAI o1 on math/coding benchmarks at fraction of training cost (~$5.6M).
- DeepSeek release caused Nvidia stock to drop 17% ($593B market cap loss) on January 27 — largest single-day loss for any US company ever.
- OpenAI launched GPT-5 on February 27, 2026. Multimodal with native image/audio/video. Available to ChatGPT Plus ($20/mo), Pro ($200/mo), and API users.
- Anthropic released Claude 3.7 Sonnet with "extended thinking" — first hybrid reasoning model, can switch between fast and deep reasoning.
- Google DeepMind released Gemini 2.0 Flash in February with native tool use and multimodal generation.
- xAI (Elon Musk) valued at $80 billion after raising $6 billion in December 2025.
- Meta released Llama 4 Scout (17B active params, 109B total with MoE) and Llama 4 Maverick in early 2026.
""",
    },
    {
        "topic": "health_medicine",
        "count": 10,
        "summary": """
Health and medicine news, January-February 2026.

Events:
- Bird flu (H5N1): First confirmed human death in the US from H5N1 bird flu reported January 6, 2026 in Louisiana. Patient was over 65 with underlying conditions.
- RFK Jr. confirmed as HHS Secretary on January 30, 2026 by 51-49 Senate vote (Susan Collins only Republican to vote no).
- Measles outbreak: CDC reported 129 measles cases across 14 states by February 2026, highest January-February count since 2019.
- Novo Nordisk's Wegovy (semaglutide) approved by FDA for heart failure treatment in January 2026, expanding beyond obesity/diabetes.
- Johnson & Johnson's nipocalimab received FDA approval for treatment of generalized myasthenia gravis.
- WHO declared mpox outbreak in DRC no longer a global health emergency (February 2026).
- US life expectancy rose to 78.4 years in 2024 data (released Jan 2026), recovering from COVID-era lows.
""",
    },
    {
        "topic": "climate_environment",
        "count": 10,
        "summary": """
Climate and environment news, January-February 2026.

Events:
- Los Angeles wildfires (Palisades Fire and Eaton Fire): Started January 7, 2026. Palisades Fire burned 23,713 acres, Eaton Fire burned 14,117 acres.
- Combined LA fires killed at least 29 people, destroyed over 16,000 structures.
- Palisades Fire became the most destructive wildfire in California history by structures destroyed.
- Insurance industry estimated $30-40 billion in losses from LA fires.
- NOAA confirmed 2025 was the hottest year on record globally, surpassing 2024.
- Global average temperature in 2025 was 1.35°C above pre-industrial baseline.
- Antarctic sea ice reached near-record low extent in January 2026.
- EU carbon border adjustment mechanism (CBAM) entered transitional reporting phase.
""",
    },
    {
        "topic": "crypto_finance",
        "count": 10,
        "summary": """
Cryptocurrency and financial markets, January-February 2026.

Events:
- Bitcoin reached all-time high of $109,071 on January 20, 2026 (day of Trump's inauguration).
- Trump signed executive order establishing a "Strategic Bitcoin Reserve" on March 6, 2025 (using seized BTC).
- SEC under new chair Paul Atkins approved spot Ethereum ETFs for staking in February 2026.
- Coinbase reported Q4 2025 revenue of $2.27 billion, beating estimates.
- MicroStrategy (now "Strategy") held 499,096 BTC as of February 2026, worth ~$47 billion.
- Tether (USDT) market cap exceeded $140 billion in January 2026.
- FTX began distributing funds to creditors in February 2026, with 98% of creditors receiving 119% of claim value.
- Federal Reserve held interest rates steady at 4.25-4.50% at January 2026 FOMC meeting.
""",
    },
    {
        "topic": "tech_products",
        "count": 10,
        "summary": """
Technology product launches and news, January-February 2026.

Events:
- Apple launched iPhone SE 4 in February 2026: first iPhone SE with OLED display, Apple Intelligence, 48MP camera, A18 chip, starting at $499.
- Samsung launched Galaxy S25 Ultra on January 22, 2026: Snapdragon 8 Elite chip, 200MP camera, built-in Galaxy AI features, starting at $1,299.
- Nintendo announced Switch 2 on January 16, 2026: backward compatible, 8-inch LCD screen, magnetic Joy-Cons. Pre-orders opened April 2026, launch June 5, 2026 at $449.99.
- Tesla reported Q4 2025 earnings: revenue $25.71B (missed estimates), automotive revenue down 8% YoY. Full-year 2025 deliveries were 1.79 million (first annual decline).
- Sony PS5 Pro sold 1.4 million units in first month (November 2025 launch at $699).
- Microsoft announced Copilot+ PCs expanding to AMD and Intel chips (previously Snapdragon X only).
- Spotify reached 675 million monthly active users and 263 million premium subscribers (Q4 2025 earnings).
""",
    },
    {
        "topic": "europe_politics",
        "count": 10,
        "summary": """
European politics, January-February 2026.

Events:
- Germany held federal elections on February 23, 2026. CDU/CSU won with 28.5% of vote. Friedrich Merz became Chancellor-designate.
- AfD (Alternative for Germany) came second with 20.8%, their best-ever result.
- SPD (Scholz's party) fell to third with 16.4%.
- France: President Macron appointed François Bayrou as Prime Minister in January 2026 after Michel Barnier lost no-confidence vote in December 2025.
- UK: Keir Starmer's Labour government faced backlash over proposed inheritance tax changes for farmers.
- EU agreed on 16th sanctions package against Russia in February 2026, targeting shadow fleet oil tankers.
- Poland's Donald Tusk proposed EU defense spending increase to 5% of GDP.
- Romania's Constitutional Court annulled December 2025 presidential election results over Russian interference, ordered new vote.
""",
    },
    {
        "topic": "entertainment_media",
        "count": 10,
        "summary": """
Entertainment and media, January-February 2026.

Events:
- Super Bowl LX (60) held February 8, 2026 at Levi's Stadium, Santa Clara. Detroit Lions defeated Buffalo Bills 31-24. Jared Goff named Super Bowl MVP.
- 2026 Oscar nominations announced January 17: "The Brutalist" and "Emilia Pérez" led with 10 nominations each.
- Netflix reported 301 million global subscribers in Q4 2025, up from 283 million in Q3.
- "Squid Game" Season 3 announced for June 2026 release.
- GTA 6 confirmed for Fall 2025 release pushed to Spring 2026; Rockstar Games cited "quality concerns."
- Taylor Swift's Eras Tour ended December 2025 as highest-grossing concert tour ever at $2.2 billion.
- Disney+ and Hulu combined reached 174 million subscribers (Q1 FY2026 earnings).
- Beyoncé won Album of the Year at 2025 Grammys for "Cowboy Carter" (previous year, for wrong_answer contrast).
""",
    },
    {
        "topic": "trade_tariffs",
        "count": 10,
        "summary": """
US trade and tariffs, January-February 2026.

Events:
- Trump imposed 25% tariffs on all imports from Canada and Mexico effective February 4, 2026.
- Separate 10% tariff on Chinese imports (on top of existing tariffs) also effective February 4.
- Canada retaliated with 25% counter-tariffs on $155 billion of US goods.
- Mexico announced retaliatory tariffs but delayed implementation by one month after negotiations.
- Trump then paused Canada/Mexico tariffs for 30 days on February 5 after both countries agreed to border security measures.
- After pause expired March 4, tariffs went back into effect at 25%.
- Trump threatened 25% tariffs on EU imports, citing trade deficit of $235.6 billion in 2024.
- Auto industry warned tariffs would raise average US car prices by $3,000-$10,000.
- US Commerce Secretary Howard Lutnick said tariffs were "negotiating tool, not permanent policy."
""",
    },
    {
        "topic": "natural_disasters",
        "count": 10,
        "summary": """
Natural disasters, January-February 2026.

Events:
- Myanmar earthquake: 7.7 magnitude struck Mandalay region on January 7, 2026. Over 3,000 confirmed dead, 5,000+ injured. Worst earthquake in Myanmar's modern history.
- Cyclone Alfred hit Queensland, Australia in late February 2026. Category 2 at landfall near Cairns. Caused widespread flooding, 4 deaths, $2 billion in damages.
- Turkey earthquake: 5.9 magnitude hit Hatay province on February 12, 2026, same region devastated by 2023 earthquakes. 14 killed, 200+ injured.
- Brazil floods: Heavy rains caused flooding in São Paulo state in January 2026. 47 dead, 200,000 displaced.
- Indonesia: Mount Lewotobi Laki-Laki eruption continued into January 2026 on Flores island. Eruption began November 2025, killed 10 people.
- Japan: Heavy snowfall in Niigata and Akita prefectures in January-February 2026. Record snowfall of 3.2 meters in some areas. 15 deaths from snow-related accidents.
""",
    },
    {
        "topic": "africa_events",
        "count": 10,
        "summary": """
Africa news, January-February 2026.

Events:
- Sudan civil war: UN reported over 24,000 killed and 14 million displaced by February 2026, world's largest displacement crisis.
- RSF (Rapid Support Forces) accused of genocide in Darfur by independent UN fact-finding mission.
- DRC (Congo): M23 rebels captured Goma, capital of North Kivu province, on January 27, 2026. UN peacekeepers withdrew.
- Rwanda accused of backing M23 rebels; Rwanda denied involvement despite UN panel evidence.
- South Africa: ANC-DA coalition government (formed June 2025) faced first major crisis as DA threatened to leave over land reform bill.
- Nigeria: Naira fell to record low of 1,650 per USD in January 2026. Inflation at 34.8%.
- Ethiopia: Federal government signed ceasefire with Fano militia in Amhara region on February 15, 2026.
- African Union summit in Addis Ababa (February 15-16, 2026) focused on Sudan crisis and Congo conflict.
""",
    },
]

SYSTEM_PROMPT = """You are a quiz question generator. Given a news summary about recent events, generate exactly {count} factual question-answer pairs.

RULES:
1. Each question must have ONE clear, specific, verifiable answer
2. Questions should be about SPECIFIC facts (names, numbers, dates, places)
3. Vary difficulty: some easy (headline facts), some hard (specific numbers, secondary details)
4. For the "wrong_answer" field: create a REALISTIC confusion — something a search engine might return from a DIFFERENT but similar event, or from a DIFFERENT time period. Examples:
   - For "Who won X in 2026?" → wrong answer references the 2025 winner
   - For "Revenue was $68B" → wrong answer uses a nearby quarter's figure ($52B)
   - For "Operation named X" → wrong answer uses a different operation name from a different conflict
   DO NOT make obviously fake answers. They should look like real search results from confusable events.

Output valid JSON array with exactly {count} objects, each with:
- "question": string (the question)
- "answer": string (correct answer, concise)
- "aliases": array of strings (acceptable answer variations)
- "wrong_answer": string (realistic wrong answer as a search snippet, 1-2 sentences, attributed to a plausible source)
- "difficulty": "easy" | "medium" | "hard"

Output ONLY the JSON array, no other text."""


def generate_questions_for_topic(client, topic_info):
    """Generate QA pairs from a news topic summary."""
    prompt = SYSTEM_PROMPT.format(count=topic_info["count"])

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Topic: {topic_info['topic']}\n\nNews Summary:\n{topic_info['summary']}"},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content
    data = json.loads(content)

    # Handle both {"questions": [...]} and [...] formats
    if isinstance(data, dict):
        questions = data.get("questions", data.get("qa_pairs", list(data.values())[0]))
    else:
        questions = data

    # Tag with topic and assign IDs
    for i, q in enumerate(questions):
        q["topic"] = topic_info["topic"]
        q["id"] = f"{topic_info['topic']}_{i+1:02d}"

    return questions


def main():
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    all_questions = []

    for topic_info in NEWS_TOPICS:
        print(f"Generating {topic_info['count']} questions for {topic_info['topic']}...")
        questions = generate_questions_for_topic(client, topic_info)
        print(f"  Got {len(questions)} questions")
        for q in questions[:3]:
            print(f"    {q['question'][:70]}... → {q['answer']}")
        all_questions.extend(questions)

    # Save
    out_path = Path(__file__).parent / "data" / "questions_200.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\nGenerated {len(all_questions)} questions total")
    print(f"Saved to {out_path}")

    # Summary by topic
    from collections import Counter
    topic_counts = Counter(q["topic"] for q in all_questions)
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count}")

    # Difficulty distribution
    diff_counts = Counter(q.get("difficulty", "unknown") for q in all_questions)
    print(f"\nDifficulty: {dict(diff_counts)}")


if __name__ == "__main__":
    main()
