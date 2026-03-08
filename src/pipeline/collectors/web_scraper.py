"""Web-scraping fallback for live race data when OpenF1 requires paid auth.

Scrapes driver positions, race status (SC/VSC/red flag), retirements,
and lap count from publicly accessible F1 results pages. Feeds data
into the same LiveRaceState dataclass used by the OpenF1 collector.

Sources tried in order:
1. Firecrawl (if configured) — handles JS-rendered pages like formula1.com
2. Direct HTML scraping — formulaonehistory.com, The Race (no JS needed)
3. Claude API fallback — fetches raw HTML, uses LLM to extract structured data
"""

from __future__ import annotations

import re
import time

import requests
from bs4 import BeautifulSoup

from config.settings import CLAUDE_API_KEY, FIRECRAWL_API_KEY, FIRECRAWL_API_URL
from src.pipeline.collectors.live_collector import (
    DriverLiveState,
    LiveRaceState,
    RaceControlEvent,
)
from src.utils.logger import logger

# User-agent to avoid blocks
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
    ),
}


class LiveTimingScraper:
    """Scrapes live race positions from public F1 websites.

    Tries multiple sources in order. Each source parser returns a
    partial LiveRaceState — the first successful one wins.
    """

    def __init__(self, season: int, round_num: int, total_laps: int = 0) -> None:
        self.season = season
        self.round_num = round_num
        self.total_laps = total_laps
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self._last_state: LiveRaceState | None = None

    def fetch_live_state(self) -> LiveRaceState | None:
        """Fetch current race state by trying multiple sources.

        Returns:
            LiveRaceState with positions and status, or None if all fail.
        """
        # Try Firecrawl first if configured (handles JS pages)
        if FIRECRAWL_API_URL:
            state = self._try_firecrawl()
            if state and state.drivers:
                self._last_state = state
                return state

        # Try direct HTML scrapers (no JS needed)
        scrapers = [
            ("formulaonehistory.com", self._scrape_formulaonehistory),
            ("the-race.com", self._scrape_the_race),
        ]

        for name, scraper_fn in scrapers:
            try:
                state = scraper_fn()
                if state and state.drivers:
                    logger.info(f"Live data from {name}: {len(state.drivers)} drivers, lap {state.leader_lap}")
                    self._last_state = state
                    return state
            except Exception as e:
                logger.debug(f"Scraper {name} failed: {e}")
                continue

        # Try Claude API as final fallback
        state = self._try_claude_parse()
        if state and state.drivers:
            logger.info(
                f"Live data from Claude API parse: "
                f"{len(state.drivers)} drivers, lap {state.leader_lap}"
            )
            self._last_state = state
            return state

        logger.warning("All live data scrapers failed")
        return self._last_state  # Return stale data rather than nothing

    def _try_firecrawl(self) -> LiveRaceState | None:
        """Use Firecrawl API to scrape JS-rendered F1 live timing page."""
        url = f"https://www.formula1.com/en/racing/{self.season}/race-results"

        headers = {"Content-Type": "application/json"}
        if FIRECRAWL_API_KEY:
            headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"

        try:
            resp = self.session.post(
                f"{FIRECRAWL_API_URL}/v1/scrape",
                json={
                    "url": url,
                    "formats": ["markdown"],
                    "waitFor": 5000,
                },
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            markdown = data.get("data", {}).get("markdown", "")
            if markdown:
                return self._parse_markdown_results(markdown)
        except Exception as e:
            logger.debug(f"Firecrawl scrape failed: {e}")
        return None

    def _scrape_formulaonehistory(self) -> LiveRaceState | None:
        """Scrape race results from formulaonehistory.com.

        This site has clean HTML tables with race results that are
        accessible without JavaScript rendering. The race results table
        has columns: Pos., No., Driver, Team, Laps, Time / Retired, Pts.
        """
        # Get race name from DB to build URL slug
        race_name = self._get_race_name_slug()
        urls = [
            f"https://www.formulaonehistory.com/{self.season}-{race_name}/",
        ]

        for url in urls:
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code != 200:
                    continue

                soup = BeautifulSoup(resp.text, "html.parser")
                return self._parse_formulaonehistory_tables(soup)
            except Exception as e:
                logger.debug(f"formulaonehistory.com failed for {url}: {e}")
                continue
        return None

    def _get_race_name_slug(self) -> str:
        """Get URL slug for the race from DB race_name."""
        try:
            from src.database.connection import get_session
            from src.database.queries import get_race

            with get_session() as session:
                race = get_race(session, self.season, self.round_num)
                if race:
                    # "Australian Grand Prix" -> "australian-grand-prix"
                    return race.race_name.lower().replace(" ", "-")
        except Exception:
            pass
        return f"round-{self.round_num}"

    def _parse_formulaonehistory_tables(
        self, soup: BeautifulSoup
    ) -> LiveRaceState | None:
        """Parse formulaonehistory.com — find the race results table.

        The race results table has header:
        Pos., No., Driver, Team, Laps, Time / Retired, Pts.
        """
        state = LiveRaceState(session_key=0, total_laps=self.total_laps)
        state.race_started = True

        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 5:
                continue

            # Check header for race result signature
            header_cells = [
                c.get_text(strip=True).upper()
                for c in rows[0].find_all(["td", "th"])
            ]
            header_text = " ".join(header_cells)

            # Race results table has "Laps" + "Pts" (FP/quali tables lack "Pts")
            is_race_table = (
                "LAPS" in header_text
                and "POS" in header_text
                and ("PTS" in header_text or "RETIRED" in header_text)
            )
            if not is_race_table:
                continue

            # Parse each row
            for row in rows[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if len(cells) < 5:
                    continue

                # Expected format: Pos, No, Driver, Team, Laps, Time/Retired, Pts
                pos_text = cells[0]
                driver_num_text = cells[1]
                driver_name = cells[2]
                laps_text = cells[4] if len(cells) > 4 else "0"
                time_text = cells[5] if len(cells) > 5 else ""

                # Parse position
                position = None
                retired = False
                if re.match(r"^\d{1,2}$", pos_text):
                    position = int(pos_text)
                elif pos_text.upper() in ("DNF", "RET", "NC", "DNS", "DSQ"):
                    retired = True

                # Parse driver number
                driver_number = 0
                if re.match(r"^\d{1,2}$", driver_num_text):
                    driver_number = int(driver_num_text)

                # Parse laps
                laps = 0
                if re.match(r"^\d+$", laps_text):
                    laps = int(laps_text)

                # Check retirement from time column
                if time_text and not time_text.startswith("+") and ":" not in time_text[:5]:
                    if any(kw in time_text.upper() for kw in ["DNF", "RET", "DNS"]):
                        retired = True

                if not driver_name or len(driver_name) < 3:
                    continue

                # Build acronym from last name
                parts = driver_name.split()
                acronym = parts[-1][:3].upper() if parts else f"D{driver_number}"

                ds = DriverLiveState(
                    driver_number=driver_number,
                    name_acronym=acronym,
                    position=position,
                    lap_number=laps,
                    retired=retired,
                )
                state.drivers[driver_number] = ds
                if laps > state.leader_lap:
                    state.leader_lap = laps

            if len(state.drivers) >= 10:
                break

        # Detect race status
        page_text = soup.get_text().upper()
        if "CHEQUERED" in page_text or "FINAL" in page_text:
            state.race_finished = True

        return state if state.drivers else None

    def _try_claude_parse(self) -> LiveRaceState | None:
        """Fetch HTML from F1 sites and use Claude API to extract race data.

        Falls back to this when structured BS4 parsing fails — Claude can
        handle arbitrary HTML layouts and extract positions reliably.
        """
        import json
        import os

        api_key = CLAUDE_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.debug("No Claude API key — skipping LLM fallback")
            return None

        # Fetch HTML from a results page
        race_slug = self._get_race_name_slug()
        urls = [
            f"https://www.formulaonehistory.com/{self.season}-{race_slug}/",
            (
                f"https://www.the-race.com/formula-1/"
                f"{self.season}-{race_slug}-f1-race-results/"
            ),
        ]

        html_content = None
        for url in urls:
            try:
                resp = self.session.get(url, timeout=15)
                if resp.status_code == 200:
                    # Truncate to ~30k chars to stay within token limits
                    html_content = resp.text[:30_000]
                    break
            except Exception:
                continue

        if not html_content:
            return None

        # Strip script/style tags to reduce noise
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        cleaned_text = soup.get_text(separator="\n", strip=True)
        # Keep only the first ~8000 chars of text
        cleaned_text = cleaned_text[:8_000]

        # Ask Claude to extract structured race data
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract the F1 RACE RESULTS (not practice or qualifying) "
                        "from this page text. Return ONLY a JSON array, no other text. "
                        "Each element: {\"pos\": int or null, \"number\": int, "
                        "\"driver\": \"Full Name\", \"team\": \"Team\", "
                        "\"laps\": int, \"retired\": bool, "
                        "\"safety_car\": bool, \"race_finished\": bool}. "
                        "Set safety_car and race_finished on the FIRST element only. "
                        "Set retired=true for DNF/DNS/NC/DSQ drivers. "
                        f"Page text:\n\n{cleaned_text}"
                    ),
                }],
            )

            response_text = message.content[0].text.strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in response_text:
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            results = json.loads(response_text)
            return self._build_state_from_json(results)

        except Exception as e:
            logger.warning(f"Claude API parse failed: {e}")
            return None

    def _build_state_from_json(self, results: list[dict]) -> LiveRaceState | None:
        """Convert Claude-extracted JSON into LiveRaceState."""
        if not results:
            return None

        state = LiveRaceState(session_key=0, total_laps=self.total_laps)
        state.race_started = True

        # Check flags from first element
        first = results[0]
        if first.get("safety_car"):
            state.safety_car_active = True
        if first.get("race_finished"):
            state.race_finished = True

        for entry in results:
            driver_number = entry.get("number", 0)
            driver_name = entry.get("driver", "")
            if not driver_name:
                continue

            parts = driver_name.split()
            acronym = parts[-1][:3].upper() if parts else f"D{driver_number}"
            laps = entry.get("laps", 0) or 0

            ds = DriverLiveState(
                driver_number=driver_number,
                name_acronym=acronym,
                position=entry.get("pos"),
                lap_number=laps,
                retired=entry.get("retired", False),
            )
            state.drivers[driver_number] = ds
            if laps > state.leader_lap:
                state.leader_lap = laps

        return state if state.drivers else None

    def _scrape_the_race(self) -> LiveRaceState | None:
        """Scrape race results from the-race.com.

        The-race.com publishes clean HTML results tables during races.
        """
        url = (
            f"https://www.the-race.com/formula-1/"
            f"{self.season}-round-{self.round_num}-f1-race-results/"
        )

        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            return self._parse_results_table(soup)
        except Exception as e:
            logger.debug(f"the-race.com scrape failed: {e}")
        return None

    def _parse_results_table(self, soup: BeautifulSoup) -> LiveRaceState | None:
        """Parse an HTML results table into LiveRaceState.

        Looks for <table> elements containing driver names and positions.
        Handles common formats from multiple F1 results sites.
        """
        state = LiveRaceState(session_key=0, total_laps=self.total_laps)
        state.race_started = True

        # Find all tables and look for one with race-result-like content
        tables = soup.find_all("table")

        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 5:  # Need at least a few drivers
                continue

            driver_count = 0
            for row in rows[1:]:  # Skip header
                cells = row.find_all(["td", "th"])
                if len(cells) < 3:
                    continue

                parsed = self._parse_result_row(cells, driver_count)
                if parsed:
                    driver_number, driver_state = parsed
                    state.drivers[driver_number] = driver_state
                    if driver_state.lap_number > state.leader_lap:
                        state.leader_lap = driver_state.lap_number
                    driver_count += 1

            if driver_count >= 10:  # Found a valid results table
                break

        # Detect race status from page text
        page_text = soup.get_text().upper()
        if "SAFETY CAR" in page_text and "VIRTUAL" not in page_text:
            state.safety_car_active = True
        if "VIRTUAL SAFETY CAR" in page_text or "VSC" in page_text:
            state.virtual_safety_car = True
        if "RED FLAG" in page_text:
            state.red_flag = True
        if "CHEQUERED" in page_text or "FINAL CLASSIFICATION" in page_text:
            state.race_finished = True

        return state if state.drivers else None

    def _parse_result_row(
        self, cells: list, fallback_number: int
    ) -> tuple[int, DriverLiveState] | None:
        """Parse a single table row into a DriverLiveState.

        Args:
            cells: List of <td>/<th> elements from a table row.
            fallback_number: Number to use if driver number isn't found.

        Returns:
            (driver_number, DriverLiveState) or None if unparseable.
        """
        texts = [c.get_text(strip=True) for c in cells]

        # Try to identify position, driver name, and laps from the row
        position = None
        driver_name = None
        laps = 0
        retired = False
        driver_number = fallback_number

        for i, text in enumerate(texts):
            # Position: first numeric value, typically 1-22
            if position is None and re.match(r"^\d{1,2}$", text):
                val = int(text)
                if 1 <= val <= 22:
                    position = val
                    continue

            # Driver number (usually 1-99)
            if re.match(r"^\d{1,2}$", text):
                val = int(text)
                if 1 <= val <= 99 and val != position:
                    driver_number = val
                    continue

            # Driver name: contains at least 2 alpha words
            if (
                driver_name is None
                and len(text) > 3
                and re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", text)
            ):
                driver_name = text
                continue

            # Laps completed
            if re.match(r"^\d{1,2}$", text) and position is not None:
                val = int(text)
                if 10 <= val <= 80:  # Reasonable lap count
                    laps = val

            # Retirement indicators
            upper = text.upper()
            if upper in ("DNF", "RET", "NC", "DNS", "DSQ"):
                retired = True
            if "DNF" in upper or "RET" in upper or "DNS" in upper:
                retired = True

        if driver_name is None:
            return None

        # Extract 3-letter acronym from name
        parts = driver_name.split()
        acronym = parts[-1][:3].upper() if parts else f"D{driver_number}"

        ds = DriverLiveState(
            driver_number=driver_number,
            name_acronym=acronym,
            position=position,
            lap_number=laps,
            retired=retired,
        )
        return driver_number, ds

    def _parse_markdown_results(self, markdown: str) -> LiveRaceState | None:
        """Parse Firecrawl markdown output into LiveRaceState.

        Firecrawl returns clean markdown tables that are easier to parse
        than raw HTML from JS-rendered pages.
        """
        state = LiveRaceState(session_key=0, total_laps=self.total_laps)
        state.race_started = True

        lines = markdown.split("\n")
        driver_count = 0

        for line in lines:
            # Look for markdown table rows: | pos | ... | driver | ...
            if not line.strip().startswith("|"):
                continue
            if "---" in line:  # Header separator
                continue

            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 3:
                continue

            # Try to parse as a result row
            position = None
            driver_name = None
            laps = 0
            retired = False

            for part in parts:
                if position is None and re.match(r"^\d{1,2}$", part):
                    val = int(part)
                    if 1 <= val <= 22:
                        position = val
                        continue

                if (
                    driver_name is None
                    and len(part) > 3
                    and re.search(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}", part)
                ):
                    driver_name = part
                    continue

                if re.match(r"^\d{1,2}$", part):
                    val = int(part)
                    if 10 <= val <= 80:
                        laps = val

                upper = part.upper()
                if upper in ("DNF", "RET", "NC", "DNS", "DSQ"):
                    retired = True

            if driver_name:
                parts_name = driver_name.split()
                acronym = parts_name[-1][:3].upper()
                ds = DriverLiveState(
                    driver_number=driver_count,
                    name_acronym=acronym,
                    position=position,
                    lap_number=laps,
                    retired=retired,
                )
                state.drivers[driver_count] = ds
                if laps > state.leader_lap:
                    state.leader_lap = laps
                driver_count += 1

        # Detect flags from markdown text
        md_upper = markdown.upper()
        if "SAFETY CAR" in md_upper:
            state.safety_car_active = True
        if "RED FLAG" in md_upper:
            state.red_flag = True
        if "CHEQUERED" in md_upper or "FINAL" in md_upper:
            state.race_finished = True

        return state if state.drivers else None


def build_scraped_race_state(
    scraper: LiveTimingScraper,
    driver_name_map: dict[str, str] | None = None,
) -> LiveRaceState | None:
    """High-level function to get live state via web scraping.

    This is the scraper equivalent of build_live_race_state() from
    the OpenF1 collector.

    Args:
        scraper: Configured LiveTimingScraper instance.
        driver_name_map: Optional mapping of scraped names to our DB names.

    Returns:
        LiveRaceState or None if scraping failed.
    """
    state = scraper.fetch_live_state()
    if state is None:
        return None

    # Remap driver names if we have a mapping
    if driver_name_map:
        new_drivers: dict[int, DriverLiveState] = {}
        for dn, ds in state.drivers.items():
            # Try to match by acronym or partial name
            matched = False
            for scraped_name, db_name in driver_name_map.items():
                if (
                    ds.name_acronym.upper() in scraped_name.upper()
                    or scraped_name.upper() in ds.name_acronym.upper()
                ):
                    ds.name_acronym = db_name[:3].upper()
                    new_drivers[dn] = ds
                    matched = True
                    break
            if not matched:
                new_drivers[dn] = ds
        state.drivers = new_drivers

    return state
