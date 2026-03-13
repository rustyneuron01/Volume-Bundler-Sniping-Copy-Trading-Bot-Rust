import os
import random
import time
import urllib.parse
from urllib.parse import quote_plus
import json
import re

import bittensor as bt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
import stamina

from gas.scraping.base import BaseScraper
from gas.types import MediaType


class GoogleScraper(BaseScraper):
    """
    Google Images scraper using Selenium with reverse image search support
    """

    def __init__(
        self,
        user_agent=None,
        scroll_delay=500,
        headless=True,
        tbs=None,
        max_year=2017,
        min_width=128,
        min_height=128,
        media_type=None,
        retry_attempts=3,
        base_delay=2.0,
        max_delay=10.0,
        jitter_factor=0.3,
    ):
        super().__init__(min_width, min_height, media_type=MediaType.REAL)

        self.user_agents = user_agent
        if self.user_agents is None:
            self.user_agents = [
                "Mozilla/5.0 (X11; Linux i686; rv:64.0) Gecko/20100101 Firefox/64.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Safari/605.1.15",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            ]

        self.scroll_delay = scroll_delay
        self.headless = headless

        self.retry_attempts = retry_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

        if tbs is None:
            tbs = {}

        if max_year is not None:
            current_year = time.localtime().tm_year
            if max_year < current_year:
                tbs["cdr"] = "1"  # enable custom date range
                tbs["cd_max"] = f"12/31/{max_year}"

        if min_width is not None and min_height is not None:
            if min_width >= 400 and min_height >= 300:
                tbs["isz"] = "l"  # large
            elif min_width >= 128:
                tbs["isz"] = "m"  # medium

        self.tbs = self._parse_request_parameters(tbs)

    def _random_delay(self, base_delay=None, max_delay=None):
        """Add random delay with jitter to avoid bot detection"""
        if base_delay is None:
            base_delay = self.base_delay
        if max_delay is None:
            max_delay = self.max_delay

        # Add jitter
        jitter = random.uniform(0, base_delay * self.jitter_factor)
        delay = min(base_delay + jitter, max_delay)
        delay += random.uniform(0, 0.5)

        time.sleep(delay)
        return delay

    def _random_scroll_delay(self):
        """Get a randomized scroll delay"""
        base_delay = self.scroll_delay / 1000.0
        jitter = random.uniform(0, base_delay * 0.5)
        return base_delay + jitter

    def _get_random_user_agent(self):
        """Get a random user agent with some variation"""
        user_agent = random.choice(self.user_agents)

        # Sometimes add a small random string to make it more unique
        if random.random() < 0.3:
            user_agent += f" {random.randint(1000, 9999)}"

        return user_agent

    def _randomize_chrome_options(self, chrome_options):
        """Add random Chrome options to avoid fingerprinting"""
        width = random.choice([1366, 1440, 1536, 1920, 2560])
        height = random.choice([768, 900, 864, 1080, 1440])
        chrome_options.add_argument(f"--window-size={width},{height}")

        languages = ["en-US,en;q=0.9", "en-GB,en;q=0.9", "en-CA,en;q=0.9"]
        chrome_options.add_argument(f"--lang={random.choice(languages)}")

        timezones = [
            "America/New_York",
            "Europe/London",
            "Asia/Tokyo",
            "Australia/Sydney",
        ]
        chrome_options.add_argument(f"--timezone={random.choice(timezones)}")

        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_argument("--disable-ipc-flooding-protection")

        return chrome_options

    def get_image_urls(self, queries=None, query_ids=None, source_image_paths=None, limit=5):
        """
        Get image URLs from Google Images using either text queries or reverse image search.

        Parameters:
        -----------
        queries : str or list, optional
            Search query or list of queries (mutually exclusive with source_image_paths)
        query_ids: str or list
            Query ids for tracking
        source_image_paths : str or list, optional
            Path(s) to image(s) to use for reverse image search (mutually exclusive with queries)
        limit : int, default=5
            Maximum number of images to return per query

        Returns:
        --------
        dict
            Dictionary mapping query keys to lists of image data dictionaries

        Raises:
        -------
        ValueError
            If neither or both queries and source_image_paths are provided
        """
        if sum(x is not None for x in [queries, source_image_paths]) != 1:
            raise ValueError(
                "Either queries or source_image must be provided (mutually exclusive)"
            )

        if queries is not None:
            return self.image_search(queries, query_ids, limit)
        elif source_image_paths is not None:
            raise NotImplementedError

    def image_search(self, queries, query_ids, limit=5):
        """
        Perform Google image search with text queries and extract full-size image URLs.

        Parameters:
        -----------
        queries : str or list
            Search query or list of queries to search for
        limit : int, default=5
            Maximum number of images to return per query

        Returns:
        --------
        dict
            Dictionary mapping query keys to lists of image data dictionaries
            Each image dict contains:
            - query: Original search query
            - url: Full-size image URL
            - source: Always "google"
            - thumbnail_url: Thumbnail URL (optional)
        """
        results = {}

        if not isinstance(queries, list):
            queries = [queries]
            query_ids = [query_ids]

        for i in range(len(queries)):
            query = queries[i]
            query_id = query_ids[i] if query_ids is not None else None
            query_id = query.replace(" ", "") if query_id is None else query_id

            bt.logging.info(f"Searching for: {query}")

            # Use stamina for retry logic with exponential backoff and jitter
            @stamina.retry(
                on=Exception,
                attempts=self.retry_attempts,
                wait_initial=self.base_delay,
                wait_max=self.max_delay,
                wait_jitter=self.base_delay * self.jitter_factor,
            )
            def search_single_query(query, limit):
                return self._perform_single_search(query, limit)

            try:
                query_results = search_single_query(query, limit)
                results[query_id] = query_results
                bt.logging.info(f"Found {len(query_results)} images for query: {query}")

                # Add random delay between queries
                if len(queries) > 1:
                    self._random_delay(base_delay=1.0, max_delay=3.0)

            except Exception as e:
                bt.logging.error(f"Failed to search for query '{query}': {str(e)}")
                results[query_id] = []

        return results

    def _perform_single_search(self, query, query_id=None, limit=1):
        """Perform a single search with randomization and anti-bot measures"""
        user_agent = self._get_random_user_agent()
        browser = None

        try:
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"user-agent={user_agent}")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option(
                "excludeSwitches", ["enable-automation"]
            )
            chrome_options.add_experimental_option("useAutomationExtension", False)

            # Add randomization to Chrome options
            chrome_options = self._randomize_chrome_options(chrome_options)

            browser = webdriver.Chrome(options=chrome_options)
            browser.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            page_url = f"https://www.google.com/search?&safe=active&source=lnms&tbs={self.tbs}&tbm=isch&q={self._parse_request_queries(query)}"
            browser.get(page_url)

            # Wait for page to load with random delay
            self._random_delay(base_delay=1.5, max_delay=3.0)

            # Scroll to load more images with randomized delays
            scroll_count = random.randint(3, 6)  # Random number of scrolls
            for i in range(scroll_count):
                # Random scroll distance
                scroll_distance = random.randint(
                    int(browser.execute_script("return window.innerHeight;") * 0.7),
                    int(browser.execute_script("return window.innerHeight;") * 1.2),
                )
                browser.execute_script(f"window.scrollBy(0, {scroll_distance})")

                # Random delay between scrolls
                time.sleep(self._random_scroll_delay())

                # Occasionally add a small pause
                if random.random() < 0.3:
                    time.sleep(random.uniform(0.5, 1.5))

                # Add human behavior simulation
                if random.random() < 0.4:
                    self._add_human_behavior_simulation(browser)

                # Check for rate limiting
                if self._handle_rate_limiting(browser):
                    bt.logging.warning(
                        "Rate limiting detected, continuing with caution"
                    )

            # Method 1: Click on images to get full-size URLs
            image_urls = self._extract_full_size_urls_by_clicking(browser, limit)

            # Method 2: If clicking method fails, try extracting from page source
            if len(image_urls) < limit:
                bt.logging.info("Trying alternative extraction method...")
                additional_urls = self._extract_urls_from_page_source(
                    browser, limit - len(image_urls)
                )
                image_urls.extend(additional_urls)

            return [
                {"query": query, "query_id": query_id, "url": url, "source": "google"}
                for url in image_urls[:limit]
            ]

        except Exception as e:
            bt.logging.error(f"Google scraper error: {str(e)}")
            raise  # Re-raise for stamina retry logic
        finally:
            if browser:
                browser.quit()

    def _extract_full_size_urls_by_clicking(self, browser, limit):
        """Extract full-size image URLs by clicking on thumbnails"""
        image_urls = []

        try:
            # Find clickable image containers
            image_containers = browser.find_elements(By.CSS_SELECTOR, "[data-ri]")

            # Randomize the order of containers to click
            random.shuffle(image_containers)

            for i, container in enumerate(
                image_containers[: limit * 2]
            ):  # Get more than needed
                if len(image_urls) >= limit:
                    break

                try:
                    # Scroll element into view with random offset
                    browser.execute_script(
                        "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
                        container,
                    )
                    time.sleep(random.uniform(0.3, 1.2))

                    # Sometimes move mouse to element first (more human-like)
                    if random.random() < 0.7:
                        ActionChains(browser).move_to_element(container).pause(
                            random.uniform(0.1, 0.5)
                        ).click().perform()
                    else:
                        container.click()

                    # Look for the full-size image in the preview pane
                    time.sleep(random.uniform(0.8, 2.0))
                    full_size_img = None
                    selectors = [
                        "img[src*='https://'][style*='max-width']",
                        ".irc_mi img",
                        ".v4dQwb img",
                        "img[jsname]",
                    ]

                    for selector in selectors:
                        try:
                            elements = browser.find_elements(By.CSS_SELECTOR, selector)
                            for elem in elements:
                                src = elem.get_attribute("src")
                                if (
                                    src
                                    and src.startswith("http")
                                    and "google" not in src
                                    and "gstatic" not in src
                                    and len(src) > 50
                                ):  # Longer URLs are usually full-size
                                    full_size_img = src
                                    break
                            if full_size_img:
                                break
                        except:
                            continue

                    if full_size_img and full_size_img not in image_urls:
                        image_urls.append(full_size_img)
                        bt.logging.debug(
                            f"Found full-size image: {full_size_img[:100]}..."
                        )

                    # Close the preview if it opened (with random delay)
                    try:
                        close_button = browser.find_element(
                            By.CSS_SELECTOR, "[aria-label='Close']"
                        )
                        time.sleep(random.uniform(0.2, 0.8))
                        close_button.click()
                        time.sleep(random.uniform(0.3, 0.7))
                    except:
                        pass

                except Exception as e:
                    bt.logging.debug(f"Error clicking image {i}: {str(e)}")
                    continue

        except Exception as e:
            bt.logging.error(f"Error in clicking method: {str(e)}")

        return image_urls

    def _extract_urls_from_page_source(self, browser, limit):
        """Extract image URLs from page source using regex"""
        image_urls = []

        try:
            page_source = browser.page_source

            # Look for image URLs in various patterns
            patterns = [
                r'"(https://[^"]*\.(?:jpg|jpeg|png|gif|webp)[^"]*)"',
                r"'(https://[^']*\.(?:jpg|jpeg|png|gif|webp)[^']*)'",
                r'src="(https://[^"]*\.(?:jpg|jpeg|png|gif|webp)[^"]*)"',
                r'data-src="(https://[^"]*\.(?:jpg|jpeg|png|gif|webp)[^"]*)"',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, page_source, re.IGNORECASE)
                for match in matches:
                    if (
                        match not in image_urls
                        and "google" not in match
                        and "gstatic" not in match
                        and len(match) > 50
                    ):  # Filter out small/thumbnail URLs
                        image_urls.append(match)
                        if len(image_urls) >= limit:
                            break
                if len(image_urls) >= limit:
                    break

        except Exception as e:
            bt.logging.error(f"Error extracting from page source: {str(e)}")

        return image_urls[:limit]

    def _parse_request_parameters(self, tbs):
        """Parse TBS parameters for Google search URL"""
        if not tbs:
            return ""
        param_parts = []
        for key, value in tbs.items():
            if value:
                param_parts.append(f"{key}:{value}")
        return urllib.parse.quote(",".join(param_parts))

    def _parse_request_queries(self, query):
        """Encode query for URL"""
        return quote_plus(query) if query else ""

    def _handle_rate_limiting(self, browser):
        """Handle potential rate limiting by adding longer delays"""
        try:
            page_source = browser.page_source.lower()
            rate_limit_indicators = [
                "unusual traffic",
                "automated requests",
                "captcha",
                "verify you are human",
                "rate limit",
                "too many requests",
            ]

            if any(indicator in page_source for indicator in rate_limit_indicators):
                bt.logging.warning(
                    "Detected potential rate limiting, adding longer delay"
                )
                time.sleep(random.uniform(30, 60))
                return True

        except Exception as e:
            bt.logging.debug(f"Error checking for rate limiting: {str(e)}")

        return False

    def _add_human_behavior_simulation(self, browser):
        """Add various human-like behaviors to avoid detection"""
        try:
            if random.random() < 0.4:
                browser.execute_script("window.scrollBy(0, -100)")
                time.sleep(random.uniform(0.5, 1.5))

            if random.random() < 0.3:
                actions = ActionChains(browser)
                x = random.randint(50, 200)
                y = random.randint(50, 200)
                actions.move_by_offset(x, y).pause(random.uniform(0.2, 0.8)).perform()

            if random.random() < 0.2:
                browser.execute_script("document.activeElement.blur();")
                time.sleep(random.uniform(0.1, 0.3))

        except Exception as e:
            bt.logging.debug(f"Human behavior simulation failed: {str(e)}")
