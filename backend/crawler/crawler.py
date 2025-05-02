import httpx
import asyncio
from urllib.parse import urljoin, urlparse
from typing import Set, List, Dict, Optional, Any
from bs4 import BeautifulSoup
import logging
from dataclasses import dataclass, field

from ..config import Config
from .scraper import WebScraper
from ..database.models import WebPageData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CrawlResult:
    """Container for crawl results"""
    pages: List[WebPageData] = field(default_factory=list)
    errors: Dict[str, str] = field(default_factory=dict)

class WebCrawler:
    """Web crawler that performs a 1-hop crawl from given URLs"""
    
    def __init__(
        self, 
        max_depth: int = None,
        timeout: int = None,
        headers: Dict[str, str] = None
    ):
        self.max_depth = max_depth or Config.CRAWLER_MAX_DEPTH
        self.timeout = timeout or Config.CRAWLER_TIMEOUT
        self.headers = headers or Config.CRAWLER_HEADERS
        self.scraper = WebScraper()
        
    async def crawl_urls(self, urls: List[str]) -> CrawlResult:
        """Crawl a list of URLs with a 1-hop depth"""
        result = CrawlResult()
        visited_urls: Set[str] = set()
        
        # First level: process initial URLs
        initial_tasks = []
        for url in urls:
            if url not in visited_urls:
                visited_urls.add(url)
                task = self._process_url(url, None, result)
                initial_tasks.append(task)
        
        # Wait for all initial tasks to complete
        if initial_tasks:
            initial_pages = await asyncio.gather(*initial_tasks, return_exceptions=True)
            
            # Filter out exceptions and None values
            initial_pages = [page for page in initial_pages if isinstance(page, WebPageData)]
            result.pages.extend(initial_pages)
            
            # If depth is 0, we're done
            if self.max_depth <= 0:
                return result
            
            # Second level: process links from initial pages
            second_level_urls = []
            for page in initial_pages:
                if page and page.content:
                    page_links = self._extract_links(page.url, page.content)
                    for link in page_links:
                        if link not in visited_urls:
                            visited_urls.add(link)
                            second_level_urls.append((link, page.url))
            
            # Process second level urls in batches to avoid overwhelming resources
            batch_size = 10
            for i in range(0, len(second_level_urls), batch_size):
                batch = second_level_urls[i:i+batch_size]
                batch_tasks = [self._process_url(url, source_url, result) for url, source_url in batch]
                batch_pages = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Filter out exceptions and None values
                batch_pages = [page for page in batch_pages if isinstance(page, WebPageData)]
                result.pages.extend(batch_pages)
        
        return result
    
    async def _process_url(
        self, 
        url: str, 
        source_url: Optional[str], 
        result: CrawlResult
    ) -> Optional[WebPageData]:
        """Process a single URL: fetch, scrape, and extract data"""
        try:
            logger.info(f"Processing URL: {url}")
            
            # Create an HTTP client with timeout
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self.headers, follow_redirects=True)
                response.raise_for_status()
                
                # Get content type to ensure we're processing HTML
                content_type = response.headers.get('content-type', '')
                if 'text/html' not in content_type.lower():
                    result.errors[url] = f"Not HTML content: {content_type}"
                    return None
                
                # Scrape the page content
                html_content = response.text
                page_data = self.scraper.scrape(html_content, url)
                
                # Add source URL
                if source_url:
                    page_data.source_url = source_url
                
                return page_data
                
        except httpx.RequestError as e:
            error_msg = f"Request error: {str(e)}"
            logger.error(f"Error processing {url}: {error_msg}")
            result.errors[url] = error_msg
            return None
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code}"
            logger.error(f"Error processing {url}: {error_msg}")
            result.errors[url] = error_msg
            return None
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Error processing {url}: {error_msg}")
            result.errors[url] = error_msg
            return None
    
    def _extract_links(self, base_url: str, html_content: str) -> List[str]:
        """Extract links from HTML content"""
        links = []
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Find all anchor tags
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            
            # Skip empty, fragment-only, or javascript links
            if not href or href.startswith('#') or href.startswith('javascript:'):
                continue
            
            # Normalize URL
            full_url = urljoin(base_url, href)
            
            # Parse and filter URLs
            parsed_url = urlparse(full_url)
            
            # Skip non-HTTP/HTTPS URLs
            if parsed_url.scheme not in ('http', 'https'):
                continue
            
            # Remove fragments
            clean_url = parsed_url._replace(fragment='').geturl()
            links.append(clean_url)
        
        return links