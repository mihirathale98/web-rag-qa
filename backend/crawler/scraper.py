from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
from typing import Optional

from ..database.models import WebPageData

class WebScraper:
    """Extract and clean content from web pages"""
    
    def scrape(self, html_content: str, url: str) -> WebPageData:
        """Scrape and clean content from HTML"""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Extract title
        title = self._extract_title(soup)
        
        # Clean the content
        cleaned_content = self._clean_content(soup)
        
        # Create WebPageData object
        return WebPageData(
            url=url,
            title=title,
            content=cleaned_content
        )
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the title from the page"""
        # Try to get title from title tag
        title_tag = soup.title
        if title_tag and title_tag.string:
            return title_tag.string.strip()
        
        # Try to get title from og:title meta tag
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()
        
        # Try to get from h1
        h1 = soup.find('h1')
        if h1 and h1.text:
            return h1.text.strip()
        
        # No title found
        return None
    
    def _clean_content(self, soup: BeautifulSoup) -> str:
        """Clean and extract main content from the page"""
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'noscript', 'iframe', 'footer', 'nav', 'aside']):
            script_or_style.decompose()
        
        # Remove hidden elements
        for hidden in soup.find_all(style=re.compile(r'display:\s*none')):
            hidden.decompose()
            
        # Try to find the main content
        main_content = None
        
        # Common content containers
        content_elements = [
            soup.find('main'),
            soup.find('article'),
            soup.find(id=re.compile(r'content|main|article', re.I)),
            soup.find(class_=re.compile(r'content|main|article', re.I)),
            soup.find('div', {'role': 'main'})
        ]
        
        # Use the first valid content container found
        for element in content_elements:
            if element and len(element.get_text(strip=True)) > 100:  # Ensure sufficient content
                main_content = element
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.body or soup
        
        # Get all paragraphs and headings from the main content
        text_elements = []
        for elem in main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = elem.get_text(strip=True)
            if text and len(text) > 10:  # Ignore very short text
                if elem.name.startswith('h'):
                    # Add extra newlines for headings for better formatting
                    text_elements.append(f"\n\n{text}\n")
                else:
                    text_elements.append(text)
        
        # Join all text elements with newlines
        content = '\n\n'.join(text_elements)
        
        # Additional cleaning
        content = self._post_process_text(content)
        
        return content
    
    def _post_process_text(self, text: str) -> str:
        """Additional text cleaning"""
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove redundant newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Fix spacing after periods
        text = re.sub(r'\.(?! |\n|$)', '. ', text)
        
        # Remove very short lines (likely not useful content)
        lines = text.split('\n')
        lines = [line for line in lines if len(line.strip()) > 15]
        
        return '\n'.join(lines).strip()