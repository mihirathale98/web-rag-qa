import re
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextCleaner:
    """Clean and normalize text content from web pages"""
    
    def clean(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Apply cleaning operations in sequence
        text = self._remove_extra_whitespace(text)
        text = self._normalize_unicode(text)
        text = self._fix_common_issues(text)
        
        return text.strip()
    
    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove redundant whitespace"""
        # Replace multiple spaces with a single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with at most two
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove spaces at the beginning of lines
        text = re.sub(r'\n +', '\n', text)
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')
        
        # Normalize ellipsis
        text = text.replace('…', '...')
        
        return text
    
    def _fix_common_issues(self, text: str) -> str:
        """Fix common text issues from web scraping"""
        # Fix missing spaces after punctuation
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Fix broken sentences across lines
        text = re.sub(r'([a-z])\n([a-z])', r'\1 \2', text)
        
        # Remove header/footer/navigation boilerplate
        text = re.sub(r'(Site Map|Privacy Policy|Terms of Use|Contact Us|All Rights Reserved)(\s*\|)*\s*', '', text, flags=re.I)
        
        # Remove excessive symbols often used as separators
        text = re.sub(r'[-=_*]{3,}', '', text)
        
        return text