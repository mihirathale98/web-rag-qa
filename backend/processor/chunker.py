from typing import List, Dict, Any, Optional
import re
from dataclasses import dataclass

from ..config import Config
from ..database.models import WebPageData, TextChunkData

@dataclass
class ChunkMetadata:
    """Metadata for text chunks"""
    title: Optional[str] = None
    source_url: Optional[str] = None
    chunk_index: int = 0
    section_title: Optional[str] = None
    total_chunks: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class TextChunker:
    """Split text content into chunks suitable for vector search"""
    
    def __init__(
        self, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
    
    def chunk_webpage(self, webpage: WebPageData) -> List[TextChunkData]:
        """Process a webpage and split its content into chunks"""
        if not webpage.content:
            return []
        
        # Extract sections with their titles
        sections = self._extract_sections(webpage.content)
        
        # Create chunks from sections
        chunks = []
        chunk_index = 0
        
        for section_title, section_text in sections:
            section_chunks = self._create_chunks(section_text)
            
            for chunk_text in section_chunks:
                # Create metadata for this chunk
                metadata = ChunkMetadata(
                    title=webpage.title,
                    source_url=webpage.url,
                    chunk_index=chunk_index,
                    section_title=section_title
                ).to_dict()
                
                # Create the chunk data object
                chunk = TextChunkData(
                    content=chunk_text,
                    web_page_id=0,  # This will be set after the webpage is saved
                    chunk_index=chunk_index,
                    metadata=metadata
                )
                
                chunks.append(chunk)
                chunk_index += 1
        
        # Update total_chunks in metadata
        for chunk in chunks:
            if chunk.metadata:
                chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _extract_sections(self, content: str) -> List[tuple]:
        """Extract sections from content based on headers"""
        # Split content by headings (looking for markdown-style headers)
        heading_pattern = re.compile(r'\n\s*(#{1,6}|[A-Z][A-Z\s]+:)\s*(.*?)\s*(?=\n)')
        
        # Find all headings
        headings = list(heading_pattern.finditer(content))
        
        if not headings:
            # No headings found, treat as one section
            return [("Document", content)]
        
        # Extract sections
        sections = []
        for i, match in enumerate(headings):
            section_title = match.group(2).strip()
            start_pos = match.end()
            
            # End position is the start of the next heading or the end of the content
            end_pos = headings[i + 1].start() if i < len(headings) - 1 else len(content)
            
            # Extract section text
            section_text = content[start_pos:end_pos].strip()
            
            if section_text:  # Only add non-empty sections
                sections.append((section_title, section_text))
        
        # Check if there's content before the first heading
        if headings[0].start() > 0:
            intro_text = content[:headings[0].start()].strip()
            if intro_text:
                sections.insert(0, ("Introduction", intro_text))
        
        return sections
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks of specified size"""
        if not text:
            return []
        
        # If text is already smaller than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk end position
            end = start + self.chunk_size
            
            if end >= len(text):
                # We're at the end of the text
                chunks.append(text[start:])
                break
            
            # Try to find a good break point (end of sentence, paragraph)
            break_points = [
                text.rfind('\n\n', start, end),  # Paragraph break
                text.rfind('. ', start, end),    # Sentence break
                text.rfind('! ', start, end),    # Exclamation break
                text.rfind('? ', start, end),    # Question break
                text.rfind(', ', start, end),    # Comma break
                text.rfind(' ', start, end)      # Word break
            ]
            
            # Find the first valid break point
            break_point = next((bp for bp in break_points if bp != -1), end)
            
            # Add chunk and move to next start position
            if break_point > start:
                # Add the character(s) after the break point
                if text[break_point] in ['.', '!', '?', ',']:
                    break_point += 2  # Include the punctuation and space
                elif text[break_point] == '\n':
                    break_point += 2  # Include the paragraph break
                else:
                    break_point += 1  # Include the space or whatever character
                
                chunks.append(text[start:break_point])
                start = break_point - self.chunk_overlap
            else:
                # No good break point found, just chunk at the end
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
        
        return chunks