import os
import requests
import time
from abc import ABC, abstractmethod
from PIL import Image
from io import BytesIO
import bittensor as bt
import json

from gas.types import Modality


class BaseScraper(ABC):
    """
    Base class for image scrapers with common functionality
    
    Parameters:
    -----------
    min_width : int
        Minimum width for downloaded images (default: 128)
    min_height : int
        Minimum height for downloaded images (default: 128)
    media_type : MediaType, optional
        Type of media being scraped
    """
    
    def __init__(self, min_width=128, min_height=128, media_type=None):
        self.min_width = min_width
        self.min_height = min_height
        self.media_type = media_type
        self.modality = Modality.IMAGE   # does not support video yet
    
    def _check_image_size(self, url):
        """
        Check if image meets minimum size requirements
        
        Parameters:
        -----------
        url : str
            Image URL to check
            
        Returns:
        --------
        tuple
            (meets_requirements: bool, width: int, height: int)
        """
        try:
            response = requests.get(url, timeout=10, stream=True)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                width, height = img.size
                meets_requirements = width >= self.min_width and height >= self.min_height
                return meets_requirements, width, height
            else:
                bt.logging.warning(f"Failed to fetch image for size check: {url}")
                return False, 0, 0
        except Exception as e:
            bt.logging.warning(f"Error checking image size for {url}: {str(e)}")
            return False, 0, 0
    
    def _get_file_extension(self, url):
        """Get appropriate file extension based on content type"""
        extension = '.jpg'  # Default
        try:
            response = requests.head(url, timeout=10)
            content_type = response.headers.get('content-type', '')
            
            if 'image/jpeg' in content_type:
                extension = '.jpg'
            elif 'image/png' in content_type:
                extension = '.png'
            elif 'image/gif' in content_type:
                extension = '.gif'
            elif 'image/webp' in content_type:
                extension = '.webp'
        except Exception as e:
            bt.logging.info(f"Error fetching headers for {url}: {str(e)}")
        
        return extension
    
    def _download_single_image(self, url, width=None, height=None):
        """
        Download a single image and yield the image data
        
        Parameters:
        -----------
        url : str
            Image URL to download
        width : int, optional
            Image width for logging
        height : int, optional
            Image height for logging
            
        Yields:
        -------
        tuple
            (success: bool, image_data: PIL.Image or None, metadata: dict or None)
        """
        try:
            image_response = requests.get(url, timeout=10)
            if image_response.status_code == 200:
                # Convert to PIL Image
                img = Image.open(BytesIO(image_response.content))
                size_info = f" ({width}x{height})" if width and height else ""
                bt.logging.trace(f"Downloaded image from {url}{size_info}")
                
                metadata = {
                    'url': url,
                    'width': width,
                    'height': height,
                    'content_type': image_response.headers.get('content-type', ''),
                    'content_length': len(image_response.content)
                }
                
                return True, img, metadata
            else:
                bt.logging.error(f"Failed to download {url}, status code: {image_response.status_code}")
                return False, None, None
        except Exception as e:
            bt.logging.error(f"Error downloading image from {url}: {str(e)}")
            return False, None, None

    @abstractmethod
    def get_image_urls(self, queries, query_ids=None, limit=5):
        """
        Abstract method to get image URLs - must be implemented by subclasses
        
        Parameters:
        -----------
        queries : str or list
            Search query or list of queries
        query_ids: str or list
            Query ids for tracking
        limit : int
            Maximum number of images per query
            
        Returns:
        --------
        dict
            Dictionary with query keys and lists of image data
        """
        pass
    
    def download_images(self, queries=None, query_ids=None, urls=None, source_image_paths=None, limit=5):
        """
        Download images based on queries with size constraints
        
        Parameters:
        -----------
        queries : str or list
            Search query or list of queries (set if not using source_images or urls)
        query_ids: str or list
            Query id for tracking
        urls: str or list
            Pre-fetched image urls (set if not using source_images or queries)
        source_image_paths: str or list
            Local path to image(s) with which to perform reverse image search
        limit : int
            Maximum number of images to download per query
            
        Yields:
        -------
        tuple
            (query_id: str, image_data: dict) where image_data contains:
            - url: str
            - image_content: PIL.Image
            - width: int
            - height: int
            - metadata: dict (query, source_url, title, source, etc.)
        """
        if sum(x is not None for x in [queries, urls, source_image_paths]) != 1:
            raise ValueError("Either queries, urls, or source_image must be provided (mutually exclusive)")
        
        if urls is None:
            # Get more URLs than needed to account for size filtering
            image_urls = self.get_image_urls(
                queries=queries,
                query_ids=query_ids,
                source_image_paths=source_image_paths,
                limit=limit * 3
            )
        
        for query_id in image_urls:
            downloaded_count = 0
            
            for img_data in image_urls[query_id]:
                if downloaded_count >= limit:
                    break
                
                url = img_data['url']
                
                meets_size_req, width, height = self._check_image_size(url)
                
                if not meets_size_req:
                    bt.logging.trace(f"Skipping image {url} - size {width}x{height} below minimum {self.min_width}x{self.min_height}")
                    continue
                
                # Download the image
                success, image_content, download_metadata = self._download_single_image(url, width, height)
                
                if not success or image_content is None:
                    continue
                
                # Prepare metadata
                metadata = {
                    'query': img_data.get('query', query_id),
                    'source_url': img_data.get('source_url'),
                    'title': img_data.get('title', ''),
                    'source': img_data.get('source', 'unknown'),
                    'download_url': url,
                    **(download_metadata or {})
                }
                
                # Yield the downloaded image data
                image_data = {
                    'url': url,
                    'image_content': image_content,
                    'width': width,
                    'height': height,
                    'metadata': metadata
                }
                
                downloaded_count += 1
                yield query_id, image_data
                
            bt.logging.debug(f"Downloaded {downloaded_count} images for query {query_id}")