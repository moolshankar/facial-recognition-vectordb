import asyncio
from typing import Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import threading

class CacheService:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                print(f"Cache hit for key: {key}, value: {value}")
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    return value
                else:
                    del self.cache[key]
            print(f"Cache miss for key: {key}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        # return if value is None or empty
        if value is None or value == []:
            return
        with self._lock:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            self.cache[key] = (value, datetime.now())
            print(f"Cache set for key: {key}, value: {value}")

    
    def clear(self) -> None:
        with self._lock:
            self.cache.clear()
    
    def size(self) -> int:
        with self._lock:
            return len(self.cache)