import io
import asyncio
import aiohttp
import base64
from PIL import Image

def base64_to_image(base64_string: str) -> Image:
    image_data = base64.b64decode(base64_string)
    image_io = io.BytesIO(image_data)
    return Image.open(image_io)


async def download_image_from_url(url: str) -> Image.Image:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail=f"Error downloading image: {url}")
            image_data = await response.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return image
