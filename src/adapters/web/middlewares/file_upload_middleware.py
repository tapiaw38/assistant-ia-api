from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from typing import Awaitable, Callable
import asyncio
import tempfile
import os
import aiofiles
from pathlib import Path


class FileUploadMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle large file uploads efficiently with streaming support
    """

    def __init__(
        self,
        app,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default
        max_files: int = 10,
        allowed_extensions: set = None,
        chunk_size: int = 8192,
        temp_dir: str = None,
    ):
        super().__init__(app)
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.allowed_extensions = allowed_extensions or {
            ".pdf",
            ".xlsx",
            ".xls",
            ".csv",
            ".png",
            ".jpg",
            ".jpeg",
        }
        self.chunk_size = chunk_size
        self.temp_dir = temp_dir or tempfile.gettempdir()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process request and handle file uploads with streaming
        """
        # Only process multipart/form-data requests
        if not self._is_file_upload_request(request):
            return await call_next(request)

        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_file_size:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "File too large",
                    "max_size": self.max_file_size,
                    "received_size": int(content_length),
                },
            )

        # Process file upload
        try:
            # Add file processing metadata to request state
            request.state.file_processing = {
                "streaming_enabled": True,
                "chunk_size": self.chunk_size,
                "temp_dir": self.temp_dir,
                "max_file_size": self.max_file_size,
            }

            return await call_next(request)

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": "File processing error", "details": str(e)},
            )

    def _is_file_upload_request(self, request: Request) -> bool:
        """Check if request is a file upload"""
        content_type = request.headers.get("content-type", "")
        return "multipart/form-data" in content_type

    async def stream_file_to_temp(self, file_data: bytes, filename: str) -> str:
        """
        Stream file data to temporary file for processing large files
        """
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(filename).suffix, dir=self.temp_dir
            )

            # Write file data in chunks
            chunk_size = self.chunk_size

            async with aiofiles.open(temp_file.name, "wb") as f:
                for i in range(0, len(file_data), chunk_size):
                    chunk = file_data[i : i + chunk_size]
                    await f.write(chunk)

            temp_file.close()
            return temp_file.name

        except Exception as e:
            # Clean up on error
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e

    async def validate_file_stream(self, file_stream, filename: str) -> dict:
        """
        Validate file stream for security and format compliance
        """
        validation_result = {"valid": True, "errors": [], "file_info": {}}

        try:
            # Check file extension
            file_extension = Path(filename).suffix.lower()
            if file_extension not in self.allowed_extensions:
                validation_result["valid"] = False
                validation_result["errors"].append(
                    f"File extension {file_extension} not allowed"
                )
                return validation_result

            # Read first chunk to validate file header
            first_chunk = await file_stream.read(1024)
            await file_stream.seek(0)  # Reset stream position

            # Basic file header validation
            if file_extension in [".pdf"]:
                if not first_chunk.startswith(b"%PDF"):
                    validation_result["valid"] = False
                    validation_result["errors"].append("Invalid PDF file header")

            elif file_extension in [".xlsx"]:
                # Excel files are ZIP archives
                if not first_chunk.startswith(b"PK"):
                    validation_result["valid"] = False
                    validation_result["errors"].append("Invalid Excel file header")

            # Add file info
            validation_result["file_info"] = {
                "filename": filename,
                "extension": file_extension,
                "header_valid": validation_result["valid"],
            }

        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")

        return validation_result

    async def cleanup_temp_files(self, temp_files: list):
        """
        Clean up temporary files asynchronously
        """
        cleanup_tasks = []

        for temp_file in temp_files:
            if os.path.exists(temp_file):
                cleanup_tasks.append(self._remove_file_async(temp_file))

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    async def _remove_file_async(self, file_path: str):
        """Remove file asynchronously"""
        try:
            await asyncio.to_thread(os.unlink, file_path)
        except Exception as e:
            print(f"Error removing temporary file {file_path}: {e}")

    def get_file_processing_config(self) -> dict:
        """Get file processing configuration"""
        return {
            "max_file_size": self.max_file_size,
            "max_files": self.max_files,
            "allowed_extensions": list(self.allowed_extensions),
            "chunk_size": self.chunk_size,
            "temp_dir": self.temp_dir,
        }


class StreamingFileHandler:
    """
    Handler for streaming file processing operations
    """

    def __init__(self, chunk_size: int = 8192):
        self.chunk_size = chunk_size

    async def process_file_stream(self, file_stream, processor_func, *args, **kwargs):
        """
        Process file stream with a given processor function
        """
        try:
            # Create temporary file for processing
            temp_file = tempfile.NamedTemporaryFile(delete=False)

            # Stream file content to temporary file
            async for chunk in self._read_file_chunks(file_stream):
                temp_file.write(chunk)

            temp_file.close()

            # Process file
            result = await processor_func(temp_file.name, *args, **kwargs)

            # Clean up
            os.unlink(temp_file.name)

            return result

        except Exception as e:
            # Clean up on error
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e

    async def _read_file_chunks(self, file_stream):
        """Read file in chunks asynchronously"""
        while True:
            chunk = await file_stream.read(self.chunk_size)
            if not chunk:
                break
            yield chunk

    async def stream_to_processor(
        self, file_data: bytes, processor_class, *args, **kwargs
    ):
        """
        Stream file data directly to processor without temporary file
        """
        from io import BytesIO

        # Create BytesIO stream
        file_stream = BytesIO(file_data)

        # Process with the given processor
        processor = processor_class(*args, **kwargs)

        # Call processor methods that accept BytesIO
        if hasattr(processor, "process_stream"):
            return await processor.process_stream(file_stream)
        else:
            # Fallback to traditional processing
            return await processor.process_file(file_stream)

    def estimate_processing_time(self, file_size: int, file_type: str) -> float:
        """
        Estimate processing time based on file size and type
        """
        # Base processing times per MB
        base_times = {
            ".pdf": 2.0,  # seconds per MB
            ".xlsx": 1.5,  # seconds per MB
            ".xls": 2.0,  # seconds per MB
            ".csv": 0.5,  # seconds per MB
        }

        file_size_mb = file_size / (1024 * 1024)
        base_time = base_times.get(file_type, 2.0)

        # Add overhead for very large files
        if file_size_mb > 50:
            base_time *= 1.5

        return file_size_mb * base_time

    def get_memory_usage_estimate(self, file_size: int, file_type: str) -> dict:
        """
        Estimate memory usage for processing
        """
        # Memory multipliers based on file type
        multipliers = {
            ".pdf": 3.0,  # PDF processing can use 3x file size
            ".xlsx": 4.0,  # Excel processing can use 4x file size
            ".xls": 3.5,  # XLS processing
            ".csv": 2.0,  # CSV processing
        }

        multiplier = multipliers.get(file_type, 3.0)
        estimated_memory = file_size * multiplier

        return {
            "file_size": file_size,
            "estimated_memory": estimated_memory,
            "multiplier": multiplier,
            "recommended_streaming": estimated_memory
            > 50 * 1024 * 1024,  # 50MB threshold
        }
