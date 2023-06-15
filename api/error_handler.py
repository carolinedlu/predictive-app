from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class CustomAPIError(Exception):
    def __init__(self, message):
        super().__init__(message)


class BadRequestError(CustomAPIError):
    def __init__(self, message):
        super().__init__(message)


class NotFoundError(CustomAPIError):
    def __init__(self, message):
        super().__init__(message)


async def handle_bad_request_error(request, exc):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )


async def handle_not_found_error(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": str(exc)},
    )
