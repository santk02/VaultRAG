@echo off
REM Setup script for VaultRAG environment
REM This script creates a .env file from .env.example

echo Creating .env file from .env.example...
copy .env.example .env
echo .env file created successfully.
echo.
echo Please edit .env to configure your specific settings.
echo Default settings are configured for local development with Ollama.
pause
