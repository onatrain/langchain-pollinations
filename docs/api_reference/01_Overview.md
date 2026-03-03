## Overview

The `langchain-pollinations` library provides a comprehensive LangChain-compatible interface to the Pollinations.ai API. This reference covers the main classes exposed by the library, their instantiation parameters, methods, and return types.

## Installation

```bash
pip install langchain-pollinations
```

## Authentication

All classes require authentication via an API key, which can be provided in two ways:

1. **Environment variable**: Set `POLLINATIONS_API_KEY` in your environment
2. **Constructor parameter**: Pass `api_key="your-key"` when instantiating

If no key is provided, a `ValueError` will be raised immediately.
