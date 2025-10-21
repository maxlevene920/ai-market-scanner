# AI Market Scanner

An intelligent Python application that scans various market sources for mentions and uses AI agents to analyze market sentiment, trends, and provide actionable insights.

## Features

- **Multi-Source Scanning**: Collects market mentions from RSS feeds, Reddit, and news APIs
- **AI-Powered Analysis**: Uses OpenAI or Anthropic models for sentiment and trend analysis
- **Comprehensive Insights**: Provides market sentiment, trend analysis, and recommendations
- **Data Management**: Saves and exports results in JSON and CSV formats
- **Command-Line Interface**: Easy-to-use CLI for all operations
- **Configurable**: Customizable keywords, sources, and analysis parameters

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-market-scanner
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Create a `.env` file with your API keys:

```env
# AI Provider API Keys (at least one required)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: News API Key
NEWS_API_KEY=your_news_api_key_here

# Optional: Social Media API Keys
TWITTER_API_KEY=your_twitter_api_key_here
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
```

## Usage

### Basic Commands

#### Scan Markets
```bash
# Scan with default keywords
python main.py scan

# Scan with custom keywords
python main.py scan -k "crypto" -k "bitcoin" -k "ethereum"

# Scan and save results
python main.py scan -k "stocks" -s --output-format json
```

#### Analyze Market Data
```bash
# Perform comprehensive analysis
python main.py analyze

# Analyze with specific keywords
python main.py analyze -k "crypto" -k "defi"

# Perform sentiment analysis only
python main.py analyze -t sentiment

# Analyze existing mentions file
python main.py analyze -f mentions_20241201_120000.json
```

#### System Management
```bash
# Check system status
python main.py status

# List saved files
python main.py list-files

# Clean up old files
python main.py cleanup --days 30
```

### Advanced Usage

#### Custom Analysis
```python
import asyncio
from src.scanner import MarketScanner
from src.agent import MarketAgent

async def custom_analysis():
    # Initialize scanner and agent
    scanner = MarketScanner()
    agent = MarketAgent()
    
    # Scan for mentions
    mentions = await scanner.scan_markets(["crypto", "defi"], max_results=100)
    
    # Analyze mentions
    result = await agent.analyze(mentions)
    
    # Print insights
    for insight in result.insights:
        print(f"Insight: {insight}")
    
    for recommendation in result.recommendations:
        print(f"Recommendation: {recommendation}")

# Run the analysis
asyncio.run(custom_analysis())
```

## Architecture

### Core Components

1. **Scanner Module** (`src/scanner/`)
   - `BaseScanner`: Abstract base class for all scanners
   - `NewsScanner`: Scans news APIs for market mentions
   - `RedditScanner`: Scans Reddit for market discussions
   - `RSSScanner`: Scans RSS feeds for market news
   - `MarketScanner`: Orchestrates multiple scanners

2. **Agent Module** (`src/agent/`)
   - `BaseAgent`: Abstract base class for AI agents
   - `SentimentAnalyzer`: Analyzes market sentiment
   - `TrendAnalyzer`: Identifies market trends
   - `MarketAgent`: Comprehensive market analysis

3. **Configuration** (`src/config/`)
   - `Settings`: Centralized configuration management
   - Environment variable handling
   - API key management

4. **Utilities** (`src/utils/`)
   - `Logger`: Structured logging system
   - `DataProcessor`: Data cleaning and processing
   - `FileManager`: File operations and persistence

### Data Flow

```
Market Sources → Scanners → Market Mentions → AI Agents → Analysis Results
     ↓              ↓           ↓              ↓            ↓
   RSS/Reddit   Rate Limiting  Cleaning    Sentiment    Insights &
   News APIs    Error Handling Validation  Trend Analysis Recommendations
```

## API Integration

### Supported AI Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo
- **Anthropic**: Claude-3, Claude-2

### Supported Data Sources

- **RSS Feeds**: Yahoo Finance, Reuters, Bloomberg, CNBC, MarketWatch
- **Reddit**: Investment and crypto subreddits
- **News APIs**: NewsAPI.org (requires API key)

## Data Formats

### Market Mention
```json
{
  "source": "Yahoo Finance",
  "title": "Bitcoin Reaches New High",
  "content": "Bitcoin price surges to $50,000...",
  "url": "https://finance.yahoo.com/news/bitcoin-reaches-new-high",
  "published_at": "2024-01-01T12:00:00",
  "sentiment_score": 0.8,
  "keywords_found": ["bitcoin", "crypto"],
  "relevance_score": 0.9
}
```

### Analysis Result
```json
{
  "analysis_type": "comprehensive",
  "timestamp": "2024-01-01T12:00:00",
  "confidence_score": 0.85,
  "insights": [
    "Market sentiment is bullish",
    "Crypto mentions increased 30%"
  ],
  "recommendations": [
    "Consider increasing crypto allocation",
    "Monitor regulatory developments"
  ],
  "data": {
    "sentiment_scores": {...},
    "trend_metrics": {...}
  }
}
```

## Configuration Options

### Scanner Configuration
```python
# In src/config/settings.py
@dataclass
class ScannerConfig:
    scan_interval_minutes: int = 30
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_attempts: int = 3
    keywords: List[str] = ["market", "trading", "investment"]
    exclude_keywords: List[str] = ["spam", "scam"]
    min_mention_threshold: int = 5
```

### Agent Configuration
```python
@dataclass
class AgentConfig:
    model_provider: str = "openai"
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    analysis_depth: str = "medium"
```

## Error Handling

The application includes comprehensive error handling:

- **Network Errors**: Automatic retries with exponential backoff
- **API Errors**: Graceful degradation when APIs are unavailable
- **Data Validation**: Input validation and cleaning
- **Logging**: Detailed logging for debugging and monitoring

## Performance Considerations

- **Rate Limiting**: Respects API rate limits for all sources
- **Concurrent Processing**: Uses asyncio for efficient I/O operations
- **Caching**: Implements caching to reduce API calls
- **Memory Management**: Efficient data structures and cleanup

## Security

- **API Key Protection**: Environment variables for sensitive data
- **Input Validation**: Sanitizes all inputs
- **Error Handling**: Prevents information leakage in error messages
- **Rate Limiting**: Prevents abuse of external APIs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the repository
- Check the documentation
- Review the logs for error details

## Roadmap

- [ ] Web interface for analysis results
- [ ] Real-time market monitoring
- [ ] Advanced sentiment analysis models
- [ ] Integration with trading platforms
- [ ] Machine learning trend prediction
- [ ] Social media sentiment analysis
- [ ] Portfolio impact analysis
