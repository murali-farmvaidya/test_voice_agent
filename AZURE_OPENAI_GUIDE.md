# Azure OpenAI Integration Guide

## Quick Setup

### 1. Get Azure OpenAI Access

1. **Sign up for Azure**: https://azure.microsoft.com
2. **Request Azure OpenAI access**: https://aka.ms/oai/access
   - Usually approved within 1-2 business days for business accounts
3. **Create Azure OpenAI Resource**:
   - Go to Azure Portal: https://portal.azure.com
   - Create resource → AI + Machine Learning → Azure OpenAI
   - Choose region (e.g., East US, UK South)
   - Select pricing tier (Standard S0)

### 2. Create a Deployment

1. Go to **Azure OpenAI Studio**: https://oai.azure.com/
2. Select your resource
3. Navigate to **Deployments** → **Create new deployment**
4. Configure:
   - **Model**: gpt-4o (recommended) or gpt-4
   - **Deployment name**: `gpt-4o` (you'll use this in .env)
   - **Capacity**: Start with default, increase if needed
5. Click **Create**

### 3. Get Your Credentials

1. Go to **Azure Portal** → Your OpenAI Resource
2. Click **Keys and Endpoint** in left menu
3. Copy:
   - **KEY 1** or **KEY 2** (either works)
   - **Endpoint** (e.g., `https://your-resource.openai.azure.com/`)

### 4. Configure Your Bot

Update your `.env` file:

```env
# Set provider to Azure
LLM_PROVIDER=azure

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=PASTE_YOUR_KEY_HERE
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

### 5. Test It

```bash
# Start the bot
uv run bot.py

# Open browser
# Go to http://localhost:7860/client
# Start talking!
```

---

## Switching Between Providers

### Use Azure OpenAI

```env
LLM_PROVIDER=azure
```

### Use Standard OpenAI

```env
LLM_PROVIDER=openai
```

That's it! Just change one line and restart the bot.

---

## Troubleshooting

### Error: 404 - Resource not found

**Cause**: Deployment name mismatch

**Fix**:
1. Go to Azure OpenAI Studio
2. Check your deployment name under "Deployments"
3. Update `.env`: `AZURE_OPENAI_DEPLOYMENT=exact-deployment-name`

### Error: Authentication failed

**Cause**: Wrong API key or expired

**Fix**:
1. Go to Azure Portal → Your Resource → Keys and Endpoint
2. Copy fresh KEY 1
3. Update `.env`: `AZURE_OPENAI_API_KEY=new_key`
4. Restart bot

### Error: Rate limit exceeded

**Cause**: Too many requests, need more capacity

**Fix**:
1. Go to Azure OpenAI Studio → Deployments
2. Increase tokens-per-minute (TPM) capacity
3. Or wait a minute and retry

---

## Azure vs Standard OpenAI

| Feature | Azure OpenAI | Standard OpenAI |
|---------|-------------|----------------|
| **Setup** | More complex (Azure account, approval) | Simple (API key) |
| **Pricing** | Similar, pay via Azure billing | Pay-as-you-go |
| **Security** | Enterprise-grade, private network | Standard cloud |
| **Compliance** | HIPAA, SOC 2, ISO compliant | Standard |
| **SLA** | 99.9% uptime guarantee | Best effort |
| **Data Privacy** | Data stays in your Azure tenant | Processed by OpenAI |
| **Best For** | Enterprise, regulated industries | Developers, startups |

---

## Cost Optimization

### Azure OpenAI Pricing

- **GPT-4o**: ~$0.005 per 1K tokens input, ~$0.015 per 1K tokens output
- **Average conversation**: ~5K tokens = ~$0.05

### Save Money

1. **Use PTU (Provisioned Throughput)**
   - Fixed monthly cost for guaranteed capacity
   - Cost-effective for high volume (>1M tokens/month)

2. **Use GPT-4o-mini** (if available)
   - Cheaper than GPT-4o
   - Good for simpler conversations

3. **Optimize prompts**
   - Keep system prompts concise
   - Limit conversation history

---

## Advanced Configuration

### Custom API Version

If you need a specific API version:

```env
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

Check latest versions: https://learn.microsoft.com/azure/ai-services/openai/reference

### Multiple Deployments

Create separate deployments for:
- **Production**: High capacity, always available
- **Development**: Lower cost, testing
- **Staging**: Mid-tier, pre-prod testing

Switch in `.env`:
```env
# Use production deployment
AZURE_OPENAI_DEPLOYMENT=gpt-4o-prod

# Or development deployment
AZURE_OPENAI_DEPLOYMENT=gpt-4o-dev
```

---

## Technical Details

### How It Works

The bot uses a custom `AzureOpenAILLMService` class that:

1. Extends Pipecat's `OpenAILLMService`
2. Overrides `create_client()` method
3. Adds `default_query={"api-version": "..."}` parameter
4. Constructs proper Azure OpenAI URL format

This ensures compatibility with Azure's API while using standard OpenAI SDK.

### Code Reference

See `bot.py` lines 65-95:

```python
class AzureOpenAILLMService(OpenAILLMService):
    """Custom OpenAI LLM Service for Azure OpenAI."""
    
    def create_client(self, ...):
        # Adds api-version query parameter
        client_kwargs["default_query"] = {
            "api-version": self._azure_api_version
        }
        return AsyncOpenAI(**client_kwargs)
```

---

## Resources

- **Azure OpenAI Docs**: https://learn.microsoft.com/azure/ai-services/openai/
- **Azure OpenAI Studio**: https://oai.azure.com/
- **Azure Portal**: https://portal.azure.com
- **API Reference**: https://learn.microsoft.com/azure/ai-services/openai/reference

---

**Questions?** Open an issue: https://github.com/farmvaidya-ai/pipecat_agent_v2/issues
