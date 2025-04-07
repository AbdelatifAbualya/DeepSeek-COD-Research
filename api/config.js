// api/config.js
// This file handles secure environment variable access for API routes

/**
 * Get environment variables safely
 * This function centralizes access to environment variables and provides
 * clear error messages when required variables are missing
 */
function getEnvVar(name, required = true) {
  const value = process.env[name];
  
  if (!value && required) {
    throw new Error(`Required environment variable ${name} is not set. Please configure it in your Vercel dashboard.`);
  }
  
  return value;
}

/**
 * Configuration object with all environment variables
 * Centralized access to prevent scattered process.env calls throughout the codebase
 */
const config = {
  // API Keys
  fireworks: {
    apiKey: getEnvVar('FIREWORKS_API_KEY'),
    endpoint: 'https://api.fireworks.ai/inference/v1/chat/completions'
  },
  
  mcp: {
    apiKey: getEnvVar('MCP_API_KEY'),
    endpoint: 'https://api.mcp.so/v1'
  },
  
  // Optional API keys
  perplexity: {
    apiKey: getEnvVar('PERPLEXITY_API_KEY', false),
    endpoint: 'https://api.perplexity.ai/chat/completions'
  },
  
  // Development flags
  isDevelopment: process.env.NODE_ENV === 'development',
  
  // Access control (optional)
  accessToken: getEnvVar('ACCESS_TOKEN', false),
  
  // Rate limiting
  rateLimit: {
    enabled: process.env.RATE_LIMIT_ENABLED === 'true',
    requestsPerMinute: parseInt(process.env.RATE_LIMIT_RPM || '30', 10)
  }
};

module.exports = config;

// api/fireworks-proxy.js
// Example of using the config in an API route
const config = require('./config');

module.exports = async (req, res) => {
  try {
    // Access API key through config
    const apiKey = config.fireworks.apiKey;
    
    // Rest of your API proxy implementation
    const response = await fetch(config.fireworks.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(req.body)
    });
    
    // Handle response...
  } catch (error) {
    console.error('API proxy error:', error);
    res.status(500).json({ error: error.message });
  }
};

// api/mcp-proxy.js 
// Example of MCP integration with environment variables
const config = require('./config');

module.exports = async (req, res) => {
  try {
    // Access MCP API key through config
    const apiKey = config.mcp.apiKey;
    
    // Get tool name from request
    const { toolName, query, options } = req.body;
    
    if (!toolName || !query) {
      return res.status(400).json({ 
        error: 'Missing required parameters: toolName and query are required' 
      });
    }
    
    // Call MCP API
    const response = await fetch(`${config.mcp.endpoint}/agents/${toolName}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({
        query: query,
        options: options || {}
      })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`MCP API error (${response.status}): ${errorText}`);
    }
    
    const data = await response.json();
    
    // Return the MCP tool results
    res.status(200).json(data);
  } catch (error) {
    console.error('MCP proxy error:', error);
    res.status(500).json({ error: error.message });
  }
};

// api/chain-of-draft.js
// Example of Chain of Draft implementation with environment variables
const config = require('./config');

module.exports = async (req, res) => {
  try {
    // Access API keys through config
    const fireworksApiKey = config.fireworks.apiKey;
    const mcpApiKey = config.mcp.apiKey;
    
    // Get request parameters
    const { query, draftCount = 3, useMcp = false, tool = null } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Missing required parameter: query' });
    }
    
    // Check if we need to use MCP tools
    if (useMcp && tool) {
      try {
        // Call MCP API with the specified tool
        const mcpResponse = await fetch(`${config.mcp.endpoint}/agents/${tool}`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${mcpApiKey}`
          },
          body: JSON.stringify({
            query: query,
            options: {
              reasoning_method: 'chain_of_draft',
              draft_count: draftCount
            }
          })
        });
        
        if (!mcpResponse.ok) {
          throw new Error(`MCP API error: ${mcpResponse.status}`);
        }
        
        const mcpData = await mcpResponse.json();
        
        // Return the MCP results
        return res.status(200).json({
          response: mcpData.result,
          tool: tool,
          metadata: mcpData.metadata || {}
        });
      } catch (mcpError) {
        console.error('MCP error:', mcpError);
        // Fall back to standard processing if MCP fails
      }
    }
    
    // Standard Chain of Draft processing with DeepSeek
    const chainOfDraftPrompt = `
      I want you to answer the query using Chain of Draft reasoning.
      Query: ${query}
      
      For your response:
      1. Create ${draftCount} progressive drafts of your reasoning
      2. Each draft should build upon and refine the previous draft
      3. Label each draft as "Draft 1:", "Draft 2:", etc.
      4. Make your reasoning more precise and focused with each draft
      
      Before providing your final answer, add a reflection step starting with "Reflection:" 
      to verify your work and catch any potential errors.
      
      Write your final answer after the #### separator.
    `;
    
    // Call Fireworks API with DeepSeek V3
    const fireworksResponse = await fetch(config.fireworks.endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${fireworksApiKey}`
      },
      body: JSON.stringify({
        model: "accounts/fireworks/models/deepseek-v3-0324",
        messages: [
          {
            role: "user",
            content: chainOfDraftPrompt
          }
        ],
        temperature: 0.7,
        max_tokens: 4096
      })
    });
    
    if (!fireworksResponse.ok) {
      throw new Error(`Fireworks API error: ${fireworksResponse.status}`);
    }
    
    const fireworksData = await fireworksResponse.json();
    const modelResponse = fireworksData.choices[0]?.message?.content || '';
    
    // Process the response to extract drafts, reflection, and final answer
    const processedResponse = processChainOfDraft(modelResponse);
    
    // Return the processed response
    res.status(200).json({
      response: modelResponse,
      processed: processedResponse
    });
  } catch (error) {
    console.error('Chain of Draft API error:', error);
    res.status(500).json({ error: error.message });
  }
};

// Helper function to process Chain of Draft responses
function processChainOfDraft(content) {
  // Extract final answer
  const separatorIndex = content.indexOf("####");
  if (separatorIndex === -1) {
    return {
      drafts: [],
      reflection: null,
      answer: content,
      hasReflection: false
    };
  }
  
  const drafting = content.substring(0, separatorIndex).trim();
  const answer = content.substring(separatorIndex + 4).trim();
  
  // Extract reflection
  const reflectionIndex = drafting.toLowerCase().lastIndexOf("reflection:");
  const hasReflection = reflectionIndex !== -1;
  let reflection = null;
  
  if (hasReflection) {
    reflection = drafting.substring(reflectionIndex).trim();
  }
  
  // Extract drafts
  const drafts = [];
  const draftRegex = /Draft\s+(\d+):/gi;
  
  let match;
  let lastIndex = 0;
  let indices = [];
  
  // Find all draft markers
  while ((match = draftRegex.exec(drafting)) !== null) {
    indices.push({
      index: match.index,
      draftNumber: parseInt(match[1])
    });
  }
  
  // Extract each draft's content
  for (let i = 0; i < indices.length; i++) {
    const startIndex = indices[i].index;
    const endIndex = (i < indices.length - 1) ? indices[i + 1].index : 
                     (hasReflection ? reflectionIndex : drafting.length);
    
    const draftContent = drafting.substring(startIndex, endIndex).trim();
    drafts.push({
      number: indices[i].draftNumber,
      content: draftContent
    });
  }
  
  return {
    drafts,
    reflection,
    answer,
    hasReflection
  };
}
