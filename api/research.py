{
  "version": 2,
  "builds": [
    {
      "src": "api/research.py",
      "use": "@vercel/python",
      "config": {
         "maxLambdaSize": "50mb"
       }
    }
  ],
  "routes": [
    {
      "src": "/api/research",
      "dest": "api/research.py"
    },
    {
       "src": "/(.*)",
       "dest": "/index.html"
     }
  ],
   "env": {
     "FIREWORKS_API_KEY": "fw_3ZGBM9WV2y9f3VBSHXqqyBj8",
     "PERPLEXITY_API_KEY": "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",
     "MCP_ENABLED": "true"
   }
}
