import { synthesizeResearch } from '../utils/fireworks-api';

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { topic, depth, style, model, temperature, maxTokens } = req.body;
    
    const response = await synthesizeResearch({
      topic,
      depth,
      style,
      model,
      temperature,
      maxTokens
    });
    
    return res.status(200).json(response);
  } catch (error) {
    console.error('Error in research synthesis:', error);
    return res.status(500).json({ error: 'Failed to synthesize research' });
  }
}
