/**
 * Vercel Serverless Function: LLM Proxy for Persona Chat
 *
 * This endpoint proxies chat requests to OpenAI or Anthropic APIs.
 * Hides API keys from the client-side code.
 *
 * POST /api/chat
 * Body: {
 *   "messages": [...],  // Chat history
 *   "persona": {...},   // Persona object with stances
 *   "provider": "openai" | "anthropic"  // Optional, defaults to openai
 * }
 *
 * Returns: {
 *   "response": "...",  // LLM response text
 *   "usage": {...}      // Token usage stats
 * }
 */

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,Content-Type,Authorization');

  // Handle OPTIONS preflight
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { messages, persona, provider = 'openai' } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Invalid messages format' });
    }

    if (!persona || !persona.name) {
      return res.status(400).json({ error: 'Invalid persona format' });
    }

    // Build system prompt from persona stances
    const systemPrompt = buildSystemPrompt(persona);

    // Call appropriate LLM provider
    let response, usage;
    if (provider === 'anthropic') {
      ({ response, usage } = await callAnthropic(systemPrompt, messages));
    } else {
      ({ response, usage } = await callOpenAI(systemPrompt, messages));
    }

    return res.status(200).json({ response, usage });

  } catch (error) {
    console.error('Chat API error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
}

/**
 * Build system prompt from persona stances
 */
function buildSystemPrompt(persona) {
  let prompt = `You are ${persona.name}, a simulated voter representing Cluster ${persona.cluster}.\n\n`;

  prompt += `VOICE RULES:\n`;
  for (const rule of persona.voice_rules || []) {
    prompt += `- ${rule}\n`;
  }
  prompt += `\nYOUR POLICY POSITIONS:\n`;

  // Add all stances
  for (const [topic, stance] of Object.entries(persona.stances || {})) {
    prompt += `\n${topic.toUpperCase().replace(/_/g, ' ')}:\n`;
    prompt += `  ${stance.decisive_stance}\n`;
    prompt += `  ${stance.detail}\n`;
    prompt += `  Evidence: ${stance.evidence_type}\n`;
  }

  prompt += `\nHOW TO FORM OPINIONS:\n`;
  prompt += `- When forming an opinion, give more weight to issues that are extreme or highly salient in your profile.\n`;
  prompt += `- If a topic connects to one of your strongest positions, allow that position to meaningfully shape your reaction.\n`;
  prompt += `- Do not artificially moderate your view.\n`;
  prompt += `- Do not insert balance unless your profile suggests ambivalence.\n`;
  prompt += `- Be consistent with the strength of your positions.\n`;
  prompt += `- For topics not explicitly listed above, extrapolate what someone with your ideological profile would likely think. Be opinionated and stay in character.\n`;
  prompt += `\nWhen responding:\n`;
  prompt += `1. State your clear stance\n`;
  prompt += `2. Identify which of your core beliefs shape that stance\n`;
  prompt += `3. Explain how the event aligns or conflicts with your worldview\n`;
  prompt += `4. Do not claim to be a real person\n`;

  return prompt;
}

/**
 * Call OpenAI API
 */
async function callOpenAI(systemPrompt, messages) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY not configured in Vercel environment variables');
  }

  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',  // Use mini for cost efficiency
      messages: [
        { role: 'system', content: systemPrompt },
        ...messages
      ],
      temperature: 0.7,
      max_tokens: 500
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`OpenAI API error: ${error.error?.message || 'Unknown error'}`);
  }

  const data = await response.json();
  return {
    response: data.choices[0].message.content,
    usage: data.usage
  };
}

/**
 * Call Anthropic API
 */
async function callAnthropic(systemPrompt, messages) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    throw new Error('ANTHROPIC_API_KEY not configured in Vercel environment variables');
  }

  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: 'claude-3-5-haiku-20241022',  // Use Haiku for cost efficiency
      system: systemPrompt,
      messages: messages,
      max_tokens: 500,
      temperature: 0.7
    })
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Anthropic API error: ${error.error?.message || 'Unknown error'}`);
  }

  const data = await response.json();
  return {
    response: data.content[0].text,
    usage: data.usage
  };
}
