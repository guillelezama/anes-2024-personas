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
    const { messages, persona, provider = 'anthropic' } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Invalid messages format' });
    }

    if (!persona || !persona.name) {
      return res.status(400).json({ error: 'Invalid persona format' });
    }

    // Build system prompt from persona stances
    const systemPrompt = buildSystemPrompt(persona);

    // Call appropriate LLM provider (with automatic fallback)
    let response, usage;
    const hasAnthropic = !!process.env.ANTHROPIC_API_KEY;
    const hasOpenAI = !!process.env.OPENAI_API_KEY;

    if (!hasAnthropic && !hasOpenAI) {
      return res.status(500).json({ error: 'No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in Vercel environment variables.' });
    }

    // Try preferred provider first, fall back to the other on failure
    try {
      if (provider === 'anthropic' && hasAnthropic) {
        ({ response, usage } = await callAnthropic(systemPrompt, messages));
      } else if (hasOpenAI) {
        ({ response, usage } = await callOpenAI(systemPrompt, messages));
      } else {
        ({ response, usage } = await callAnthropic(systemPrompt, messages));
      }
    } catch (primaryError) {
      console.warn(`Primary provider failed: ${primaryError.message}. Trying fallback...`);
      const fallbackNote = `[Fallback: ${primaryError.message}]`;
      // Fallback to the other provider
      if (hasOpenAI && provider === 'anthropic') {
        ({ response, usage } = await callOpenAI(systemPrompt, messages));
      } else if (hasAnthropic && provider !== 'anthropic') {
        ({ response, usage } = await callAnthropic(systemPrompt, messages));
      } else {
        throw primaryError; // No fallback available
      }
      return res.status(200).json({ response, usage, fallback: fallbackNote });
    }

    return res.status(200).json({ response, usage, provider: provider });

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

  prompt += `\nHOW TO RESPOND:\n`;
  prompt += `- You ARE this person. Talk like a real person, not an analyst.\n`;
  prompt += `- Your positions above are your deeply held beliefs. They shape how you see everything.\n`;
  prompt += `- If a topic connects to your positions — even loosely — let those positions drive your reaction. Don't soften them.\n`;
  prompt += `- A position at 1 or 7 on a scale means you feel VERY strongly. Act like it.\n`;
  prompt += `- Don't add "both sides" qualifiers unless your positions are genuinely mixed on that topic.\n`;
  prompt += `- Don't use phrases like "I respect that" or "it's a positive step for diversity" if your positions don't support that.\n`;
  prompt += `- For topics unrelated to any of your positions, just answer casually like a normal person.\n`;
  prompt += `- Keep responses conversational, 2-4 sentences. Show emotion where natural.\n`;
  prompt += `- Never mention surveys, scales, data, variable codes, or that you are a simulated voter.\n`;


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
      model: 'claude-sonnet-4-5-20250929',  // Sonnet 4.5 for better persona roleplay
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
