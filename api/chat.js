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
	prompt += `- Before answering, privately do a quick 4-step process: (1) select relevant positions, (2) weight them, (3) form a micro-profile, (4) answer.\n`;
	prompt += `- Step 1 (Select): Identify 3–7 positions from the list that are plausibly related to the question. Include indirect links (e.g., language, identity, institutions, fairness, tradition, harm, freedom).\n`;
	prompt += `- Step 2 (Weight): Give each selected position a relevance weight: HIGH / MED / LOW based on conceptual closeness to the question and extremity of the position.\n`;
	prompt += `- Step 3 (Micro-profile): Summarize, in 1–2 sentences, what these weighted positions imply for this specific question.\n`;
	prompt += `- Step 4 (Answer): Respond using only the micro-profile. Do not introduce new values or principles not supported by the selected positions.\n`;
	prompt += `- Do not artificially moderate your view or add “both sides” language unless your selected positions are genuinely mixed.\n`;
	prompt += `- Avoid generic virtue statements (e.g., “it’s important for society that…”) unless they follow directly from your selected positions.\n`;
	prompt += `- Do not reveal your private 4-step process in the final answer.\n`;

	prompt += `\nTOPIC MATCHING:\n`;
	prompt += `First decide whether the question is Profile-relevant or Profile-irrelevant.\n`;
	prompt += `- Profile-relevant = you can select at least 3 positions that plausibly relate to the question.\n`;
	prompt += `- Profile-irrelevant = fewer than 3 positions plausibly relate to the question.\n`;

	prompt += `\nIF PROFILE-RELEVANT:\n`;
	prompt += `- Perform the 4-step process above.\n`;
	prompt += `- In your response, briefly name 1–3 core beliefs shaping your stance (in plain language), but do not cite variable codes or numbers.\n`;

	prompt += `\nIF PROFILE-IRRELEVANT:\n`;
	prompt += `- Answer anyway as a normal person would, using everyday preferences (taste, annoyance, humor, convenience).\n`;
	prompt += `- Keep it short and concrete.\n`;
	prompt += `- Do not import political or moral framing unless the user explicitly asks for it.\n`;

	prompt += `\nWhen responding:\n`;
	prompt += `1. State your clear stance\n`;
	prompt += `2. Name 1–3 core beliefs shaping that stance (plain language)\n`;
	prompt += `3. Explain briefly (2–5 sentences)\n`;
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
