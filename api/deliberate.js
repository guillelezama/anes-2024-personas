/**
 * Vercel Serverless Function: Deliberation Orchestrator
 *
 * Orchestrates a 4-phase structured deliberation between 2-3 voter personas.
 * Each persona is grounded in their stored stances from avatars.json.
 *
 * POST /api/deliberate
 * Body: { topic: string, personas: PersonaObject[] }
 * Response: {
 *   transcript: Array<{ phase, speaker, personaId?, text }>,
 *   summary: { agreements, disagreements, assumptions, evidenceToChange }
 * }
 */

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'X-Requested-With,Content-Type,Authorization');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { topic, personas } = req.body;

    if (!topic || typeof topic !== 'string' || !topic.trim()) {
      return res.status(400).json({ error: 'topic is required' });
    }
    if (!Array.isArray(personas) || personas.length < 2 || personas.length > 3) {
      return res.status(400).json({ error: 'personas must be an array of 2 or 3 persona objects' });
    }
    for (const p of personas) {
      if (!p.name || !p.stances) {
        return res.status(400).json({ error: 'Each persona must have name and stances fields' });
      }
    }

    const transcript = [];

    // Phase 1: Positions -- each persona states their position
    for (const persona of personas) {
      const text = await callPersona(persona, buildPositionPrompt(topic));
      transcript.push({ phase: 'positions', speaker: persona.name, personaId: String(persona.cluster), text });
    }

    // Phase 2: Challenges -- each persona critiques the next (rotating)
    for (let i = 0; i < personas.length; i++) {
      const speaker = personas[i];
      const target = personas[(i + 1) % personas.length];
      const targetStatement = transcript.find(t => t.speaker === target.name && t.phase === 'positions')?.text || '';
      const text = await callPersona(speaker, buildChallengePrompt(topic, target.name, targetStatement));
      transcript.push({ phase: 'challenges', speaker: speaker.name, personaId: String(speaker.cluster), text });
    }

    // Phase 3: Compromise -- each persona proposes what they can live with
    const debateSoFar = formatTranscript(transcript);
    for (const persona of personas) {
      const text = await callPersona(persona, buildCompromisePrompt(topic, debateSoFar));
      transcript.push({ phase: 'compromise', speaker: persona.name, personaId: String(persona.cluster), text });
    }

    // Phase 4: Mediator -- joint statement + structured summary
    const fullTranscript = formatTranscript(transcript);
    const names = personas.map(p => p.name).join(', ');
    const { jointStatement, summary } = await callMediator(topic, names, fullTranscript);
    transcript.push({ phase: 'joint', speaker: 'Mediator', text: jointStatement });

    return res.status(200).json({ transcript, summary });

  } catch (error) {
    console.error('Deliberate API error:', error);
    return res.status(500).json({ error: 'Internal server error', message: error.message });
  }
}

// ============================================================================
// PERSONA SYSTEM PROMPT
// ============================================================================

function buildPersonaSystem(persona) {
  let prompt = `You are ${persona.name}, an American voter taking part in a structured political deliberation.\n\n`;

  prompt += `YOUR POLICY POSITIONS:\n`;
  for (const [key, stance] of Object.entries(persona.stances || {})) {
    prompt += `- ${key.replace(/_/g, ' ')}: ${stance.decisive_stance}`;
    if (stance.value !== undefined) prompt += ` (value: ${stance.value.toFixed(1)})`;
    prompt += '\n';
  }

  prompt += `\nRULES:\n`;
  prompt += `- Speak only in first person as this voter. Never mention surveys, data, clusters, scales, or AI.\n`;
  prompt += `- Do not invent statistics or cite specific studies. Speak from lived experience and values.\n`;
  prompt += `- Be direct and honest. Point out real tradeoffs and acknowledge genuine uncertainty.\n`;
  prompt += `- Do not soften your disagreements artificially or add "I respect that" unless it follows from your actual positions.\n`;
  prompt += `- Do not use "both sides" framing unless your positions are genuinely mixed on this topic.\n`;
  prompt += `- Stay under 120 words.\n`;

  return prompt;
}

// ============================================================================
// PHASE PROMPT BUILDERS
// ============================================================================

function buildPositionPrompt(topic) {
  return `The topic is: "${topic}"

State your position in four parts:
1. My position: (one clear sentence)
2. Why people like me think this: (one or two sentences grounded in your values and experience)
3. My biggest concern: (one sentence)
4. What would change my mind: (be honest -- one sentence)

Stay under 120 words total.`;
}

function buildChallengePrompt(topic, targetName, targetStatement) {
  return `The topic is: "${topic}"

${targetName} said:
"${targetStatement}"

Respond directly to what ${targetName} said. Identify the part you find weakest or most wrong and explain why from your own values and experience. Do not agree for the sake of politeness. Note any tradeoffs or uncertainties you think they are ignoring. Stay under 120 words.`;
}

function buildCompromisePrompt(topic, debateSoFar) {
  return `The topic is: "${topic}"

Here is the debate so far:
${debateSoFar}

Now propose a compromise in two parts:
1. A version I could live with: (a concrete policy or outcome you could accept, even if not ideal)
2. My red line: (the one thing that would make any deal unacceptable to you)

Be honest. Do not pretend to agree more than you actually do. Stay under 120 words.`;
}

// ============================================================================
// MEDIATOR
// ============================================================================

function buildMediatorSystem() {
  return `You are an impartial political science mediator. Your job is to analyze a structured deliberation and produce a fair, accurate synthesis. Do not favor any participant. Do not invent facts or statistics. Identify real agreements, real disagreements, and the underlying values driving differences. Use plain language.`;
}

function buildMediatorPrompt(topic, names, fullTranscript) {
  return `Topic: "${topic}"
Participants: ${names}

Full deliberation transcript:
${fullTranscript}

Produce a synthesis as valid JSON with exactly this structure:
{
  "jointStatement": ["bullet 1", "bullet 2", "bullet 3"],
  "agreements": ["..."],
  "disagreements": ["..."],
  "assumptions": ["..."],
  "evidenceToChange": ["..."]
}

Rules:
- jointStatement: 3 to 6 bullets. Each under 30 words. Identify genuine shared ground and shared facts, even where views differ. Total under 180 words.
- agreements: 2 to 5 bullets of genuine shared positions or shared concerns.
- disagreements: 2 to 5 bullets of unresolved differences that persist after the compromise phase.
- assumptions: 2 to 5 bullets naming the core values or empirical assumptions driving disagreement (e.g. "X assumes enforcement deters behavior; Y assumes it does not").
- evidenceToChange: one item per participant by name, describing what kind of evidence or argument would realistically shift their position.

Return only the JSON object. No text before or after.`;
}

async function callMediator(topic, names, fullTranscript) {
  const systemPrompt = buildMediatorSystem();
  const userPrompt = buildMediatorPrompt(topic, names, fullTranscript);
  const raw = await callLLM(systemPrompt, [{ role: 'user', content: userPrompt }], 0);
  return parseMediatorOutput(raw);
}

// ============================================================================
// LLM HELPERS
// ============================================================================

async function callPersona(persona, userPrompt) {
  const systemPrompt = buildPersonaSystem(persona);
  return await callLLM(systemPrompt, [{ role: 'user', content: userPrompt }], 0.7);
}

async function callLLM(systemPrompt, messages, temperature = 0.7) {
  const hasAnthropic = !!process.env.ANTHROPIC_API_KEY;
  const hasOpenAI = !!process.env.OPENAI_API_KEY;

  if (!hasAnthropic && !hasOpenAI) {
    throw new Error('No API key configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in environment variables.');
  }

  try {
    if (hasAnthropic) return await callAnthropic(systemPrompt, messages, temperature);
    return await callOpenAI(systemPrompt, messages, temperature);
  } catch (primaryError) {
    if (hasAnthropic && hasOpenAI) {
      console.warn(`Primary provider failed: ${primaryError.message}. Trying fallback...`);
      return await callOpenAI(systemPrompt, messages, temperature);
    }
    throw primaryError;
  }
}

async function callAnthropic(systemPrompt, messages, temperature) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: 'claude-sonnet-4-5-20250929',
      system: systemPrompt,
      messages,
      max_tokens: 600,
      temperature
    })
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(`Anthropic API error: ${err.error?.message || 'Unknown error'}`);
  }
  const data = await response.json();
  return data.content[0].text;
}

async function callOpenAI(systemPrompt, messages, temperature) {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      model: 'gpt-4o-mini',
      messages: [{ role: 'system', content: systemPrompt }, ...messages],
      temperature,
      max_tokens: 600
    })
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(`OpenAI API error: ${err.error?.message || 'Unknown error'}`);
  }
  const data = await response.json();
  return data.choices[0].message.content;
}

// ============================================================================
// UTILITIES
// ============================================================================

function formatTranscript(transcript) {
  return transcript
    .map(t => `[${t.phase.toUpperCase()}] ${t.speaker}: ${t.text}`)
    .join('\n\n');
}

function trimToWordLimit(text, limit) {
  const words = text.trim().split(/\s+/);
  if (words.length <= limit) return text.trim();
  return words.slice(0, limit).join(' ') + '...';
}

function parseMediatorOutput(raw) {
  try {
    let cleaned = raw
      .replace(/^```json\s*/i, '')
      .replace(/^```\s*/i, '')
      .replace(/```\s*$/i, '')
      .trim();
    // If the model added preamble text, extract the first {...} block
    if (!cleaned.startsWith('{')) {
      const match = cleaned.match(/\{[\s\S]*\}/);
      if (match) cleaned = match[0];
    }
    const parsed = JSON.parse(cleaned);
    const jointStatement = (parsed.jointStatement || []).map(b => `- ${b}`).join('\n');
    return {
      jointStatement,
      summary: {
        agreements: parsed.agreements || [],
        disagreements: parsed.disagreements || [],
        assumptions: parsed.assumptions || [],
        evidenceToChange: parsed.evidenceToChange || []
      }
    };
  } catch {
    return {
      jointStatement: '(Joint statement could not be generated. Please try again.)',
      summary: {
        agreements: ['Summary generation failed. Please try again.'],
        disagreements: [],
        assumptions: [],
        evidenceToChange: []
      }
    };
  }
}
