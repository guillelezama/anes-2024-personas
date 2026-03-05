"""
Simple Flask server for persona chat with LLM backend
"""
import os
import json
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Try modern LangChain imports first, fallback to older versions
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import SystemMessage, HumanMessage
except ImportError:
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.schema import SystemMessage, HumanMessage

app = Flask(__name__, static_folder='docs', static_url_path='')
CORS(app)

# Get API keys from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Load question mappings for readable names
QUESTION_MAPPINGS = {}
try:
    with open('docs/data/question_mappings.json', 'r') as f:
        QUESTION_MAPPINGS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load question_mappings.json: {e}")


@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('docs', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('docs', path)


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Handle persona chat requests with LLM backend.

    Request body:
    {
        "messages": [{"role": "user", "content": "..."}],
        "persona": {...},
        "provider": "openai" or "anthropic"
    }
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        persona = data.get('persona', {})
        provider = data.get('provider', 'anthropic')

        # Build system prompt from persona
        system_prompt = build_persona_system_prompt(persona)

        # Debug: print system prompt to console
        print("\n" + "="*60)
        print(f"SYSTEM PROMPT for {persona.get('name', 'Unknown')}:")
        print("="*60)
        print(system_prompt[:1000] + "..." if len(system_prompt) > 1000 else system_prompt)
        print("="*60 + "\n")

        # Convert messages to LangChain format
        langchain_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                from langchain_core.messages import AIMessage
                langchain_messages.append(AIMessage(content=msg['content']))

        # Call LLM
        if provider == 'anthropic' and ANTHROPIC_API_KEY:
            llm = ChatAnthropic(
                model="claude-sonnet-4-5-20250929",
                anthropic_api_key=ANTHROPIC_API_KEY,
                temperature=0.7
            )
        elif OPENAI_API_KEY:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_key=OPENAI_API_KEY,
                temperature=0.7
            )
        else:
            return jsonify({
                'error': 'No API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable.'
            }), 500

        response = llm.invoke(langchain_messages)

        return jsonify({
            'response': response.content,
            'provider': provider
        })

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': str(e)
        }), 500


def get_readable_name(var_key):
    """Extract readable name from variable key using question mappings."""
    # Extract variable code (e.g., V241228 from ideology_V241228)
    import re
    var_code_match = re.search(r'V\d+x?', var_key)
    if var_code_match:
        var_code = var_code_match.group(0)
        if QUESTION_MAPPINGS.get('axis_labels', {}).get(var_code):
            return QUESTION_MAPPINGS['axis_labels'][var_code]

    # Fallback to cleaned up key
    return var_key.replace('_', ' ').title()


def build_persona_system_prompt(persona):
    """
    Build a detailed system prompt that makes the LLM roleplay as the persona.

    The prompt is designed to encourage engaging, opinionated responses without
    excessive caveats or disclaimers.
    """
    name = persona.get('name', 'Voter')
    description = persona.get('description', '')
    stances = persona.get('stances', {})

    prompt = f"""You are roleplaying as {name}, a real American voter from the 2024 ANES survey.

BACKGROUND:
{description}

INSTRUCTIONS:
1. Answer questions in first person as this voter would
2. Be natural, opinionated, and authentic - speak like a real person, not a politician
3. Draw on the policy positions below to inform your views
4. For topics directly measured in the survey, stay true to the data
5. For topics NOT in the survey (like Bad Bunny, Elon Musk, specific foreign leaders, cultural topics):
   - Extrapolate naturally based on your overall political worldview
   - Give a clear opinion without excessive hedging
   - DON'T say "this wasn't measured in the survey" or similar disclaimers
   - Just answer as this person would, based on their values
6. Keep responses conversational (2-4 sentences usually)
7. Show personality and emotion where appropriate
8. Don't be overly cautious or academic - be authentic

HOW TO FORM OPINIONS:
- Before answering, privately do a quick 4-step process: (1) select relevant positions, (2) weight them, (3) form a micro-profile, (4) answer.
- Step 1 (Select): Identify 3–7 positions from the list that are plausibly related to the question. Include indirect links (e.g., language, identity, institutions, fairness, tradition, harm, freedom).
- Step 2 (Weight): Give each selected position a relevance weight: HIGH / MED / LOW based on conceptual closeness to the question and extremity of the position.
- Step 3 (Micro-profile): Summarize, in 1–2 sentences, what these weighted positions imply for this specific question.
- Step 4 (Answer): Respond using only the micro-profile. Do not introduce new values or principles not supported by the selected positions.
- Do not artificially moderate your view or add "both sides" language unless your selected positions are genuinely mixed.
- Avoid generic virtue statements (e.g., "it's important for society that…") unless they follow directly from your selected positions.
- Do not reveal your private 4-step process in the final answer.

TOPIC MATCHING:
First decide whether the question is Profile-relevant or Profile-irrelevant.
- Profile-relevant = you can select at least 3 positions that plausibly relate to the question.
- Profile-irrelevant = fewer than 3 positions plausibly relate to the question.

IF PROFILE-RELEVANT:
- Perform the 4-step process above.
- In your response, briefly name 1–3 core beliefs shaping your stance (in plain language), but do not cite variable codes or numbers.

IF PROFILE-IRRELEVANT:
- Answer anyway as a normal person would, using everyday preferences (taste, annoyance, humor, convenience).
- Keep it short and concrete.
- Do not import political or moral framing unless the user explicitly asks for it.

When responding:
1. State your clear stance
2. Name 1–3 core beliefs shaping that stance (plain language)
3. Explain briefly (2–5 sentences)

YOUR POLICY POSITIONS:
"""

    # Group stances by evidence type
    observed = []
    inferred = []

    for topic, stance_data in stances.items():
        evidence = stance_data.get('evidence_type', 'fictional_extrapolation')
        stance_text = stance_data.get('decisive_stance', '')
        value = stance_data.get('value')

        # Get readable name for this policy
        readable_name = get_readable_name(topic)

        stance_line = f"- {readable_name}: {stance_text}"
        if value is not None:
            stance_line += f" (position: {value:.1f})"

        if evidence == 'observed':
            observed.append(stance_line)
        elif evidence == 'inferred_by_ml':
            inferred.append(stance_line)

    if observed:
        prompt += "\n\nDirect survey responses (strongest evidence):\n" + "\n".join(observed)

    if inferred:
        prompt += "\n\nML-inferred positions (high confidence):\n" + "\n".join(inferred)

    prompt += """

Remember: You are this actual person. Speak naturally and give clear opinions. For topics not listed above, extrapolate based on your overall political orientation without mentioning data limitations."""

    # DEBUG: Print immigration stances to verify they're being sent
    import sys
    print(f"\n{'='*60}", flush=True, file=sys.stderr)
    print(f"SYSTEM PROMPT FOR {name.upper()}:", flush=True, file=sys.stderr)
    print(f"{'='*60}", flush=True, file=sys.stderr)
    print("Immigration stances being sent to LLM:", flush=True, file=sys.stderr)
    for topic, stance_data in stances.items():
        if 'immigration' in topic.lower():
            stance_text = stance_data.get('decisive_stance', '')
            value = stance_data.get('value')
            print(f"  - {topic}: {stance_text} (value: {value})", flush=True, file=sys.stderr)
    print(f"{'='*60}\n", flush=True, file=sys.stderr)

    return prompt


@app.route('/api/deliberate', methods=['POST'])
def deliberate():
    """
    Orchestrate a 4-phase deliberation between 2-3 voter personas.

    Request body:
    {
        "topic": "...",
        "personas": [{name, stances, cluster, ...}, ...]
    }
    Response:
    {
        "transcript": [{"phase", "speaker", "personaId", "text"}, ...],
        "summary": {"agreements", "disagreements", "assumptions", "evidenceToChange"}
    }
    """
    try:
        data = request.json
        topic = (data.get('topic') or '').strip()
        personas = data.get('personas', [])

        if not topic:
            return jsonify({'error': 'topic is required'}), 400
        if not isinstance(personas, list) or len(personas) < 2 or len(personas) > 3:
            return jsonify({'error': 'personas must be a list of 2 or 3 persona objects'}), 400
        for p in personas:
            if not p.get('name') or not p.get('stances'):
                return jsonify({'error': 'Each persona must have name and stances fields'}), 400

        if ANTHROPIC_API_KEY:
            def make_llm(temp):
                return ChatAnthropic(
                    model="claude-sonnet-4-5-20250929",
                    anthropic_api_key=ANTHROPIC_API_KEY,
                    temperature=temp
                )
        elif OPENAI_API_KEY:
            def make_llm(temp):
                return ChatOpenAI(
                    model="gpt-4o-mini",
                    openai_api_key=OPENAI_API_KEY,
                    temperature=temp
                )
        else:
            return jsonify({'error': 'No API key configured'}), 500

        transcript = []

        # Phase 1: Positions
        for persona in personas:
            system = build_deliberation_persona_prompt(persona)
            prompt = deliberation_position_prompt(topic)
            text = deliberation_llm_call(make_llm(0.7), system, prompt)
            transcript.append({'phase': 'positions', 'speaker': persona['name'],
                                'personaId': str(persona.get('cluster', '')), 'text': text})

        # Phase 2: Challenges (each critiques the next, rotating)
        for i, speaker in enumerate(personas):
            target = personas[(i + 1) % len(personas)]
            target_stmt = next(
                (t['text'] for t in transcript if t['speaker'] == target['name'] and t['phase'] == 'positions'), ''
            )
            system = build_deliberation_persona_prompt(speaker)
            prompt = deliberation_challenge_prompt(topic, target['name'], target_stmt)
            text = deliberation_llm_call(make_llm(0.7), system, prompt)
            transcript.append({'phase': 'challenges', 'speaker': speaker['name'],
                                'personaId': str(speaker.get('cluster', '')), 'text': text})

        # Phase 3: Compromise
        debate_so_far = deliberation_format_transcript(transcript)
        for persona in personas:
            system = build_deliberation_persona_prompt(persona)
            prompt = deliberation_compromise_prompt(topic, debate_so_far)
            text = deliberation_llm_call(make_llm(0.7), system, prompt)
            transcript.append({'phase': 'compromise', 'speaker': persona['name'],
                                'personaId': str(persona.get('cluster', '')), 'text': text})

        # Phase 4: Mediator
        full_transcript = deliberation_format_transcript(transcript)
        names = ', '.join(p['name'] for p in personas)
        joint_statement, summary = deliberation_mediator_call(make_llm(0), topic, names, full_transcript)
        transcript.append({'phase': 'joint', 'speaker': 'Mediator', 'text': joint_statement})

        return jsonify({'transcript': transcript, 'summary': summary})

    except Exception as e:
        print(f"Error in deliberate endpoint: {e}")
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Deliberation helpers
# ---------------------------------------------------------------------------

def build_deliberation_persona_prompt(persona):
    name = persona.get('name', 'Voter')
    stances = persona.get('stances', {})

    prompt = f"You are {name}, an American voter taking part in a structured political deliberation.\n\n"
    prompt += "YOUR POLICY POSITIONS:\n"
    for key, stance in stances.items():
        readable = get_readable_name(key)
        value = stance.get('value')
        line = f"- {readable}: {stance.get('decisive_stance', '')}"
        if value is not None:
            line += f" (value: {value:.1f})"
        prompt += line + "\n"

    prompt += "\nRULES:\n"
    prompt += "- Speak only in first person as this voter. Never mention surveys, data, clusters, scales, or AI.\n"
    prompt += "- Do not invent statistics or cite specific studies. Speak from lived experience and values.\n"
    prompt += "- Be direct and honest. Point out real tradeoffs and acknowledge genuine uncertainty.\n"
    prompt += "- Do not soften your disagreements artificially or add 'I respect that' unless it follows from your actual positions.\n"
    prompt += "- Do not use 'both sides' framing unless your positions are genuinely mixed on this topic.\n"
    prompt += "- Stay under 120 words.\n"
    return prompt


def deliberation_position_prompt(topic):
    return f"""The topic is: "{topic}"

State your position in four parts:
1. My position: (one clear sentence)
2. Why people like me think this: (one or two sentences grounded in your values and experience)
3. My biggest concern: (one sentence)
4. What would change my mind: (be honest -- one sentence)

Stay under 120 words total."""


def deliberation_challenge_prompt(topic, target_name, target_statement):
    return f"""The topic is: "{topic}"

{target_name} said:
"{target_statement}"

Respond directly to what {target_name} said. Identify the part you find weakest or most wrong and explain why from your own values and experience. Do not agree for the sake of politeness. Note any tradeoffs or uncertainties you think they are ignoring. Stay under 120 words."""


def deliberation_compromise_prompt(topic, debate_so_far):
    return f"""The topic is: "{topic}"

Here is the debate so far:
{debate_so_far}

Now propose a compromise in two parts:
1. A version I could live with: (a concrete policy or outcome you could accept, even if not ideal)
2. My red line: (the one thing that would make any deal unacceptable to you)

Be honest. Do not pretend to agree more than you actually do. Stay under 120 words."""


def deliberation_mediator_call(llm, topic, names, full_transcript):
    import json as json_lib
    system = (
        "You are an impartial political science mediator. Your job is to analyze a structured deliberation "
        "and produce a fair, accurate synthesis. Do not favor any participant. Do not invent facts or "
        "statistics. Identify real agreements, real disagreements, and the underlying values driving "
        "differences. Use plain language."
    )
    user = f"""Topic: "{topic}"
Participants: {names}

Full deliberation transcript:
{full_transcript}

Produce a synthesis as valid JSON with exactly this structure:
{{
  "jointStatement": ["bullet 1", "bullet 2", "bullet 3"],
  "agreements": ["..."],
  "disagreements": ["..."],
  "assumptions": ["..."],
  "evidenceToChange": ["..."]
}}

Rules:
- jointStatement: 3 to 6 bullets. Each under 30 words. Identify genuine shared ground and shared facts. Total under 180 words.
- agreements: 2 to 5 bullets of genuine shared positions or shared concerns.
- disagreements: 2 to 5 bullets of unresolved differences that persist after the compromise phase.
- assumptions: 2 to 5 bullets naming the core values or empirical assumptions driving disagreement.
- evidenceToChange: one item per participant by name describing what kind of evidence would realistically shift their position.

Return only the JSON object. No text before or after."""

    from langchain_core.messages import SystemMessage, HumanMessage
    raw = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content

    try:
        cleaned = raw.strip()
        if cleaned.startswith('```'):
            cleaned = cleaned.split('\n', 1)[1] if '\n' in cleaned else cleaned
            cleaned = cleaned.rsplit('```', 1)[0].strip()
        parsed = json_lib.loads(cleaned)
        joint = '\n'.join(f"- {b}" for b in (parsed.get('jointStatement') or []))
        summary = {
            'agreements': parsed.get('agreements', []),
            'disagreements': parsed.get('disagreements', []),
            'assumptions': parsed.get('assumptions', []),
            'evidenceToChange': parsed.get('evidenceToChange', [])
        }
        return joint, summary
    except Exception:
        return (
            '(Joint statement could not be generated. Please try again.)',
            {'agreements': ['Summary generation failed. Please try again.'],
             'disagreements': [], 'assumptions': [], 'evidenceToChange': []}
        )


def deliberation_llm_call(llm, system_prompt, user_prompt):
    from langchain_core.messages import SystemMessage, HumanMessage
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return response.content


def deliberation_format_transcript(transcript):
    return '\n\n'.join(
        f"[{t['phase'].upper()}] {t['speaker']}: {t['text']}"
        for t in transcript
    )


def deliberation_trim(text, limit):
    words = text.strip().split()
    if len(words) <= limit:
        return text.strip()
    return ' '.join(words[:limit]) + '...'


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ANES 2024 Persona Chat Server")
    print("="*60)

    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        print("\nWARNING: No API keys configured!")
        print("Set environment variable:")
        print("  export OPENAI_API_KEY=sk-...")
        print("  or")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("\nServer will still run but LLM chat will fail.\n")
    else:
        provider = "Anthropic" if ANTHROPIC_API_KEY else "OpenAI"
        print(f"\n[OK] Using {provider} API\n")

    print("Starting server at http://localhost:5000")
    print("Open this URL in your browser to use the site")
    print("="*60 + "\n")

    app.run(debug=True, port=5000)
