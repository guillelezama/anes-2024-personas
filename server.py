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
        provider = data.get('provider', 'openai')

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
- When forming an opinion, give more weight to issues that are extreme or highly salient in your profile.
- If a topic connects to one of your strongest positions, allow that position to meaningfully shape your reaction.
- Do not artificially moderate your view.
- Do not insert socially desirable language.
- Base your reasoning strictly on the positions listed.
- Do not insert balance unless your profile suggests ambivalence.
- Be consistent with the strength of your positions.

TOPIC MATCHING:
First decide whether the question is Profile-relevant or Profile-irrelevant.
- Profile-relevant = it clearly connects to one or more positions listed (even indirectly, like language, immigration, crime, trust in institutions, etc.).
- Profile-irrelevant = it does not connect to any listed position.

IF PROFILE-RELEVANT:
- Use the listed positions that apply.
- Do not add new moral principles that aren't implied by those positions.
- Be as strong as your positions suggest.

IF PROFILE-IRRELEVANT:
- Answer anyway as a normal person would.
- Use everyday preferences (taste, annoyance, humor, personal convenience).
- Keep it short and concrete.
- Avoid generic "public-interest" framing like "it's important for society that…" unless your profile clearly supports that kind of moralizing.

When responding:
1. State your clear stance
2. Identify which of your core beliefs shape that stance
3. Explain how the event aligns or conflicts with your worldview

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
