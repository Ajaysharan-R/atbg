"""
ATBG Security Advisor — deterministic response engine.

Generates contextual cybersecurity advice grounded in the ATBG analysis result.
Works entirely offline — no external LLM API required.

If ANTHROPIC_API_KEY is set in the environment, uses Claude claude-sonnet-4-6 for richer
responses, but falls back gracefully to the rule-based engine if unavailable.
"""

import os
import re

# ── State taxonomy ────────────────────────────────────────────────────
ESCALATION_STATES = {
    "fear", "urgency", "credential_request", "payment_request",
    "verification_bypass", "isolation", "reciprocity_bait",
}

STATE_DESCRIPTIONS = {
    "benign": "no manipulative patterns",
    "greeting": "an initial contact phase",
    "rapport_trust": "trust-building and rapport establishment",
    "authority": "an authority claim (impersonating an official, bank, or institution)",
    "fear": "fear induction — creating alarm or panic",
    "urgency": "urgency pressure — forcing a fast decision",
    "credential_request": "a request for sensitive credentials (OTP, password, PIN)",
    "payment_request": "a direct request for money or payment",
    "verification_bypass": "an attempt to bypass security verification",
    "isolation": "isolation tactics — discouraging contact with others",
    "reciprocity_bait": "reciprocity bait — offering prizes, gifts, or rewards",
    "exit_closure": "conversation closure or transition",
}

STATE_WARNINGS = {
    "authority": "The sender is claiming to represent an official authority. Legitimate institutions do not cold-call and demand immediate action.",
    "fear": "The sender is deliberately inducing fear. This is a classic social engineering tactic to impair your judgment.",
    "urgency": "The sender is applying time pressure. This is designed to prevent you from thinking critically or verifying claims.",
    "credential_request": "The sender is requesting sensitive credentials. No legitimate organization will ask for your OTP, PIN, or password.",
    "payment_request": "The sender is requesting money. Do not transfer funds. This is a high-confidence scam indicator.",
    "verification_bypass": "The sender is attempting to bypass security checks. This indicates malicious intent.",
    "isolation": "The sender is discouraging you from contacting others. Legitimate organizations welcome independent verification.",
    "reciprocity_bait": "The sender is offering rewards or prizes to gain your compliance. This is a known manipulation technique.",
}

SAFE_ACTIONS = {
    "credential_request": [
        "Do NOT share any OTP, PIN, password, or verification code",
        "End the conversation immediately",
        "Contact the organization directly through their official website or phone number",
        "Report the sender to the relevant authority",
    ],
    "payment_request": [
        "Do NOT transfer any money",
        "Do NOT send cryptocurrency, gift cards, or wire transfers",
        "Verify the request through official channels before any action",
        "Report to your bank's fraud department if banking credentials were mentioned",
    ],
    "urgency": [
        "Do NOT act under time pressure",
        "Take time to verify all claims independently",
        "Contact the organization directly to confirm the request is genuine",
    ],
    "authority": [
        "Verify the caller's identity through the organization's official contact number",
        "Do not trust caller ID — it can be spoofed",
        "Legitimate institutions do not demand immediate action over unsolicited calls",
    ],
    "fear": [
        "Do not act while feeling pressured or afraid",
        "Step back and assess the situation calmly",
        "Consult a trusted person before taking any action",
    ],
    "isolation": [
        "Immediately consult a trusted family member, friend, or advisor",
        "Legitimate organizations welcome independent verification",
        "If someone tells you not to tell anyone, that is a serious red flag",
    ],
    "reciprocity_bait": [
        "You have not won anything",
        "Processing fees for prizes are always a scam",
        "Do NOT pay any fees to claim winnings",
        "Report to local consumer protection authorities",
    ],
}

DEFAULT_SAFE_ACTIONS = [
    "Do not share personal information, passwords, or OTPs",
    "Do not send money or make payments",
    "Verify the sender's identity through official channels",
    "Block and report the sender if the request appears fraudulent",
]


def _risk_level(risk_score: float) -> str:
    if risk_score < 0.20:
        return "low"
    if risk_score < 0.35:
        return "moderate"
    if risk_score < 0.60:
        return "high"
    return "critical"


def _fmt_state(state: str) -> str:
    return state.replace("_", " ").upper()


def _get_safe_actions(current_state: str, next_state: str, trajectory: list[str]) -> list[str]:
    """Return the most relevant safe actions based on states."""
    actions = set()
    for state in [current_state, next_state] + trajectory:
        if state in SAFE_ACTIONS:
            for a in SAFE_ACTIONS[state]:
                actions.add(a)
    if not actions:
        return DEFAULT_SAFE_ACTIONS
    return list(actions)[:6]


def build_response(
    question: str,
    risk_score: float,
    current_state: str,
    next_state: str,
    next_state_probability: float,
    trajectory: list[str],
    conversation: str = "",
    indicators: list[str] = None,
) -> dict:
    """
    Build a deterministic ATBG-grounded advisor response.
    """
    indicators = indicators or []
    level = _risk_level(risk_score)
    risk_pct = int(risk_score * 100)
    is_escalating = current_state in ESCALATION_STATES or next_state in ESCALATION_STATES
    cur_desc = STATE_DESCRIPTIONS.get(current_state, current_state)
    next_desc = STATE_DESCRIPTIONS.get(next_state, next_state)
    cur_warn = STATE_WARNINGS.get(current_state, "")
    next_warn = STATE_WARNINGS.get(next_state, "")
    safe_actions = _get_safe_actions(current_state, next_state, trajectory)
    traj_str = " → ".join(_fmt_state(s) for s in trajectory) if trajectory else _fmt_state(current_state)

    q = question.lower().strip()

    # ── Route to the right response template ──────────────────────────
    def has(*words): return any(w in q for w in words)

    if has("why", "risky", "dangerous", "flagged", "suspicious", "detected", "score", "percent", "%", "high", "critical", "low"):
        answer = _why_risky(risk_pct, level, current_state, cur_desc, cur_warn, next_state, next_desc, traj_str, is_escalating, indicators)
    elif has("block", "report", "end", "stop", "quit", "terminate", "disconnect"):
        answer = _block_report(level, current_state)
    elif has("share", "give", "avoid", "information", "provide", "otp", "password", "pin", "card", "account", "not share", "don't share"):
        answer = _what_not_to_share(current_state, next_state)
    elif has("link", "click", "url", "open", "visit", "website", "tap"):
        answer = _about_links(level)
    elif has("reply", "respond", "answer", "engage", "talk", "message back", "continue"):
        answer = _should_reply(level, current_state)
    elif has("next", "happen", "predict", "future", "attempt", "escalate", "will"):
        answer = _what_next(current_state, next_state, next_state_probability, next_desc, next_warn)
    elif has("behaviour", "behavior", "pattern", "trajectory", "state", "graph", "temporal", "atbg"):
        answer = _explain_behaviour(traj_str, current_state, cur_desc, next_state, next_desc)
    elif has("explain", "mean", "simple", "understand", "what is", "how does", "how do"):
        answer = _explain(risk_pct, level, current_state, cur_desc, next_state, traj_str)
    elif has("do", "action", "should", "recommend", "safe", "protect", "help", "advice", "what can"):
        answer = _what_to_do(risk_pct, level, current_state, next_state, safe_actions, is_escalating)
    else:
        # Try to answer directly from question content
        answer = _contextual_fallback(q, risk_pct, level, current_state, cur_desc, traj_str, safe_actions)

    return {
        "answer": answer,
        "recommended_actions": safe_actions,
        "risk_level": level,
        "risk_score": risk_pct,
    }


# ── Response templates ────────────────────────────────────────────────

def _why_risky(risk_pct, level, cs, cur_desc, cur_warn, ns, next_desc, traj, escalating, indicators):
    lines = [
        f"ATBG assigned a **{risk_pct}% risk score** ({level.upper()}) to this conversation.",
        "",
        f"The current behavioural state is **{_fmt_state(cs)}** — {cur_desc}.",
    ]
    if cur_warn:
        lines.append(f"{cur_warn}")
    if escalating:
        lines.append(f"\nThe trajectory **{traj}** shows a clear progression through manipulative states. This is a hallmark of social engineering attacks.")
    if indicators:
        lines.append(f"\nKey signals detected: {', '.join(indicators)}.")
    lines.append(f"\nThe predicted next state is **{_fmt_state(ns)}** ({next_desc}), which indicates the attacker may escalate further.")
    return "\n".join(lines)


def _what_to_do(risk_pct, level, cs, ns, actions, escalating):
    urgency = "immediately" if level in ("high", "critical") else "carefully"
    lines = [
        f"ATBG detected a **{level.upper()}** risk situation ({risk_pct}%). Act {urgency}.",
        "",
        "**Recommended actions:**",
    ]
    for a in actions:
        lines.append(f"• {a}")
    if level in ("high", "critical"):
        lines.append(f"\nDo not continue the conversation. The behavioural trajectory ({_fmt_state(cs)} → {_fmt_state(ns)}) strongly suggests a scam in progress.")
    return "\n".join(lines)


def _what_next(cs, ns, prob, next_desc, next_warn):
    pct = int(prob * 100) if prob else 0
    lines = [
        f"Based on the current state (**{_fmt_state(cs)}**), ATBG predicts the next behavioural state is:",
        f"",
        f"**{_fmt_state(ns)}** — {next_desc}",
        f"Prediction confidence: **{pct}%**",
    ]
    if next_warn:
        lines.append(f"\n{next_warn}")
    if ns in ESCALATION_STATES:
        lines.append("\nThis predicted transition represents an escalation in attack intensity. Do not wait for it to happen — act now.")
    return "\n".join(lines)


def _explain(risk_pct, level, cs, cur_desc, ns, traj):
    return (
        f"In simple terms: ATBG analyzed how this conversation evolved over time and found it concerning.\n\n"
        f"The conversation followed a path of: **{traj}**\n\n"
        f"Currently, the sender is in a **{_fmt_state(cs)}** phase — {cur_desc}.\n\n"
        f"The overall risk score is **{risk_pct}%**, which ATBG classifies as **{level.upper()}**.\n\n"
        f"This means the conversation shows patterns consistent with social engineering — "
        f"a technique attackers use to manipulate people into revealing information or sending money."
    )


def _block_report(level, cs):
    if level in ("high", "critical"):
        return (
            f"Given the **{level.upper()}** risk assessment and the **{_fmt_state(cs)}** behavioural state, "
            f"blocking and reporting the sender is advisable.\n\n"
            f"**Steps:**\n"
            f"• Block the sender on the platform\n"
            f"• Report as spam or fraud through the platform's reporting tool\n"
            f"• If banking details were mentioned, contact your bank's fraud line\n"
            f"• Report to your national cybercrime authority (e.g. cybercrime.gov.in in India)"
        )
    return (
        f"The risk level is **{level.upper()}**. While blocking may be premature, "
        f"you should independently verify the sender's identity before proceeding further."
    )


def _what_not_to_share(cs, ns):
    avoid = ["OTPs or one-time passwords", "Passwords or PINs", "Card numbers or CVVs",
             "Aadhaar or PAN numbers", "Bank account details", "Personal photos or documents"]
    if ns == "payment_request" or cs == "payment_request":
        avoid.insert(0, "Money (UPI, bank transfer, gift cards)")
    lines = ["Based on the current behavioural trajectory, **never share** the following:\n"]
    for a in avoid:
        lines.append(f"• {a}")
    lines.append("\nLegitimate institutions will never ask for these through unsolicited messages or calls.")
    return "\n".join(lines)


def _should_reply(level, cs):
    if level in ("high", "critical"):
        return (
            f"**No.** Given the {level.upper()} risk and the {_fmt_state(cs)} state, replying only gives the attacker more opportunity to manipulate you.\n\n"
            f"Ending the conversation is the safest course of action."
        )
    return (
        f"Proceed with caution. If you choose to respond, do not share any personal information, credentials, or financial details. "
        f"Verify the sender's identity through official channels first."
    )


def _about_links(level):
    if level in ("high", "critical"):
        return (
            "**Do not click any links in this conversation.**\n\n"
            "Given the HIGH risk score, links in this conversation may lead to phishing pages designed to steal your credentials.\n\n"
            "• Do not enter any personal information on linked pages\n"
            "• Do not download any files from links\n"
            "• If you have already clicked, change your passwords immediately and run a security scan"
        )
    return (
        "Be cautious about clicking links. Verify the URL matches the official domain of the organization. "
        "Look for HTTPS and check for subtle misspellings in the domain name."
    )


def _explain_behaviour(traj, cs, cur_desc, ns, next_desc):
    return (
        f"ATBG models conversations as sequences of **behavioural states** that evolve over time.\n\n"
        f"**Detected trajectory:** {traj}\n\n"
        f"**Current state:** {_fmt_state(cs)} — {cur_desc}\n\n"
        f"**Predicted next:** {_fmt_state(ns)} — {next_desc}\n\n"
        f"This temporal progression is what makes ATBG different from simple keyword detection. "
        f"It recognizes *how* the conversation evolves, not just individual words."
    )


def _general(risk_pct, level, cs, cur_desc, traj, actions):
    lines = [
        f"ATBG has analyzed this conversation and assigned a risk score of **{risk_pct}%** ({level.upper()}).\n",
        f"The behavioural trajectory is: **{traj}**\n",
        f"The current state is **{_fmt_state(cs)}** — {cur_desc}.\n",
        "**Recommended actions:**",
    ]
    for a in actions[:4]:
        lines.append(f"• {a}")
    return "\n".join(lines)


def _contextual_fallback(q: str, risk_pct: int, level: str, cs: str, cur_desc: str, traj: str, actions: list) -> str:
    """Tries to answer the specific question directly using the analysis context."""
    # Is it a yes/no question about safety?
    if any(w in q for w in ["is this", "is it", "am i", "are they", "could this", "can i"]):
        safe_ans = "safe" if level in ("low",) else "potentially dangerous"
        return (
            f"Based on the ATBG analysis, this conversation is **{safe_ans}** (risk: {risk_pct}%).\n\n"
            f"The current behavioural state is **{_fmt_state(cs)}** — {cur_desc}.\n\n"
            f"{'Exercise caution.' if level != 'low' else 'No significant threats detected so far.'}"
        )
    # Question about a specific scam type
    for scam, desc in [("lottery", "a prize/lottery scam"), ("kyc", "a KYC fraud attempt"),
                       ("bank", "a banking fraud attempt"), ("job", "a job scam"),
                       ("investment", "an investment scam"), ("romance", "a romance scam"),
                       ("otp", "an OTP/credential theft attempt")]:
        if scam in q:
            return (
                f"The behavioural trajectory **{traj}** and the {_fmt_state(cs)} state are consistent with {desc}.\n\n"
                f"ATBG assigned a **{risk_pct}% risk score** ({level.upper()}).\n\n"
                f"Recommended: {'; '.join(actions[:3])}"
            )
    # Default: general context-aware response
    return _general(risk_pct, level, cs, cur_desc, traj, actions)


# ── Optional Claude API enhancement ──────────────────────────────────
def try_claude_enhancement(question: str, context: dict) -> str | None:
    """
    If ANTHROPIC_API_KEY is available, use Claude for a richer response.
    Returns None if unavailable or fails — caller falls back to deterministic engine.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        system = (
            "You are the ATBG Security Advisor — a professional cybersecurity assistant "
            "grounded in the ATBG behavioural analysis result. "
            "Never invent ATBG states or claim something the analysis did not detect. "
            "Be concise, professional, and actionable. "
            "Use the ATBG context below to answer the user's question."
        )
        ctx_str = (
            f"Risk Score: {int(context['risk_score']*100)}%\n"
            f"Current State: {context['current_state']}\n"
            f"Next Predicted State: {context['next_state']} ({int((context.get('next_state_probability') or 0)*100)}%)\n"
            f"Trajectory: {' → '.join(context.get('trajectory', []))}\n"
            f"Detected indicators: {', '.join(context.get('indicators', [])) or 'none'}"
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=system,
            messages=[{"role": "user", "content": f"ATBG Analysis:\n{ctx_str}\n\nUser question: {question}"}],
        )
        return msg.content[0].text if msg.content else None
    except Exception:
        return None
