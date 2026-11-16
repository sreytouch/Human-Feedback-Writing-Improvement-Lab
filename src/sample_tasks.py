"""Practice tasks and a separate fixed set for the v1 improvement test.

Held-out tasks are evaluation inputs only. They are never used as demonstrations
inside generation prompts. Once their results inform a prompt change, they are
no longer an untouched test set for that changed strategy.
"""

HELD_OUT_PROTOCOL_VERSION = "v1"

SAMPLE_TASKS = [
    {
        "id": "sample-01", "title": "Polite deadline",
        "original_text": "I need your answer today.",
        "rewrite_instruction": "Make this more polite.",
        "constraints": "Preserve the deadline.",
    },
    {
        "id": "sample-02", "title": "Missing file",
        "original_text": "You forgot to send the file.",
        "rewrite_instruction": "Make this professional and non-accusatory.",
        "constraints": "",
    },
    {
        "id": "sample-03", "title": "Incomplete application",
        "original_text": "Your application is incomplete.",
        "rewrite_instruction": "Make this friendlier while keeping it concise.",
        "constraints": "",
    },
    {
        "id": "sample-04", "title": "Respectful disagreement",
        "original_text": "I disagree with your decision.",
        "rewrite_instruction": "Make this confident but respectful.",
        "constraints": "",
    },
    {
        "id": "sample-05", "title": "Workplace progress check",
        "original_text": "Why haven’t you finished this yet?",
        "rewrite_instruction": "Rewrite this for a professional workplace Slack message.",
        "constraints": "",
    },
    {
        "id": "sample-06", "title": "Urgent phone call",
        "original_text": "Call me ASAP.",
        "rewrite_instruction": "Make this polite while preserving urgency.",
        "constraints": "",
    },
    {
        "id": "sample-07", "title": "Plain-language postponement",
        "original_text": "Due to unforeseen circumstances, implementation of the aforementioned procedure has been temporarily postponed.",
        "rewrite_instruction": "Make this easier to understand without removing important information.",
        "constraints": "",
    },
    {
        "id": "sample-08", "title": "Empathetic refund refusal",
        "original_text": "We can’t give you a refund.",
        "rewrite_instruction": "Rewrite this for a customer using an empathetic tone.",
        "constraints": "Do not promise a refund or compensation.",
    },
    {
        "id": "sample-09", "title": "Short meeting request",
        "original_text": "I wanted to reach out to see whether you might have some time available tomorrow to discuss the project.",
        "rewrite_instruction": "Shorten this.",
        "constraints": "Use fewer than 25 words and preserve the meaning.",
    },
    {
        "id": "sample-10", "title": "Student-friendly eligibility",
        "original_text": "Participants are responsible for ensuring compliance with applicable eligibility requirements.",
        "rewrite_instruction": "Rewrite for a college-student audience using simple language.",
        "constraints": "",
    },
]

HELD_OUT_TASKS = [
    {
        "id": "heldout-v1-01", "title": "Library service notice",
        "original_text": "The library's second floor will be closed on Tuesday for repairs, but the ground-floor desk will remain open.",
        "rewrite_instruction": "Rewrite as a clear, welcoming notice for library visitors.",
        "constraints": "Keep Tuesday, the second-floor closure, and the open ground-floor desk. Use no more than two sentences.",
    },
    {
        "id": "heldout-v1-02", "title": "Museum photography policy",
        "original_text": "Flash photography is prohibited in the gallery. Photography without flash is permitted.",
        "rewrite_instruction": "Make this a friendly sign for visitors.",
        "constraints": "Preserve the difference between flash and non-flash photography. Do not ban all photography.",
    },
    {
        "id": "heldout-v1-03", "title": "Workshop registration",
        "original_text": "Registration opens at 09:00 UTC on Thursday and closes when all 30 places have been booked.",
        "rewrite_instruction": "Rewrite this in simple language for a community newsletter.",
        "constraints": "Retain the opening time, timezone, weekday, and 30-place limit. Do not promise a place.",
    },
    {
        "id": "heldout-v1-04", "title": "Software maintenance window",
        "original_text": "The dashboard will be read-only from 14:00 to 14:30 UTC while we update the reporting service.",
        "rewrite_instruction": "Rewrite this as a concise status notification.",
        "constraints": "Keep both times and UTC. Say read-only, not fully offline. Use fewer than 30 words.",
    },
    {
        "id": "heldout-v1-05", "title": "Volunteer weather update",
        "original_text": "If rain begins before noon, the volunteer orientation will move from the courtyard to Room 12.",
        "rewrite_instruction": "Make this easier for first-time volunteers to follow.",
        "constraints": "Preserve the condition, the before-noon timing, and Room 12. Do not say the move is already decided.",
    },
    {
        "id": "heldout-v1-06", "title": "Meal attendance request",
        "original_text": "Please confirm by Friday whether you will attend the team lunch and whether you need a vegetarian meal.",
        "rewrite_instruction": "Make this warm and conversational.",
        "constraints": "Keep the Friday deadline and both questions. Do not add a lunch date or menu.",
    },
    {
        "id": "heldout-v1-07", "title": "Shared equipment reminder",
        "original_text": "After using the studio camera, return it to cabinet B and place the battery on the charger.",
        "rewrite_instruction": "Turn this into two clear numbered steps.",
        "constraints": "Use exactly two numbered steps. Retain cabinet B and the instruction to charge the battery.",
    },
    {
        "id": "heldout-v1-08", "title": "Tentative delivery estimate",
        "original_text": "The replacement keyboard is expected to arrive within five business days, but this estimate is not guaranteed.",
        "rewrite_instruction": "Make this sound helpful and reassuring for a customer.",
        "constraints": "Retain five business days and the uncertainty. Do not guarantee delivery or offer compensation.",
    },
    {
        "id": "heldout-v1-09", "title": "Feedback with a specific request",
        "original_text": "The draft explains the problem well, but the chart labels are too small to read on a phone.",
        "rewrite_instruction": "Rewrite this as constructive feedback to a colleague.",
        "constraints": "Preserve the positive feedback and the chart-label readability issue. Do not invent other problems.",
    },
    {
        "id": "heldout-v1-10", "title": "Room booking cancellation",
        "original_text": "Room bookings may be canceled without a fee up to 24 hours before the reserved start time.",
        "rewrite_instruction": "Explain this policy in plain language for students.",
        "constraints": "Preserve the 24-hour cutoff and free cancellation condition. Do not claim cancellation is always free.",
    },
]

for task in HELD_OUT_TASKS:
    task["protocol_version"] = HELD_OUT_PROTOCOL_VERSION
