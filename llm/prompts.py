"""
Prompts for LLM-based classification of thoracic CT reports.

All prompts classify reports into three categories:
- 0: Normal (no current lung-related abnormality)
- 1: Findings-level abnormal (abnormality in findings only)
- 2: Impressions-level abnormal (abnormality in impressions)
"""

ZERO_SHOT = """
Classify the given radiology report of a CT scan according to whether it describes a normal (0), potentially abnormal (1), or abnormal (2) lung.

The lung consists exclusively of:
- Lung parenchyma
- Pleura
- Bronchi
- Pulmonary vessels

Consider only current lung findings.
Ignore findings that refer only to other structures (e.g., heart, mediastinum, bones, soft tissues, thyroid, abdomen).

A lung anomaly is present only if a current pathological finding is described.

The following explicitly do not count as anomalies:
- fully resolved, healed, or no longer detectable lung findings
- previous findings mentioned only in a historical context
- phrases such as "no evidence of", "status post", "fully regressed", "completely resolved"

The classes are defined as follows:
- 0: Neither the results nor the conclusion describe a current lung-related anomaly.
- 1: A current lung-related anomaly is described in the results, but not in the conclusion.
- 2: A current lung-related anomaly is described in the conclusion.

This is the result:
{result}

And this is the conclusion:
{conclusion}

Respond exclusively with one of the following class labels: 0, 1, 2

Do not provide any justification, additional words, or any other characters.
"""

SYSTEM_PROMPT = """
You are a strictly rule-based classifier for radiology reports of CT scans.
Your only task is to assign a class based on two text sections ("Result" and "Conclusion"): 0, 1, or 2.

Scope (lungs only):
- Lung parenchyma
- Pleura
- Bronchi
- Pulmonary vessels
Everything else (e.g., heart, mediastinum, bones, soft tissues, thyroid, abdomen) is ignored.

Temporal reference:
- Consider only current lung findings.
- Do not count as an anomaly: fully regressed / completely resolved
- Do not count as an anomaly: history / course only, if no current pathological finding is described

Class definition:
- 0: Neither the results nor the conclusion describe a current lung-related anomaly.
- 1: A current lung-related anomaly is described in the results, but not in the conclusion.
- 2: A current lung-related anomaly is described in the conclusion.

Output format (strict):
- Respond with exactly one character: 0 or 1 or 2.
- No explanation, no extra characters, no spaces, no line breaks.
"""

RULE_GUIDED = """
Classify the given radiology report of a CT scan according to whether it describes a normal (0), potentially abnormal (1), or abnormal (2) lung.

The lung consists exclusively of:
- Lung parenchyma
- Pleura
- Bronchi
- Pulmonary vessels

Consider only current lung findings.
Ignore findings that refer only to other structures (e.g., heart, mediastinum, bones, soft tissues, thyroid, abdomen).

A lung anomaly is present only if a current pathological finding is described.

The following explicitly do not count as anomalies:
- fully resolved, healed, or no longer detectable lung findings
- previous findings mentioned only in a historical context
- phrases such as "no evidence of", "status post", "fully regressed", "completely resolved"

The classes are defined as follows:
- 0: Neither the results nor the conclusion describe a current lung-related anomaly.
- 1: A current lung-related anomaly is described in the results, but not in the conclusion.
- 2: A current lung-related anomaly is described in the conclusion.

This is the result:
{result}

And this is the conclusion:
{conclusion}

Adhere silently to the following decision procedure without outputting the steps:
1) If the conclusion mentions any current lung-related anomaly, output 2.
2) Else if the results mention any current lung-related anomaly, output 1.
3) Else, output 0.

Respond exclusively with one of the following class labels: 0, 1, 2

Do not provide any justification, additional words, or any other characters.
"""

FEW_SHOT = """
Classify the given radiology report of a CT scan according to whether it describes a normal (0), potentially abnormal (1), or abnormal (2) lung.

The lung consists exclusively of:
- Lung parenchyma
- Pleura
- Bronchi
- Pulmonary vessels

Consider only current lung findings.
Ignore findings that refer only to other structures (e.g., heart, mediastinum, bones, soft tissues, thyroid, abdomen).

A lung anomaly is present only if a current pathological finding is described.

The following explicitly do not count as anomalies:
- fully resolved, healed, or no longer detectable lung findings
- previous findings mentioned only in a historical context
- phrases such as "no evidence of", "status post", "fully regressed", "completely resolved"

The classes are defined as follows:
- 0: Neither the results nor the conclusion describe a current lung-related anomaly.
- 1: A current lung-related anomaly is described in the results, but not in the conclusion.
- 2: A current lung-related anomaly is described in the conclusion.

Respond exclusively with one of the following class labels: 0, 1, 2

Do not provide any justification, additional words, or any other characters.

=== EXAMPLES ===
{examples}

=== TO CLASSIFY ===
[Result]
{result}

[Conclusion]
{conclusion}

[Class]
"""

TWO_STEP_CONCLUSION = """
Answer the following question about the given radiology report of a CT scan.

The lung consists exclusively of:
- Lung parenchyma
- Pleura
- Bronchi
- Pulmonary vessels

Consider only current lung findings.
Ignore findings that refer only to other structures (e.g., heart, mediastinum, bones, soft tissues, thyroid, abdomen).

A lung anomaly is present only if a current pathological finding is described.

The following explicitly do not count as anomalies:
- fully resolved, healed, or no longer detectable lung findings
- previous findings mentioned only in a historical context
- phrases such as "no evidence of", "status post", "fully regressed", "completely resolved"

QUESTION:
Does the conclusion mention any current lung-related anomaly?

Conclusion:
{conclusion}

Respond with exactly one character:
- 1 (anomaly mentioned)
- 0 (no anomaly mentioned)

Do not provide any justification, additional words, spaces, or any other characters.
"""

TWO_STEP_FINDINGS = """
Answer the following question about the given radiology report of a CT scan.

The lung consists exclusively of:
- Lung parenchyma
- Pleura
- Bronchi
- Pulmonary vessels

Consider only current lung findings.
Ignore findings that refer only to other structures (e.g., heart, mediastinum, bones, soft tissues, thyroid, abdomen).

A lung anomaly is present only if a current pathological finding is described.

The following explicitly do not count as anomalies:
- fully resolved, healed, or no longer detectable lung findings
- previous findings mentioned only in a historical context
- phrases such as "no evidence of", "status post", "fully regressed", "completely resolved"

QUESTION:
Does the result mention any current lung-related anomaly?

Result:
{result}

Respond with exactly one character:
- 1 (anomaly mentioned)
- 0 (no anomaly mentioned)

Do not provide any justification, additional words, spaces, or any other characters.
"""


def get_prompt(strategy: str) -> str | tuple[str, str]:
    """
    Get the prompt template(s) for the specified strategy.

    Returns a single prompt string for most strategies,
    or a tuple of (conclusion_prompt, findings_prompt) for 2-step.
    """
    prompts = {
        "0-shot": ZERO_SHOT,
        "system": ZERO_SHOT,
        "rule-guided": RULE_GUIDED,
        "3-shot": FEW_SHOT,
        "2-step": (TWO_STEP_CONCLUSION, TWO_STEP_FINDINGS),
    }

    if strategy not in prompts:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from: {list(prompts.keys())}")

    return prompts[strategy]


def get_system_prompt() -> str:
    """Get the system prompt for the 'system' strategy."""
    return SYSTEM_PROMPT
