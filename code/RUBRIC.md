# Rhetoric Classification Codebook (LLM rubric)

Each candidate **sentence** receives three INDEPENDENT binary labels. A sentence may
have more than one label set to 1, or none. Judge the sentence **in the meaning it
conveys**, not by keywords.

## aggressive = 1 when
The sentence is confrontational, assertive, or dominance-displaying toward an opponent
or institution — direct accusations of failure, lying, or incompetence; blunt
challenges; "you" attacks — **without** emotional/moral escalation or group derogation.
- ✓ "You said it, and you know it isn't true."
- ✓ "He failed the American people on the economy."
- ✗ "I think we should invest in roads." (not confrontational)

## inflammatory = 1 when
The sentence is emotionally provocative, morally loaded, hyperbolic, or designed to
arouse fear/anger/outrage — catastrophizing, moral condemnation, threat framing,
charged name-calling.
- ✓ "This is an absolute disaster that will destroy our country."
- ✓ "His agenda is radical, dangerous, and corrupt."
- ✗ "Unemployment rose by two percent last year." (factual, unloaded)

## discriminatory = 1 ONLY when
The speaker **themselves demeans, dehumanizes, or derogates a PROTECTED GROUP** —
defined by race, ethnicity, religion, national origin, gender, or sexual orientation —
**as a group**.
EXCLUDE (these are discriminatory = 0):
- Statements **condemning, denouncing, or describing** discrimination by others.
- **Quoting or attributing** a discriminatory statement to someone else.
- Neutral or positive uses of phrases like "those people" (e.g., frontline workers).
- Policy discussion (immigration, abortion, etc.) **without** group derogation.
- Attacks on **individuals** or **foreign leaders/governments** (not a protected group).
Examples:
- ✓ "Those immigrants are animals who don't belong here." (speaker derogates a group)
- ✗ "He launched his campaign calling Mexicans rapists." (condemning/quoting → 0)
- ✗ "He praises thugs like the leader of North Korea." (foreign leader → 0)
- ✗ "Those people on the front lines saved our lives." (positive → 0)

## none
If none of the three apply, all three labels are 0.

When uncertain about discriminatory, default to 0 unless the group-derogation is explicit.
