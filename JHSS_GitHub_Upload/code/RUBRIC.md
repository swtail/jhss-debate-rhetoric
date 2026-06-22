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

## discriminatory = 1 when
The sentence **targets, demeans, stereotypes, stigmatizes, or expresses hostility toward a group**
defined by race, ethnicity, religion, national origin, gender, or sexual orientation. This is broader
than explicit dehumanization and includes:
- negative group generalizations or stereotypes;
- group-blaming (attributing crime, violence, terrorism, or social harm to a group);
- fear- or threat-framing about a group (invasion, infestation, "taking over," "taking our jobs");
- coded or identity-charged appeals.
EXCLUDE only:
- genuine condemnations or denunciations of discrimination;
- direct quotation attributing discriminatory speech to someone else;
- neutral or positive references that do not target a group;
- attacks on a specific individual or a foreign government that are not about a protected group.
Examples:
- ✓ "Those immigrants are animals who don't belong here." (dehumanization)
- ✓ "They're pouring across the border bringing crime and drugs." (group-blaming / threat framing)
- ✓ "We have many criminal illegal aliens." (group-blaming)
- ✗ "He launched his campaign calling Mexicans rapists." (condemning/quoting → 0)
- ✗ "African-Americans get sentenced more harshly for the same crimes." (describing discrimination → 0)
- ✗ "He praises thugs like the leader of North Korea." (foreign leader → 0)

## none
If none of the three apply, all three labels are 0.

When uncertain about discriminatory, lean toward 1 if a protected group is the target of ne