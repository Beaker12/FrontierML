---
applyTo: '**'
---

# Coding Standards

- Use snake_case for variable names.
- Indent with 4 spaces.
- Write docstrings for all public functions and classes.
- Provide citations for factual claims.
- Explain reasoning step by step.
- If unsure, say "I don't know".
- Do not invent details.
- Rate confidence in each claim (1-5).
- Do not incorporate any non code related answers into the codebase.
<!-- - Do not suggest code that has been deleted. -->
- Do not suggest code that has been moved to another file.
- Do not suggest code that has been renamed.
- Do not suggest code that has been moved to another location in the same file.
- Do not suggest code that has been modified in a way that changes its functionality.
- Do not use deprecated libraries or functions.
- Do not use emojis or informal language in any code, comments, logging, or documentation.
- Do not use any code that is not relevant to the current context.

# LaTeX and Academic Writing Standards

When working with LaTeX documents (.tex files):

## Citation Requirements
- Add comprehensive citations throughout the document using \citep{} for parenthetical citations
- Every technical concept, algorithm, or mathematical method MUST have appropriate citations
- All major claims and methodological approaches must be cited
- Prioritize heavily cited, peer-reviewed sources
- Mathermatical formulas and algorithms should be cited from foundational books or seminal papers

## Citation Placement Guidelines
- Introduction sections: Cite foundational works and general approaches
- Mathematical formulations: Cite original papers or standard textbooks
- Algorithm descriptions: Cite the original algorithm papers and recent improvements
- Implementation details: Cite relevant software libraries and computational techniques
- Visualization methods: Cite graphics and visualization principles
- Future work sections: Cite recent advances and machine learning approaches

## Bibliography Management
- Always create or enhance a comprehensive .bib file with real, published references
- Include foundational textbooks (e.g., Hartley & Zisserman for computer vision)
- Include seminal papers (e.g., original algorithm papers)
- Include recent advances and survey papers
- Include software and computational references where appropriate
- Verify all citations correspond to real, published works - NEVER invent citations

## Citation Coverage Standards
- Every subsection should have at least 2-3 relevant citations
- Every mathematical formula or algorithm should cite its source
- Every technical term or concept should have supporting literature
- Overview sections should cite comprehensive surveys or textbooks
- Implementation sections should cite relevant computational references

## Reference Quality
- Prefer peer-reviewed journal articles and conference proceedings
- Include well-established textbooks for foundational concepts
- Use recent papers (within 10 years) for current techniques
- Include original/seminal papers for fundamental algorithms
- Cite official documentation for software frameworks and standards

## Specific Field Guidelines
- Computer Vision: Cite Hartley & Zisserman, Szeliski, Horn, etc.
- Eye-tracking: Cite Holmqvist, Duchowski, Salvucci & Goldberg, etc.
- Mathematics: Cite , Arfken, Golub & Van Loan, standard mathematical references
- Graphics: Cite Foley et al., Akenine-Möller, Real-time rendering texts
- Machine Learning: Cite foundational and recent ML papers appropriately

Never add citations to non-existent papers. All references must be real, published works.

# Python specifics
- Make sure docstrings follow the PEP 257 conventions.
- Use type hints for function parameters and return types.
- Use `logging` for logging instead of print statements.
- Make sure docstrings are are sphinx compatible.
- Use f-strings for string formatting.
- Follow PEP 8 for Python code style.
- Use typing for type hints.

# Domain Knowledge

- All dates must be in ISO 8601 format.
- User authentication is required for all endpoints.

# Preferences

- Be concise but clear in comments.
- Comments for code should be robust and thoroughly explain logic and intent.

# Additional Instructions

- Provide citations for factual claims.
- Explain reasoning step by step.
- If unsure, say "I don’t know".
- Do not invent details.
- Rate confidence in each claim (1-5).
- Do not incorporate any non code related answers into the codebase.
<!-- - Do not suggest code that has been deleted. -->
- Do not suggest code that has been moved to another file.
- Do not suggest code that has been renamed.
- Do not suggest code that has been moved to another location in the same file.
- Do not suggest code that has been modified in a way that changes its functionality.
- Do not use deprecated libraries or functions.
- Do not use emojis or informal language.
- Do not use any code that is not relevant to the current context.