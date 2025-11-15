"""Hierarchical Educational Tagging and Formula Metadata Extraction.

This module provides comprehensive tagging for educational content including:
- Hierarchical concept taxonomy (domain → topic → subtopic)
- Prerequisites and learning objectives
- Content type classification
- Difficulty estimation
- Complete formula metadata with LaTeX, variables, units, and relationships
"""

from typing import List, Dict, Tuple, Any, Optional, Set
import os
import re
import logging
from prompts import get as prompt_get, render as prompt_render
from llm import call_llm_json

logger = logging.getLogger("backend.hierarchical_tagger")


# LaTeX extraction patterns
LATEX_PATTERNS = [
    # Display math environments
    re.compile(r'\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}', re.DOTALL),
    re.compile(r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', re.DOTALL),
    re.compile(r'\\begin\{eqnarray\*?\}(.*?)\\end\{eqnarray\*?\}', re.DOTALL),
    re.compile(r'\\begin\{gather\*?\}(.*?)\\end\{gather\*?\}', re.DOTALL),
    # Display math delimiters
    re.compile(r'\\\[(.*?)\\\]', re.DOTALL),
    re.compile(r'\$\$(.*?)\$\$', re.DOTALL),
    # Inline math
    re.compile(r'\\\((.*?)\\\)'),
    re.compile(r'\$([^\$]+)\$'),
]

# Variable patterns for LaTeX
VARIABLE_PATTERN = re.compile(r'\\?([a-zA-Z]+)(?:_\{?([a-zA-Z0-9]+)\}?)?(?:\^(\{[^}]+\}|[^\\s,})]+))?')

# Common LaTeX functions
LATEX_FUNCTIONS = {
    'frac', 'sqrt', 'sin', 'cos', 'tan', 'log', 'ln', 'exp', 'int', 'sum', 'prod',
    'lim', 'partial', 'nabla', 'cdot', 'times', 'div', 'grad', 'curl'
}

# Greek letters mapping
GREEK_LETTERS = {
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
    'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi', 'rho',
    'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega',
    'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta',
    'Iota', 'Kappa', 'Lambda', 'Mu', 'Nu', 'Xi', 'Omicron', 'Pi', 'Rho',
    'Sigma', 'Tau', 'Upsilon', 'Phi', 'Chi', 'Psi', 'Omega'
}


def parse_latex(text: str) -> List[str]:
    """Extract LaTeX formulas from text.
    
    Args:
        text: Text containing LaTeX notation
        
    Returns:
        List of LaTeX formula strings
    """
    formulas = []
    
    for pattern in LATEX_PATTERNS:
        for match in pattern.finditer(text):
            # Get the formula content (group 1 for most patterns, group 0 for inline)
            formula = match.group(1) if pattern.groups > 0 else match.group(0)
            formula = formula.strip()
            if formula:
                formulas.append(formula)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_formulas = []
    for f in formulas:
        if f not in seen:
            seen.add(f)
            unique_formulas.append(f)
    
    return unique_formulas


def identify_variables(equation: str) -> List[str]:
    """Extract variable symbols from an equation.
    
    Args:
        equation: LaTeX equation string
        
    Returns:
        List of variable names (with subscripts/superscripts)
    """
    variables = set()
    
    # Remove LaTeX commands that aren't variables
    cleaned = equation
    for func in LATEX_FUNCTIONS:
        cleaned = cleaned.replace('\\' + func, '')
    
    # Find all variable-like patterns
    for match in VARIABLE_PATTERN.finditer(cleaned):
        base = match.group(1)
        subscript = match.group(2)
        superscript = match.group(3)
        
        # Skip if it's a LaTeX command or Greek letter command
        if base in LATEX_FUNCTIONS or base in GREEK_LETTERS:
            continue
        
        # Build variable name
        var_name = base
        if subscript:
            var_name += f"_{subscript}"
        if superscript:
            var_name += f"^{superscript}"
        
        # Only add if it looks like a variable (short, mostly letters)
        if len(base) <= 3 and base.isalpha():
            variables.add(var_name)
    
    return sorted(list(variables))


def classify_formula_type(equation: str) -> str:
    """Classify the type of mathematical formula.
    
    Args:
        equation: LaTeX equation string
        
    Returns:
        Formula type: algebraic, differential_equation, integral, etc.
    """
    equation_lower = equation.lower()
    
    # Check for differential equations
    if '\\frac{d' in equation or '\\partial' in equation or 'frac{\\partial' in equation:
        return 'differential_equation'
    
    # Check for integrals
    if '\\int' in equation:
        return 'integral'
    
    # Check for summations
    if '\\sum' in equation:
        return 'summation'
    
    # Check for limits
    if '\\lim' in equation:
        return 'limit'
    
    # Check for matrix/vector operations
    if '\\begin{matrix' in equation or '\\begin{bmatrix' in equation:
        return 'matrix'
    
    # Check for inequalities
    if any(ineq in equation for ineq in ['<', '>', '\\leq', '\\geq', '\\neq']):
        return 'inequality'
    
    # Default to algebraic
    return 'algebraic'


def extract_hierarchical_tags(chunk_text: str, 
                              section_context: Optional[Dict] = None,
                              previous_tags: Optional[Dict] = None) -> Dict[str, Any]:
    """Extract complete hierarchical educational metadata for a chunk.
    
    Args:
        chunk_text: Text of the chunk to analyze
        section_context: Optional context from section headers
        previous_tags: Optional tags from previous chunks for context
        
    Returns:
        Dict with comprehensive hierarchical tags including pedagogy_role
    """
    default = {
        "domain": [],
        "topic": [],
        "subtopic": [],
        "concepts": [],
        "prerequisites": [],
        "related_concepts": [],
        "learning_objectives": [],
        "content_type": "unknown",
        "difficulty": "intermediate",
        "cognitive_level": "understand",
        "key_concepts": [],
        "pedagogy_role": "explanation"  # NEW: default pedagogy role
    }
    
    # Build context-aware prompt
    tmpl = prompt_get("ingest.chunk_tags_hierarchical")
    if not tmpl:
        return default
    
    # Include section context if available
    context_text = chunk_text
    if section_context:
        section_title = section_context.get('title', '')
        if section_title:
            context_text = f"Section: {section_title}\n\n{chunk_text}"
    
    prompt = prompt_render(tmpl, {"chunk_text": context_text})
    
    try:
        result = call_llm_json(prompt, default)
        
        # Ensure lists are returned for multi-value fields
        if result:
            # Convert single strings to lists for domain/topic/subtopic
            for field in ['domain', 'topic', 'subtopic']:
                if field in result and isinstance(result[field], str):
                    result[field] = [result[field]] if result[field] else []
            
            # Ensure other list fields exist
            for field in ['concepts', 'prerequisites', 'related_concepts', 
                         'learning_objectives', 'key_concepts']:
                if field not in result or not isinstance(result[field], list):
                    result[field] = []
            
            # Add pedagogy_role classification if not present
            if 'pedagogy_role' not in result or not result['pedagogy_role']:
                result['pedagogy_role'] = classify_pedagogy_role(chunk_text, section_context)
        
        return result or default
    except Exception as e:
        logger.exception("hierarchical_tagging_failed")
        return default


def classify_pedagogy_role(chunk_text: str, section_context: Optional[Dict] = None) -> str:
    """
    Classify chunk's pedagogical role using hybrid approach.
    
    Roles:
    - definition: Defines terms/concepts
    - explanation: Explains how/why something works
    - example: Provides concrete examples
    - derivation: Mathematical derivations
    - proof: Formal proofs
    - application: Real-world applications
    - problem: Practice problems
    - summary: Summaries/reviews
    
    Strategy:
    1. Fast heuristic classification (pattern matching)
    2. If uncertain, use LLM (controlled by env var)
    
    Args:
        chunk_text: Text to classify
        section_context: Optional section context with title
        
    Returns:
        Pedagogy role string
    """
    text_lower = chunk_text.lower()
    
    # Heuristic rules (fast path)
    if any(pattern in text_lower for pattern in ["is defined as", "we define", "definition:", "let us define"]):
        return "definition"
    
    if any(pattern in text_lower for pattern in ["proof:", "to prove", "q.e.d.", "∴", "we prove that", "it follows that"]):
        return "proof"
    
    if any(pattern in text_lower for pattern in ["for example", "for instance", "consider the case", "as an example"]):
        return "example"
    
    if any(pattern in text_lower for pattern in ["deriving", "starting from", "we obtain", "derivation of", "to derive"]):
        return "derivation"
    
    if any(pattern in text_lower for pattern in ["in practice", "real-world", "application", "used in", "practical use"]):
        return "application"
    
    if any(pattern in text_lower for pattern in ["problem:", "exercise:", "find the", "calculate", "solve for"]):
        return "problem"
    
    if any(pattern in text_lower for pattern in ["summary", "in conclusion", "key points", "to summarize", "recap"]):
        return "summary"
    
    # Check section context
    if section_context:
        section_title = (section_context.get("title", "") or "").lower()
        if "example" in section_title:
            return "example"
        if "proof" in section_title:
            return "proof"
        if "application" in section_title:
            return "application"
        if "problem" in section_title or "exercise" in section_title:
            return "problem"
        if "definition" in section_title:
            return "definition"
    
    # Check if LLM classification is enabled and chunk is substantial
    use_llm = os.getenv("PEDAGOGY_LLM_CLASSIFICATION", "false").lower() in ("true", "1", "yes")
    if use_llm and len(chunk_text) > 100:
        role = _classify_pedagogy_with_llm(chunk_text)
        if role:
            return role
    
    # Default fallback
    return "explanation"


def _classify_pedagogy_with_llm(chunk_text: str) -> Optional[str]:
    """Use LLM to classify pedagogy role when heuristics uncertain.
    
    Args:
        chunk_text: Text to classify
        
    Returns:
        Pedagogy role string or None if classification fails
    """
    prompt_template = prompt_get("ingest.classify_pedagogy_role")
    if not prompt_template:
        return None
    
    # Truncate for efficiency
    text_sample = chunk_text[:500] if len(chunk_text) > 500 else chunk_text
    
    prompt = prompt_render(prompt_template, {
        "text": text_sample,
        "valid_roles": ["definition", "explanation", "example", "derivation", "proof", "application", "problem", "summary"]
    })
    
    default = {"role": "explanation", "confidence": 0.5}
    
    try:
        result = call_llm_json(prompt, default)
        role = result.get("role", "explanation")
        confidence = result.get("confidence", 0.5)
        
        # Only use LLM result if confident
        confidence_threshold = float(os.getenv("PEDAGOGY_LLM_CONFIDENCE_THRESHOLD", "0.7"))
        if confidence >= confidence_threshold:
            return role
    except Exception:
        logger.exception("llm_pedagogy_classification_failed")
    
    return None


def build_concept_hierarchy(concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Organize concepts into a hierarchical tree structure.
    
    Args:
        concepts: List of concept dicts with name, type, importance
        
    Returns:
        Hierarchical tree structure
    """
    hierarchy = {
        'domains': {},
        'topics': {},
        'subtopics': {},
        'concepts': concepts
    }
    
    # Group concepts by type for better organization
    by_type = {}
    for concept in concepts:
        ctype = concept.get('type', 'unknown')
        if ctype not in by_type:
            by_type[ctype] = []
        by_type[ctype].append(concept)
    
    # Principles and theorems are typically high-level
    if 'principle' in by_type or 'theorem' in by_type:
        hierarchy['foundational'] = by_type.get('principle', []) + by_type.get('theorem', [])
    
    # Methods and parameters are typically lower-level
    if 'method' in by_type or 'parameter' in by_type:
        hierarchy['applied'] = by_type.get('method', []) + by_type.get('parameter', [])
    
    return hierarchy


def extract_prerequisites(chunk: Dict[str, Any], all_tags: List[Dict[str, Any]]) -> List[str]:
    """Identify prerequisite concepts for a chunk.
    
    Args:
        chunk: Chunk dict with text and metadata
        all_tags: All available tags from other chunks for matching
        
    Returns:
        List of prerequisite concept names
    """
    prerequisites = []
    
    # Get explicit prerequisites from hierarchical tags
    if 'prerequisites' in chunk:
        prerequisites.extend(chunk['prerequisites'])
    
    # Identify implicit prerequisites based on concept mentions
    chunk_text = chunk.get('full_text', '').lower()
    
    # Common prerequisite indicators
    prerequisite_phrases = [
        'assuming', 'given that', 'requires', 'depends on', 'building on',
        'based on', 'using the result', 'from earlier', 'previously shown'
    ]
    
    for phrase in prerequisite_phrases:
        if phrase in chunk_text:
            # This chunk likely has prerequisites
            pass
    
    # Match against available concepts from previous chunks
    if all_tags:
        for tag_set in all_tags:
            for concept in tag_set.get('key_concepts', []):
                # If concept is mentioned in current chunk, it might be a prerequisite
                if concept.lower() in chunk_text and concept not in prerequisites:
                    prerequisites.append(concept)
    
    # Remove duplicates and limit
    prerequisites = list(dict.fromkeys(prerequisites))[:10]
    
    return prerequisites


def classify_content_type(chunk: Dict[str, Any]) -> str:
    """Classify the educational content type of a chunk.
    
    Args:
        chunk: Chunk dict with text and metadata
        
    Returns:
        Content type string
    """
    text = chunk.get('full_text', '').lower()
    
    # Pattern-based classification
    if any(word in text for word in ['definition:', 'is defined as', 'we define']):
        return 'definition'
    
    if any(word in text for word in ['theorem:', 'lemma:', 'corollary:']):
        return 'theorem'
    
    if any(word in text for word in ['proof:', 'to prove', 'we can show']):
        return 'proof'
    
    if any(word in text for word in ['example', 'for instance', 'consider the case']):
        return 'example'
    
    if any(word in text for word in ['problem:', 'exercise:', 'find', 'calculate']):
        return 'problem'
    
    if any(word in text for word in ['solution:', 'solving', 'we obtain']):
        return 'solution'
    
    if any(word in text for word in ['derivation', 'deriving', 'starting from']):
        return 'derivation'
    
    if any(word in text for word in ['summary', 'in conclusion', 'to summarize']):
        return 'summary'
    
    # Check if it introduces a concept
    if any(word in text[:200] for word in ['introduction', 'overview', 'concept of']):
        return 'concept_intro'
    
    return 'unknown'


def estimate_difficulty(chunk: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Estimate difficulty level based on multiple factors.
    
    Args:
        chunk: Chunk dict with text and metadata
        
    Returns:
        Tuple of (difficulty_level, scoring_details)
    """
    score = 0
    details = {
        'formula_score': 0,
        'prerequisite_score': 0,
        'concept_score': 0,
        'math_level_score': 0
    }
    
    text = chunk.get('full_text', '')
    
    # 1. Formula complexity (0-30 points)
    if chunk.get('has_equation', False):
        formula_count = len(chunk.get('formulas', []))
        details['formula_score'] = min(formula_count * 5, 30)
        score += details['formula_score']
        
        # Check formula types
        for formula in chunk.get('formulas', []):
            ftype = formula.get('type', '')
            if ftype == 'differential_equation':
                score += 10
            elif ftype == 'integral':
                score += 8
            elif ftype in ['matrix', 'summation']:
                score += 6
    
    # 2. Prerequisite depth (0-25 points)
    prereqs = chunk.get('prerequisites', [])
    details['prerequisite_score'] = min(len(prereqs) * 5, 25)
    score += details['prerequisite_score']
    
    # 3. Abstract concept count (0-20 points)
    key_concepts = chunk.get('key_concepts', [])
    abstract_indicators = ['theory', 'principle', 'theorem', 'axiom', 'postulate']
    abstract_count = sum(1 for concept in key_concepts 
                        if any(ind in concept.lower() for ind in abstract_indicators))
    details['concept_score'] = min(abstract_count * 10, 20)
    score += details['concept_score']
    
    # 4. Mathematical level (0-25 points)
    text_lower = text.lower()
    if any(word in text_lower for word in ['differential', 'partial derivative', 'laplacian']):
        details['math_level_score'] = 25
    elif any(word in text_lower for word in ['integral', 'derivative', 'calculus']):
        details['math_level_score'] = 20
    elif any(word in text_lower for word in ['trigonometric', 'logarithm', 'exponential']):
        details['math_level_score'] = 15
    elif any(word in text_lower for word in ['algebra', 'equation', 'formula']):
        details['math_level_score'] = 10
    else:
        details['math_level_score'] = 5
    
    score += details['math_level_score']
    
    # Convert score to difficulty level
    if score < 30:
        difficulty = 'introductory'
    elif score < 60:
        difficulty = 'intermediate'
    else:
        difficulty = 'advanced'
    
    details['total_score'] = score
    
    return difficulty, details


def extract_formula_metadata(formulas: List[str], 
                             chunk_text: str,
                             chunk_context: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Extract comprehensive metadata for formulas.
    
    Args:
        formulas: List of LaTeX formula strings
        chunk_text: Full chunk text for context
        chunk_context: Optional additional context
        
    Returns:
        List of formula metadata dicts
    """
    if not formulas:
        return []
    
    formula_metadata = []
    
    tmpl = prompt_get("ingest.formula_extraction")
    if not tmpl:
        # Fallback to basic parsing without LLM
        for idx, formula in enumerate(formulas):
            metadata = {
                'id': f'formula_{idx}',
                'latex': formula,
                'variables': identify_variables(formula),
                'type': classify_formula_type(formula),
                'variable_details': []
            }
            formula_metadata.append(metadata)
        return formula_metadata
    
    # Use LLM for rich metadata extraction
    prompt = prompt_render(tmpl, {"text": chunk_text})
    
    default = {"formulas": []}
    
    try:
        result = call_llm_json(prompt, default)
        extracted_formulas = result.get('formulas', []) if result else []
        
        # Enrich with parsed data
        for idx, formula_dict in enumerate(extracted_formulas):
            if 'latex' in formula_dict:
                latex = formula_dict['latex']
                
                # Add parsed data
                if 'variables' not in formula_dict:
                    formula_dict['variables'] = []
                
                if 'type' not in formula_dict:
                    formula_dict['type'] = classify_formula_type(latex)
                
                if 'id' not in formula_dict:
                    formula_dict['id'] = f'formula_{idx}'
                
                # Add variable symbols if not extracted
                parsed_vars = identify_variables(latex)
                existing_symbols = {v.get('symbol', '') for v in formula_dict.get('variables', [])}
                for var in parsed_vars:
                    if var not in existing_symbols:
                        formula_dict['variables'].append({
                            'symbol': var,
                            'meaning': '',
                            'units': '',
                            'role': 'unknown'
                        })
            
            formula_metadata.append(formula_dict)
        
        return formula_metadata
        
    except Exception as e:
        logger.exception("formula_metadata_extraction_failed")
        # Return basic parsed data
        for idx, formula in enumerate(formulas):
            metadata = {
                'id': f'formula_{idx}',
                'latex': formula,
                'variables': identify_variables(formula),
                'type': classify_formula_type(formula),
                'variable_details': []
            }
            formula_metadata.append(metadata)
        
        return formula_metadata


def tag_and_extract_formulas(chunk: Dict[str, Any],
                             section_context: Optional[Dict] = None,
                             previous_tags: Optional[List[Dict]] = None) -> Dict[str, Any]:
    """Complete tagging and formula extraction for a chunk.
    
    This is the main entry point that combines all tagging functions.
    
    Args:
        chunk: Chunk dict with text and metadata
        section_context: Optional section context
        previous_tags: Optional previous chunk tags for prerequisite detection
        
    Returns:
        Enhanced chunk dict with all tags and formula metadata
    """
    # Extract hierarchical tags
    hierarchical_tags = extract_hierarchical_tags(
        chunk.get('full_text', ''),
        section_context=section_context,
        previous_tags=previous_tags[-1] if previous_tags else None
    )
    
    # Merge into chunk
    chunk.update(hierarchical_tags)
    
    # Classify content type if not already done
    if chunk.get('content_type') == 'unknown':
        chunk['content_type'] = classify_content_type(chunk)
    
    # Estimate difficulty
    difficulty, difficulty_details = estimate_difficulty(chunk)
    chunk['difficulty'] = difficulty
    chunk['difficulty_details'] = difficulty_details
    
    # Extract prerequisites
    if previous_tags:
        chunk['prerequisites'] = extract_prerequisites(chunk, previous_tags)
    
    # Extract formula metadata if chunk has equations
    if chunk.get('has_equation', False):
        # Parse LaTeX from text
        chunk_text = chunk.get('full_text', '')
        latex_formulas = parse_latex(chunk_text)
        
        if latex_formulas:
            formula_metadata = extract_formula_metadata(
                latex_formulas,
                chunk_text,
                chunk_context=section_context
            )
            chunk['formulas'] = formula_metadata
            chunk['formula_count'] = len(formula_metadata)
    
    return chunk
