#!/usr/bin/env python3
"""
RL Training Data Generation Script

End-to-end script to generate training data for tutor RL by:
1. Loading resources from the database
2. Extracting concept sequences
3. Simulating tutoring sessions with diverse student models
4. Outputting trajectories in multiple formats (SFT, DPO, PPO)

Usage:
    # Generate from specific concepts
    python scripts/generate_rl_data.py --concepts "convection,conduction,radiation" --students 5
    
    # Generate from resources
    python scripts/generate_rl_data.py --resources "resource_id_1,resource_id_2" --students 10
    
    # Generate with specific learner types
    python scripts/generate_rl_data.py --concepts "thermodynamics" --learners "struggling,average,fast_learner"
    
    # Batch generation with progress
    python scripts/generate_rl_data.py --concepts "heat_transfer" --students 20 --output data/training_batch1

Examples with SOTA Models:
    # Use GPT-4o as expert tutor for high-quality SFT data
    python scripts/generate_rl_data.py --concepts "convection" --students 5 \\
        --tutor-model "gpt-4o" \\
        --student-model "gpt-4o-mini" \\
        --prefix "gpt4o_expert"
    
    # Use Claude 3.5 Sonnet for pedagogical responses
    python scripts/generate_rl_data.py --concepts "thermodynamics" --students 3 \\
        --tutor-model "claude-3-5-sonnet-20241022" \\
        --student-model "gemini-2.0-flash-lite" \\
        --learners "struggling,average"
    
    # Use same SOTA model for all components
    python scripts/generate_rl_data.py --concepts "heat_transfer" \\
        --all-models "gpt-4o" \\
        --students 10
    
    # Quick test with single concept (no LLM students for speed)
    python scripts/generate_rl_data.py --concepts "natural_convection" --students 2 --no-llm-students
    
    # Full curriculum from textbook
    python scripts/generate_rl_data.py --resources "heat_transfer_ch6" --students 5 --output data/ht_curriculum
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate RL training data from tutoring simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--concepts",
        type=str,
        help="Comma-separated list of concept IDs to cover"
    )
    input_group.add_argument(
        "--resources",
        type=str,
        help="Comma-separated list of resource IDs to extract concepts from"
    )
    input_group.add_argument(
        "--concepts-file",
        type=str,
        help="JSON file with list of concepts"
    )
    
    # Student configuration
    parser.add_argument(
        "--students",
        type=int,
        default=3,
        help="Number of different students per concept (default: 3)"
    )
    parser.add_argument(
        "--learners",
        type=str,
        default=None,
        help="Comma-separated learner types: fast_learner,average,struggling,anxious,rusher,deep_thinker,passive"
    )
    parser.add_argument(
        "--no-llm-students",
        action="store_true",
        help="Use template-based students instead of LLM (faster but less diverse)"
    )
    
    # Session parameters
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns per concept (default: 20)"
    )
    parser.add_argument(
        "--target-mastery",
        type=float,
        default=0.8,
        help="Target mastery level (default: 0.8)"
    )
    parser.add_argument(
        "--max-concepts",
        type=int,
        default=10,
        help="Maximum concepts per session (default: 10)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default="data/trajectories",
        help="Output directory for trajectory files (default: data/trajectories)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="rl_data",
        help="Prefix for output files (default: rl_data)"
    )
    
    # LLM Model configuration
    parser.add_argument(
        "--tutor-model",
        type=str,
        default=None,
        help="LLM model for tutor (policy + responses). E.g., gpt-4o, claude-3-5-sonnet-20241022, gemini-2.0-flash"
    )
    parser.add_argument(
        "--student-model",
        type=str,
        default=None,
        help="LLM model for student simulation. E.g., gemini-2.0-flash-lite, gpt-4o-mini"
    )
    parser.add_argument(
        "--analyzer-model",
        type=str,
        default=None,
        help="LLM model for input analysis (can use cheaper model)"
    )
    parser.add_argument(
        "--all-models",
        type=str,
        default=None,
        help="Use same model for all components (convenience flag)"
    )
    
    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration and exit without running"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse inputs
    concept_ids = []
    resource_ids = []
    
    if args.concepts:
        concept_ids = [c.strip() for c in args.concepts.split(",") if c.strip()]
    elif args.resources:
        resource_ids = [r.strip() for r in args.resources.split(",") if r.strip()]
    elif args.concepts_file:
        with open(args.concepts_file) as f:
            data = json.load(f)
            concept_ids = data if isinstance(data, list) else data.get("concepts", [])
    else:
        # Default: use sample concepts for testing
        logger.warning("No concepts or resources specified. Using sample concepts for demo.")
        concept_ids = ["convection", "conduction", "radiation"]
    
    # Parse learner types
    learner_types = None
    if args.learners:
        learner_types = [lt.strip() for lt in args.learners.split(",") if lt.strip()]
    
    # Resolve model configuration
    tutor_model = args.all_models or args.tutor_model
    student_model = args.all_models or args.student_model
    analyzer_model = args.all_models or args.analyzer_model
    
    # Print configuration
    print("\n" + "=" * 60)
    print("RL Training Data Generation")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Concepts: {concept_ids if concept_ids else '(from resources)'}")
    print(f"  Resources: {resource_ids if resource_ids else '(none)'}")
    print(f"  Students per concept: {args.students}")
    print(f"  Learner types: {learner_types or 'all (balanced)'}")
    print(f"  Max turns/concept: {args.max_turns}")
    print(f"  Target mastery: {args.target_mastery}")
    print(f"  Use LLM students: {not args.no_llm_students}")
    print(f"  Output directory: {args.output}")
    print(f"  Output prefix: {args.prefix}")
    print(f"\nLLM Models:")
    print(f"  Tutor model: {tutor_model or '(default from env)'}")
    print(f"  Student model: {student_model or '(default: gemini-2.0-flash-lite)'}")
    print(f"  Analyzer model: {analyzer_model or '(default from env)'}")
    print()
    
    if args.dry_run:
        print("DRY RUN - exiting without execution")
        return 0
    
    # Import here to avoid slow startup for --help
    from mdp.curriculum_runner import CurriculumRunner, CurriculumConfig
    from mdp.llm_client import LLMClient
    
    # Create config with model specifications
    config = CurriculumConfig(
        resource_ids=resource_ids,
        concept_ids=concept_ids,
        max_turns_per_concept=args.max_turns,
        target_mastery=args.target_mastery,
        max_concepts_per_session=args.max_concepts,
        students_per_concept=args.students,
        learner_types=learner_types,
        use_llm_students=not args.no_llm_students,
        output_dir=args.output,
        output_prefix=args.prefix,
        # LLM model configuration
        tutor_model=tutor_model,
        student_model=student_model,
        analyzer_model=analyzer_model,
    )
    
    # Run
    print("Starting data generation...")
    start_time = time.time()
    
    try:
        runner = CurriculumRunner(config)
        results = runner.run()
    except Exception as e:
        logger.error(f"Data generation failed: {e}", exc_info=True)
        return 1
    
    duration = time.time() - start_time
    
    # Print results
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    
    if results.get("status") == "success":
        stats = results.get("statistics", {})
        files = results.get("output_files", {})
        
        print(f"\nSummary:")
        print(f"  Sessions run: {results.get('sessions_run', 0)}")
        print(f"  Concepts covered: {results.get('concepts_covered', 0)}")
        print(f"  Total transitions: {results.get('total_transitions', 0)}")
        print(f"  Duration: {duration:.1f}s")
        
        print(f"\nStatistics:")
        print(f"  Mastery rate: {stats.get('mastery_rate', 0):.1%}")
        print(f"  Avg mastery gain: {stats.get('avg_mastery_gain', 0):.3f}")
        print(f"  Avg turns/concept: {stats.get('avg_turns_per_concept', 0):.1f}")
        
        reward_stats = stats.get("reward_stats", {})
        print(f"  Reward mean: {reward_stats.get('mean', 0):.3f}")
        print(f"  Reward range: [{reward_stats.get('min', 0):.3f}, {reward_stats.get('max', 0):.3f}]")
        
        print(f"\nLearner distribution:")
        for lt, count in stats.get("learner_distribution", {}).items():
            print(f"    {lt}: {count}")
        
        print(f"\nOutput files:")
        for fmt, path in files.items():
            print(f"  {fmt}: {path}")
        
        print(f"\nTo use for training:")
        print(f"  SFT: {files.get('sft', 'N/A')}")
        print(f"  DPO: {files.get('dpo', 'N/A')}")
        print(f"  PPO: {files.get('ppo', 'N/A')}")
        
    else:
        print(f"\nGeneration failed: {results.get('error', 'Unknown error')}")
        return 1
    
    return 0


def list_concepts_from_resources(resource_ids: list):
    """Utility to list available concepts from resources"""
    from mdp.rag import RAGTools
    
    rag = RAGTools()
    concepts = rag.get_concepts_for_resources(resource_ids)
    
    print(f"\nConcepts from resources {resource_ids}:")
    for c in concepts:
        print(f"  - {c.get('name')} (id: {c.get('id')})")
    
    return concepts


def list_available_resources():
    """Utility to list available resources in the database"""
    try:
        from kg_pipeline.base import managed_driver
        
        with managed_driver() as driver:
            if not driver:
                print("Could not connect to database")
                return []
            
            with driver.session() as session:
                result = session.run("""
                    MATCH (r:Resource)
                    RETURN r.id as id, r.title as title
                    LIMIT 20
                """)
                resources = [r.data() for r in result]
        
        print("\nAvailable resources:")
        for r in resources:
            print(f"  - {r.get('title', 'Untitled')} (id: {r.get('id')})")
        
        return resources
        
    except Exception as e:
        logger.error(f"Could not list resources: {e}")
        return []


if __name__ == "__main__":
    sys.exit(main())
