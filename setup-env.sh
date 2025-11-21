#!/bin/bash

# Setup Environment Variables for Tutor Agent
# This script helps you quickly set up your environment

set -e

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║           Tutor Agent Environment Setup - TUTOR-IMPROVE-18               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env from .env.example..."
    cp .env.example .env
    echo "✅ Created .env"
    echo ""
fi

# Ask for environment type
echo "🎯 Select your environment:"
echo "  1) Development (debug mode with verbose logging)"
echo "  2) Production (intelligent mode, full AI)"
echo "  3) Testing (simple mode, fast, no LLM)"
echo "  4) Structured Learning (step-by-step mode)"
echo "  5) Custom (enter mode manually)"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        MODE="debug"
        echo "✅ Selected: Development (TUTOR_MODE=debug)"
        ;;
    2)
        MODE="intelligent"
        echo "✅ Selected: Production (TUTOR_MODE=intelligent)"
        ;;
    3)
        MODE="simple"
        echo "✅ Selected: Testing (TUTOR_MODE=simple)"
        ;;
    4)
        MODE="step_by_step"
        echo "✅ Selected: Structured Learning (TUTOR_MODE=step_by_step)"
        ;;
    5)
        read -p "Enter mode (simple/intelligent/step_by_step/debug): " MODE
        echo "✅ Selected: Custom ($MODE)"
        ;;
    *)
        echo "❌ Invalid choice. Defaulting to intelligent."
        MODE="intelligent"
        ;;
esac

echo ""
echo "🔧 Setting up environment variables..."

# Update or add TUTOR_MODE to .env
if grep -q "^TUTOR_MODE=" .env; then
    # Replace existing TUTOR_MODE
    sed -i.bak "s/^TUTOR_MODE=.*/TUTOR_MODE=$MODE/" .env
    echo "✅ Updated TUTOR_MODE=$MODE in .env"
else
    # Add TUTOR_MODE at the beginning (after comments)
    # Find the first non-comment line and insert before it
    sed -i.bak '1s/^/TUTOR_MODE='$MODE'\n/' .env
    echo "✅ Added TUTOR_MODE=$MODE to .env"
fi

echo ""
echo "📊 Your configuration:"
echo "  ┌─────────────────────────────────────────┐"
echo "  │ TUTOR_MODE=$MODE"
grep -E "^(TUTOR_MASTERY|OPENAI_API|LLM_MODEL|DATABASE|NEO4J|REDIS)" .env 2>/dev/null | head -5 | sed 's/^/  │ /' || echo "  │ (other env vars in .env)"
echo "  │                                         │"
echo "  │ All other settings auto-configured      │"
echo "  └─────────────────────────────────────────┘"
echo ""

# Show what this mode enables
echo "🚀 This mode enables:"
case $MODE in
    simple)
        echo "  ✅ Heuristic decision-making"
        echo "  ✅ State machine"
        echo "  ✅ Mastery tracking"
        echo "  ❌ LLM policy (disabled)"
        echo "  ❌ SRL planning (disabled)"
        echo "  ℹ️ Best for: Development, testing, debugging"
        ;;
    intelligent)
        echo "  ✅ LLM policy layer"
        echo "  ✅ SRL planning"
        echo "  ✅ State machine"
        echo "  ✅ LLM-integrated grounding"
        echo "  ✅ Mastery tracking"
        echo "  ℹ️ Best for: Production, best quality"
        ;;
    step_by_step)
        echo "  ✅ SRL planning"
        echo "  ✅ Multi-step execution"
        echo "  ✅ State machine"
        echo "  ✅ LLM-integrated grounding"
        echo "  ❌ LLM policy (disabled)"
        echo "  ℹ️ Best for: Structured learning, curriculum"
        ;;
    debug)
        echo "  ✅ LLM policy layer"
        echo "  ✅ SRL planning"
        echo "  ✅ State machine"
        echo "  ✅ Debug logging (VERBOSE)"
        echo "  ✅ Decision traces in responses"
        echo "  ℹ️ Best for: Troubleshooting, development"
        ;;
    *)
        echo "  ⚠️ Unknown mode: $MODE"
        echo "  Valid modes: simple, intelligent, step_by_step, debug"
        ;;
esac

echo ""
echo "📚 Documentation:"
echo "  • ENVIRONMENT_SETUP_GUIDE.md - Comprehensive guide"
echo "  • TUTOR-IMPROVE-18-COMPLETE.md - Implementation details"
echo "  • backend/agents/tutor/config.py - Source code"
echo ""

echo "✅ Environment setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review .env file (optional adjustments)"
echo "  2. Start the backend: python backend/main.py"
echo "  3. Check logs for 'tutor_config_initialized'"
echo ""
echo "Need help? Read ENVIRONMENT_SETUP_GUIDE.md"

