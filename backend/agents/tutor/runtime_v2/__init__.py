"""Runtime v2 modules for MDP- and RL-oriented tutor execution.

This package hosts the new orchestrators and policies that treat the tutor
as a stack of MDPs (session, concept, turn) with pluggable policies.

Initial versions may delegate to legacy runtimes for behaviour parity; over
time, logic will move here while keeping external APIs unchanged.
"""
