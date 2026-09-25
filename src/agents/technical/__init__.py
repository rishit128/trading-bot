"""The technical (lead) agent, split by responsibility:

    agent.py     TechnicalAgent: which calls to make, in what order, and what to do when one fails
    prompts.py   the exact text sent to the model, and the mechanical baseline it quotes
    schemas.py   the response models and JSON schemas for each call
    phases.py    the model-free rules of the refinement phases (capping, audit details, the critic's veto)"""
from src.agents.technical.agent import TechnicalAgent
from src.agents.technical.phases import MAX_PHASE_ADJUSTMENT
from src.agents.technical.prompts import mechanical_action, mechanical_baseline
from src.agents.technical.schemas import (CONTEXT_SCHEMA, COT_SCHEMA, LEARNING_SCHEMA, REFLECT_SCHEMA, AdvancedSignal,
                                          ContextSignal, LearningSignal, ReflectionChain, ReflectionStage)

__all__ = ["TechnicalAgent", "MAX_PHASE_ADJUSTMENT", "mechanical_action", "mechanical_baseline", "COT_SCHEMA",
           "LEARNING_SCHEMA", "CONTEXT_SCHEMA", "REFLECT_SCHEMA", "AdvancedSignal", "LearningSignal", "ContextSignal",
           "ReflectionChain", "ReflectionStage"]
