"""The technical (lead) agent, split by responsibility:

    agent.py     TechnicalAgent: which calls to make, in what order, and what to do when one fails
    prompts.py   the exact text sent to the model, and the mechanical baseline it quotes
    schemas.py   the response models and JSON schemas for each call
    phases.py    the model-free rules of the review step (the critic's veto and the audit stamps)"""
from src.agents.technical.agent import TechnicalAgent
from src.agents.technical.prompts import mechanical_action, mechanical_baseline
from src.agents.technical.schemas import COT_SCHEMA, REFLECT_SCHEMA, AdvancedSignal, ReflectionChain, ReflectionStage

__all__ = ["TechnicalAgent", "mechanical_action", "mechanical_baseline", "COT_SCHEMA", "REFLECT_SCHEMA", "AdvancedSignal",
           "ReflectionChain", "ReflectionStage"]
