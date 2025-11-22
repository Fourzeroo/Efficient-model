"""
Phoenix Tracing Integration for NAS System

This module sets up Phoenix observability for LangGraph and LangChain operations.
It instruments LLM calls, chains, and custom spans for comprehensive tracing.
"""

import os
from typing import Optional
from contextlib import contextmanager

from . import config as graph_config


# Global flag to track if Phoenix is initialized
_phoenix_initialized = False


def setup_phoenix_tracing() -> bool:
    """
    Initialize Phoenix tracing for LangChain and LangGraph.
    
    Returns:
        True if Phoenix was successfully initialized, False otherwise
    """
    global _phoenix_initialized
    
    if not graph_config.PHOENIX_ENABLED:
        if graph_config.VERBOSE:
            print("Phoenix tracing disabled (PHOENIX_ENABLED=false)")
        return False
    
    if _phoenix_initialized:
        if graph_config.DEBUG:
            print("Phoenix already initialized, skipping")
        return True
    
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from openinference.instrumentation.openai import OpenAIInstrumentor
        
        # Register Phoenix tracer with custom endpoint
        tracer_provider = register(
            project_name="nas-agent-graph",
            endpoint=graph_config.PHOENIX_COLLECTOR_ENDPOINT,
        )
        
        # Instrument LangChain
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        
        # Instrument OpenAI (for ChatOpenAI calls with prompts/responses)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        
        if graph_config.VERBOSE:
            print(f"✓ Phoenix tracing enabled at {graph_config.PHOENIX_COLLECTOR_ENDPOINT}")
            print(f"  View traces at: http://{graph_config.PHOENIX_HOST}:{graph_config.PHOENIX_PORT}")
            print(f"  Instrumented: LangChain, OpenAI")
        
        _phoenix_initialized = True
        return True
        
    except ImportError as e:
        print(f"⚠ Phoenix tracing unavailable: {e}")
        print("  Install with: pip install arize-phoenix openinference-instrumentation-langchain openinference-instrumentation-openai")
        return False
        
    except Exception as e:
        print(f"⚠ Failed to initialize Phoenix tracing: {e}")
        return False


def get_tracer():
    """Get the OpenTelemetry tracer for custom spans."""
    try:
        from opentelemetry import trace
        return trace.get_tracer("nas_agent_graph")
    except ImportError:
        return None


@contextmanager
def trace_operation(operation_name: str, attributes: Optional[dict] = None):
    """
    Context manager for tracing custom operations.
    
    Args:
        operation_name: Name of the operation to trace
        attributes: Optional attributes to attach to the span
        
    Example:
        with trace_operation("config_modification", {"changes": 3}):
            apply_config_changes(...)
    """
    tracer = get_tracer()
    
    if tracer is None or not _phoenix_initialized:
        # No-op if tracing not available
        yield
        return
    
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
        
        with tracer.start_as_current_span(operation_name) as span:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    except ImportError:
        # Fallback if OpenTelemetry not available
        yield


def add_span_attribute(key: str, value: any):
    """Add an attribute to the current span if tracing is enabled."""
    if not _phoenix_initialized:
        return
    
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.is_recording():
            span.set_attribute(key, str(value))
    except (ImportError, Exception):
        pass


def add_span_event(name: str, attributes: Optional[dict] = None):
    """Add an event to the current span if tracing is enabled."""
    if not _phoenix_initialized:
        return
    
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.is_recording():
            span.add_event(name, attributes or {})
    except (ImportError, Exception):
        pass


def is_phoenix_enabled() -> bool:
    """Check if Phoenix tracing is enabled and initialized."""
    return _phoenix_initialized
