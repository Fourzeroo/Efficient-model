"""
Phoenix Integration Test Script

This script verifies that Phoenix tracing is working correctly
with the LangGraph NAS system.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


def test_phoenix_connection():
    """Test connection to Phoenix server."""
    print("=" * 80)
    print("PHOENIX INTEGRATION TEST")
    print("=" * 80)
    
    # Test 1: Check if Phoenix server is accessible
    print("\n[1/5] Testing Phoenix server connection...")
    try:
        import requests
        response = requests.get("http://localhost:6006", timeout=5)
        if response.status_code == 200:
            print("✓ Phoenix server is accessible at http://localhost:6006")
        else:
            print(f"⚠ Phoenix server responded with status {response.status_code}")
    except Exception as e:
        print(f"✗ Cannot connect to Phoenix server: {e}")
        print("  Make sure Phoenix is running: python -m phoenix.server.main serve")
        return False
    
    # Test 2: Import Phoenix tracing module
    print("\n[2/5] Testing Phoenix tracing imports...")
    try:
        from nas_agent_graph.phoenix_tracing import (
            setup_phoenix_tracing, 
            trace_operation,
            is_phoenix_enabled
        )
        print("✓ Phoenix tracing module imported successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Install dependencies: pip install -r nas_agent_graph/requirements.txt")
        return False
    
    # Test 3: Initialize Phoenix tracing
    print("\n[3/5] Initializing Phoenix tracing...")
    try:
        success = setup_phoenix_tracing()
        if success:
            print("✓ Phoenix tracing initialized successfully")
        else:
            print("⚠ Phoenix tracing initialization returned False")
            return False
    except Exception as e:
        print(f"✗ Initialization error: {e}")
        return False
    
    # Test 4: Test custom span creation
    print("\n[4/5] Testing custom span creation...")
    try:
        from nas_agent_graph.phoenix_tracing import add_span_attribute, add_span_event
        
        with trace_operation("test_operation", {"test": "value"}):
            add_span_attribute("test_attr", "test_value")
            add_span_event("test_event", {"data": 123})
        
        print("✓ Custom spans created successfully")
    except Exception as e:
        print(f"✗ Span creation error: {e}")
        return False
    
    # Test 5: Verify traces endpoint
    print("\n[5/5] Verifying traces endpoint...")
    try:
        import requests
        response = requests.get("http://localhost:6006/v1/traces", timeout=5)
        if response.status_code in [200, 405]:  # 405 is OK (POST only)
            print("✓ Traces endpoint is accessible")
        else:
            print(f"⚠ Traces endpoint status: {response.status_code}")
    except Exception as e:
        print(f"⚠ Could not verify traces endpoint: {e}")
    
    print("\n" + "=" * 80)
    print("✓ PHOENIX INTEGRATION TEST PASSED")
    print("=" * 80)
    print("\nPhoenix UI: http://localhost:6006")
    print("Project: nas-agent-graph")
    print("\nRun the NAS system to see traces:")
    print("  python quickstart_nas_graph.py")
    print("=" * 80)
    
    return True


def test_llm_tracing():
    """Test LLM call tracing with a simple example."""
    print("\n" + "=" * 80)
    print("TESTING LLM TRACING (OPTIONAL)")
    print("=" * 80)
    
    try:
        from nas_agent_graph.phoenix_tracing import setup_phoenix_tracing
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("⚠ OPENROUTER_API_KEY not set, skipping LLM tracing test")
            return
        
        print("\n[1/2] Initializing Phoenix for LLM tracing...")
        setup_phoenix_tracing()
        print("✓ Phoenix initialized")
        
        print("\n[2/2] Making test LLM call (this will appear in Phoenix)...")
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            model="anthropic/claude-3.5-sonnet",
        )
        
        response = llm.invoke([HumanMessage(content="Say 'Phoenix test successful' and nothing else.")])
        print(f"✓ LLM Response: {response.content}")
        
        print("\n✓ LLM call traced successfully!")
        print("  Check Phoenix UI to see the LLM trace")
        
    except Exception as e:
        print(f"⚠ LLM tracing test failed: {e}")


def main():
    """Run all tests."""
    success = test_phoenix_connection()
    
    if success:
        print("\n" + "=" * 80)
        print("Would you like to test LLM tracing? (requires API key)")
        print("This will make a real API call to OpenRouter.")
        print("=" * 80)
        
        # For automated testing, skip LLM test
        # Uncomment below to enable interactive LLM test
        # response = input("Test LLM tracing? (y/n): ")
        # if response.lower() == 'y':
        #     test_llm_tracing()
        
        print("\n✓ All tests completed!")
    else:
        print("\n✗ Some tests failed. Please fix the issues and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
