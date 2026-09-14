from northstar.agents import AgentRoute, calculate_expense_approval, classify_request
from northstar.rag import filter_current_policy_results


def test_router_classifies_greeting_tool_and_knowledge() -> None:
    assert classify_request("Hello") is AgentRoute.GREETING
    assert classify_request("What approval do I need for a $7,000 expense?") is AgentRoute.TOOL
    assert classify_request("Can I work overseas?") is AgentRoute.KNOWLEDGE


def test_expense_approval_tool_uses_current_policy_thresholds() -> None:
    assert "employee's manager" in calculate_expense_approval(500).approval
    assert "department budget owner" in calculate_expense_approval(5000).approval
    assert "department head and Finance Operations" in calculate_expense_approval(7000).approval


def test_superseded_results_are_filtered() -> None:
    results = [
        {"section_title": "Approval thresholds"},
        {"section_title": "Superseded version"},
    ]
    filtered = filter_current_policy_results(results)
    assert filtered == [{"section_title": "Approval thresholds"}]
