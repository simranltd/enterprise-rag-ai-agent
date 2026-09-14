"""Deterministic Expense Policy approval checker."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpenseApprovalResult:
    amount: float
    approval: str
    document_title: str = "Northstar Financial Services Expense Policy"
    section_title: str = "3. Approval thresholds"
    document_version: str = "3.0"

    @property
    def answer_text(self) -> str:
        return (
            f"For an expense of ${self.amount:,.2f} CAD, required approval is "
            f"{self.approval}."
        )


def calculate_expense_approval(amount: float) -> ExpenseApprovalResult:
    """Apply the current Expense Policy thresholds exactly."""
    if amount < 0:
        raise ValueError("amount cannot be negative")
    if amount <= 500:
        approval = "the employee's manager"
    elif amount <= 5000:
        approval = "the employee's manager and department budget owner"
    else:
        approval = "the department head and Finance Operations"
    return ExpenseApprovalResult(amount=amount, approval=approval)
