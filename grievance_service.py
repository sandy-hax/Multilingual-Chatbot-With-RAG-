"""
grievance_service.py
====================
Sequential Grievance Form Filing and Department Analysis Engine.
Analyzes user complaint intent, asks follow-up questions sequentially,
and commits completed grievance forms to PostgreSQL/SQLite DB.

Fix 2: Added session-based in-memory draft store so that the LLM agent
loop can persist the form state across multiple tool-call turns within
the same user query session.
"""

import re
import os
import uuid
import threading
from typing import Dict, Any, Optional

from db import db_service


# Department taxonomy
DEPARTMENTS = {
    "pacs":        "PACS Cooperative Society",
    "pmfby":       "PMFBY Crop Insurance",
    "agriculture": "Agriculture & Farmers Welfare",
    "pds":         "Public Distribution System (PDS)",
    "general":     "General Public Grievances",
}


class GrievanceService:
    """
    Manages grievance intent analysis and sequential form filing state.

    Each active grievance session is identified by a `session_id` string.
    The draft dict is stored in `_drafts` and persisted across multiple
    tool-call turns within the same LLM agent loop invocation.
    """

    def __init__(self):
        # [Fix 2] Thread-safe in-memory session draft store
        self._drafts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Session draft management
    # ------------------------------------------------------------------

    def get_draft(self, session_id: str) -> Dict[str, Any]:
        """Retrieve the current draft for a session (creates empty if new)."""
        with self._lock:
            return self._drafts.setdefault(session_id, {})

    def save_draft(self, session_id: str, draft: Dict[str, Any]) -> None:
        """Persist an updated draft back into the session store."""
        with self._lock:
            self._drafts[session_id] = draft

    def clear_draft(self, session_id: str) -> None:
        """Remove a completed/abandoned draft from the store."""
        with self._lock:
            self._drafts.pop(session_id, None)

    # ------------------------------------------------------------------
    # Department classification
    # ------------------------------------------------------------------

    def analyze_department(self, text: str) -> str:
        """Analyze text to classify the target government department."""
        text_lower = text.lower()

        if any(k in text_lower for k in [
            "pacs", "cooperative", "society", "loan", "fertilizer",
            "bye-law", "kisaan credit",
        ]):
            return DEPARTMENTS["pacs"]
        elif any(k in text_lower for k in [
            "insurance", "pmfby", "crop damage", "claim", "yield loss", "premium",
        ]):
            return DEPARTMENTS["pmfby"]
        elif any(k in text_lower for k in [
            "pm-kisan", "subsidy", "seed", "tractor", "agriculture", "soil",
        ]):
            return DEPARTMENTS["agriculture"]
        elif any(k in text_lower for k in [
            "ration", "pds", "rice", "wheat", "fair price shop", "card",
        ]):
            return DEPARTMENTS["pds"]
        else:
            # Default to PACS for rural cooperative assistant
            return DEPARTMENTS["pacs"]

    # ------------------------------------------------------------------
    # Sequential form collection
    # ------------------------------------------------------------------

    def process_grievance_turn(
        self,
        user_text: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """
        [Fix 2] Session-aware sequential form collection loop.

        `session_id` is used to persist draft state across multiple tool-call
        turns within the same LLM agent loop.  A completed form is saved to
        the database and the draft is cleared from the store.
        """
        # Load existing draft for this session
        draft = self.get_draft(session_id)

        # 1. Department Identification (first turn only)
        if not draft.get("department"):
            draft["department"] = self.analyze_department(user_text)

        # 2. Extract phone number heuristically from user text
        text_clean = user_text.strip()
        phone_match = re.search(r'\b[6-9]\d{9}\b', text_clean)
        if phone_match and not draft.get("phone_number"):
            draft["phone_number"] = phone_match.group(0)

        # 3. Sequential field collection — one field prompted per turn

        if not draft.get("citizen_name"):
            if draft.get("prompted_for") == "name":
                draft["citizen_name"] = text_clean
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "name"
                self.save_draft(session_id, draft)
                return {
                    "is_complete": False,
                    "prompt": (
                        f"I see you have a complaint regarding "
                        f"**{draft['department']}**. "
                        f"May I please have your **Full Name** to register the form?"
                    ),
                    "draft": draft,
                }

        if not draft.get("phone_number"):
            if draft.get("prompted_for") == "phone":
                draft["phone_number"] = text_clean
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "phone"
                self.save_draft(session_id, draft)
                return {
                    "is_complete": False,
                    "prompt": (
                        f"Thank you {draft['citizen_name']}. "
                        f"Please share your 10-digit **Phone Number** for status updates."
                    ),
                    "draft": draft,
                }

        if not draft.get("district") or not draft.get("pacs_centre"):
            if draft.get("prompted_for") == "location":
                parts = text_clean.split(",")
                draft["district"] = parts[0].strip()
                draft["pacs_centre"] = (
                    parts[1].strip() if len(parts) > 1
                    else parts[0].strip() + " PACS"
                )
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "location"
                self.save_draft(session_id, draft)
                return {
                    "is_complete": False,
                    "prompt": (
                        "Which **District** and **PACS Centre / Village** does this issue belong to? "
                        "(e.g., *Coimbatore, Periyanaickenpalayam PACS*)"
                    ),
                    "draft": draft,
                }

        if not draft.get("description"):
            if draft.get("prompted_for") == "description":
                draft["description"] = text_clean
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "description"
                self.save_draft(session_id, draft)
                return {
                    "is_complete": False,
                    "prompt": "Please describe your **Grievance / Issue** in detail.",
                    "draft": draft,
                }

        if not draft.get("desired_resolution"):
            if draft.get("prompted_for") == "resolution":
                draft["desired_resolution"] = text_clean
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "resolution"
                self.save_draft(session_id, draft)
                return {
                    "is_complete": False,
                    "prompt": (
                        "What is your **Desired Resolution**? "
                        "(e.g., *Immediate loan disbursal*, *Re-inspection of crop damage*)"
                    ),
                    "draft": draft,
                }

        # 4. Form complete — persist to DB and clear draft
        draft["issue_category"] = draft["department"]
        saved_record = db_service.create_grievance(draft)

        # [Fix 2] Clear completed session draft
        self.clear_draft(session_id)

        return {
            "is_complete": True,
            "tracking_id": saved_record["tracking_id"],
            "prompt": (
                f"✅ Your formal grievance has been filed successfully! "
                f"Tracking ID: **{saved_record['tracking_id']}**.\n"
                f"Department: {saved_record['department']}\n"
                f"PACS Centre: {saved_record['pacs_centre']}\n"
                f"Status: Pending PACS Operator Review.\n"
                f"You can track status anytime on our portal."
            ),
            "draft": saved_record,
        }


grievance_service = GrievanceService()
