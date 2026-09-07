"""
grievance_service.py
====================
Sequential Grievance Form Filing and Department Analysis Engine.
Analyzes user complaint intent, asks follow-up questions sequentially,
and commits completed grievance forms to PostgreSQL/SQLite DB.
"""

import re
import os
import uuid
from typing import Dict, Any, Tuple, Optional

from db import db_service


# Department taxonomy
DEPARTMENTS = {
    "pacs": "PACS Cooperative Society",
    "pmfby": "PMFBY Crop Insurance",
    "agriculture": "Agriculture & Farmers Welfare",
    "pds": "Public Distribution System (PDS)",
    "general": "General Public Grievances"
}


class GrievanceService:
    """
    Manages grievance intent analysis and sequential form filing state.
    """

    def analyze_department(self, text: str) -> str:
        """Analyze text to classify the target government department."""
        text_lower = text.lower()

        if any(k in text_lower for k in ["pacs", "cooperative", "society", "loan", "fertilizer", "bye-law", "kisaan credit"]):
            return DEPARTMENTS["pacs"]
        elif any(k in text_lower for k in ["insurance", "pmfby", "crop damage", "claim", "yield loss", "premium"]):
            return DEPARTMENTS["pmfby"]
        elif any(k in text_lower for k in ["pm-kisan", "subsidy", "seed", "tractor", "agriculture", "soil"]):
            return DEPARTMENTS["agriculture"]
        elif any(k in text_lower for k in ["ration", "pds", "rice", "wheat", "fair price shop", "card"]):
            return DEPARTMENTS["pds"]
        else:
            return DEPARTMENTS["pacs"] # Default to PACS for rural cooperative assistant

    def process_grievance_turn(self, user_text: str, draft: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Sequential form collection loop.
        Identifies department and prompts for missing fields one by one.
        """
        if draft is None:
            draft = {}

        # 1. Department Identification
        if not draft.get("department"):
            draft["department"] = self.analyze_department(user_text)

        # 2. Extract initial details from user text if available
        text_clean = user_text.strip()

        # Simple extraction heuristics
        phone_match = re.search(r'\b[6-9]\d{9}\b', text_clean)
        if phone_match and not draft.get("phone_number"):
            draft["phone_number"] = phone_match.group(0)

        # 3. Check for missing required fields in sequence
        if not draft.get("citizen_name"):
            # If the user just gave their name or general response
            if draft.get("prompted_for") == "name":
                draft["citizen_name"] = text_clean
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "name"
                return {
                    "is_complete": False,
                    "prompt": f"I see you have a complaint regarding **{draft['department']}**. May I please have your **Full Name** to register the form?",
                    "draft": draft
                }

        if not draft.get("phone_number"):
            if draft.get("prompted_for") == "phone":
                draft["phone_number"] = text_clean
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "phone"
                return {
                    "is_complete": False,
                    "prompt": f"Thank you {draft['citizen_name']}. Please share your 10-digit **Phone Number** for status updates.",
                    "draft": draft
                }

        if not draft.get("district") or not draft.get("pacs_centre"):
            if draft.get("prompted_for") == "location":
                parts = text_clean.split(",")
                draft["district"] = parts[0].strip()
                draft["pacs_centre"] = parts[1].strip() if len(parts) > 1 else parts[0].strip() + " PACS"
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "location"
                return {
                    "is_complete": False,
                    "prompt": "Which **District** and **PACS Centre / Village** does this issue belong to? (e.g., *Coimbatore, Periyanaickenpalayam PACS*)",
                    "draft": draft
                }

        if not draft.get("description"):
            if draft.get("prompted_for") == "description":
                draft["description"] = text_clean
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "description"
                return {
                    "is_complete": False,
                    "prompt": "Please describe your **Grievance / Issue** in detail.",
                    "draft": draft
                }

        if not draft.get("desired_resolution"):
            if draft.get("prompted_for") == "resolution":
                draft["desired_resolution"] = text_clean
                draft["prompted_for"] = None
            else:
                draft["prompted_for"] = "resolution"
                return {
                    "is_complete": False,
                    "prompt": "What is your **Desired Resolution**? (e.g., *Immediate loan disbursal*, *Re-inspection of crop damage*)",
                    "draft": draft
                }

        # 4. Form Complete -> Store in DB
        draft["issue_category"] = draft["department"]
        saved_record = db_service.create_grievance(draft)

        return {
            "is_complete": True,
            "tracking_id": saved_record["tracking_id"],
            "prompt": f"✅ Your formal grievance has been filed successfully! Tracking ID: **{saved_record['tracking_id']}**.\n"
                      f"Department: {saved_record['department']}\n"
                      f"PACS Centre: {saved_record['pacs_centre']}\n"
                      f"Status: Pending PACS Operator Review.\n"
                      f"You can track status anytime on our portal.",
            "draft": saved_record
        }


grievance_service = GrievanceService()
