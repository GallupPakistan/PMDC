import pandas as pd


def clean_status(status):

    if pd.isna(status):
        return "Unknown"

    status = str(status).lower()

    if "active" in status and "inactive" not in status:
        return "Active"

    if "inactive" in status:
        return "Inactive"

    if "suspend" in status:
        return "Suspended"

    if "cancel" in status:
        return "Cancelled"

    if "shift" in status:
        return "Shifted"

    if "duplicate" in status:
        return "Duplicate"

    if "not_found" in status:
        return "Not Found"

    return "Other"