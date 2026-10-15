"""
Single source of truth for build/runtime flavor.

Valid values used currently:
- "DEMO"
- "v41"
"""

BUILD_FLAVOR = "v41"

# Optional runtime override for patternId whitelist filter in GUI/runtime code.
# - None  -> follow flavor default
# - True  -> force whitelist ON
# - False -> force whitelist OFF
PATTERN_ID_WHITELIST_OVERRIDE = None
